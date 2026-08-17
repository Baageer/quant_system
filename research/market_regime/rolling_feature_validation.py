"""Freeze validation-selected feature subsets and evaluate later rolling windows.

Candidate sets are selected once from a completed feature-ablation summary:

* ``core`` keeps the top standalone features by validation PR-AUC.
* ``validation_pruned`` removes a feature only when its leave-one-out
  validation PR-AUC and Log Loss both improve over the full model.

The subsequent rolling folds never re-select features.  Each fold uses an
expanding training history, a preceding calibration window for regularization
selection and probability calibration, then reports the following independent
test window once.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from research.market_regime.feature_ablation import FINAL_MODEL_NAMES, SUMMARY_METRICS, restrict_samples
from research.market_regime.logistic_model import infer_feature_columns, load_dataset_splits, train_logistic_experiment
from tqdm.auto import tqdm


def _metrics_path(dataset_dir: Union[str, Path], ablation_id: str) -> Path:
    return Path(dataset_dir) / "trend_feature_ablation_metrics_{}.csv".format(ablation_id)


def load_ablation_metrics(dataset_dir: Union[str, Path], ablation_id: str) -> pd.DataFrame:
    """Load a feature-ablation report with its validation metrics intact."""

    path = _metrics_path(dataset_dir, ablation_id)
    if not path.is_file():
        raise FileNotFoundError("Missing feature ablation metrics: {}".format(path))
    metrics = pd.read_csv(path)
    required = {"analysis_type", "feature", "validation_trend_macro_pr_auc", "validation_log_loss"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError("Feature ablation metrics missing columns: {}".format(missing))
    return metrics


def build_validation_frozen_candidates(
    ablation_metrics: pd.DataFrame,
    full_feature_columns: Sequence[str],
    core_count: int = 8,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Freeze core and pruned sets using validation metrics only.

    The full set is included as a rolling benchmark.  Candidate selection does
    not inspect any ``test_*`` column in the ablation report.
    """

    if int(core_count) < 1:
        raise ValueError("core_count must be positive")
    full_features = list(full_feature_columns)
    full_rows = ablation_metrics.loc[ablation_metrics["analysis_type"].eq("full")]
    if len(full_rows) != 1:
        raise ValueError("Expected exactly one full-model row in feature ablation metrics")
    full = full_rows.iloc[0]
    single = ablation_metrics.loc[ablation_metrics["analysis_type"].eq("single_feature")].copy()
    single = single.loc[single["feature"].isin(full_features)]
    if len(single) < int(core_count):
        raise ValueError("Not enough standalone feature rows for core_count={}".format(core_count))
    single = single.sort_values(
        ["validation_trend_macro_pr_auc", "validation_log_loss", "feature"],
        ascending=[False, True, True],
    )
    core_features = single.head(int(core_count))["feature"].tolist()

    leave_one_out = ablation_metrics.loc[ablation_metrics["analysis_type"].eq("leave_one_out")].copy()
    leave_one_out = leave_one_out.loc[leave_one_out["feature"].isin(full_features)]
    improves_pr_auc = leave_one_out["validation_trend_macro_pr_auc"] >= float(full["validation_trend_macro_pr_auc"])
    improves_log_loss = leave_one_out["validation_log_loss"] <= float(full["validation_log_loss"])
    strictly_improves = (
        leave_one_out["validation_trend_macro_pr_auc"] > float(full["validation_trend_macro_pr_auc"])
    ) | (leave_one_out["validation_log_loss"] < float(full["validation_log_loss"]))
    removed_features = sorted(leave_one_out.loc[improves_pr_auc & improves_log_loss & strictly_improves, "feature"].tolist())
    pruned_features = [feature for feature in full_features if feature not in removed_features]
    if not pruned_features:
        raise ValueError("Validation pruning removed every model feature")

    candidates = {
        "full": full_features,
        "core_{}".format(int(core_count)): core_features,
        "validation_pruned": pruned_features,
    }
    rows = []
    for candidate, features in candidates.items():
        for rank, feature in enumerate(features, start=1):
            rows.append(
                {
                    "candidate": candidate,
                    "feature": feature,
                    "feature_rank": rank,
                    "feature_count": len(features),
                    "selection_rule": (
                        "full frozen set"
                        if candidate == "full"
                        else "top standalone validation PR-AUC, validation Log Loss tie-break"
                        if candidate.startswith("core_")
                        else "remove only when leave-one-out validation PR-AUC and Log Loss both improve"
                    ),
                }
            )
    return candidates, pd.DataFrame(rows)


def _dated_samples(samples: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["label_available_date"] = pd.to_datetime(result["label_available_date"], errors="coerce")
    if result[["date", "label_available_date"]].isnull().any().any():
        raise ValueError("Rolling validation requires valid date and label_available_date columns")
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_rolling_folds(
    all_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    calibration_days: int = 252,
    test_days: int = 252,
    min_test_days: int = 126,
) -> List[Dict[str, object]]:
    """Create non-overlapping test folds anchored to the original test period."""

    if min(calibration_days, test_days, min_test_days) < 1:
        raise ValueError("rolling day counts must be positive")
    available_dates = pd.Index(sorted(all_samples["date"].drop_duplicates()))
    evaluation_dates = pd.Index(sorted(test_samples["date"].drop_duplicates()))
    folds = []
    for offset in range(0, len(evaluation_dates), int(test_days)):
        test_dates = evaluation_dates[offset : offset + int(test_days)]
        if len(test_dates) < int(min_test_days):
            continue
        test_start = test_dates[0]
        prior_dates = available_dates[available_dates < test_start]
        calibration_dates = prior_dates[-int(calibration_days) :]
        if len(calibration_dates) < int(calibration_days):
            continue
        folds.append(
            {
                "fold": len(folds) + 1,
                "calibration_start": calibration_dates[0],
                "calibration_end": calibration_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            }
        )
    if not folds:
        raise ValueError("No rolling folds available with the requested calibration/test windows")
    return folds


def _fold_samples(all_samples: pd.DataFrame, test_samples: pd.DataFrame, fold: Mapping[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_start = pd.Timestamp(fold["calibration_start"])
    test_start = pd.Timestamp(fold["test_start"])
    test_end = pd.Timestamp(fold["test_end"])
    train = all_samples.loc[
        (all_samples["date"] < calibration_start) & (all_samples["label_available_date"] < calibration_start)
    ]
    calibration = all_samples.loc[
        (all_samples["date"] >= calibration_start)
        & (all_samples["date"] < test_start)
        & (all_samples["label_available_date"] < test_start)
    ]
    test = test_samples.loc[(test_samples["date"] >= test_start) & (test_samples["date"] <= test_end)]
    if min(len(train), len(calibration), len(test)) == 0:
        raise ValueError("Fold {} has an empty train, calibration or test sample".format(fold["fold"]))
    return train, calibration, test


def _final_test_metrics(report: Mapping[str, object], calibration: str) -> Mapping[str, object]:
    model_name = FINAL_MODEL_NAMES[calibration]
    rows = report["metrics"].loc[(report["metrics"]["split"] == "test") & (report["metrics"]["model"] == model_name)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one final test metrics row")
    return rows.iloc[0].to_dict()


def run_rolling_feature_validation(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    candidates: Mapping[str, Sequence[str]],
    calibration_days: int = 252,
    test_days: int = 252,
    min_test_days: int = 126,
    c_values: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    calibration: str = "multinomial",
    show_progress: bool = False,
    candidate_names: Optional[Sequence[str]] = None,
    fold_numbers: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Retrain frozen candidate sets across later rolling out-of-sample folds."""

    if calibration not in FINAL_MODEL_NAMES:
        raise ValueError("calibration must be 'none' or 'multinomial'")
    all_samples = _dated_samples(pd.concat([train_samples, validation_samples, test_samples], ignore_index=True))
    evaluation_samples = _dated_samples(test_samples)
    folds = build_rolling_folds(all_samples, evaluation_samples, calibration_days, test_days, min_test_days)
    if candidate_names is not None:
        requested_candidates = list(candidate_names)
        unknown_candidates = sorted(set(requested_candidates).difference(candidates))
        if unknown_candidates:
            raise ValueError("Unknown candidate names: {}".format(unknown_candidates))
        candidates = {name: candidates[name] for name in requested_candidates}
    if fold_numbers is not None:
        requested_folds = {int(value) for value in fold_numbers}
        folds = [fold for fold in folds if int(fold["fold"]) in requested_folds]
        if not folds:
            raise ValueError("No requested rolling folds are available")
    tasks = [(fold, candidate, list(features)) for fold in folds for candidate, features in candidates.items()]
    rows = []
    for fold, candidate, features in tqdm(
        tasks,
        desc="Running rolling feature validation",
        total=len(tasks),
        unit="model",
        disable=not show_progress,
    ):
        fold_train, fold_validation, fold_test = _fold_samples(all_samples, evaluation_samples, fold)
        report = train_logistic_experiment(
            restrict_samples(fold_train, features),
            restrict_samples(fold_validation, features),
            restrict_samples(fold_test, features),
            c_values=c_values,
            calibration=calibration,
        )
        metrics = _final_test_metrics(report, calibration)
        row = {
            "fold": fold["fold"],
            "candidate": candidate,
            "feature_count": len(features),
            "calibration_start": fold["calibration_start"],
            "calibration_end": fold["calibration_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "train_sample_count": len(fold_train),
            "calibration_sample_count": len(fold_validation),
            "test_sample_count": len(fold_test),
            "selected_c": float(report["selected_c"]),
        }
        row.update({metric: metrics[metric] for metric in SUMMARY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold", "candidate"]).reset_index(drop=True)


def summarize_rolling_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pool fold means and compare every frozen candidate with full."""

    metric_columns = [metric for metric in SUMMARY_METRICS if metric in fold_metrics.columns]
    summary = fold_metrics.groupby("candidate", as_index=False)[metric_columns].mean()
    full = summary.loc[summary["candidate"].eq("full")]
    if len(full) == 1:
        full = full.iloc[0]
        for metric in metric_columns:
            summary["delta_vs_full_{}".format(metric)] = summary[metric] - full[metric]
    elif len(full) > 1:
        raise ValueError("Rolling results contain multiple full candidates")
    return summary.sort_values("candidate").reset_index(drop=True)


def export_rolling_feature_validation(
    candidates: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    output_dir: Union[str, Path],
    run_id: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Write frozen candidate definitions, fold metrics, summary and manifest."""

    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "candidates": ("trend_feature_rolling_candidates", candidates),
        "fold_metrics": ("trend_feature_rolling_metrics", fold_metrics),
        "summary": ("trend_feature_rolling_summary", summarize_rolling_metrics(fold_metrics)),
    }
    paths = {}
    for key, (prefix, frame) in tables.items():
        path = destination / "{}_{}.csv".format(prefix, identifier)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    manifest = {
        "analysis_id": identifier,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "validation_frozen_feature_rolling_evaluation",
        "paths": {key: str(path) for key, path in paths.items()},
    }
    if metadata:
        manifest.update(dict(metadata))
    manifest_path = destination / "trend_feature_rolling_manifest_{}.json".format(identifier)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _parse_c_values(text: str) -> List[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("c_values must contain positive numbers")
    return values


def _parse_text_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    values = [value.strip() for value in text.split(",") if value.strip()]
    return values or None


def _parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    return values or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze validation-selected features and evaluate later rolling windows.")
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--core-count", type=int, default=8)
    parser.add_argument("--calibration-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--min-test-days", type=int, default=126)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3,10")
    parser.add_argument("--calibration", choices=("none", "multinomial"), default="multinomial")
    parser.add_argument("--candidates", default=None, help="Optional comma-separated frozen candidate names.")
    parser.add_argument("--folds", default=None, help="Optional comma-separated rolling fold numbers.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = parser.parse_args(argv)

    train_samples, validation_samples, test_samples = load_dataset_splits(args.dataset_dir, args.dataset_id)
    full_features = infer_feature_columns(train_samples)
    ablation_metrics = load_ablation_metrics(args.dataset_dir, args.ablation_id)
    candidate_sets, candidate_table = build_validation_frozen_candidates(ablation_metrics, full_features, args.core_count)
    c_values = _parse_c_values(args.c_values)
    fold_metrics = run_rolling_feature_validation(
        train_samples,
        validation_samples,
        test_samples,
        candidate_sets,
        calibration_days=args.calibration_days,
        test_days=args.test_days,
        min_test_days=args.min_test_days,
        c_values=c_values,
        calibration=args.calibration,
        show_progress=not args.no_progress,
        candidate_names=_parse_text_list(args.candidates),
        fold_numbers=_parse_int_list(args.folds),
    )
    paths = export_rolling_feature_validation(
        candidate_table,
        fold_metrics,
        args.output_dir or args.dataset_dir,
        args.run_id,
        {
            "dataset_id": args.dataset_id,
            "dataset_dir": str(args.dataset_dir),
            "ablation_id": args.ablation_id,
            "candidate_sets": candidate_sets,
            "core_count": args.core_count,
            "calibration_days": args.calibration_days,
            "test_days": args.test_days,
            "min_test_days": args.min_test_days,
            "c_values": c_values,
            "calibration": args.calibration,
            "requested_candidates": _parse_text_list(args.candidates),
            "requested_folds": _parse_int_list(args.folds),
            "feature_selection_data": "feature ablation validation metrics only",
            "rolling_protocol": "expanding train; preceding calibration window; non-overlapping later test windows",
        },
    )
    print("Frozen candidates: {}".format(paths["candidates"]))
    print("Rolling metrics: {}".format(paths["fold_metrics"]))
    print("Rolling summary: {}".format(paths["summary"]))
    print("Rolling manifest: {}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
