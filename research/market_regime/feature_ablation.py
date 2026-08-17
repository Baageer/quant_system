"""Measure the standalone and incremental value of frozen trend features.

The input tables must be the already split ``*_model_samples_*`` exports from
``probability_dataset``.  This module intentionally does not regenerate
labels, select candidate features, or change the train/validation/test split.
For every variant, preprocessing is fit on training data only, regularization
and calibration are chosen using validation data, and test metrics are then
reported once.

Example::

    python -m research.market_regime.feature_ablation \
        --dataset-dir output/datasets \
        --dataset-id hs300_probability_dataset \
        --run-id hs300_feature_ablation_v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from research.market_regime.logistic_model import (
    MODEL_METADATA_COLUMNS,
    infer_feature_columns,
    load_dataset_splits,
    train_logistic_experiment,
)


DEFAULT_MODES = ("full", "single_feature", "leave_one_out")
FINAL_MODEL_NAMES = {"none": "logistic_raw", "multinomial": "logistic_calibrated"}
SUMMARY_METRICS = (
    "trend_macro_pr_auc",
    "trend_macro_roc_auc",
    "log_loss",
    "macro_brier",
    "precision_at_top_5pct_up",
    "precision_at_top_5pct_down",
)


def _parse_c_values(text: str) -> List[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("c_values must contain positive numbers")
    return values


def _parse_modes(text: str) -> Tuple[str, ...]:
    modes = tuple(value.strip() for value in text.split(",") if value.strip())
    unknown = sorted(set(modes).difference(DEFAULT_MODES))
    if not modes or unknown:
        raise ValueError("modes must be selected from {}".format(", ".join(DEFAULT_MODES)))
    return modes


def _parse_features(text: Optional[str], available: Sequence[str]) -> List[str]:
    if text is None:
        return list(available)
    requested = [value.strip() for value in text.split(",") if value.strip()]
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError("Requested features are not model inputs: {}".format(unknown))
    return list(dict.fromkeys(requested))


def select_feature_columns(
    feature_columns: Sequence[str],
    mode: str,
    feature: Optional[str] = None,
) -> List[str]:
    """Return the frozen model inputs for one ablation variant."""

    available = list(feature_columns)
    if mode == "full":
        return available
    if feature not in available:
        raise ValueError("feature must be one of the supplied feature_columns")
    if mode == "single_feature":
        return [feature]
    if mode == "leave_one_out":
        selected = [column for column in available if column != feature]
        if not selected:
            raise ValueError("leave_one_out requires at least two feature columns")
        return selected
    raise ValueError("Unknown ablation mode: {}".format(mode))


def restrict_samples(samples: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Keep identifiers, labels and exactly the requested model inputs."""

    missing = [column for column in feature_columns if column not in samples.columns]
    if missing:
        raise ValueError("Samples are missing requested feature columns: {}".format(missing))
    metadata = [column for column in samples.columns if column in MODEL_METADATA_COLUMNS]
    return samples.loc[:, metadata + list(feature_columns)].copy()


def _final_metrics(report: Mapping[str, object], split: str, calibration: str) -> Dict[str, object]:
    model_name = FINAL_MODEL_NAMES[calibration]
    metrics = report["metrics"]
    row = metrics.loc[(metrics["split"] == split) & (metrics["model"] == model_name)]
    if len(row) != 1:
        raise ValueError("Expected exactly one {} metrics row for {}".format(model_name, split))
    return row.iloc[0].to_dict()


def _base_rate_metrics(report: Mapping[str, object], split: str) -> Dict[str, object]:
    metrics = report["metrics"]
    row = metrics.loc[(metrics["split"] == split) & (metrics["model"] == "base_rate")]
    if len(row) != 1:
        raise ValueError("Expected exactly one base_rate metrics row")
    return row.iloc[0].to_dict()


def run_feature_ablation(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    modes: Sequence[str] = DEFAULT_MODES,
    features: Optional[Sequence[str]] = None,
    c_values: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    calibration: str = "multinomial",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run full, standalone and leave-one-feature-out Logistic experiments.

    The returned rows contain validation and test metrics for each variant.
    Test deltas are measured relative to the separately fitted full model;
    positive deltas improve PR-AUC/ROC-AUC/precision, while negative deltas
    improve loss and Brier score.
    """

    if calibration not in FINAL_MODEL_NAMES:
        raise ValueError("calibration must be 'none' or 'multinomial'")
    all_features = infer_feature_columns(train_samples)
    selected_features = list(features) if features is not None else all_features
    unknown = sorted(set(selected_features).difference(all_features))
    if unknown:
        raise ValueError("Features are not numeric model inputs: {}".format(unknown))
    variants = [("full", None, select_feature_columns(all_features, "full"))]
    for mode in modes:
        if mode == "full":
            continue
        for feature in selected_features:
            variants.append((mode, feature, select_feature_columns(all_features, mode, feature)))

    rows = []
    base_rate_test = None
    for mode, feature, feature_set in tqdm(
        variants,
        desc="Running feature ablations",
        total=len(variants),
        unit="model",
        disable=not show_progress,
    ):
        report = train_logistic_experiment(
            restrict_samples(train_samples, feature_set),
            restrict_samples(validation_samples, feature_set),
            restrict_samples(test_samples, feature_set),
            c_values=c_values,
            calibration=calibration,
        )
        validation = _final_metrics(report, "validation", calibration)
        test = _final_metrics(report, "test", calibration)
        if base_rate_test is None:
            base_rate_test = _base_rate_metrics(report, "test")
        row = {
            "analysis_type": mode,
            "feature": feature,
            "feature_count": len(feature_set),
            "feature_columns": json.dumps(feature_set, ensure_ascii=False),
            "selected_c": float(report["selected_c"]),
        }
        row.update({"validation_{}".format(key): value for key, value in validation.items() if key in SUMMARY_METRICS})
        row.update({"test_{}".format(key): value for key, value in test.items() if key in SUMMARY_METRICS})
        rows.append(row)

    results = pd.DataFrame(rows)
    full = results.loc[results["analysis_type"] == "full"].iloc[0]
    for metric in SUMMARY_METRICS:
        test_column = "test_{}".format(metric)
        results["test_delta_vs_full_{}".format(metric)] = results[test_column] - full[test_column]
        results["test_base_rate_{}".format(metric)] = base_rate_test[metric]
        results["test_delta_vs_base_rate_{}".format(metric)] = results[test_column] - base_rate_test[metric]
    return results.sort_values(["analysis_type", "feature"], na_position="first").reset_index(drop=True)


def export_feature_ablation(
    results: pd.DataFrame,
    output_dir: Union[str, Path],
    run_id: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Export the summary table and a manifest that records frozen inputs."""

    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "trend_feature_ablation_metrics_{}.csv".format(identifier)
    results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis_id": identifier,
        "created_at": datetime.now().isoformat(),
        "analysis_type": "frozen_feature_ablation",
        "metric_delta_convention": "variant test metric minus full-model test metric",
        "positive_delta_improves": ["trend_macro_pr_auc", "trend_macro_roc_auc", "precision_at_top_5pct_up", "precision_at_top_5pct_down"],
        "negative_delta_improves": ["log_loss", "macro_brier"],
        "paths": {"metrics": str(summary_path)},
    }
    if metadata:
        manifest.update(dict(metadata))
    manifest_path = destination / "trend_feature_ablation_manifest_{}.json".format(identifier)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"metrics": summary_path, "manifest": manifest_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate standalone and leave-one-out values of frozen trend features.")
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--features", default=None, help="Optional comma-separated subset for single/leave-one-out runs.")
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3,10")
    parser.add_argument("--calibration", choices=("none", "multinomial"), default="multinomial")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = parser.parse_args(argv)

    train_samples, validation_samples, test_samples = load_dataset_splits(args.dataset_dir, args.dataset_id)
    all_features = infer_feature_columns(train_samples)
    modes = _parse_modes(args.modes)
    selected_features = _parse_features(args.features, all_features)
    c_values = _parse_c_values(args.c_values)
    results = run_feature_ablation(
        train_samples,
        validation_samples,
        test_samples,
        modes=modes,
        features=selected_features,
        c_values=c_values,
        calibration=args.calibration,
        show_progress=not args.no_progress,
    )
    paths = export_feature_ablation(
        results,
        args.output_dir or args.dataset_dir,
        args.run_id,
        {
            "dataset_id": args.dataset_id,
            "dataset_dir": str(args.dataset_dir),
            "modes": list(modes),
            "features": selected_features,
            "c_values": c_values,
            "calibration": args.calibration,
            "validation_selection_metric": "trend_macro_pr_auc, then log_loss",
        },
    )
    print("Feature ablation metrics: {}".format(paths["metrics"]))
    print("Feature ablation manifest: {}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
