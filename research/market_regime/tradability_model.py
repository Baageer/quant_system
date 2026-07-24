"""Train a secondary tradability classifier and combine it with ``p_up``.

The classifier predicts ``tradability_target_10d`` using the frozen feature
list from an existing trend Logistic model.  Future-return columns remain
labels only.  Validation selects the regularization strength and calibrates
the binary probability; test is evaluated once with all decisions frozen.
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.market_regime.logistic_model import (
    _average_precision,
    _binary_auc_ranks,
    fit_preprocessor,
    transform_features,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols


TARGET_COLUMN = "tradability_target_10d"
POSITIVE_CLASS = "tradable_up"
NEGATIVE_CLASS = "not_tradable"


def load_trend_feature_columns(dataset_dir: Path, trend_model_id: str) -> List[str]:
    path = dataset_dir / "trend_logistic_manifest_{}.json".format(trend_model_id)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    features = list(manifest["feature_columns"])
    if not features:
        raise ValueError("Trend model manifest has no feature columns")
    return features


def load_tradability_split(
    dataset_dir: Path,
    dataset_id: str,
    split: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Load only inputs and labels required for the secondary classifier."""

    path = dataset_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
    required = ["date", "symbol", "target", "sample_weight", "auxiliary_label_eligible", TARGET_COLUMN]
    usecols = list(dict.fromkeys(required + list(feature_columns)))
    samples = pd.read_csv(path, usecols=usecols, dtype={"symbol": str})
    samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
    samples["symbol"] = normalize_a_share_symbols(samples["symbol"])
    valid = (
        samples["auxiliary_label_eligible"].astype(str).str.lower().eq("true")
        & samples[TARGET_COLUMN].isin([POSITIVE_CLASS, NEGATIVE_CLASS])
    )
    return samples.loc[valid].dropna(subset=["date", "symbol"]).reset_index(drop=True)


def _weights(samples: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(samples.get("sample_weight", 1.0), errors="coerce")
    result = np.asarray(values, dtype=float)
    result[~np.isfinite(result) | (result <= 0)] = 1.0
    return result


def _balanced_weights(labels: np.ndarray, event_weights: np.ndarray) -> np.ndarray:
    positive_weight = float(np.sum(event_weights[labels]))
    negative_weight = float(np.sum(event_weights[~labels]))
    if positive_weight <= 0 or negative_weight <= 0:
        raise ValueError("Training split must contain both tradability classes")
    total = positive_weight + negative_weight
    factors = {True: total / (2.0 * positive_weight), False: total / (2.0 * negative_weight)}
    return event_weights * np.asarray([factors[bool(label)] for label in labels], dtype=float)


def _fit_binary(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(C=float(c_value), penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(features, labels.astype(int), sample_weight=weights)
    return model


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    position = list(model.classes_).index(1)
    return probabilities[:, position]


def evaluate_binary(targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, split: str, model: str) -> Dict[str, object]:
    """Evaluate a binary probability with event-weighted metrics."""

    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    top_count = max(1, int(np.ceil(0.05 * len(targets))))
    top = np.argsort(-clipped, kind="mergesort")[:top_count]
    return {
        "split": split,
        "model": model,
        "sample_count": int(len(targets)),
        "weighted_sample_count": float(np.sum(weights)),
        "base_rate": float(np.sum(weights[targets]) / np.sum(weights)),
        "log_loss": float(np.sum(weights * -(targets * np.log(clipped) + (~targets) * np.log(1 - clipped))) / np.sum(weights)),
        "brier": float(np.sum(weights * (clipped - targets.astype(float)) ** 2) / np.sum(weights)),
        "pr_auc": _average_precision(targets, clipped, weights),
        "roc_auc": _binary_auc_ranks(targets, clipped, weights),
        "precision_at_top_5pct": float(np.sum(weights[top] * targets[top]) / np.sum(weights[top])),
    }


def fit_binary_calibrator(validation_probabilities: np.ndarray, validation_labels: np.ndarray, weights: np.ndarray):
    logits = np.log(np.clip(validation_probabilities, 1e-12, 1 - 1e-12) / np.clip(1 - validation_probabilities, 1e-12, 1.0))
    calibrator = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(logits.reshape(-1, 1), validation_labels.astype(int), sample_weight=weights)
    return calibrator


def apply_binary_calibrator(calibrator, probabilities: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1 - 1e-12) / np.clip(1 - probabilities, 1e-12, 1.0))
    return _positive_probability(calibrator, logits.reshape(-1, 1))


def train_tradability_experiment(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    feature_columns: Sequence[str],
    c_values: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
) -> Dict[str, object]:
    """Fit, select and calibrate the binary secondary model."""

    preprocessor = fit_preprocessor(train_samples, feature_columns)
    train_features = transform_features(train_samples, preprocessor)
    validation_features = transform_features(validation_samples, preprocessor)
    test_features = transform_features(test_samples, preprocessor)
    train_labels = np.asarray(train_samples[TARGET_COLUMN].eq(POSITIVE_CLASS), dtype=bool)
    validation_labels = np.asarray(validation_samples[TARGET_COLUMN].eq(POSITIVE_CLASS), dtype=bool)
    test_labels = np.asarray(test_samples[TARGET_COLUMN].eq(POSITIVE_CLASS), dtype=bool)
    train_weights = _weights(train_samples)
    validation_weights = _weights(validation_samples)
    test_weights = _weights(test_samples)
    fit_weights = _balanced_weights(train_labels, train_weights)

    candidates = []
    grid = []
    for c_value in c_values:
        if float(c_value) <= 0:
            raise ValueError("All C values must be positive")
        classifier = _fit_binary(train_features, train_labels, fit_weights, float(c_value))
        validation_probabilities = _positive_probability(classifier, validation_features)
        metrics = evaluate_binary(validation_labels, validation_probabilities, validation_weights, "validation", "tradability_raw")
        metrics["c_value"] = float(c_value)
        candidates.append((float(c_value), classifier, validation_probabilities))
        grid.append(metrics)
    grid_frame = pd.DataFrame(grid).sort_values(["pr_auc", "log_loss", "c_value"], ascending=[False, True, True])
    selected_c = float(grid_frame.iloc[0]["c_value"])
    _, classifier, validation_raw = next(item for item in candidates if item[0] == selected_c)
    test_raw = _positive_probability(classifier, test_features)
    calibrator = fit_binary_calibrator(validation_raw, validation_labels, validation_weights)
    validation_probabilities = apply_binary_calibrator(calibrator, validation_raw)
    test_probabilities = apply_binary_calibrator(calibrator, test_raw)
    metrics = pd.DataFrame([
        evaluate_binary(validation_labels, validation_raw, validation_weights, "validation", "tradability_raw"),
        evaluate_binary(test_labels, test_raw, test_weights, "test", "tradability_raw"),
        evaluate_binary(validation_labels, validation_probabilities, validation_weights, "validation", "tradability_calibrated"),
        evaluate_binary(test_labels, test_probabilities, test_weights, "test", "tradability_calibrated"),
    ])
    return {
        "selected_c": selected_c,
        "feature_columns": list(feature_columns),
        "preprocessor": preprocessor,
        "classifier": classifier,
        "calibrator": calibrator,
        "validation_grid": grid_frame,
        "metrics": metrics,
        "validation_predictions": build_tradability_predictions(validation_samples, validation_probabilities, "tradability_calibrated"),
        "test_predictions": build_tradability_predictions(test_samples, test_probabilities, "tradability_calibrated"),
    }


def build_tradability_predictions(samples: pd.DataFrame, probabilities: np.ndarray, model_name: str) -> pd.DataFrame:
    result = samples.loc[:, ["date", "symbol", "target", TARGET_COLUMN, "sample_weight"]].copy()
    result["p_tradable"] = probabilities
    result["predicted_tradability"] = np.where(probabilities >= 0.5, POSITIVE_CLASS, NEGATIVE_CLASS)
    result["model"] = model_name
    return result


def load_trend_predictions(dataset_dir: Path, trend_model_id: str, split: str) -> pd.DataFrame:
    path = dataset_dir / "trend_logistic_{}_predictions_{}.csv".format(split, trend_model_id)
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    return frame.dropna(subset=["date", "symbol"])


def combine_entry_scores(
    tradability_predictions: pd.DataFrame,
    trend_predictions: pd.DataFrame,
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Merge frozen ``p_up`` and calibrated ``p_tradable`` into entry score."""

    trend = trend_predictions.loc[:, ["date", "symbol", "p_up", "p_down", "p_none"]].copy()
    keys = ["date", "symbol"]
    if tradability_predictions.duplicated(keys).any() or trend.duplicated(keys).any():
        raise ValueError("Entry-score inputs must have unique date and symbol keys")
    combined = tradability_predictions.merge(trend, on=keys, how="inner")
    if combined.empty:
        raise ValueError("No rows matched when combining {} entry scores".format(split))
    combined["entry_score"] = combined["p_up"] * combined["p_tradable"]
    labels = np.asarray(combined[TARGET_COLUMN].eq(POSITIVE_CLASS), dtype=bool)
    weights = _weights(combined)
    metrics = evaluate_binary(labels, np.asarray(combined["entry_score"], dtype=float), weights, split, "entry_score")
    metrics["matched_prediction_count"] = int(len(combined))
    return combined.sort_values(["date", "symbol"]).reset_index(drop=True), metrics


def export_tradability_experiment(
    report: Mapping[str, object],
    combined_validation: pd.DataFrame,
    combined_test: pd.DataFrame,
    entry_metrics: Sequence[Mapping[str, object]],
    output_dir: Path,
    run_id: str,
    metadata: Mapping[str, object],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_grid": report["validation_grid"],
        "metrics": pd.concat([report["metrics"], pd.DataFrame(entry_metrics)], ignore_index=True),
        "validation_predictions": report["validation_predictions"],
        "test_predictions": report["test_predictions"],
        "validation_entry_scores": combined_validation,
        "test_entry_scores": combined_test,
    }
    paths = {}
    for name, table in tables.items():
        path = output_dir / "trend_tradability_{}_{}.csv".format(name, run_id)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)
    model_path = output_dir / "trend_tradability_model_{}.pkl".format(run_id)
    with model_path.open("wb") as handle:
        pickle.dump({key: report[key] for key in ("feature_columns", "preprocessor", "classifier", "calibrator")}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    paths["model"] = str(model_path)
    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "model_type": "binary_l2_logistic_regression",
        "target": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "selected_c": report["selected_c"],
        "combination": "entry_score = p_up_from_frozen_trend_model * calibrated_p_tradable",
        "feature_columns": report["feature_columns"],
        "paths": paths,
        **dict(metadata),
    }
    manifest_path = output_dir / "trend_tradability_manifest_{}.json".format(run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def _parse_c_values(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one C value is required")
    return values


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="hs300_probability_dataset")
    parser.add_argument("--trend-model-id", default="hs300_logistic_v2")
    parser.add_argument("--run-id", default="hs300_tradability_v1")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3")
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    features = load_trend_feature_columns(dataset_dir, args.trend_model_id)
    splits = [load_tradability_split(dataset_dir, args.dataset_id, split, features) for split in ("train", "validation", "test")]
    report = train_tradability_experiment(*splits, feature_columns=features, c_values=_parse_c_values(args.c_values))
    trend_validation = load_trend_predictions(dataset_dir, args.trend_model_id, "validation")
    trend_test = load_trend_predictions(dataset_dir, args.trend_model_id, "test")
    combined_validation, validation_entry_metrics = combine_entry_scores(report["validation_predictions"], trend_validation, "validation")
    combined_test, test_entry_metrics = combine_entry_scores(report["test_predictions"], trend_test, "test")
    paths = export_tradability_experiment(
        report, combined_validation, combined_test, [validation_entry_metrics, test_entry_metrics], Path(args.output_dir), args.run_id,
        {"dataset_id": args.dataset_id, "trend_model_id": args.trend_model_id, "c_values": _parse_c_values(args.c_values), "selection_metric": "validation_pr_auc_then_log_loss"},
    )
    print("Selected C={}; test entry-score PR-AUC={:.6f}; manifest={}".format(report["selected_c"], test_entry_metrics["pr_auc"], paths["manifest"]))


if __name__ == "__main__":
    main()
