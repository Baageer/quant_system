"""Multinomial Logistic baseline for the future trend probability dataset.

The module intentionally fits all preprocessing statistics on the training
split only.  The validation split selects the L2 regularization strength and
fits an optional post-hoc multinomial probability calibrator; the test split
is evaluated once after those decisions are frozen.

Example::

    D:\\Anaconda3\\python.exe -m research.market_regime.logistic_model \
        --dataset-dir output/batchs --dataset-id hs300_probability_dataset
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for logistic_model. Install requirements.txt "
        "or run with the project's Python environment."
    ) from exc


CLASSES = ("down", "none", "up")
TREND_CLASSES = ("down", "up")
MODEL_METADATA_COLUMNS = {
    "date",
    "symbol",
    "target",
    "target_event_id",
    "target_event_date",
    "days_to_target",
    "label_available_date",
    "sample_weight",
    "split",
    "split_reason",
    "market_state",
}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    return float(np.sum(values * weights) / denominator) if denominator > 0 else np.nan


def infer_feature_columns(samples: pd.DataFrame) -> List[str]:
    """Identify numeric model inputs while excluding labels and identifiers."""

    candidates = [column for column in samples.columns if column not in MODEL_METADATA_COLUMNS]
    numeric_columns = []
    for column in candidates:
        converted = pd.to_numeric(samples[column], errors="coerce")
        if converted.notnull().any():
            numeric_columns.append(column)
    if not numeric_columns:
        raise ValueError("No numeric model features found in sample table")
    return numeric_columns


def fit_preprocessor(train_samples: pd.DataFrame, feature_columns: Sequence[str]) -> Dict[str, object]:
    """Fit median imputation and standardization using the training split only."""

    values = np.asarray(train_samples.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce"), dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    filled = np.where(np.isfinite(values), values, medians)
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return {
        "feature_columns": list(feature_columns),
        "medians": medians,
        "means": means,
        "scales": scales,
    }


def transform_features(samples: pd.DataFrame, preprocessor: Mapping[str, object]) -> np.ndarray:
    """Impute and standardize a split with frozen training parameters."""

    feature_columns = list(preprocessor["feature_columns"])
    values = np.asarray(samples.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce"), dtype=float)
    medians = np.asarray(preprocessor["medians"], dtype=float)
    means = np.asarray(preprocessor["means"], dtype=float)
    scales = np.asarray(preprocessor["scales"], dtype=float)
    filled = np.where(np.isfinite(values), values, medians)
    return (filled - means) / scales


def _get_targets_and_weights(samples: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if "target" not in samples.columns:
        raise ValueError("Sample table must contain target")
    targets = np.asarray(samples["target"].astype(str), dtype=str)
    unknown = sorted(set(targets).difference(CLASSES))
    if unknown:
        raise ValueError("Unexpected target classes: {}".format(unknown))
    weights = pd.to_numeric(samples.get("sample_weight", 1.0), errors="coerce")
    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
    return targets, weights


def _balanced_fit_weights(targets: np.ndarray, event_weights: np.ndarray) -> np.ndarray:
    total_weight = float(np.sum(event_weights))
    class_totals = {
        target_class: float(np.sum(event_weights[targets == target_class]))
        for target_class in CLASSES
    }
    if any(class_totals[target_class] <= 0 for target_class in CLASSES):
        raise ValueError("Training split must contain all classes: {}".format(class_totals))
    factors = {
        target_class: total_weight / (len(CLASSES) * class_totals[target_class])
        for target_class in CLASSES
    }
    return event_weights * np.asarray([factors[target] for target in targets], dtype=float)


def _fit_classifier(features: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray, c_value: float) -> LogisticRegression:
    classifier = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(features, targets, sample_weight=sample_weight)
    return classifier


def aligned_probabilities(classifier: LogisticRegression, features: np.ndarray) -> np.ndarray:
    """Return probabilities in fixed ``down, none, up`` column order."""

    raw_probabilities = classifier.predict_proba(features)
    result = np.zeros((len(features), len(CLASSES)), dtype=float)
    for source_position, target_class in enumerate(classifier.classes_):
        result[:, CLASSES.index(str(target_class))] = raw_probabilities[:, source_position]
    return result


def class_prevalence(targets: np.ndarray, event_weights: np.ndarray) -> np.ndarray:
    total_weight = float(np.sum(event_weights))
    return np.asarray(
        [float(np.sum(event_weights[targets == target_class])) / total_weight for target_class in CLASSES],
        dtype=float,
    )


def _binary_auc_ranks(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ROC-AUC without relying on sklearn-version-specific multiclass APIs."""

    positive_weight = float(np.sum(weights[labels]))
    negative_weight = float(np.sum(weights[~labels]))
    if positive_weight <= 0 or negative_weight <= 0:
        return np.nan
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    cumulative_negative = 0.0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        group_positive = float(np.sum(sorted_weights[start:end][sorted_labels[start:end]]))
        group_negative = float(np.sum(sorted_weights[start:end][~sorted_labels[start:end]]))
        concordant += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = end
    return concordant / (positive_weight * negative_weight)


def _average_precision(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    positive_weight = float(np.sum(weights[labels]))
    if positive_weight <= 0:
        return np.nan
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    cumulative_positive = np.cumsum(sorted_weights * sorted_labels.astype(float))
    cumulative_weight = np.cumsum(sorted_weights)
    precision = cumulative_positive / cumulative_weight
    recall_step = sorted_weights * sorted_labels.astype(float) / positive_weight
    return float(np.sum(precision * recall_step))


def evaluate_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    event_weights: np.ndarray,
    split: str,
    model_name: str,
) -> Dict[str, object]:
    """Calculate weighted classification, ranking and probability metrics."""

    true_positions = np.asarray([CLASSES.index(target) for target in targets], dtype=int)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    predicted_positions = np.argmax(clipped, axis=1)
    metrics: Dict[str, object] = {
        "split": split,
        "model": model_name,
        "sample_count": int(len(targets)),
        "weighted_sample_count": float(np.sum(event_weights)),
        "accuracy": _weighted_mean((predicted_positions == true_positions).astype(float), event_weights),
        "log_loss": _weighted_mean(-np.log(clipped[np.arange(len(targets)), true_positions]), event_weights),
    }
    briers = []
    trend_aps = []
    trend_rocs = []
    for position, target_class in enumerate(CLASSES):
        labels = targets == target_class
        class_probability = clipped[:, position]
        brier = _weighted_mean((class_probability - labels.astype(float)) ** 2, event_weights)
        average_precision = _average_precision(labels, class_probability, event_weights)
        roc_auc = _binary_auc_ranks(labels, class_probability, event_weights)
        top_count = max(1, int(np.ceil(0.05 * len(targets))))
        top_positions = np.argsort(-class_probability, kind="mergesort")[:top_count]
        precision_at_top = _weighted_mean(labels[top_positions].astype(float), event_weights[top_positions])
        metrics["brier_{}".format(target_class)] = brier
        metrics["pr_auc_{}".format(target_class)] = average_precision
        metrics["roc_auc_{}".format(target_class)] = roc_auc
        metrics["precision_at_top_5pct_{}".format(target_class)] = precision_at_top
        briers.append(brier)
        if target_class in TREND_CLASSES:
            trend_aps.append(average_precision)
            trend_rocs.append(roc_auc)
    metrics["macro_brier"] = float(np.nanmean(briers))
    metrics["trend_macro_pr_auc"] = float(np.nanmean(trend_aps))
    metrics["trend_macro_roc_auc"] = float(np.nanmean(trend_rocs))
    return metrics


def fit_multinomial_calibrator(
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    event_weights: np.ndarray,
) -> LogisticRegression:
    """Fit a validation-only multinomial Platt-style calibrator on log probabilities."""

    calibration_features = np.log(np.clip(validation_probabilities, 1e-12, 1.0))
    return _fit_classifier(calibration_features, validation_targets, event_weights, c_value=1.0)


def apply_multinomial_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    calibration_features = np.log(np.clip(probabilities, 1e-12, 1.0))
    return aligned_probabilities(calibrator, calibration_features)


def build_calibration_table(
    targets: np.ndarray,
    probabilities: np.ndarray,
    event_weights: np.ndarray,
    split: str,
    model_name: str,
    bin_count: int = 10,
) -> pd.DataFrame:
    rows = []
    boundaries = np.linspace(0.0, 1.0, bin_count + 1)
    for position, target_class in enumerate(CLASSES):
        labels = (targets == target_class).astype(float)
        values = probabilities[:, position]
        for bin_index in range(bin_count):
            lower, upper = boundaries[bin_index], boundaries[bin_index + 1]
            mask = (values >= lower) & ((values < upper) if bin_index < bin_count - 1 else (values <= upper))
            if not np.any(mask):
                continue
            weights = event_weights[mask]
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "target_class": target_class,
                    "bin": bin_index + 1,
                    "lower_probability": lower,
                    "upper_probability": upper,
                    "sample_count": int(np.sum(mask)),
                    "weighted_sample_count": float(np.sum(weights)),
                    "mean_predicted_probability": _weighted_mean(values[mask], weights),
                    "observed_frequency": _weighted_mean(labels[mask], weights),
                }
            )
    return pd.DataFrame(rows)


def coefficient_table(classifier: LogisticRegression, preprocessor: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for class_position, target_class in enumerate(classifier.classes_):
        for feature, coefficient in zip(preprocessor["feature_columns"], classifier.coef_[class_position]):
            rows.append(
                {
                    "target_class": str(target_class),
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "odds_ratio_per_standard_deviation": float(np.exp(np.clip(coefficient, -50, 50))),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_class", "standardized_coefficient"], ascending=[True, False])


def prediction_table(samples: pd.DataFrame, probabilities: np.ndarray, model_name: str) -> pd.DataFrame:
    columns = [column for column in MODEL_METADATA_COLUMNS if column in samples.columns]
    result = samples.loc[:, columns].copy()
    for position, target_class in enumerate(CLASSES):
        result["p_{}".format(target_class)] = probabilities[:, position]
    result["predicted_target"] = np.asarray(CLASSES, dtype=object)[np.argmax(probabilities, axis=1)]
    result["model"] = model_name
    return result


def train_logistic_experiment(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    c_values: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    calibration: str = "multinomial",
) -> Dict[str, object]:
    """Train, select, calibrate and evaluate a leakage-controlled Logistic model."""

    if calibration not in {"none", "multinomial"}:
        raise ValueError("calibration must be 'none' or 'multinomial'")
    feature_columns = infer_feature_columns(train_samples)
    preprocessor = fit_preprocessor(train_samples, feature_columns)
    train_features = transform_features(train_samples, preprocessor)
    validation_features = transform_features(validation_samples, preprocessor)
    test_features = transform_features(test_samples, preprocessor)
    train_targets, train_event_weights = _get_targets_and_weights(train_samples)
    validation_targets, validation_event_weights = _get_targets_and_weights(validation_samples)
    test_targets, test_event_weights = _get_targets_and_weights(test_samples)
    train_fit_weights = _balanced_fit_weights(train_targets, train_event_weights)

    validation_grid = []
    candidate_models = []
    for c_value in c_values:
        if float(c_value) <= 0:
            raise ValueError("All c_values must be positive")
        classifier = _fit_classifier(train_features, train_targets, train_fit_weights, float(c_value))
        validation_probabilities = aligned_probabilities(classifier, validation_features)
        metrics = evaluate_probabilities(
            validation_targets, validation_probabilities, validation_event_weights, "validation", "logistic_raw"
        )
        metrics["c_value"] = float(c_value)
        validation_grid.append(metrics)
        candidate_models.append((float(c_value), classifier, validation_probabilities, metrics))
    validation_grid_frame = pd.DataFrame(validation_grid).sort_values(
        ["trend_macro_pr_auc", "log_loss", "c_value"], ascending=[False, True, True]
    )
    selected_c = float(validation_grid_frame.iloc[0]["c_value"])
    selected_c, selected_model, selected_validation_probabilities, _ = next(
        model for model in candidate_models if model[0] == selected_c
    )

    train_prevalence = class_prevalence(train_targets, train_event_weights)
    validation_baseline = np.tile(train_prevalence, (len(validation_targets), 1))
    test_baseline = np.tile(train_prevalence, (len(test_targets), 1))
    test_raw_probabilities = aligned_probabilities(selected_model, test_features)
    metrics_rows = [
        evaluate_probabilities(validation_targets, validation_baseline, validation_event_weights, "validation", "base_rate"),
        evaluate_probabilities(validation_targets, selected_validation_probabilities, validation_event_weights, "validation", "logistic_raw"),
        evaluate_probabilities(test_targets, test_baseline, test_event_weights, "test", "base_rate"),
        evaluate_probabilities(test_targets, test_raw_probabilities, test_event_weights, "test", "logistic_raw"),
    ]
    calibration_tables = [
        build_calibration_table(validation_targets, selected_validation_probabilities, validation_event_weights, "validation", "logistic_raw"),
        build_calibration_table(test_targets, test_raw_probabilities, test_event_weights, "test", "logistic_raw"),
    ]
    calibrator = None
    test_probabilities = test_raw_probabilities
    validation_probabilities = selected_validation_probabilities
    final_model_name = "logistic_raw"
    if calibration == "multinomial":
        calibrator = fit_multinomial_calibrator(
            selected_validation_probabilities, validation_targets, validation_event_weights
        )
        validation_probabilities = apply_multinomial_calibrator(calibrator, selected_validation_probabilities)
        test_probabilities = apply_multinomial_calibrator(calibrator, test_raw_probabilities)
        final_model_name = "logistic_calibrated"
        metrics_rows.extend(
            [
                evaluate_probabilities(validation_targets, validation_probabilities, validation_event_weights, "validation", final_model_name),
                evaluate_probabilities(test_targets, test_probabilities, test_event_weights, "test", final_model_name),
            ]
        )
        calibration_tables.extend(
            [
                build_calibration_table(validation_targets, validation_probabilities, validation_event_weights, "validation", final_model_name),
                build_calibration_table(test_targets, test_probabilities, test_event_weights, "test", final_model_name),
            ]
        )
    test_predictions = prediction_table(test_samples, test_probabilities, final_model_name)
    validation_predictions = prediction_table(validation_samples, validation_probabilities, final_model_name)
    market_state_metrics = pd.DataFrame()
    if "market_state" in test_samples.columns:
        from research.market_regime.market_environment import stratify_probability_predictions

        market_state_metrics = stratify_probability_predictions(
            test_predictions, test_samples, evaluate_probabilities, split="test"
        )
    return {
        "selected_c": selected_c,
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "classifier": selected_model,
        "calibrator": calibrator,
        "validation_grid": validation_grid_frame,
        "metrics": pd.DataFrame(metrics_rows),
        "calibration": pd.concat(calibration_tables, ignore_index=True),
        "coefficients": coefficient_table(selected_model, preprocessor),
        "test_predictions": test_predictions,
        "validation_predictions": validation_predictions,
        "market_state_metrics": market_state_metrics,
        "test_model_name": final_model_name,
        "train_prevalence": train_prevalence,
    }


def _dataset_path(dataset_dir: Union[str, Path], split: str, dataset_id: str) -> Path:
    return Path(dataset_dir) / "trend_probability_{}_model_samples_{}.csv".format(split, dataset_id)


def load_dataset_splits(dataset_dir: Union[str, Path], dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {split: _dataset_path(dataset_dir, split, dataset_id) for split in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Logistic model samples: {}".format(missing))
    return tuple(pd.read_csv(paths[split], low_memory=False) for split in ("train", "validation", "test"))


def export_logistic_experiment(
    report: Mapping[str, object],
    output_dir: Union[str, Path],
    run_id: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Export model, predictions, metrics, calibration and provenance manifest."""

    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_grid": "trend_logistic_validation_grid",
        "metrics": "trend_logistic_metrics",
        "calibration": "trend_logistic_calibration",
        "coefficients": "trend_logistic_coefficients",
        "validation_predictions": "trend_logistic_validation_predictions",
        "test_predictions": "trend_logistic_test_predictions",
        "market_state_metrics": "trend_logistic_market_state_metrics",
    }
    paths = {}
    for key, prefix in tables.items():
        path = destination / "{}_{}.csv".format(prefix, identifier)
        report[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    model_path = destination / "trend_logistic_model_{}.pkl".format(identifier)
    with model_path.open("wb") as output_file:
        pickle.dump(
            {
                "classes": CLASSES,
                "feature_columns": report["feature_columns"],
                "preprocessor": report["preprocessor"],
                "classifier": report["classifier"],
                "calibrator": report["calibrator"],
            },
            output_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    paths["model"] = model_path
    manifest = {
        "analysis_id": identifier,
        "created_at": datetime.now().isoformat(),
        "model_type": "multinomial_l2_logistic_regression",
        "classes": list(CLASSES),
        "selected_c": float(report["selected_c"]),
        "calibration": "multinomial" if report["calibrator"] is not None else "none",
        "feature_columns": list(report["feature_columns"]),
        "train_prevalence": [float(value) for value in report["train_prevalence"]],
        "paths": {key: str(path) for key, path in paths.items()},
    }
    if metadata:
        manifest.update(dict(metadata))
    manifest_path = destination / "trend_logistic_manifest_{}.json".format(identifier)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _parse_c_values(text: str) -> List[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one C value is required")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the multinomial trend Logistic baseline.")
    parser.add_argument("--dataset-dir", default="output/batchs")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3,10")
    parser.add_argument("--calibration", choices=("none", "multinomial"), default="multinomial")
    args = parser.parse_args(argv)
    train_samples, validation_samples, test_samples = load_dataset_splits(args.dataset_dir, args.dataset_id)
    report = train_logistic_experiment(
        train_samples,
        validation_samples,
        test_samples,
        c_values=_parse_c_values(args.c_values),
        calibration=args.calibration,
    )
    destination = args.output_dir or args.dataset_dir
    paths = export_logistic_experiment(
        report,
        destination,
        args.run_id,
        {
            "dataset_id": args.dataset_id,
            "dataset_dir": str(args.dataset_dir),
            "c_values": _parse_c_values(args.c_values),
            "validation_selection_metric": "trend_macro_pr_auc, then log_loss",
        },
    )
    selected_metrics = report["metrics"]
    selected_metrics = selected_metrics[
        (selected_metrics["split"] == "test") & (selected_metrics["model"] == report["test_model_name"])
    ].iloc[0]
    print("Selected C={:.6g}; test trend macro PR-AUC={:.6f}; test log loss={:.6f}".format(
        report["selected_c"], selected_metrics["trend_macro_pr_auc"], selected_metrics["log_loss"]
    ))
    print("Manifest: {}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
