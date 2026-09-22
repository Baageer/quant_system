"""Diagnose whether tradeable-window quality is causally predictable.

The ``excess_hs300_v1`` dataset labels every date x symbol row with two future
views that were measured against the HS300 index:

* ``window_grade_{W}w_{h}d`` -- five ordinal buckets of the best h-day excess
  achievable by entering on any session in ``signal+1 .. signal+W``;
* ``window_entry_offset_{W}w_{h}d`` -- which of those offsets achieved the best
  excess (so ``W - 1`` of every ``W`` entries are ex-post suboptimal).

This module asks, using only the frozen causal 28-feature panel available at
the signal date, whether those two quantities are separable before T+1:

A. **Window quality**: a balanced 5-class multinomial over ``window_grade``,
   reported per class (ROC-AUC / PR-AUC / top-5% precision) plus two binary
   collapses that matter for a filter: "good window" (up | strong_up) and
   "strong window" (strong_up) against the rest.
B. **Entry offset**: a balanced 5-class multinomial over the best offset, plus
   a realized-premium back-of-envelope: entering at the *predicted* offset vs
   always T+1 (offset 1) vs the ex-post best offset.  This bounds how much of
   the window premium a causal timing rule could actually harvest.

Training uses only eligible rows, uniform per-row weight, median/standardised
features fit on training data, per-class balanced weights and a validation
C-grid (macro PR-AUC, then macro ROC-AUC).  Everything on test is reported
once.  Rows are the same frozen rows consumed by ``trade_outcome_model``.

Example::

    python -m research.market_regime.window_predictability_diagnostic \
        --dataset-dir output/datasets --dataset-id excess_hs300_v1 \
        --trend-model-id hs300_logistic_v2 \
        --benchmark-file data/benchmarks/000300.csv --run-id window_diag_v1
"""

import argparse
import json
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
from research.market_regime.tradability_labels import load_price_histories, normalize_a_share_symbols
from research.market_regime.tradability_model import load_trend_feature_columns
from research.market_regime.excess_target_dataset import load_hs300_benchmark


GRADE_ORDER = ("strong_down", "down", "none", "up", "strong_up")
OFFSET_ORDER = ("1", "2", "3", "4", "5")
RANKING_TOP = 0.05


def load_excess_split(dataset_dir: Path, dataset_id: str, split: str) -> pd.DataFrame:
    """Load eligible rows of one split, keeping only complete window labels."""

    path = dataset_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    eligible = frame["auxiliary_label_eligible"].astype(str).str.lower().eq("true")
    kept = eligible & frame["window_grade_5w_10d"].notnull() & frame["window_excess_5w_10d"].notnull()
    result = frame.loc[kept].copy()
    if result.empty:
        raise ValueError("No eligible window-labelled rows in {}".format(split))
    return result.reset_index(drop=True)


def class_balanced_weights(targets: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Return per-class balanced weights (each class receives equal total weight)."""

    targets = np.asarray(targets)
    weights = np.ones(len(targets), dtype=float)
    total = float(len(targets))
    factors = {}
    for target_class in order:
        count = float(np.sum(targets == target_class))
        if count <= 0:
            raise ValueError("Missing target class {} in training rows".format(target_class))
        factors[target_class] = total / (len(order) * count)
    return weights * np.asarray([factors[target] for target in targets], dtype=float)


def stratify_cap(targets: np.ndarray, order: Sequence[str], max_per_class: int, seed: int = 42) -> np.ndarray:
    """Return a boolean mask capping large classes for a faster diagnostic fit."""

    generator = np.random.default_rng(seed)
    keep = np.zeros(len(targets), dtype=bool)
    for target_class in order:
        positions = np.flatnonzero(targets == target_class)
        if positions.size > max_per_class:
            positions = generator.choice(positions, size=max_per_class, replace=False)
        keep[positions] = True
    return keep


def aligned_multiclass_probabilities(classifier: LogisticRegression, features: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Return predict_proba columns reordered to ``order``."""

    raw = classifier.predict_proba(features)
    result = np.zeros((len(features), len(order)), dtype=float)
    for source_position, target_class in enumerate(classifier.classes_):
        result[:, order.index(str(target_class))] = raw[:, source_position]
    return result


def per_class_ranking_metrics(targets: np.ndarray, probabilities: np.ndarray, order: Sequence[str], split: str, model: str) -> Dict[str, object]:
    """Weighted ranking metrics for every class plus macro summaries."""

    weights = np.ones(len(targets), dtype=float)
    rows = {"split": split, "model": model}
    rocs, prs, top5s = [], [], []
    top_count = max(1, int(np.ceil(RANKING_TOP * len(targets))))
    for position, target_class in enumerate(order):
        labels = targets == target_class
        class_probability = np.clip(probabilities[:, position], 1e-12, 1.0)
        roc_auc = _binary_auc_ranks(labels, class_probability, weights)
        pr_auc = _average_precision(labels, class_probability, weights)
        top_positions = np.argsort(-class_probability, kind="mergesort")[:top_count]
        precision_at_top = float(np.mean(labels[top_positions]))
        rows["roc_auc_{}".format(target_class)] = roc_auc
        rows["pr_auc_{}".format(target_class)] = pr_auc
        rows["precision_top5_{}".format(target_class)] = precision_at_top
        rocs.append(roc_auc)
        prs.append(pr_auc)
        top5s.append(precision_at_top)
    predicted = np.asarray(order)[np.argmax(probabilities, axis=1)]
    rows["accuracy"] = float(np.mean(predicted == targets))
    rows["macro_roc_auc"] = float(np.nanmean(rocs))
    rows["macro_pr_auc"] = float(np.nanmean(prs))
    rows["macro_precision_top5"] = float(np.nanmean(top5s))
    rows["order"] = ",".join(order)
    return rows


def binary_collapse_metrics(
    probabilities: np.ndarray, order: Sequence[str], targets: np.ndarray, positives: Sequence[str],
    split: str, model: str,
) -> Dict[str, object]:
    """Ranking metrics for a union of classes used as the binary positive set."""

    labels = np.asarray([target in positives for target in targets])
    class_probability = np.clip(np.sum(probabilities[:, [order.index(cls) for cls in positives]], axis=1), 1e-12, 1.0)
    weights = np.ones(len(targets), dtype=float)
    top_count = max(1, int(np.ceil(RANKING_TOP * len(targets))))
    top_positions = np.argsort(-class_probability, kind="mergesort")[:top_count]
    return {
        "split": split,
        "model": model,
        "positive_classes": ",".join(positives),
        "positive_prevalence": float(np.mean(labels)),
        "roc_auc": float(_binary_auc_ranks(labels, class_probability, weights)),
        "pr_auc": float(_average_precision(labels, class_probability, weights)),
        "precision_top5": float(np.mean(labels[top_positions])),
        "recall_top5": float(np.mean(labels[top_positions])) / float(np.mean(labels)) if np.mean(labels) > 0 else np.nan,
    }


def select_c_and_fit(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    order: Sequence[str],
    c_values: Sequence[float],
) -> Tuple[float, LogisticRegression, Dict[str, float]]:
    """Pick C by validation macro PR-AUC, then macro ROC-AUC, on balanced classes."""

    train_weights = class_balanced_weights(train_targets, order)
    best_c, best_score, best_tiebreak, best_model, best_metrics = None, -np.inf, -np.inf, None, None
    for c_value in c_values:
        classifier = LogisticRegression(
            C=float(c_value), penalty="l2", solver="lbfgs", max_iter=1000, random_state=42,
            multi_class="multinomial",
        ).fit(train_features, train_targets, sample_weight=train_weights)
        validation_probabilities = aligned_multiclass_probabilities(classifier, validation_features, order)
        metrics = per_class_ranking_metrics(validation_targets, validation_probabilities, order, "validation", "logistic")
        score = float(metrics["macro_pr_auc"])
        tiebreak = float(metrics["macro_roc_auc"])
        if score > best_score or (score == best_score and tiebreak > best_tiebreak):
            best_c, best_score, best_tiebreak, best_model, best_metrics = float(c_value), score, tiebreak, classifier, metrics
    if best_model is None:
        raise ValueError("No C value produced a valid validation score")
    return best_c, best_model, {key: best_metrics[key] for key in ("macro_pr_auc", "macro_roc_auc")}


def offset_excess_matrix(
    samples: pd.DataFrame, benchmark_file: Path, price_history_dir: Optional[str] = None,
    window: int = 5, horizon: int = 10,
) -> np.ndarray:
    """Recompute the per-offset h-day HS300 excess for each row.

    Returns a float array (n_rows, window) whose column k is the excess of
    entering ``k+1`` sessions after the signal.  Only close levels are needed,
    matching the label construction in ``excess_target_dataset``.
    """

    window, horizon = int(window), int(horizon)
    histories, _ = load_price_histories(samples["symbol"], price_history_dir)
    benchmark = load_hs300_benchmark(benchmark_file)
    excess_matrix = np.full((len(samples), window), np.nan, dtype=float)
    for symbol, group in samples.groupby("symbol", sort=False):
        prices = histories.get(symbol)
        if prices is None or prices.empty:
            continue
        positions = group.index.to_numpy()
        close = np.asarray(pd.to_numeric(prices["close"], errors="coerce"), dtype=float)
        length = len(close)
        local_positions = prices.index.get_indexer(pd.DatetimeIndex(group["date"]))
        benchmark_level = np.asarray(benchmark.reindex(prices.index), dtype=float)
        for offset in range(1, window + 1):
            entries = local_positions + offset
            exits = entries + horizon
            valid = (local_positions >= 0) & (entries < length) & (exits < length)
            selected = np.flatnonzero(valid)
            if selected.size == 0:
                continue
            entry_close = close[entries[selected]]
            exit_close = close[exits[selected]]
            benchmark_entry = benchmark_level[entries[selected]]
            benchmark_exit = benchmark_level[exits[selected]]
            good = (
                np.isfinite(entry_close) & np.isfinite(exit_close) & (entry_close > 0)
                & np.isfinite(benchmark_entry) & np.isfinite(benchmark_exit) & (benchmark_entry > 0)
            )
            row_positions = positions[selected]
            values = np.full(selected.shape, np.nan, dtype=float)
            if np.any(good):
                values[good] = (
                    exit_close[good] / entry_close[good] - benchmark_exit[good] / benchmark_entry[good]
                )
            excess_matrix[row_positions, offset - 1] = values
    return excess_matrix


def evaluate_offset_policy(excess_matrix: np.ndarray, predicted_offsets: np.ndarray) -> Dict[str, float]:
    """Delivered 10-day excess of predicted offsets vs always-T+1 vs ex-post best."""

    t1 = excess_matrix[:, 0]
    finite = np.isfinite(excess_matrix).all(axis=1) & np.isfinite(predicted_offsets)
    t1 = t1[finite]
    matrix = excess_matrix[finite]
    predicted = np.asarray(predicted_offsets[finite], dtype=int) - 1
    delivered = matrix[np.arange(matrix.shape[0]), predicted]
    best = np.max(matrix, axis=1)
    return {
        "row_count": int(finite.sum()),
        "mean_always_t1_excess": float(np.mean(t1)),
        "mean_predicted_offset_excess": float(np.mean(delivered)),
        "mean_best_offset_excess": float(np.mean(best)),
        "median_always_t1_excess": float(np.median(t1)),
        "median_predicted_offset_excess": float(np.median(delivered)),
        "harvested_vs_t1": float(np.mean(delivered) - np.mean(t1)),
        "best_vs_t1": float(np.mean(best) - np.mean(t1)),
        "fraction_pred_ge_t1": float(np.mean(delivered >= t1)),
        "fraction_best_ge_t1": float(np.mean(best >= t1)),
    }


def build_diagnostic(
    dataset_dir: Path,
    dataset_id: str,
    benchmark_file: Path,
    trend_model_id: str,
    feature_columns: Sequence[str],
    horizon: int = 10,
    window: int = 5,
    c_values: Sequence[float] = (0.3, 1.0, 3.0),
    max_per_class: int = 60000,
    price_history_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Run both diagnostic tasks and assemble result tables."""

    splits = {name: load_excess_split(dataset_dir, dataset_id, name) for name in ("train", "validation", "test")}
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    preprocessor = fit_preprocessor(train, feature_columns)
    train_X = transform_features(train, preprocessor)
    validation_X = transform_features(validation, preprocessor)
    test_X = transform_features(test, preprocessor)

    # ---- A: window grade ----------------------------------------------------
    grade_column = "window_grade_5w_10d"
    train_targets = np.asarray(train[grade_column].astype(str), dtype=str)
    keep_train = stratify_cap(train_targets, GRADE_ORDER, max_per_class)
    train_X_capped = train_X[keep_train]
    train_targets_capped = train_targets[keep_train]
    validation_targets = np.asarray(validation[grade_column].astype(str), dtype=str)
    test_targets = np.asarray(test[grade_column].astype(str), dtype=str)
    grade_c, grade_classifier, _ = select_c_and_fit(
        train_X_capped, train_targets_capped, validation_X, validation_targets, GRADE_ORDER, c_values,
    )
    grade_rows = []
    grade_rows.append(per_class_ranking_metrics(validation_targets, aligned_multiclass_probabilities(grade_classifier, validation_X, GRADE_ORDER), GRADE_ORDER, "validation", "grade_5way"))
    test_grade_probs = aligned_multiclass_probabilities(grade_classifier, test_X, GRADE_ORDER)
    grade_rows.append(per_class_ranking_metrics(test_targets, test_grade_probs, GRADE_ORDER, "test", "grade_5way"))
    for positives, name in ((("up", "strong_up"), "good_window"), (("strong_up",), "strong_window"), (("down", "strong_down"), "bad_window")):
        grade_rows.append(binary_collapse_metrics(test_grade_probs, GRADE_ORDER, test_targets, positives, "test", name))
    grade_metrics = pd.DataFrame(grade_rows)

    # ---- B: entry offset ----------------------------------------------------
    offset_column = "window_entry_offset_5w_10d"
    train_offsets = np.asarray(train[offset_column].astype(int).astype(str), dtype=str)
    keep_offset = stratify_cap(train_offsets, OFFSET_ORDER, max_per_class)
    train_X_offset = train_X[keep_offset]
    train_offset_targets = train_offsets[keep_offset]
    validation_offsets = np.asarray(validation[offset_column].astype(int).astype(str), dtype=str)
    test_offsets = np.asarray(test[offset_column].astype(int).astype(str), dtype=str)
    offset_c, offset_classifier, _ = select_c_and_fit(
        train_X_offset, train_offset_targets, validation_X, validation_offsets, OFFSET_ORDER, c_values,
    )
    offset_rows = []
    offset_rows.append(per_class_ranking_metrics(validation_offsets, aligned_multiclass_probabilities(offset_classifier, validation_X, OFFSET_ORDER), OFFSET_ORDER, "validation", "offset_5way"))
    test_offset_probs = aligned_multiclass_probabilities(offset_classifier, test_X, OFFSET_ORDER)
    offset_rows.append(per_class_ranking_metrics(test_offsets, test_offset_probs, OFFSET_ORDER, "test", "offset_5way"))
    offset_metrics = pd.DataFrame(offset_rows)

    test_excess = offset_excess_matrix(test, benchmark_file, price_history_dir, window=window, horizon=horizon)
    test_predicted_offsets = np.asarray([int(o) for o in OFFSET_ORDER])[np.argmax(test_offset_probs, axis=1)].astype(float)
    policy = evaluate_offset_policy(test_excess, test_predicted_offsets)
    policy["model"] = "offset_5way"
    policy["selected_c"] = offset_c
    policy["grade_selected_c"] = grade_c

    # predicted-grade score vs realized window excess rank IC (daily mean), as headroom marker
    good_score = np.sum(test_grade_probs[:, [GRADE_ORDER.index(c) for c in ("up", "strong_up")]], axis=1)
    ic_rows = []
    frame = pd.DataFrame({
        "date": test["date"], "good_score": good_score, "window_excess": test["window_excess_5w_10d"],
        "future_excess": test["future_excess_return_10d"],
    }).dropna()
    for label, value_column in (("good_score_vs_window_excess", "window_excess"), ("good_score_vs_future_excess", "future_excess")):
        daily = []
        for _, group in frame.groupby("date"):
            if len(group) < 20 or group[value_column].nunique() < 2 or group["good_score"].nunique() < 2:
                continue
            daily.append(group["good_score"].rank().corr(group[value_column].rank()))
        ic_rows.append({"split": "test", "model": "grade_good_score", "metric": label, "daily_mean_rank_ic": float(np.mean(daily)) if daily else np.nan})
    rank_ics = pd.DataFrame(ic_rows)

    return {
        "grade_metrics": grade_metrics,
        "offset_metrics": offset_metrics,
        "rank_ics": rank_ics,
        "policy": policy,
        "grade_selected_c": grade_c,
        "offset_selected_c": offset_c,
        "feature_count": int(len(feature_columns)),
        "train_grade_counts": {cls: int(np.sum(train_targets == cls)) for cls in GRADE_ORDER},
        "train_offset_counts": {cls: int(np.sum(train_offsets == cls)) for cls in OFFSET_ORDER},
        "row_counts": {name: int(len(splits[name])) for name in ("train", "validation", "test")},
    }


def export_diagnostic(report: Mapping[str, object], output_dir: Path, run_id: str, metadata: Mapping[str, object]) -> Dict[str, str]:
    """Write metric/policy tables and a reproducible manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "grade_metrics": report["grade_metrics"],
        "offset_metrics": report["offset_metrics"],
        "rank_ics": report["rank_ics"],
    }
    paths = {}
    for name, table in tables.items():
        path = output_dir / "window_diagnostic_{}_{}.csv".format(name, run_id)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)
    policy_path = output_dir / "window_diagnostic_policy_{}.csv".format(run_id)
    pd.DataFrame([report["policy"]]).to_csv(policy_path, index=False, encoding="utf-8-sig")
    paths["policy"] = str(policy_path)
    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "question": "is window quality / best offset causally predictable before T+1",
        "targets": {"grade": "window_grade_5w_10d", "offset": "window_entry_offset_5w_10d"},
        "grade_selected_c": report["grade_selected_c"],
        "offset_selected_c": report["offset_selected_c"],
        "feature_count": report["feature_count"],
        "balanced_class_weights": True,
        "uniform_row_weights": True,
        "train_cap_per_class": metadata.get("max_per_class"),
        "row_counts": report["row_counts"],
        "train_grade_counts": report["train_grade_counts"],
        "train_offset_counts": report["train_offset_counts"],
        "policy_note": "realized test premium of entering at predicted offset vs always T+1 vs ex-post best; offset 1 == T+1",
        "paths": paths,
        **dict(metadata),
    }
    manifest_path = output_dir / "window_diagnostic_manifest_{}.json".format(run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="excess_hs300_v1")
    parser.add_argument("--trend-model-id", default="hs300_logistic_v2")
    parser.add_argument("--benchmark-file", default="data/benchmarks/000300.csv")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="window_diag_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--c-values", default="0.3,1,3")
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    c_values = [float(value.strip()) for value in args.c_values.split(",") if value.strip()]
    feature_columns = load_trend_feature_columns(Path(args.dataset_dir), args.trend_model_id)
    report = build_diagnostic(
        Path(args.dataset_dir), args.dataset_id, Path(args.benchmark_file), args.trend_model_id,
        feature_columns, horizon=args.horizon, window=args.window, c_values=c_values,
        max_per_class=args.max_per_class, price_history_dir=args.price_history_dir,
    )
    paths = export_diagnostic(
        report, Path(args.output_dir), args.run_id,
        {
            "dataset_id": args.dataset_id,
            "trend_model_id": args.trend_model_id,
            "horizon": args.horizon,
            "window": args.window,
            "c_values": c_values,
            "max_per_class": args.max_per_class,
            "selection": "validation macro PR-AUC then macro ROC-AUC",
        },
    )
    policy = report["policy"]
    print(
        "Window diagnostic done. Test policy (10d HS300 excess): always-T+1={t1:.5f} "
        "predicted-offset={pred:.5f} best={best:.5f}; grade/offset selected C={gc},{oc}; manifest={manifest}".format(
            t1=policy["mean_always_t1_excess"], pred=policy["mean_predicted_offset_excess"],
            best=policy["mean_best_offset_excess"], gc=report["grade_selected_c"], oc=report["offset_selected_c"],
            manifest=paths["manifest"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
