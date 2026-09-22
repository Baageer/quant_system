"""A/B a defensive window-grade filter on top of the outcome-model ranking.

``trade_outcome_ranking_evaluation`` already turns the predicted ``entry_score``
(excess Ridge head ``hs300_excess_v1_outcome``) into a daily cross-sectional
Top-K basket and backtests it.  The window diagnostic (``window_diag_v1``)
found that *within-window entry timing* is not causally predictable, but that
the extreme window tail -- in particular a *bad* tradeable window
(``down | strong_down`` grade, test ROC-AUC 0.61) -- carries a small real
pre-separation signal.  This module tests whether using that tail signal as a
defensive overlay changes the end-to-end backtest.

Overlay rule: on every signal date, ban the cross-sectional worst ``q`` share
of eligible names by predicted ``P(bad_window)`` -- i.e. set those names'
``entry_score`` so low they can never enter the daily Top-K.  All other names
keep their exact predicted ``entry_score``, so a banned slot is replaced by the
next-best name.  ``q = 0`` is the untouched baseline arm.  The overlay consumes
nothing a real decision at ``t`` would not have: the grade classifier is fit on
train-eligible rows only, C is chosen on validation, and ``P(bad_window)`` is
predicted from the frozen t-features.

The drop share ``q`` is chosen on validation from a small grid and then frozen to
the test split exactly once; the other test arms are reported for visibility but
are not the selected policy.  The choice goes through
``selection_rule.select_configuration``, so ``q`` must **beat the unfiltered
baseline (``q = 0``)** to be adopted -- ranking the grid alone would always
adopt some filter, even when the overlay has no increment.

Example::

    python -m research.market_regime.window_grade_filter_ab \
        --dataset-id excess_hs300_v1 --trend-model-id hs300_logistic_v2 \
        --outcome-model-id hs300_excess_v1_outcome --run-id window_grade_filter_v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.logistic_model import fit_preprocessor, transform_features
from research.market_regime.selection_rule import select_configuration
from research.market_regime.probability_trade_evaluation import (
    _run_engine,
    make_passive_strategy,
    select_passive_symbols,
    summarize_backtest,
)
from research.market_regime.trade_outcome_model import target_columns
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_model import load_trend_feature_columns
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.window_predictability_diagnostic import (
    GRADE_ORDER,
    aligned_multiclass_probabilities,
    load_excess_split,
    select_c_and_fit,
    stratify_cap,
)


SCORE_COLUMN = "entry_score"
GRADE_COLUMN = "window_grade_5w_10d"
BAD_CLASSES = ("down", "strong_down")
DEFAULT_DROP_QUANTILES = (0.0, 0.05, 0.10, 0.20)
# ``drop_quantile == 0.0`` bans nothing; this is the arm the filter has to beat.
BASELINE_ARM = "none"


def fit_grade_classifier(
    dataset_dir: Path,
    dataset_id: str,
    feature_columns: Sequence[str],
    c_values: Sequence[float],
    max_per_class: int,
) -> Tuple[object, object, float]:
    """Refit the diagnostic's window-grade classifier identically (train only)."""

    train = load_excess_split(dataset_dir, dataset_id, "train")
    validation = load_excess_split(dataset_dir, dataset_id, "validation")
    preprocessor = fit_preprocessor(train, feature_columns)
    train_X = transform_features(train, preprocessor)
    validation_X = transform_features(validation, preprocessor)
    train_targets = np.asarray(train[GRADE_COLUMN].astype(str), dtype=str)
    keep_train = stratify_cap(train_targets, GRADE_ORDER, max_per_class)
    validation_targets = np.asarray(validation[GRADE_COLUMN].astype(str), dtype=str)
    selected_c, classifier, _ = select_c_and_fit(
        train_X[keep_train], train_targets[keep_train], validation_X, validation_targets,
        GRADE_ORDER, c_values,
    )
    return classifier, preprocessor, selected_c


def add_bad_window_probability(
    frame: pd.DataFrame, classifier, preprocessor: object, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Append the causal ``P(bad_window)`` for every row of ``frame``."""

    features = transform_features(frame, preprocessor)
    probabilities = aligned_multiclass_probabilities(classifier, features, GRADE_ORDER)
    result = frame.copy()
    result["p_bad_window"] = probabilities[:, [GRADE_ORDER.index(cls) for cls in BAD_CLASSES]].sum(axis=1)
    return result


def load_combined_split(
    dataset_dir: Path,
    dataset_id: str,
    model_id: str,
    split: str,
    classifier,
    preprocessor: object,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Join outcome-model ``entry_score`` with the causal grade tail probability."""

    samples = load_excess_split(dataset_dir, dataset_id, split)
    samples = add_bad_window_probability(samples, classifier, preprocessor, feature_columns)
    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    predictions = pd.read_csv(path, dtype={"symbol": str})
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["symbol"] = normalize_a_share_symbols(predictions["symbol"])
    scores = predictions.loc[:, ["date", "symbol", SCORE_COLUMN]].copy()
    scores[SCORE_COLUMN] = pd.to_numeric(scores[SCORE_COLUMN], errors="coerce")
    merged = samples.merge(scores, on=["date", "symbol"], how="inner")
    if len(merged) != len(scores):
        raise ValueError(
            "Split {} grade/score alignment incomplete: {} vs {}".format(split, len(merged), len(scores))
        )
    if merged[SCORE_COLUMN].isnull().any():
        raise ValueError("Split {} has rows without an entry_score".format(split))
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def bad_window_ban_mask(frame: pd.DataFrame, quantile: float) -> np.ndarray:
    """True for the daily worst ``quantile`` share of rows by P(bad_window)."""

    if quantile <= 0.0:
        return np.zeros(len(frame), dtype=bool)
    if not 0.0 < quantile < 1.0:
        raise ValueError("drop quantile must be in (0, 1)")
    rank_pct = frame.groupby("date", sort=False)["p_bad_window"].rank(method="first", pct=True)
    return np.asarray(rank_pct > (1.0 - quantile), dtype=bool)


def adjusted_scores(frame: pd.DataFrame, ban: np.ndarray) -> pd.DataFrame:
    """Return a copy whose banned rows can never enter the daily Top-K."""

    result = frame.copy()
    result[SCORE_COLUMN] = result[SCORE_COLUMN].where(~ban, -np.inf)
    return result


def realized_outcome(frame: pd.DataFrame, selected: pd.Series, horizon: int) -> Dict[str, float]:
    """Forward outcomes of the equal-weight daily basket of a filter arm."""

    excess_column, mae_column = target_columns(horizon)
    mask = np.asarray(selected, dtype=bool)
    selected_excess = np.asarray(frame.loc[mask, excess_column], dtype=float)
    selected_mae = np.asarray(frame.loc[mask, mae_column], dtype=float)
    universe_excess = np.asarray(frame[excess_column], dtype=float)
    return {
        "selected_count": int(mask.sum()),
        "selected_mean_excess_return": float(np.mean(selected_excess)),
        "selected_mean_mae": float(np.mean(selected_mae)),
        "selected_excess_over_universe": float(np.mean(selected_excess) - np.mean(universe_excess)),
    }


def run_arm(
    name: str,
    frame: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    selected: pd.Series,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    horizon: int,
    frictionless: bool = False,
) -> Dict[str, object]:
    """Backtest one Top-K arm through the unchanged ranking engine."""

    prepared = apply_rank_selection(raw_market_data, frame, selected)
    start_date, end_date = frame["date"].min(), frame["date"].max()
    target_notional = initial_capital / max_positions
    results, trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
        frictionless=frictionless,
    )
    return summarize_backtest(name, results, trades, initial_capital)


def choose_drop_quantile_on_validation(
    validation: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    drop_quantiles: Sequence[float],
    initial_capital: float,
    max_positions: int,
    config_path: str,
    horizon: int,
    top_k: float,
) -> Tuple[float, pd.DataFrame]:
    """Freeze the drop share, if and only if it beats not filtering at all.

    Returns ``0.0`` when the rule declines, which is the unfiltered baseline.
    """

    rows = []
    for quantile in drop_quantiles:
        name = "none" if quantile == 0.0 else "drop_bad_top{:.0f}".format(quantile * 100)
        ban = bad_window_ban_mask(validation, quantile)
        arm = adjusted_scores(validation, ban)
        selected = select_top_k(arm, top_k, SCORE_COLUMN)
        metrics = run_arm(
            name, arm, raw_market_data, selected, initial_capital, max_positions,
            config_path, horizon,
        )
        rows.append({
            "drop_quantile": float(quantile),
            "arm": name,
            "banned_row_count": int(ban.sum()),
            "banned_share": float(ban.mean()),
            **realized_outcome(validation, selected, horizon),
            **{"validation_" + key: value for key, value in metrics.items() if key != "strategy"},
        })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_total_return", "validation_turnover_multiple", "drop_quantile"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # Selection is by rule, not by rank order: the grid's top row is only the
    # inspection view.  ``drop_quantile == 0.0`` (arm ``"none"``) is the
    # untouched baseline, and the rule has to be shown to beat it -- selecting by
    # rank alone can only ever adopt *some* filter, never "don't filter".
    outcome = select_configuration(
        grid,
        baseline_key={"arm": BASELINE_ARM},
        return_column="validation_total_return",
        turnover_column="validation_turnover_multiple",
    )
    if outcome.row is None:
        print("\nselection: NO CHANGE (no filtering) -- {}".format(outcome.reason))
        return 0.0, outcome.table
    print("\nselection: {}".format(outcome.reason))
    return float(outcome.row["drop_quantile"]), outcome.table


def run_window_grade_filter_ab(
    dataset_dir: Path,
    dataset_id: str,
    trend_model_id: str,
    outcome_model_id: str,
    output_dir: Path,
    horizon: int = 10,
    top_k: float = 0.1,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    c_values: Sequence[float] = (0.3, 1.0, 3.0),
    max_per_class: int = 60000,
    drop_quantiles: Sequence[float] = DEFAULT_DROP_QUANTILES,
) -> Dict[str, object]:
    """Fit the tail classifier once, pick ``q`` on validation, freeze to test."""

    if any(not 0.0 <= q < 1.0 for q in drop_quantiles):
        raise ValueError("drop_quantiles must lie in [0, 1)")
    if len(set(drop_quantiles)) != len(drop_quantiles):
        raise ValueError("drop_quantiles must be distinct")

    feature_columns = load_trend_feature_columns(dataset_dir, trend_model_id)
    classifier, preprocessor, selected_c = fit_grade_classifier(
        dataset_dir, dataset_id, feature_columns, c_values, max_per_class
    )

    # ---- validation: freeze the drop share ------------------------------------
    validation = load_combined_split(
        dataset_dir, dataset_id, outcome_model_id, "validation", classifier, preprocessor, feature_columns
    )
    validation_raw, validation_skipped = build_rank_market_data(validation, price_history_dir, SCORE_COLUMN)
    if not validation_raw:
        raise ValueError("No validation price data")
    selected_drop_quantile, validation_grid = choose_drop_quantile_on_validation(
        validation, validation_raw, drop_quantiles, initial_capital, max_positions,
        config_path, horizon, top_k,
    )

    # ---- test: baseline arm + freeze the selected arm, then report the rest ---
    test = load_combined_split(
        dataset_dir, dataset_id, outcome_model_id, "test", classifier, preprocessor, feature_columns
    )
    test_raw, test_skipped = build_rank_market_data(test, price_history_dir, SCORE_COLUMN)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions

    test_rows = []
    prepared_frames = {}
    for quantile in drop_quantiles:
        name = "none" if quantile == 0.0 else "drop_bad_top{:.0f}".format(quantile * 100)
        ban = bad_window_ban_mask(test, quantile)
        arm = adjusted_scores(test, ban)
        selected = select_top_k(arm, top_k, SCORE_COLUMN)
        metrics = run_arm(
            name, arm, test_raw, selected, initial_capital, max_positions, config_path, horizon
        )
        prepared_frames[name] = apply_rank_selection(test_raw, arm, selected)
        test_rows.append({
            "arm": name,
            "drop_quantile": float(quantile),
            "is_selected_on_validation": bool(quantile == selected_drop_quantile),
            "banned_row_count": int(ban.sum()),
            "banned_share": float(ban.mean()),
            "selected_count": int(np.asarray(selected, dtype=bool).sum()),
            **realized_outcome(test, selected, horizon),
            **{"test_" + key: value for key, value in metrics.items() if key != "strategy"},
        })
    test_arms = pd.DataFrame(test_rows)

    selected_name = "none" if selected_drop_quantile == 0.0 else "drop_bad_top{:.0f}".format(
        selected_drop_quantile * 100
    )
    baseline_name = BASELINE_ARM
    frictionless = run_arm(
        "trade_outcome_top_k_frictionless", test, test_raw,
        select_top_k(test, top_k, SCORE_COLUMN), initial_capital, max_positions, config_path, horizon,
        frictionless=True,
    )
    selected_prepared = prepared_frames[selected_name]
    passive_symbols = select_passive_symbols(test_raw, max_positions)
    passive_results, passive_trades = _run_engine(
        selected_prepared,
        make_passive_strategy(passive_symbols, target_notional),
        initial_capital, config_path, start_date, end_date,
    )
    passive = summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital)

    comparison = {
        "baseline_test_total_return": float(
            test_arms.loc[test_arms["arm"] == baseline_name, "test_total_return"].iloc[0]
        ),
        "selected_test_total_return": float(
            test_arms.loc[test_arms["arm"] == selected_name, "test_total_return"].iloc[0]
        ),
        "net_excess_vs_baseline": float(
            test_arms.loc[test_arms["arm"] == selected_name, "test_total_return"].iloc[0]
            - test_arms.loc[test_arms["arm"] == baseline_name, "test_total_return"].iloc[0]
        ),
        "friction_drag_selected": float(frictionless["total_return"]) - float(
            test_arms.loc[test_arms["arm"] == selected_name, "test_total_return"].iloc[0]
        ),
        "passive_total_return": float(passive["total_return"]),
        "selected_vs_passive": float(
            test_arms.loc[test_arms["arm"] == selected_name, "test_total_return"].iloc[0]
            - float(passive["total_return"])
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "window_grade_filter_{}".format(outcome_model_id)
    validation_grid.to_csv(output_dir / (prefix + "_validation_grid.csv"), index=False, encoding="utf-8-sig")
    test_arms.to_csv(output_dir / (prefix + "_test_arms.csv"), index=False, encoding="utf-8-sig")
    test.assign(selected_arm=selected_name, selected_top_k=top_k).to_csv(
        output_dir / (prefix + "_test_scores.csv"), index=False, encoding="utf-8-sig"
    )

    manifest = {
        "analysis_id": "window_grade_filter_ab",
        "created_at": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "trend_model_id": trend_model_id,
        "outcome_model_id": outcome_model_id,
        "overlay": "daily cross-sectional drop of the worst q share by P(bad_window); all other names keep exact entry_score",
        "grade_classifier": {
            "target": GRADE_COLUMN,
            "positive_classes_for_bad": list(BAD_CLASSES),
            "selected_c_validation": float(selected_c),
            "training": "train-eligible only, balanced class weights, capped",
            "max_per_class": int(max_per_class),
            "c_values": [float(c) for c in c_values],
        },
        "selection": {
            "top_k_frozen_from_baseline": float(top_k),
            "drop_quantile_selected_on_validation": float(selected_drop_quantile),
            "objective": "validation_net_total_return_then_lower_turnover",
        },
        "selected_arm": selected_name,
        "baseline_arm": baseline_name,
        "comparison": comparison,
        "execution": {
            "signal_timing": "EOD score at t, order at next session t+1",
            "min_holding_days": int(horizon),
            "rebalance": "weekly",
            "max_positions": max_positions,
            "initial_capital": initial_capital,
            "validation_skipped_prices": validation_skipped,
            "test_skipped_prices": test_skipped,
        },
        "paths": {
            "validation_grid": str(output_dir / (prefix + "_validation_grid.csv")),
            "test_arms": str(output_dir / (prefix + "_test_arms.csv")),
            "test_scores": str(output_dir / (prefix + "_test_scores.csv")),
        },
    }
    manifest_path = output_dir / (prefix + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="excess_hs300_v1")
    parser.add_argument("--trend-model-id", default="hs300_logistic_v2")
    parser.add_argument("--outcome-model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--c-values", default="0.3,1,3")
    parser.add_argument("--max-per-class", type=int, default=60000)
    parser.add_argument("--drop-quantiles", default="0,0.05,0.1,0.2")
    return parser.parse_args(argv)


def _float_values(value: str) -> list:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_window_grade_filter_ab(
        Path(args.dataset_dir), args.dataset_id, args.trend_model_id, args.outcome_model_id,
        Path(args.output_dir), horizon=args.horizon, top_k=args.top_k, max_positions=args.max_positions,
        initial_capital=args.initial_capital, price_history_dir=args.price_history_dir,
        config_path=args.config_path, c_values=_float_values(args.c_values),
        max_per_class=args.max_per_class, drop_quantiles=_float_values(args.drop_quantiles),
    )
    comparison = manifest["comparison"]
    print(
        "Filter A/B done. Selected drop={q}; baseline net={base:.2%} -> selected net={sel:.2%} "
        "(delta {delta:+.2%}); vs passive {passive:.2%}; test_arms={arms}".format(
            q=manifest["selection"]["drop_quantile_selected_on_validation"],
            base=comparison["baseline_test_total_return"],
            sel=comparison["selected_test_total_return"],
            delta=comparison["net_excess_vs_baseline"],
            passive=comparison["passive_total_return"],
            arms=manifest["paths"]["test_arms"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
