"""Validate a two-stage ``p_up`` trigger plus ``p_tradable`` confirmation."""

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.probability_trade_evaluation import _run_engine, make_passive_strategy, select_passive_symbols, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    load_score_predictions,
    make_rank_strategy,
    ranking_quality,
)


P_UP_TOP_K_GRID = (0.01, 0.02, 0.03, 0.05)
P_TRADABLE_TOP_K_GRID = (0.005, 0.01, 0.02, 0.03, 0.05)


def _daily_top_k(scores: pd.DataFrame, column: str, top_k: float) -> pd.Series:
    ranks = scores.groupby("date", sort=False)[column].rank(method="first", ascending=False)
    counts = scores.groupby("date", sort=False)[column].transform("size")
    return (ranks <= np.ceil(counts * top_k)).astype(bool)


def select_joint_signals(scores: pd.DataFrame, p_up_top_k: float, p_tradable_top_k: float) -> pd.Series:
    """Require both trend-event ranking and tradability ranking on the same EOD."""

    if "p_up" not in scores.columns:
        raise ValueError("Joint signal evaluation requires p_up")
    return _daily_top_k(scores, "p_up", p_up_top_k) & _daily_top_k(scores, "p_tradable", p_tradable_top_k)


def evaluate_joint_validation_grid(
    validation_scores: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    initial_capital: float,
    max_positions: int,
    config_path: str,
    min_filled_buys: int = 20,
) -> pd.DataFrame:
    """Choose a two-stage filter using validation net results only."""

    start_date, end_date = validation_scores["date"].min(), validation_scores["date"].max()
    target_notional = initial_capital / max_positions
    records = []
    for p_up_top_k in P_UP_TOP_K_GRID:
        for p_tradable_top_k in P_TRADABLE_TOP_K_GRID:
            selected = select_joint_signals(validation_scores, p_up_top_k, p_tradable_top_k)
            prepared = apply_rank_selection(raw_market_data, validation_scores, selected)
            results, trades = _run_engine(
                prepared, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date
            )
            performance = summarize_backtest("joint", results, trades, initial_capital)
            records.append(
                {
                    "p_up_top_k": p_up_top_k,
                    "p_tradable_top_k": p_tradable_top_k,
                    "eligible": performance["filled_buy_count"] >= min_filled_buys,
                    **ranking_quality(validation_scores, selected),
                    **{"validation_" + key: value for key, value in performance.items() if key != "strategy"},
                }
            )
    return pd.DataFrame(records).sort_values(
        ["eligible", "validation_total_return", "validation_turnover_multiple", "p_up_top_k", "p_tradable_top_k"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def run_joint_evaluation(
    dataset_dir: Path,
    model_id: str,
    output_dir: Path,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
):
    validation = load_score_predictions(dataset_dir, model_id, "validation")
    test = load_score_predictions(dataset_dir, model_id, "test")
    validation_raw, validation_skipped = build_rank_market_data(validation, price_history_dir)
    grid = evaluate_joint_validation_grid(validation, validation_raw, initial_capital, max_positions, config_path)
    choice = grid.iloc[0]
    p_up_top_k = float(choice["p_up_top_k"])
    p_tradable_top_k = float(choice["p_tradable_top_k"])
    test_selected = select_joint_signals(test, p_up_top_k, p_tradable_top_k)
    test_raw, test_skipped = build_rank_market_data(test, price_history_dir)
    prepared = apply_rank_selection(test_raw, test, test_selected)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    net_results, net_trades = _run_engine(prepared, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date)
    free_results, free_trades = _run_engine(prepared, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date, frictionless=True)
    passive_symbols = select_passive_symbols(prepared, max_positions)
    passive_results, passive_trades = _run_engine(prepared, make_passive_strategy(passive_symbols, target_notional), initial_capital, config_path, start_date, end_date)
    metrics = pd.DataFrame([
        summarize_backtest("p_up_and_p_tradable_net", net_results, net_trades, initial_capital),
        summarize_backtest("p_up_and_p_tradable_frictionless", free_results, free_trades, initial_capital),
        summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital),
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "trend_joint_signal_{}".format(model_id)
    grid.to_csv(output_dir / (prefix + "_validation_grid.csv"), index=False)
    test.assign(selected_joint=np.asarray(test_selected, dtype=bool)).to_csv(output_dir / (prefix + "_test_scores.csv"), index=False)
    metrics.to_csv(output_dir / (prefix + "_backtest_metrics.csv"), index=False)
    net_results.to_csv(output_dir / (prefix + "_equity_net.csv"))
    free_results.to_csv(output_dir / (prefix + "_equity_frictionless.csv"))
    net_trades.to_csv(output_dir / (prefix + "_trades_net.csv"), index=False)
    manifest = {
        "model_id": model_id,
        "selected_validation_only": {"p_up_top_k": p_up_top_k, "p_tradable_top_k": p_tradable_top_k},
        "selection_objective": "validation_net_total_return_then_lower_turnover_with_minimum_buys",
        "test_ranking_quality": ranking_quality(test, test_selected),
        "execution": {"signal_timing": "EOD joint signal at t, order at t+1", "weekly_rebalance": True, "min_holding_days": 5, "max_positions": max_positions, "initial_capital": initial_capital, "validation_skipped_prices": validation_skipped, "test_skipped_prices": test_skipped},
        "paths": {"validation_grid": str(output_dir / (prefix + "_validation_grid.csv")), "test_scores": str(output_dir / (prefix + "_test_scores.csv")), "backtest_metrics": str(output_dir / (prefix + "_backtest_metrics.csv"))},
    }
    with (output_dir / (prefix + "_manifest.json")).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    return manifest


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_tradability_v1")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    manifest = run_joint_evaluation(Path(args.dataset_dir), args.model_id, Path(args.output_dir), args.initial_capital, args.max_positions, args.price_history_dir, args.config_path)
    choice = manifest["selected_validation_only"]
    print("Selected p_up Top-K={:.2%}, p_tradable Top-K={:.2%}".format(choice["p_up_top_k"], choice["p_tradable_top_k"]))


if __name__ == "__main__":
    main()
