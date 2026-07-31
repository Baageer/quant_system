"""Backtest the direct T+1 outcome model's daily Top-K ``entry_score``."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.probability_trade_evaluation import (
    _run_engine,
    make_passive_strategy,
    select_passive_symbols,
    summarize_backtest,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_ranking_evaluation import (
    TOP_K_GRID,
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_model import target_columns


SCORE_COLUMN = "entry_score"


def load_outcome_scores(dataset_dir: Path, model_id: str, split: str, horizon: int) -> pd.DataFrame:
    """Load a direct outcome score table exported by ``trade_outcome_model``."""

    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    excess_column, mae_column = target_columns(horizon)
    frame = pd.read_csv(path, dtype={"symbol": str})
    required = {"date", "symbol", SCORE_COLUMN, excess_column, mae_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Outcome score file missing columns: {}".format(sorted(missing)))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    frame[SCORE_COLUMN] = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    frame[excess_column] = pd.to_numeric(frame[excess_column], errors="coerce")
    frame[mae_column] = pd.to_numeric(frame[mae_column], errors="coerce")
    return frame.dropna(subset=["date", "symbol", SCORE_COLUMN, excess_column, mae_column]).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def outcome_quality(scores: pd.DataFrame, selected: pd.Series, horizon: int) -> Dict[str, float]:
    """Report forward outcomes for an equal-weight daily selected basket."""

    excess_column, mae_column = target_columns(horizon)
    mask = np.asarray(selected, dtype=bool)
    selected_excess = np.asarray(scores.loc[mask, excess_column], dtype=float)
    selected_mae = np.asarray(scores.loc[mask, mae_column], dtype=float)
    universe_excess = np.asarray(scores[excess_column], dtype=float)
    return {
        "selected_count": int(mask.sum()),
        "selected_mean_excess_return": float(np.mean(selected_excess)),
        "selected_mean_mae": float(np.mean(selected_mae)),
        "selected_excess_over_universe": float(np.mean(selected_excess) - np.mean(universe_excess)),
    }


def evaluate_outcome_validation_grid(
    validation_scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
) -> pd.DataFrame:
    """Choose Top-K by validation net return after applying A-share constraints."""

    start_date, end_date = validation_scores["date"].min(), validation_scores["date"].max()
    target_notional = initial_capital / max_positions
    rows = []
    for top_k in TOP_K_GRID:
        selected = select_top_k(validation_scores, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, validation_scores, selected)
        results, trades = _run_engine(
            prepared,
            make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
            initial_capital,
            config_path,
            start_date,
            end_date,
        )
        performance = summarize_backtest("top_k", results, trades, initial_capital)
        rows.append(
            {
                "top_k": top_k,
                **outcome_quality(validation_scores, selected, horizon),
                **{"validation_" + key: value for key, value in performance.items() if key != "strategy"},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_total_return", "validation_turnover_multiple", "top_k"], ascending=[False, True, True]
    ).reset_index(drop=True)


def run_outcome_ranking_evaluation(
    dataset_dir: Path,
    model_id: str,
    output_dir: Path,
    horizon: int = 10,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
):
    """Select a validation Top-K and evaluate its direct score on the test split."""

    validation = load_outcome_scores(dataset_dir, model_id, "validation", horizon)
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    validation_raw, validation_skipped = build_rank_market_data(validation, price_history_dir, SCORE_COLUMN)
    if not validation_raw:
        raise ValueError("No validation price data")
    validation_grid = evaluate_outcome_validation_grid(
        validation, validation_raw, horizon, initial_capital, max_positions, config_path
    )
    selected_top_k = float(validation_grid.iloc[0]["top_k"])
    test_selected = select_top_k(test, selected_top_k, SCORE_COLUMN)
    test_raw, test_skipped = build_rank_market_data(test, price_history_dir, SCORE_COLUMN)
    prepared = apply_rank_selection(test_raw, test, test_selected)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    net_results, net_trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
    )
    free_results, free_trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
        frictionless=True,
    )
    passive_symbols = select_passive_symbols(prepared, max_positions)
    passive_results, passive_trades = _run_engine(
        prepared, make_passive_strategy(passive_symbols, target_notional), initial_capital, config_path, start_date, end_date
    )
    metrics = pd.DataFrame(
        [
            summarize_backtest("trade_outcome_top_k_net", net_results, net_trades, initial_capital),
            summarize_backtest("trade_outcome_top_k_frictionless", free_results, free_trades, initial_capital),
            summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "trade_outcome_ranking_{}".format(model_id)
    validation_grid.to_csv(output_dir / (prefix + "_validation_grid.csv"), index=False)
    test.assign(selected_top_k=np.asarray(test_selected, dtype=bool)).to_csv(
        output_dir / (prefix + "_test_scores.csv"), index=False
    )
    metrics.to_csv(output_dir / (prefix + "_backtest_metrics.csv"), index=False)
    net_results.to_csv(output_dir / (prefix + "_equity_net.csv"))
    free_results.to_csv(output_dir / (prefix + "_equity_frictionless.csv"))
    net_trades.to_csv(output_dir / (prefix + "_trades_net.csv"), index=False)
    manifest = {
        "model_id": model_id,
        "horizon": int(horizon),
        "score_column": SCORE_COLUMN,
        "selected_top_k_validation_only": selected_top_k,
        "selection_objective": "validation_net_total_return_then_lower_turnover",
        "test_forward_outcome_quality": outcome_quality(test, test_selected, horizon),
        "execution": {
            "signal_timing": "EOD score at t, order at next session t+1",
            "min_holding_days": int(horizon),
            "rebalance": "weekly",
            "max_positions": max_positions,
            "initial_capital": initial_capital,
            "validation_skipped_prices": validation_skipped,
            "test_skipped_prices": test_skipped,
        },
    }
    manifest_path = output_dir / (prefix + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_trade_outcome_v1")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    manifest = run_outcome_ranking_evaluation(
        Path(args.dataset_dir), args.model_id, Path(args.output_dir), args.horizon, args.initial_capital,
        args.max_positions, args.price_history_dir, args.config_path,
    )
    print("Selected validation Top-K={:.2%}".format(manifest["selected_top_k_validation_only"]))


if __name__ == "__main__":
    main()
