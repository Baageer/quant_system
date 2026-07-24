"""Validate ``p_tradable`` as a cross-sectional Top-K trade filter."""

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from research.market_regime.data_loader import load_local_price_history
from research.market_regime.probability_trade_evaluation import (
    _run_engine,
    make_passive_strategy,
    select_passive_symbols,
    summarize_backtest,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_model import POSITIVE_CLASS, TARGET_COLUMN, _weights


TOP_K_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def load_score_predictions(dataset_dir: Path, model_id: str, split: str) -> pd.DataFrame:
    path = dataset_dir / "trend_tradability_{}_entry_scores_{}.csv".format(split, model_id)
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    required = {"date", "symbol", "p_tradable", TARGET_COLUMN, "sample_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Score file missing columns: {}".format(sorted(missing)))
    return frame.dropna(subset=["date", "symbol", "p_tradable"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def select_top_k(scores: pd.DataFrame, top_k: float) -> pd.Series:
    """Pick the highest ``p_tradable`` observations in each daily universe."""

    if not 0 < top_k <= 1:
        raise ValueError("top_k must be in (0, 1]")
    ranks = scores.groupby("date", sort=False)["p_tradable"].rank(method="first", ascending=False)
    counts = scores.groupby("date", sort=False)["p_tradable"].transform("size")
    return (ranks <= np.ceil(counts * top_k)).astype(bool)


def ranking_quality(scores: pd.DataFrame, selected: pd.Series) -> Dict[str, float]:
    labels = np.asarray(scores[TARGET_COLUMN].eq(POSITIVE_CLASS), dtype=bool)
    candidates = np.asarray(selected, dtype=bool)
    weights = _weights(scores)
    true_positive_weight = float(np.sum(weights[candidates & labels]))
    candidate_weight = float(np.sum(weights[candidates]))
    positive_weight = float(np.sum(weights[labels]))
    precision = true_positive_weight / candidate_weight if candidate_weight else 0.0
    recall = true_positive_weight / positive_weight if positive_weight else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "selected_count": int(candidates.sum()),
        "selected_positive_count": int((candidates & labels).sum()),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }


def build_rank_market_data(scores: pd.DataFrame, price_history_dir: Optional[str] = None):
    """Load test/validation OHLCV once and attach only the causal score."""

    market_data = {}
    skipped = []
    start_date, end_date = scores["date"].min(), scores["date"].max()
    for symbol in tqdm(sorted(scores["symbol"].unique()), desc="Loading ranking prices", unit="symbol"):
        try:
            prices = load_local_price_history(symbol, price_history_dir=price_history_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)].copy()
        if prices.empty:
            skipped.append({"symbol": symbol, "reason": "no_prices_in_period"})
            continue
        symbol_scores = scores[scores["symbol"] == symbol].set_index("date")
        prices = prices.join(symbol_scores[["p_tradable"]], how="left")
        prices["p_tradable"] = prices["p_tradable"].fillna(0.0)
        market_data[symbol] = prices
    return market_data, skipped


def apply_rank_selection(
    raw_market_data: Mapping[str, pd.DataFrame], scores: pd.DataFrame, selected: pd.Series
) -> Dict[str, pd.DataFrame]:
    """Shift EOD cross-sectional membership to next-session trade signals."""

    decisions = scores.loc[:, ["date", "symbol"]].copy()
    decisions["eod_candidate"] = np.asarray(selected, dtype=bool)
    prepared = {}
    for symbol, prices in raw_market_data.items():
        frame = prices.copy()
        symbol_decisions = decisions[decisions["symbol"] == symbol].set_index("date")
        frame = frame.join(symbol_decisions, how="left")
        frame["eod_candidate"] = frame["eod_candidate"].fillna(False).astype(bool)
        frame["entry_trade"] = frame["eod_candidate"].shift(1).fillna(False).astype(bool)
        frame["hold_trade"] = frame["eod_candidate"].shift(1).fillna(False).astype(bool)
        prepared[symbol] = frame
    return prepared


def make_rank_strategy(max_positions: int, target_notional: float, min_holding_days: int = 5, lot_size: int = 100):
    """Weekly ranked entries, with a minimum holding period before replacement."""

    entry_dates: Dict[str, pd.Timestamp] = {}
    current_week = None

    def strategy(date, relevant_data, positions):
        nonlocal current_week
        iso = pd.Timestamp(date).isocalendar()
        week = (int(iso[0]), int(iso[1]))
        if week == current_week:
            return {}
        current_week = week
        held = set(positions)
        for symbol in held:
            if symbol in relevant_data:
                entry_dates.setdefault(symbol, pd.Timestamp(date))
        for symbol in list(entry_dates):
            if symbol not in held:
                del entry_dates[symbol]
        signals = {}
        for symbol in sorted(held):
            frame = relevant_data.get(symbol)
            if frame is None or bool(frame.loc[date, "hold_trade"]):
                continue
            entry_date = entry_dates.get(symbol, pd.Timestamp(date))
            entry_position = int(frame.index.searchsorted(entry_date, side="left"))
            days = int(frame.index.get_loc(date)) - entry_position
            if days >= min_holding_days:
                signals[symbol] = {"action": "sell", "shares": positions[symbol], "reason": "tradability_rank_exit"}
        slots = max(max_positions - len(held), 0)
        candidates = []
        for symbol, frame in relevant_data.items():
            if symbol not in held and bool(frame.loc[date, "entry_trade"]):
                candidates.append((symbol, float(frame.loc[date, "p_tradable"]), float(frame.loc[date, "close"])))
        for symbol, _, price in sorted(candidates, key=lambda item: (-item[1], item[0]))[:slots]:
            shares = int(np.floor(target_notional / price / lot_size) * lot_size) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "tradability_rank_entry_t_plus_1"}
        return signals

    return strategy


def evaluate_validation_grid(
    validation_scores: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    initial_capital: float,
    max_positions: int,
    config_path: str,
) -> pd.DataFrame:
    """Select Top-K by validation net return; quality metrics remain visible."""

    start_date, end_date = validation_scores["date"].min(), validation_scores["date"].max()
    target_notional = initial_capital / max_positions
    rows = []
    for top_k in TOP_K_GRID:
        selected = select_top_k(validation_scores, top_k)
        prepared = apply_rank_selection(raw_market_data, validation_scores, selected)
        results, trades = _run_engine(
            prepared, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date
        )
        performance = summarize_backtest("top_k", results, trades, initial_capital)
        rows.append({"top_k": top_k, **ranking_quality(validation_scores, selected), **{"validation_" + key: value for key, value in performance.items() if key != "strategy"}})
    return pd.DataFrame(rows).sort_values(
        ["validation_total_return", "validation_turnover_multiple", "top_k"], ascending=[False, True, True]
    ).reset_index(drop=True)


def run_ranking_evaluation(
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
    if not validation_raw:
        raise ValueError("No validation price data")
    validation_grid = evaluate_validation_grid(validation, validation_raw, initial_capital, max_positions, config_path)
    selected_top_k = float(validation_grid.iloc[0]["top_k"])
    test_selected = select_top_k(test, selected_top_k)
    test_raw, test_skipped = build_rank_market_data(test, price_history_dir)
    prepared_test = apply_rank_selection(test_raw, test, test_selected)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    net_results, net_trades = _run_engine(prepared_test, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date)
    free_results, free_trades = _run_engine(prepared_test, make_rank_strategy(max_positions, target_notional), initial_capital, config_path, start_date, end_date, frictionless=True)
    passive_symbols = select_passive_symbols(prepared_test, max_positions)
    passive_results, passive_trades = _run_engine(prepared_test, make_passive_strategy(passive_symbols, target_notional), initial_capital, config_path, start_date, end_date)
    metrics = pd.DataFrame([
        summarize_backtest("p_tradable_top_k_net", net_results, net_trades, initial_capital),
        summarize_backtest("p_tradable_top_k_frictionless", free_results, free_trades, initial_capital),
        summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital),
    ])
    test_quality = ranking_quality(test, test_selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "trend_tradability_ranking_{}".format(model_id)
    validation_grid.to_csv(output_dir / (prefix + "_validation_grid.csv"), index=False)
    test.assign(selected_top_k=np.asarray(test_selected, dtype=bool)).to_csv(output_dir / (prefix + "_test_scores.csv"), index=False)
    metrics.to_csv(output_dir / (prefix + "_backtest_metrics.csv"), index=False)
    net_results.to_csv(output_dir / (prefix + "_equity_net.csv"))
    free_results.to_csv(output_dir / (prefix + "_equity_frictionless.csv"))
    net_trades.to_csv(output_dir / (prefix + "_trades_net.csv"), index=False)
    manifest = {
        "model_id": model_id,
        "selected_top_k_validation_only": selected_top_k,
        "selection_objective": "validation_net_total_return_then_lower_turnover",
        "test_ranking_quality": test_quality,
        "execution": {"signal_timing": "EOD rank at t, order at next session t+1", "min_holding_days": 5, "rebalance": "weekly", "max_positions": max_positions, "initial_capital": initial_capital, "validation_skipped_prices": validation_skipped, "test_skipped_prices": test_skipped},
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
    manifest = run_ranking_evaluation(Path(args.dataset_dir), args.model_id, Path(args.output_dir), args.initial_capital, args.max_positions, args.price_history_dir, args.config_path)
    print("Selected validation Top-K={:.2%}; test weighted precision={:.4%}".format(manifest["selected_top_k_validation_only"], manifest["test_ranking_quality"]["weighted_precision"]))


if __name__ == "__main__":
    main()
