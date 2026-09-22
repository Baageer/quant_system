"""Replay one random draw's engine fills to locate the return's source.

``engine_churn_diagnostic`` shows the engine's random-20 arm earns far more
than a numpy random-20 under every sizing assumption tried so far (the gap
survives a 100x capital increase, so lot rounding and the cash constraint are
not it).  This probe takes the fills the engine actually produced and re-asks
the question from the trades themselves:

1. replay the fills verbatim          -> must reproduce the engine (control)
2. hold every bought name to the end  -> is the *buy* selection special?
3. same buys, equal notional          -> is the trade *sizing* special?
4. the exchange split of what was bought vs the pool

Example::

    python -m research.experiments.churn_replay_probe
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import _score_slots, _write_rank_score
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


def _exchange(symbol: str) -> str:
    if symbol[:3] in ("000", "002", "300", "301"):
        return "SZ"
    if symbol[:3] in ("600", "601", "603", "605"):
        return "SH"
    return "other"


def _closes(raw_market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index()


def _replay(trades: pd.DataFrame, closes: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    """Cash ledger replay of the engine's own fills."""

    cash = initial_capital
    holdings: Dict[str, int] = {}
    for row in trades.itertuples():
        if row.action == "buy":
            cash -= row.shares * row.price + row.commission
            holdings[row.symbol] = holdings.get(row.symbol, 0) + row.shares
        elif row.action == "sell":
            cash += row.shares * row.price - row.commission - row.stamp_duty
            holdings[row.symbol] = holdings.get(row.symbol, 0) - row.shares
    final = closes.ffill().iloc[-1]
    stock_value = sum(shares * final.get(symbol, np.nan) for symbol, shares in holdings.items() if shares > 0)
    return {"replay_final": float(cash + stock_value), "replay_return": float((cash + stock_value) / initial_capital - 1.0)}


def _hold_to_end(trades: pd.DataFrame, closes: pd.DataFrame, initial_capital: float, max_positions: int) -> Dict[str, float]:
    """Buy what the engine bought, when it bought, and never sell.

    Each buy still deploys ``initial_capital / max_positions``; with repeated
    buys of the same name these stack, so the position count is not capped --
    this is deliberately the naive "just own the picks" counterfactual.
    """

    prices = closes.ffill()
    notional = initial_capital / max_positions
    holdings: Dict[str, float] = {}
    for row in trades[trades["action"] == "buy"].itertuples():
        holdings[row.symbol] = holdings.get(row.symbol, 0.0) + notional
    total = 0.0
    for symbol, amount in holdings.items():
        series = prices[symbol].dropna()
        if series.empty:
            total += amount
            continue
        first_date = trades[(trades["symbol"] == symbol) & (trades["action"] == "buy")]["date"].min()
        entry = prices[symbol].asof(first_date)
        exit_price = series.iloc[-1]
        total += amount * (exit_price / entry) if entry and entry > 0 else amount
    return {"hold_to_end_value": float(total), "hold_to_end_return": float(total / initial_capital - 1.0)}


def run_probe(
    dataset_dir: Path,
    model_id: str,
    seed: int = 20260915,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    config_path: str = "config/settings.yaml",
) -> Dict[str, object]:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw_market_data, _ = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    shuffled = np.random.default_rng(seed).permutation(values)

    probe = test.copy()
    probe[SCORE_COLUMN] = shuffled
    selected = select_top_k(probe, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(raw_market_data, probe, selected)
    _write_rank_score(prepared, shuffled, slots)

    results, trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, initial_capital / max_positions, min_holding_days=horizon),
        initial_capital,
        config_path,
        test["date"].min(),
        test["date"].max(),
    )
    performance = summarize_backtest("random", results, trades, initial_capital)
    closes = _closes(raw_market_data)

    buys = trades[(trades["action"] == "buy") & (trades["status"] == "filled")].copy()
    buys["exchange"] = buys["symbol"].map(_exchange)
    pool = pd.Series({s: _exchange(s) for s in closes.columns})
    pool_ret = (closes.ffill().iloc[-1] / closes.ffill().iloc[0] - 1.0)

    # Holding period per round trip, matched symbol-by-symbol in fill order.
    holding_days = []
    open_lots: Dict[str, int] = {}
    for row in trades[trades["status"] == "filled"].itertuples():
        if row.action == "buy":
            open_lots.setdefault(row.symbol, row.date)
        elif row.symbol in open_lots:
            holding_days.append((row.date - open_lots.pop(row.symbol)).days)

    return {
        "engine_total_return": float(performance["total_return"]),
        "engine_turnover": float(performance["turnover_multiple"]),
        "replay": _replay(trades, closes, initial_capital),
        "hold_to_end": _hold_to_end(trades, closes, initial_capital, max_positions),
        "buy_exchange_mix": buys["exchange"].value_counts(normalize=True).round(4).to_dict(),
        "pool_exchange_mix": pool.value_counts(normalize=True).round(4).to_dict(),
        "pool_exchange_return": {k: float(pool_ret[pool[pool == k].index].mean()) for k in sorted(pool.unique())},
        "mean_holding_days": float(np.mean(holding_days)) if holding_days else float("nan"),
        "median_holding_days": float(np.median(holding_days)) if holding_days else float("nan"),
        "round_trips": len(holding_days),
    }


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--config-path", default="config/settings.yaml")
    args = parser.parse_args(argv)
    report = run_probe(
        Path(args.dataset_dir), args.model_id, args.seed, args.top_k, args.horizon,
        args.max_positions, args.initial_capital, args.config_path,
    )
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
