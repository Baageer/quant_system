"""Decompose the unexplained "engine random-20 beats numpy random-20" gap.

``random_portfolio_baseline`` reports the HS300 random arm at +15.85% (200-draw
mean) while ``pool_return_audit.simulate_churn`` -- the same "random 20 names,
rebalance every N days" idea in plain numpy -- lands at +2.15%.  The static
variant (``--min-holding-days 100000``) agrees with the pool buy-and-hold
(+6.41% vs +6.37%), so the two price sources are consistent; the disagreement
lives in the trading dynamics, not the data.

This script runs ONE draw through the production chain and reports the things
that a scalar return hides: realised exposure, position count, the price
profile of what actually got bought, and a beta/alpha split of the daily
portfolio return against the equal-weight pool.  It then replays the same
selection under progressively more engine-like assumptions to see which one
moves the number.

Example::

    python -m research.experiments.engine_churn_diagnostic
"""

import argparse
import json
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


def _exposure(results: pd.DataFrame) -> Dict[str, float]:
    """How much of the book was actually in stocks, and in how many names."""

    counts = results["positions"].apply(lambda d: sum(1 for v in d.values() if v > 0))
    invested = 1.0 - results["cash"] / results["portfolio_value"]
    return {
        "sessions": int(len(results)),
        "mean_positions": float(counts.mean()),
        "median_positions": float(counts.median()),
        "max_positions": int(counts.max()),
        "min_positions": int(counts.min()),
        "mean_exposure": float(invested.mean()),
        "median_exposure": float(invested.median()),
        "min_exposure": float(invested.min()),
        "max_exposure": float(invested.max()),
        "mean_cash_fraction": float((results["cash"] / results["portfolio_value"]).mean()),
    }


def _pool_daily(raw_market_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight pool daily return, for the beta/alpha split."""

    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in raw_market_data.items()}).sort_index()
    returns = closes.pct_change()
    return returns.mean(axis=1, skipna=True)


def _beta_alpha(portfolio: pd.Series, pool: pd.Series) -> Dict[str, float]:
    joined = pd.concat([portfolio.rename("p"), pool.rename("m")], axis=1).dropna()
    covariance = np.cov(joined["p"], joined["m"])
    beta = float(covariance[0, 1] / covariance[1, 1]) if covariance[1, 1] > 0 else float("nan")
    alpha_daily = float(joined["p"].mean() - beta * joined["m"].mean())
    return {
        "beta_vs_pool": beta,
        "alpha_daily": alpha_daily,
        "alpha_annualized": float((1.0 + alpha_daily) ** 252 - 1.0),
        "corr": float(joined["p"].corr(joined["m"])),
        "sessions": int(len(joined)),
    }


def _buy_profile(trades: pd.DataFrame) -> Dict[str, object]:
    """What the engine actually bought -- price levels and order sizes."""

    buys = trades[trades["action"] == "buy"]
    filled = buys[buys["status"] == "filled"]
    rejected = buys[buys["status"] == "rejected"]
    notional = filled["shares"] * filled["price"]
    return {
        "buy_signals": int(len(buys)),
        "filled": int(len(filled)),
        "rejected": int(len(rejected)),
        "reject_reasons": rejected["rejection_reason"].value_counts().to_dict() if len(rejected) else {},
        "distinct_symbols_bought": int(filled["symbol"].nunique()),
        "mean_buy_price": float(filled["price"].mean()),
        "median_buy_price": float(filled["price"].median()),
        "mean_notional": float(notional.mean()),
        "median_notional": float(notional.median()),
        "p10_notional": float(notional.quantile(0.10)),
        "p90_notional": float(notional.quantile(0.90)),
    }


def run_diagnostic(
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
    raw_market_data, skipped = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values)

    probe = test.copy()
    probe[SCORE_COLUMN] = shuffled
    selected = select_top_k(probe, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(raw_market_data, probe, selected)
    _write_rank_score(prepared, shuffled, slots)

    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    results, trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
    )
    performance = summarize_backtest("random", results, trades, initial_capital)

    portfolio_daily = results["portfolio_value"].pct_change()
    pool_daily = _pool_daily(raw_market_data).reindex(portfolio_daily.index)

    # How many candidates does a day actually offer?  A small daily universe
    # would make "top 10%" much narrower than the pool.
    per_date = test.groupby("date").size()
    selected_per_date = probe.groupby("date")[SCORE_COLUMN].apply(lambda s: int(np.ceil(len(s) * top_k)))

    report = {
        "model_id": model_id,
        "seed": int(seed),
        "window": [str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date())],
        "skipped_prices": len(skipped),
        "total_return": float(performance["total_return"]),
        "turnover_multiple": float(performance["turnover_multiple"]),
        "filled_buy_count": float(performance["filled_buy_count"]),
        "exposure": _exposure(results),
        "beta_alpha": _beta_alpha(portfolio_daily, pool_daily),
        "buy_profile": _buy_profile(trades),
        "universe": {
            "test_rows": int(len(test)),
            "test_symbols": int(test["symbol"].nunique()),
            "price_symbols": int(len(raw_market_data)),
            "mean_rows_per_date": float(per_date.mean()),
            "min_rows_per_date": int(per_date.min()),
            "max_rows_per_date": int(per_date.max()),
            "mean_selected_per_date": float(selected_per_date.mean()),
        },
        "pool_buy_and_hold": float(
            (pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index().iloc[-1]
             / pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index().iloc[0] - 1.0).mean()
        ),
    }
    return report


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

    report = run_diagnostic(
        Path(args.dataset_dir),
        args.model_id,
        args.seed,
        args.top_k,
        args.horizon,
        args.max_positions,
        args.initial_capital,
        args.config_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
