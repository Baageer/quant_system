"""Reconcile the two "no skill" baselines: pool buy-and-hold vs random 20 names.

``random_portfolio_baseline`` draws uniform-random 20-name portfolios and runs
them through the production execution chain.  On HS300 its 200-draw mean is
**+15.9%**, while the pool equal-weight figure quoted in
``docs/research/02_model_and_target/zz1000_full_pipeline_summary.md`` (+1.47%) was derived from the excess labels
by compounding the daily cross-sectional mean of ``future_excess_return_10d``
over non-overlapping 10-day windows.  Both are supposed to describe "buy the
pool and hold it".  A 14pp disagreement with a ~0.8pp standard error is
structural, not noise, and one of the two estimators is wrong -- so the earlier
write-up cannot be trusted until this is settled.

Three candidate explanations, all testable from data already on disk:

1. **Naive compounding of excess.**  ``prod(1 + mean_i(r_i - b))`` is not
   ``prod(1 + mean_i r_i) / prod(1 + b)``.  The two agree only to first order;
   with 10-day index moves of +-10% the second-order term accumulates over ~82
   windows.  ``excess_naive`` vs ``excess_exact`` below isolates it.
2. **The label universe is not the price universe.**  Rows survive into the
   score file only when score, excess AND mae are all non-null, so a name
   suspended or delisted near a window boundary silently leaves the pool.
3. **The 20-name basket is not the 300-name basket.**  Concentration changes the
   *realised* (geometric) path even when the cross-sectional mean is identical.

The script reports, on the same window: the price-weighted equal-weight pool
return (buy-and-hold and 10-day rebalanced), the index return, and both label
compositions of the excess, so the three can be told apart.

Example::

    python -m research.experiments.pool_return_audit --arm hs300
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.passive_benchmark import DEFAULT_BENCHMARK_FILE, load_index_levels
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_model import target_columns
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


ARMS = {
    "hs300": ("hs300_excess_v1_outcome", DEFAULT_BENCHMARK_FILE),
    "zz1000": ("zz1000_excess_v1_outcome", "data/benchmarks/000852.csv"),
}


def _non_overlapping(rebalance_days: int, length: int):
    return list(range(0, length, rebalance_days))


def pool_price_returns(
    market_data: Dict[str, pd.DataFrame], rebalance_days: int = 10
) -> Dict[str, float]:
    """Equal-weight pool return from prices: buy-and-hold and 10-day rebalanced."""

    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in market_data.items()})
    closes = closes.sort_index()
    hold = closes.iloc[-1] / closes.iloc[0] - 1.0

    starts = _non_overlapping(rebalance_days, len(closes) - rebalance_days)
    per_window = []
    for start in starts:
        begin = closes.iloc[start]
        end = closes.iloc[start + rebalance_days]
        window = (end / begin - 1.0).replace([np.inf, -np.inf], np.nan)
        per_window.append(float(window.mean(skipna=True)))
    rebalanced = float(np.prod([1.0 + value for value in per_window]) - 1.0)
    return {
        "buy_and_hold_equal_weight": float(hold.mean(skipna=True)),
        "buy_and_hold_median": float(hold.median(skipna=True)),
        "rebalanced_10d_equal_weight": rebalanced,
        "windows": len(per_window),
        "symbols": int(closes.shape[1]),
    }


def pool_excess_returns(
    scores: pd.DataFrame, levels: pd.Series, horizon: int, rebalance_days: int = 10
) -> Dict[str, float]:
    """The label-based pool excess, composed naively and exactly.

    ``future_excess_return_{h}d`` is already an *h*-day forward excess, so window
    *k* contributes ``e_k = mean_i(r_ik - b_k)`` at the window's start date and
    the matching index move ``b_k``.  The naive composition ``prod(1 + e_k)``
    ignores that the excess has to ride on top of the index's own compounding;
    the exact one is ``prod(1 + b_k + e_k)``.
    """

    excess_column, _ = target_columns(horizon)
    frame = scores[["date", "symbol", excess_column]].dropna(subset=[excess_column])
    per_date = frame.groupby("date")[excess_column].mean().sort_index()
    dates = per_date.index
    levels = levels.reindex(levels.index.union(dates)).ffill()
    if levels.loc[dates.min()] != levels.loc[dates.min()]:
        raise ValueError("Benchmark does not cover the score window")

    starts = _non_overlapping(rebalance_days, len(dates) - rebalance_days)
    excess_terms, index_terms = [], []
    for start in starts:
        begin, end = dates[start], dates[min(start + rebalance_days, len(dates) - 1)]
        excess_terms.append(float(per_date.iloc[start]))
        index_terms.append(float(levels.loc[end] / levels.loc[begin] - 1.0))
    naive = float(np.prod([1.0 + value for value in excess_terms]) - 1.0)
    index_return = float(np.prod([1.0 + value for value in index_terms]) - 1.0)
    pool_exact = float(np.prod([1.0 + b + e for b, e in zip(index_terms, excess_terms)]) - 1.0)
    return {
        "excess_naive_compounded": naive,
        "implied_pool_from_naive": (1.0 + naive) * (1.0 + index_return) - 1.0,
        "pool_exact_compounded": pool_exact,
        "excess_exact_compounded": pool_exact - index_return,
        "index_return": index_return,
        "per_date_mean_excess": float(per_date.mean()),
        "share_positive_dates": float((per_date > 0).mean()),
        "mean_symbols_per_date": float(frame.groupby("date").size().mean()),
        "rows": int(len(frame)),
    }


def simulate_churn(
    market_data: Dict[str, pd.DataFrame],
    draws: int = 200,
    max_positions: int = 20,
    rebalance_days: int = 5,
    seed: int = 20260915,
) -> Dict[str, float]:
    """Random 20 names replaced every ``rebalance_days`` -- without the engine.

    The engine's churning random-20 mean sits well above both the pool's
    buy-and-hold and its 10-day rebalanced return, and the gap is large enough
    to change how the strategy's own number should be read.  This replays the
    same idea in plain numpy, so an engine-side accounting effect can be told
    apart from a real property of the universe.
    """

    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in market_data.items()}).sort_index()
    values = closes.to_numpy(dtype=float)
    dates = _non_overlapping(rebalance_days, len(closes) - rebalance_days)
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(draws):
        equity = 1.0
        for start in dates:
            begin = values[start]
            end = values[start + rebalance_days]
            usable = np.isfinite(begin) & np.isfinite(end) & (begin > 0)
            picks = rng.choice(np.flatnonzero(usable), size=max_positions, replace=False)
            equity *= float(np.mean(end[picks] / begin[picks]))
        totals.append(equity - 1.0)
    totals = np.asarray(totals)
    return {
        "churn_draws": int(draws),
        "churn_mean": float(totals.mean()),
        "churn_sd": float(totals.std(ddof=1)),
        "churn_median": float(np.median(totals)),
        "churn_rebalance_days": int(rebalance_days),
    }


def run_pool_return_audit(
    dataset_dir: Path, model_id: str, benchmark_file: str, horizon: int = 10, churn_days=(5, 10, 15)
):
    scores = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    market_data, skipped = build_rank_market_data(scores, None, SCORE_COLUMN)
    levels = load_index_levels(benchmark_file)
    report = {
        "model_id": model_id,
        "benchmark_file": benchmark_file,
        "test_window": [str(scores["date"].min().date()), str(scores["date"].max().date())],
        "label_rows": int(len(scores)),
        "skipped_prices": len(skipped),
        "prices": pool_price_returns(market_data, horizon),
        "labels": pool_excess_returns(scores, levels, horizon, horizon),
        "churn": {
            "days_{}".format(days): simulate_churn(market_data, rebalance_days=days)["churn_mean"]
            for days in churn_days
        },
    }
    print(report["model_id"], report["test_window"])
    for section in ("prices", "labels", "churn"):
        for key, value in report[section].items():
            print("  {:<28} {}".format(key, _format(key, value)))
    return report


COUNT_KEYS = ("windows", "symbols", "rows", "mean_symbols_per_date")


def _format(key, value):
    """Percentages for return/ratio fields, plain numbers for counts."""

    if isinstance(value, float) and key not in COUNT_KEYS:
        return "{:.4%}".format(value)
    if isinstance(value, float):
        return "{:.1f}".format(value)
    return value


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="hs300")
    parser.add_argument("--model-id", default=None,
                        help="Override the arm's model id (e.g. a point-in-time rerun).")
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args(argv)
    model_id, benchmark_file = ARMS[args.arm]
    run_pool_return_audit(Path(args.dataset_dir), args.model_id or model_id,
                          args.benchmark_file or benchmark_file, args.horizon)


if __name__ == "__main__":
    main()
