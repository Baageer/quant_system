"""How much of an arm's *absolute* return is its exposure path rather than its stock picking?

``engine_churn_anomaly_resolved.md`` established that the production closure's
exposure ``w = 持仓市值/(持仓市值+现金)`` is marked-to-market daily, so ``w`` rises
after the market has risen and falls after it has fallen.  In a market whose
daily returns are positively autocorrelated that makes ``cov(w_t, r_t) > 0``
mechanical, and it pays the book a term that a fully-invested index can never
collect.  Measured on HS300 test 2022-07-20 -> 2025-12-02 that term is worth
**+6pp to +19pp**, and it is the entire gap between the engine's random arm and
a numpy buy-and-hold replication of it.

That term acts on a mechanism the model arm and the random arm *share*, so it
cancels in a difference and the published ranking conclusions survive.  But any
absolute number -- above all any "excess over the index" -- still carries it,
and cannot be quoted as skill until it is stripped.

This script strips it.  It runs an arm through the untouched production chain,
then replays **the engine's own daily holdings** with the exposure pinned:

    V_t = V_{t-1} * (1 + w * r_t) - cost_t

where ``r_t`` is the gross return of exactly the names the book held at the
previous close, and ``cost_t`` the actual commissions and stamp duty that
session.  Names, entry/exit dates, the trade schedule and the costs are all
unchanged -- the *only* thing flattened is the exposure path.  Running it for
the model arm and for N score-permuted random draws gives:

* ``pollution`` = actual - constant-exposure, per arm;
* ``model - random`` under both treatments, which is what should be quoted.

Two pins are reported: ``w_bar`` (the arm's own mean exposure -- "same level,
timing destroyed") and ``1.00`` (强制满仓).  Example::

    python -m research.experiments.constant_exposure_arm --arm hs300 --split test --draws 3
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import (
    _score_slots,
    _write_rank_score,
    carry_forward_marks,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


ARMS = {
    "hs300": "hs300_excess_v1_outcome",
    "zz1000": "zz1000_excess_v1_outcome",
}

REPORT_METRICS = ("total_return", "annual_volatility", "sharpe_zero_rate", "max_drawdown")


def basket_returns(results: pd.DataFrame, marks: pd.DataFrame) -> pd.Series:
    """Gross return of the names held at the previous close, per session.

    Uses yesterday's share counts against both marks, so it is a pure price
    return -- today's trades are deliberately excluded; their effect is carried
    by the explicit cost term and by the exposure pin.
    """

    dates = list(results.index)
    holdings = list(results["positions"])
    returns = []
    for position, (previous, current) in enumerate(zip(dates[:-1], dates[1:])):
        held = holdings[position]
        if not held:
            returns.append(0.0)
            continue
        before = sum(shares * marks.at[previous, symbol] for symbol, shares in held.items())
        after = sum(shares * marks.at[current, symbol] for symbol, shares in held.items())
        returns.append(float(after / before - 1.0) if before > 0 else 0.0)
    return pd.Series(returns, index=dates[1:])


def daily_costs(trades: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Commission plus stamp duty actually paid on each session."""

    costs = pd.Series(0.0, index=index)
    if trades.empty:
        return costs
    filled = trades[trades["status"] == "filled"]
    if filled.empty:
        return costs
    per_day = filled.groupby("date")[["commission", "stamp_duty"]].sum().sum(axis=1)
    per_day.index = pd.DatetimeIndex(per_day.index)
    costs.loc[per_day.index.intersection(index)] = per_day.loc[per_day.index.intersection(index)]
    return costs


def constant_exposure_path(
    initial_value: float,
    returns: pd.Series,
    costs: pd.Series,
    exposure: float,
    index: pd.Index,
) -> pd.Series:
    """Replay the arm's own basket returns with the exposure pinned to ``exposure``."""

    value = initial_value
    values = [value]
    for date in index[1:]:
        value = value * (1.0 + exposure * float(returns.get(date, 0.0))) - float(costs.get(date, 0.0))
        values.append(value)
    return pd.Series(values, index=index)


def shuffled_exposure_path(
    initial_value: float,
    returns: pd.Series,
    costs: pd.Series,
    exposures: pd.Series,
    index: pd.Index,
    rng: np.random.Generator,
) -> pd.Series:
    """Same exposure *values* in a random order -- level kept, ordering destroyed.

    If this lands on top of the pinned path rather than the actual one, the
    book is being paid for the *order* its exposure appears in, which is exactly
    the ``T*cov(w_t, r_t)`` term, and pinning is the right strip.
    """

    shuffled = rng.permutation(exposures.to_numpy())
    value = initial_value
    values = [value]
    for position, date in enumerate(index[1:]):
        value = value * (1.0 + shuffled[position] * float(returns.get(date, 0.0))) - float(costs.get(date, 0.0))
        values.append(value)
    return pd.Series(values, index=index)


def _as_results(values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"portfolio_value": values}, index=values.index)


def measure_arm(
    prepared: Dict[str, pd.DataFrame],
    context: Dict[str, object],
    initial_capital: float,
) -> Dict[str, object]:
    """Actual path, its pinned-exposure replays, and the implied pollution."""

    results, trades = _run_engine(
        prepared,
        make_rank_strategy(
            context["max_positions"], context["target_notional"], min_holding_days=context["horizon"]
        ),
        initial_capital,
        context["config_path"],
        context["start_date"],
        context["end_date"],
    )
    marks = carry_forward_marks({s: f["close"] for s, f in prepared.items()}, results.index)
    returns = basket_returns(results, marks)
    costs = daily_costs(trades, results.index)

    sleeve = pd.Series(
        [sum(shares * marks.at[date, s] for s, shares in held.items()) for date, held in results["positions"].items()],
        index=results.index,
    )
    total = results["portfolio_value"].astype(float)
    w_bar = float((sleeve / total).mean())

    empty = trades.iloc[0:0]
    actual = summarize_backtest("actual", results, trades, initial_capital)
    pinned = {}
    for exposure in (w_bar, 1.0):
        path = constant_exposure_path(total.iloc[0], returns, costs, exposure, results.index)
        pinned[exposure] = summarize_backtest("pinned", _as_results(path), empty, initial_capital)

    rng = np.random.default_rng(context["seed"])
    exposures = (sleeve / total).reindex(returns.index)
    shuffled = [
        summarize_backtest(
            "shuffled",
            _as_results(shuffled_exposure_path(total.iloc[0], returns, costs, exposures, results.index, rng)),
            empty,
            initial_capital,
        )["total_return"]
        for _ in range(context["shuffles"])
    ]

    return {
        "w_bar": w_bar,
        "actual": actual,
        "pinned_w_bar": pinned[w_bar],
        "pinned_full": pinned[1.0],
        "shuffled_mean": float(np.mean(shuffled)),
        "pollution": float(actual["total_return"]) - float(pinned[w_bar]["total_return"]),
    }


def run(
    dataset_dir: Path,
    model_id: str,
    split: str = "test",
    draws: int = 3,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    config_path: str = "config/settings.yaml",
    seed: int = 20260915,
    shuffles: int = 20,
) -> pd.DataFrame:
    scores = load_outcome_scores(dataset_dir, model_id, split, horizon)
    raw_market_data, _ = build_rank_market_data(scores, None, SCORE_COLUMN)
    if not raw_market_data:
        raise ValueError("No price data for {} {}".format(model_id, split))
    slots = _score_slots(scores, raw_market_data)
    values = np.asarray(scores[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)
    context: Dict[str, object] = {
        "max_positions": max_positions,
        "target_notional": initial_capital / max_positions,
        "horizon": horizon,
        "config_path": config_path,
        "start_date": scores["date"].min(),
        "end_date": scores["date"].max(),
        "seed": seed,
        "shuffles": shuffles,
    }

    rows: List[Dict[str, object]] = []
    for draw in list(range(draws)) + [-1]:
        label = "REAL" if draw < 0 else "random {:>2}".format(draw)
        probe = scores.copy()
        probe[SCORE_COLUMN] = values if draw < 0 else rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        measured = measure_arm(prepared, context, initial_capital)
        rows.append({
            "label": label,
            "w_bar": measured["w_bar"],
            "actual": measured["actual"]["total_return"],
            "pinned_w_bar": measured["pinned_w_bar"]["total_return"],
            "pinned_full": measured["pinned_full"]["total_return"],
            "shuffled_mean": measured["shuffled_mean"],
            "pollution": measured["pollution"],
            "actual_sharpe": measured["actual"]["sharpe_zero_rate"],
            "pinned_sharpe": measured["pinned_w_bar"]["sharpe_zero_rate"],
        })
        frame = rows[-1]
        print(
            "{:<10} w_bar={:.3f}  actual={:>9.2%}  pinned(w_bar)={:>9.2%}  shuffled={:>9.2%}  "
            "pinned(1.00)={:>9.2%}  pollution={:>8.2%}".format(
                label, frame["w_bar"], frame["actual"], frame["pinned_w_bar"],
                frame["shuffled_mean"], frame["pinned_full"], frame["pollution"],
            ),
            flush=True,
        )

    frame = pd.DataFrame(rows)
    real = frame[frame["label"] == "REAL"].iloc[0]
    random_rows = frame[frame["label"] != "REAL"]
    print()
    print("random mean  actual={:>9.2%}  pinned(w_bar)={:>9.2%}  shuffled={:>9.2%}  pollution={:>8.2%}".format(
        random_rows["actual"].mean(), random_rows["pinned_w_bar"].mean(),
        random_rows["shuffled_mean"].mean(), random_rows["pollution"].mean()))
    print("REAL         actual={:>9.2%}  pinned(w_bar)={:>9.2%}  shuffled={:>9.2%}  pollution={:>8.2%}".format(
        real["actual"], real["pinned_w_bar"], real["shuffled_mean"], real["pollution"]))
    print()
    print("  shuffled ~ pinned  => the book is paid for the *order* of its exposure, not its level")
    print("  shuffled ~ actual  => the exposure path is worth nothing and the pin is not the right strip")
    print()
    print("model-random actual        = {:>8.2%}".format(real["actual"] - random_rows["actual"].mean()))
    print("model-random pinned(w_bar) = {:>8.2%}".format(real["pinned_w_bar"] - random_rows["pinned_w_bar"].mean()))
    print("-> the difference is the quotable stock-picking number; the two rows should agree.")
    return frame


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default="hs300")
    parser.add_argument("--model-id", default=None, help="Override the arm's model id.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args(argv)
    print("{}  split={}  {} random draws\n".format(
        args.model_id or ARMS[args.arm], args.split, args.draws))
    run(
        Path(args.dataset_dir), args.model_id or ARMS[args.arm], args.split, args.draws,
        args.top_k, args.horizon, args.max_positions, args.initial_capital, args.config_path, args.seed,
    )


if __name__ == "__main__":
    main()
