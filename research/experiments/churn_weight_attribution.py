"""Is the random-score churn arm's excess in the *holdings* or in the *weights*?

``churn_attribution_probe`` settled the holdings half of the question: the names
this closure buys have no forward-10d edge over the pool on the same dates
(edge -0.05% per buy, |t| < 0.7 across draws).  A shuffled score buys nothing
special, as it must.

So the excess has to live in *how the book is weighted*.  The production closure
buys a fixed ``initial_capital / max_positions`` of every name, never rebalances
a lot, and only ever fills free slots -- so the book is a sum of unequally-aged
25k lots sitting next to idle cash, not an equal-weighted portfolio of its own
holdings.

This probe replays one draw and, from the *same* daily records, rebuilds three
paths that differ only in the weighting rule:

* **actual**     -- what the ledger did (fixed notional lots, cash at 0%);
* **equal_wt**   -- identical holdings and identical cash, but each day's stock
  return replaced by the equal-weighted mean of the held names' returns;
* **pool_wt**    -- identical cash, but the stock sleeve earns the whole pool's
  equal-weight return.

All three share the cash path of the actual run, so the difference between them
is purely the within-book weighting.  Example::

    python -m research.experiments.churn_weight_attribution --draws 3
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import (
    _score_slots,
    _write_rank_score,
    carry_forward_marks,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


def replay(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    horizon: int,
    initial_capital: float,
    all_dates,
) -> Dict[str, object]:
    """Run the production closure, recording cash, book value and holdings daily."""

    target_notional = initial_capital / max_positions
    strategy = make_rank_strategy(max_positions, target_notional, min_holding_days=horizon)
    price_of = {symbol: frame["close"] for symbol, frame in prepared.items()}
    marks = carry_forward_marks(price_of, all_dates)

    date_to_symbols: Dict[pd.Timestamp, list] = {}
    for symbol, frame in prepared.items():
        for date in frame.index:
            date_to_symbols.setdefault(date, []).append(symbol)

    cash = initial_capital
    units: Dict[str, float] = {}
    lots: Dict[str, float] = {}  # entry price per symbol, for the lot-age readout
    records = []
    buys = 0

    for date in all_dates:
        relevant = {s: prepared[s] for s in date_to_symbols.get(date, [])}
        signals = strategy(date, relevant, units)
        for symbol in sorted(signals):
            signal = signals[symbol]
            series = price_of[symbol]
            if date not in series.index:
                continue
            price = series.loc[date]
            if not np.isfinite(price) or price <= 0:
                continue
            if signal["action"] == "buy":
                shares = target_notional / price
                units[symbol] = units.get(symbol, 0.0) + shares
                lots.setdefault(symbol, float(price))
                cash -= shares * price
                buys += 1
            else:
                held = units.get(symbol, 0.0)
                shares = min(held, float(signal["shares"]))
                if shares <= 0:
                    continue
                units[symbol] = held - shares
                cash += shares * price
                if units[symbol] <= 1e-12:
                    del units[symbol]
                    lots.pop(symbol, None)

        stock_value = 0.0
        for symbol, shares in units.items():
            price = marks.at[date, symbol]
            if np.isfinite(price):
                stock_value += shares * price
        records.append(
            {
                "date": date,
                "cash": cash,
                "stock_value": stock_value,
                "total": cash + stock_value,
                "holdings": {"%s" % s: units[s] for s in units},
            }
        )

    return {"records": records, "buys": buys}


def _price_panel(prepared: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = pd.DataFrame({s: f["close"] for s, f in prepared.items()}).sort_index()
    return panel.ffill()


def attribute(records, panel: pd.DataFrame) -> Dict[str, object]:
    """Rebuild the three weighting paths from one set of daily records."""

    dates = [r["date"] for r in records]
    rets = panel.pct_change(fill_method=None)
    pool_ret = rets.mean(axis=1, skipna=True)

    actual, equal_wt, pool_wt = [records[0]["total"]], [records[0]["total"]], [records[0]["total"]]
    held_ret, pool_ret_at, weights = [], [], []
    negative_cash_days = 0

    for i in range(1, len(records)):
        prev, row = records[i - 1], records[i]
        date = row["date"]
        gross, sizes = [], []
        for symbol in prev["holdings"]:
            p_prev = panel.at[dates[i - 1], symbol] if symbol in panel.columns else np.nan
            p_now = panel.at[date, symbol] if symbol in panel.columns else np.nan
            if np.isfinite(p_prev) and np.isfinite(p_now) and p_prev > 0:
                gross.append(p_now / p_prev - 1.0)
                sizes.append(prev["holdings"][symbol] * p_prev)

        book = prev["stock_value"]
        if gross:
            gross = np.asarray(gross, dtype=float)
            sizes = np.asarray(sizes, dtype=float)
            w = sizes / sizes.sum() if sizes.sum() > 0 else np.full(len(sizes), 1.0 / len(sizes))
            held_ret.append(float(w @ gross))
            weights.append(float(w.max()))
            stock_step_equal = book * (1.0 + float(gross.mean()))
        else:
            held_ret.append(0.0)
            weights.append(np.nan)
            stock_step_equal = book
        pool_ret_at.append(float(pool_ret.get(date, 0.0)))

        actual.append(row["total"])
        equal_wt.append(prev["cash"] + stock_step_equal)
        pool_wt.append(prev["cash"] + book * (1.0 + float(pool_ret.get(date, 0.0))))
        if row["cash"] < 0:
            negative_cash_days += 1

    initial = records[0]["total"]
    return {
        "actual": float(actual[-1] / initial - 1.0),
        "equal_weight": float(equal_wt[-1] / initial - 1.0),
        "pool_weight": float(pool_wt[-1] / initial - 1.0),
        "mean_cash_frac": float(np.mean([r["cash"] / r["total"] if r["total"] > 0 else np.nan for r in records])),
        "negative_cash_days": negative_cash_days,
        "mean_top_weight": float(np.nanmean(weights)),
        "mean_held_return": float(np.mean(held_ret)),
        "mean_pool_return": float(np.mean(pool_ret_at)),
    }


def run(
    dataset_dir: Path,
    model_id: str,
    draws: int = 3,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    seed: int = 20260915,
) -> pd.DataFrame:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw_market_data, _ = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)
    panel = _price_panel(raw_market_data)

    rows = []
    for draw in range(draws):
        probe = test.copy()
        probe[SCORE_COLUMN] = rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        out = replay(prepared, max_positions, horizon, initial_capital, all_dates)
        stats = attribute(out["records"], panel)
        stats["draw"] = draw
        stats["buys"] = out["buys"]
        rows.append(stats)
        print(
            "draw {:>3}  actual={:>8.2%}  equal_wt={:>8.2%}  pool_wt={:>8.2%}  "
            "cash={:>5.0%}  top_wt={:>5.1%}  neg_cash_days={}".format(
                draw, stats["actual"], stats["equal_weight"], stats["pool_weight"],
                stats["mean_cash_frac"], stats["mean_top_weight"], stats["negative_cash_days"],
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args(argv)
    frame = run(
        Path(args.dataset_dir), args.model_id, args.draws, args.top_k,
        args.horizon, args.max_positions, args.initial_capital, args.seed,
    )
    print()
    print(frame.to_string(index=False))
    print()
    for column in ("actual", "equal_weight", "pool_weight"):
        print("{:<14} mean {:>9.2%}".format(column, frame[column].mean()))


if __name__ == "__main__":
    main()
