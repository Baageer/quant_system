"""Why does a ~76%-invested book of *neutral* names beat a fully-invested pool?

``churn_weight_attribution`` showed the holdings and the lot weights are both
worth nothing: replacing the book's names with the whole pool, same cash, same
dates, moves the answer by under 0.6pp.  Yet that book returns ~+12.5% while the
pool buy-and-hold returns +6.37% and the daily-rebalanced pool returns ~+1.5%.

A 76%-invested book whose daily stock return equals the pool's should compound to
about the rebalanced figure.  Something in the exposure path must be paying.
There are only two candidates left:

* **leverage** -- the frictionless ledger lets cash go negative when the closure
  fires buys it cannot fund.  The production engine does *not* (it shrinks every
  order until it is affordable), so this would be an artifact of my ledger;
* **timing** -- the exposure weight w_t covaries with the pool's return.

This probe runs one draw both ways (levered, and cash-constrained like the
engine) and decomposes the log growth:

    log V_T/V_0 = sum log(1 + w_t r_t)
                ~ w_bar * sum r_t          (level effect)
                + T * cov(w_t, r_t)        (timing effect)
                - 0.5 * sum (w_t r_t)^2    (drag)

and then re-runs the *same* exposure path with w_t shuffled in time (timing
destroyed, level kept) and with w_t pinned to its mean.  If the shuffled path
matches the real one, the exposure timing is worth nothing and the gap is a
level/leverage effect.  Example::

    python -m research.experiments.churn_exposure_decomposition --draws 2
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


def run_ledger(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    horizon: int,
    initial_capital: float,
    all_dates,
    cash_constrained: bool,
) -> pd.DataFrame:
    """Replay the closure, optionally shrinking orders to affordable cash.

    ``cash_constrained=True`` reproduces the engine's buy rule exactly; with
    ``False`` the ledger funds buys by borrowing at 0%, which is what the
    frictionless probes were unknowingly doing.
    """

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
    rows = []

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
                if cash_constrained:
                    while shares > 0 and shares * price > cash:
                        shares -= 100.0
                    if shares <= 0:
                        continue
                units[symbol] = units.get(symbol, 0.0) + shares
                cash -= shares * price
            else:
                held = units.get(symbol, 0.0)
                shares = min(held, float(signal["shares"]))
                if shares <= 0:
                    continue
                units[symbol] = held - shares
                cash += shares * price
                if units[symbol] <= 1e-12:
                    del units[symbol]

        stock_value = 0.0
        for symbol, shares in units.items():
            price = marks.at[date, symbol]
            if np.isfinite(price):
                stock_value += shares * price
        rows.append({"date": date, "cash": cash, "stock_value": stock_value, "total": cash + stock_value})

    return pd.DataFrame(rows).set_index("date")


def decompose(track: pd.DataFrame, pool_daily: pd.Series) -> Dict[str, float]:
    """Split the compounded growth into level, timing and drag."""

    value = track["total"]
    r = pool_daily.reindex(value.index).fillna(0.0).to_numpy()
    w = (track["stock_value"] / track["total"]).shift(1).fillna(0.0).to_numpy()

    realised = float(value.iloc[-1] / value.iloc[0] - 1.0)
    log_real = float(np.log1p(w * r).sum())
    level = float(w.mean() * r.sum())
    timing = float(len(r) * np.cov(w, r, ddof=0)[0, 1])
    drag = float(-0.5 * np.sum((w * r) ** 2))

    rng = np.random.default_rng(7)
    shuffled = []
    for _ in range(50):
        w_shuf = rng.permutation(w)
        shuffled.append(float(np.expm1(np.log1p(w_shuf * r).sum())))
    flat = float(np.expm1(np.log1p(w.mean() * r).sum()))

    return {
        "realised": realised,
        "log_growth": log_real,
        "level_term": level,
        "timing_term": timing,
        "drag_term": drag,
        "reconstructed": level + timing + drag,
        "w_bar": float(w.mean()),
        "exp_shuffled_w": float(np.mean(shuffled)),
        "const_w": flat,
        "min_cash": float(track["cash"].min()),
        "median_cash": float(track["cash"].median()),
        "mean_cash": float(track["cash"].mean()),
        "avg_borrow_frac": float(np.mean(np.clip(-track["cash"], 0, None) / track["total"])),
        "max_borrow_frac": float((np.clip(-track["cash"], 0, None) / track["total"]).max()),
    }


def run(
    dataset_dir: Path,
    model_id: str,
    draws: int = 2,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    seed: int = 20260915,
) -> None:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw_market_data, _ = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)

    closes = pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index()
    pool_daily = closes.pct_change(fill_method=None).mean(axis=1, skipna=True)
    pool_bh = float((closes.iloc[-1] / closes.iloc[0] - 1.0).mean())
    print("pool equal-weight buy-and-hold {:>8.2%}".format(pool_bh))
    print("pool daily-rebalanced (compounded daily EW) {:>8.2%}".format(
        float((1.0 + pool_daily.reindex(closes.index).fillna(0.0)).prod() - 1.0)))
    print()

    for draw in range(draws):
        probe = test.copy()
        probe[SCORE_COLUMN] = rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        for label, constrained in (("levered  ", False), ("cash-clamp", True)):
            track = run_ledger(prepared, max_positions, horizon, initial_capital, all_dates, constrained)
            stats = decompose(track, pool_daily)
            print(
                "draw {} {}  realised={:>8.2%}  w_bar={:.3f}  cash(min/med)={:>9,.0f}/{:>8,.0f}  "
                "borrow_avg={:>6.2%}".format(
                    draw, label, stats["realised"], stats["w_bar"], stats["min_cash"],
                    stats["median_cash"], stats["avg_borrow_frac"],
                ), flush=True)
            print(
                "              level={:>8.2%}  timing={:>8.2%}  drag={:>8.2%}  recon={:>8.2%}  "
                "| const_w={:>8.2%}  shuffled_w(mean50)={:>8.2%}".format(
                    stats["level_term"], stats["timing_term"], stats["drag_term"],
                    stats["reconstructed"], stats["const_w"], stats["exp_shuffled_w"],
                ), flush=True)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args(argv)
    run(
        Path(args.dataset_dir), args.model_id, args.draws, args.top_k,
        args.horizon, args.max_positions, args.initial_capital, args.seed,
    )


if __name__ == "__main__":
    main()
