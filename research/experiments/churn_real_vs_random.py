"""Is the v1 headline stock *selection*, or the same exposure-timing effect?

The random arm's excess over the pool turned out to be neither accounting, nor
execution, nor selection, nor weighting -- it is the book's *exposure path*:
constant exposure earns +3.47% and a time-shuffled exposure path +5.4%, while
the realised ordering earns +10..22%, all against a +6.37% pool buy-and-hold.

That raises the only question that matters for the research programme: is the
model's own +45.98% the same effect with a better-looking label, or is there a
genuine selection component underneath it?

Two things separate them, and this probe reports both for real and shuffled
scores on the same chain:

* **buy edge** -- the forward-10d return of the names actually bought minus the
  pool's forward-10d return on the same dates.  Selection has an edge; exposure
  timing does not.
* **exposure timing term** -- T * cov(w_t, r_t) over the arm's own exposure path.

Example::

    python -m research.experiments.churn_real_vs_random --draws 3
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


def simulate(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    horizon: int,
    initial_capital: float,
    all_dates,
) -> Dict[str, object]:
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
    buys = []
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
                while shares > 0 and shares * price > cash:
                    shares -= 100.0
                if shares <= 0:
                    continue
                units[symbol] = units.get(symbol, 0.0) + shares
                cash -= shares * price
                buys.append((symbol, date))
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

    return {"track": pd.DataFrame(rows).set_index("date"), "buys": buys}


def buy_edge(prepared: Dict[str, pd.DataFrame], buys, horizon: int) -> Dict[str, float]:
    closes = pd.DataFrame({s: f["close"] for s, f in prepared.items()}).sort_index()
    fwd = closes.shift(-horizon) / closes - 1.0
    pool = fwd.mean(axis=1, skipna=True)
    own, bench = [], []
    for symbol, date in buys:
        if date not in fwd.index or symbol not in fwd.columns:
            continue
        a, b = fwd.at[date, symbol], pool.get(date, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            own.append(a)
            bench.append(b)
    own, bench = np.asarray(own), np.asarray(bench)
    diff = own - bench
    return {
        "buys_scored": int(len(own)),
        "mean_fwd_bought": float(own.mean()) if len(own) else float("nan"),
        "mean_fwd_pool": float(bench.mean()) if len(bench) else float("nan"),
        "edge": float(diff.mean()) if len(diff) else float("nan"),
        "t_stat": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 1 else float("nan"),
    }


def exposure_terms(track: pd.DataFrame, pool_daily: pd.Series) -> Dict[str, float]:
    value = track["total"]
    r = pool_daily.reindex(value.index).fillna(0.0).to_numpy()
    w = (track["stock_value"] / track["total"]).shift(1).fillna(0.0).to_numpy()
    n = len(r)
    return {
        "return": float(value.iloc[-1] / value.iloc[0] - 1.0),
        "w_bar": float(w.mean()),
        "level_term": float(w.mean() * r.sum()),
        "timing_term": float(n * np.cov(w, r, ddof=0)[0, 1]),
        "drag_term": float(-0.5 * np.sum((w * r) ** 2)),
        "w_first_half": float(w[: n // 2].mean()),
        "w_second_half": float(w[n // 2:].mean()),
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

    closes = pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index()
    pool_daily = closes.pct_change(fill_method=None).mean(axis=1, skipna=True)
    print("pool equal-weight buy-and-hold {:>8.2%}".format(float((closes.iloc[-1] / closes.iloc[0] - 1.0).mean())))
    print("pool daily autocorrelation (lag 1) {:>8.4f}".format(float(pool_daily.autocorr(1))))
    print()

    rows = []
    for draw in list(range(draws)) + [-1]:
        label = "REAL v1" if draw < 0 else "random {:>2}".format(draw)
        probe = test.copy()
        probe[SCORE_COLUMN] = values if draw < 0 else rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        out = simulate(prepared, max_positions, horizon, initial_capital, all_dates)
        stats = exposure_terms(out["track"], pool_daily)
        stats.update(buy_edge(prepared, out["buys"], horizon))
        stats["label"] = label
        rows.append(stats)
        print(
            "{}  return={:>8.2%}  w={:.3f} (half1 {:.3f} / half2 {:.3f})  level={:>7.2%}  "
            "timing={:>8.2%}  drag={:>7.2%}".format(
                label, stats["return"], stats["w_bar"], stats["w_first_half"], stats["w_second_half"],
                stats["level_term"], stats["timing_term"], stats["drag_term"],
            ), flush=True)
        print(
            "            buys={:<5} fwd_bought={:>7.2%}  fwd_pool={:>7.2%}  edge={:>7.3%}  t={:>5.2f}".format(
                stats["buys_scored"], stats["mean_fwd_bought"], stats["mean_fwd_pool"],
                stats["edge"], stats["t_stat"],
            ), flush=True)
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


if __name__ == "__main__":
    main()
