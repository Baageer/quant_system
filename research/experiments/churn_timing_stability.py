"""Is the random arm's exposure-timing term stable, or one regime's luck?

Established: the random arm beats the pool through its exposure *path*
(T * cov(w_t, r_t) is large and positive in every draw) while its holdings earn
exactly the pool's return and its buys have no forward edge.  That leaves one
practical question for anyone quoting a backtest number: is the timing term a
standing property of the closure, or was it earned in one stretch of this
particular window?

Reports, per calendar year, for a cash-constrained (engine-faithful) ledger:

* the pool's equal-weight buy-and-hold that year;
* the arm's realised return that year;
* the level / timing / drag split of the arm's log growth that year;
* the buy edge that year.

If the timing term is positive in every year it is a mechanism; if it lives in
one year then every absolute number in the HS300 research line inherits that
year's luck.  Example::

    python -m research.experiments.churn_timing_stability --draws 2
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.churn_real_vs_random import buy_edge, simulate
from research.market_regime.output_paths import DATASET_DIR
from research.experiments.random_portfolio_baseline import _score_slots, _write_rank_score
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


def per_year(track: pd.DataFrame, pool_daily: pd.Series) -> pd.DataFrame:
    value = track["total"]
    r = pool_daily.reindex(value.index).fillna(0.0)
    w = (track["stock_value"] / track["total"]).shift(1).fillna(0.0)
    rows = []
    for year, idx in value.groupby(value.index.year).groups.items():
        v = value.loc[idx]
        if len(v) < 2:
            continue
        rr = r.loc[idx].to_numpy()
        ww = w.loc[idx].to_numpy()
        rows.append(
            {
                "year": year,
                "arm_return": float(v.iloc[-1] / value.loc[value.index < v.index[0]].iloc[-1] - 1.0)
                if (value.index < v.index[0]).any() else float("nan"),
                "level": float(ww.mean() * rr.sum()),
                "timing": float(len(rr) * np.cov(ww, rr, ddof=0)[0, 1]),
                "drag": float(-0.5 * np.sum((ww * rr) ** 2)),
                "w_bar": float(ww.mean()),
            }
        )
    return pd.DataFrame(rows)


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
    pool_year = {}
    for year, idx in closes.groupby(closes.index.year).groups.items():
        sub = closes.loc[idx]
        pool_year[year] = float((sub.iloc[-1] / sub.iloc[0] - 1.0).mean())
    print("pool buy-and-hold by year (equal-weight mean of names):")
    for year in sorted(pool_year):
        print("  {}  {:>8.2%}   (lag-1 autocorr {:>6.3f})".format(
            year, pool_year[year], float(pool_daily.loc[str(year)].autocorr(1))))
    print()

    for draw in list(range(draws)) + [-1]:
        label = "REAL v1" if draw < 0 else "random {}".format(draw)
        probe = test.copy()
        probe[SCORE_COLUMN] = values if draw < 0 else rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        out = simulate(prepared, max_positions, horizon, initial_capital, all_dates)
        frame = per_year(out["track"], pool_daily)
        frame["pool_year"] = frame["year"].map(pool_year)
        edge = buy_edge(prepared, out["buys"], horizon)
        print("== {} ==  total {:>8.2%}   buys={}  buy edge={:>7.3%} (t={:.2f})".format(
            label, out["track"]["total"].iloc[-1] / initial_capital - 1.0,
            edge["buys_scored"], edge["edge"], edge["t_stat"]))
        print(frame.to_string(index=False, float_format=lambda x: "{:>8.3f}".format(x)))
        print()


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
