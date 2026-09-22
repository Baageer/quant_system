"""How big is the static-HS300-pool bias, and does it grow with window length?

``data/HS300.txt`` holds *today's* 299 constituents but the pipeline applies it
to the entire history (frozen at ``feature_event_study`` time and carried
downstream as the ``symbol`` column).  Nobody has quantified what that costs.

This is the cheap upper bound: hold today's 300 equal-weight over a past window
and compare with the real 000300 index over the same window.  The gap is

    survivorship/promotion bias + dividend wedge

The dividend term is real but one-directional: the pool prices are 后复权 (hfq),
so they include dividends, while ``000300.csv`` is a **price** index that does
not.  That wedge alone is worth roughly +2%/yr to the pool, so the *survivorship*
part is the raw gap minus that wedge.

Example::

    python -m research.experiments.hs300_pool_bias_audit
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_POOL = "data/HS300.txt"
DEFAULT_PRICE_DIR = "data/raw/tushare/price_history/hfq"
DEFAULT_BENCHMARK = "data/benchmarks/000300.csv"
# 后复权 (dividend-inclusive) pool vs price-only index.  Empirical A-share HS300
# dividend yield is ~2%/yr; applied to the pool and subtracted, per year.
DEFAULT_DIVIDEND_WEDGE = 0.02


def read_pool(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_closes(symbol: str, price_dir: Path) -> Optional[pd.Series]:
    """Concatenate every cached hfq file for one symbol into a close series."""

    # Cached files are named "{code}.{EX}_{start}_{end}.csv" while the pool file
    # holds bare six-digit codes, so the exchange suffix has to be matched too.
    files = sorted(price_dir.glob("{}.*_*.csv".format(symbol)))
    if not files:
        files = sorted(price_dir.glob("{}_*.csv".format(symbol)))
    if not files:
        return None
    parts = []
    for path in files:
        frame = pd.read_csv(path, usecols=["date", "close"], low_memory=False)
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    series = frame.set_index("date")["close"].astype(float).sort_index()
    return series[series > 0]


def build_panel(symbols: Sequence[str], price_dir: Path) -> pd.DataFrame:
    series = {}
    for symbol in symbols:
        closes = load_closes(symbol, price_dir)
        if closes is not None and len(closes):
            series[symbol] = closes
    return pd.DataFrame(series).sort_index()


def index_closes(benchmark: Path) -> pd.Series:
    frame = pd.read_csv(benchmark, low_memory=False)
    frame.columns = [c.strip().lstrip("﻿") for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date")["close"].astype(float).sort_index()


def window_stats(panel: pd.DataFrame, bench: pd.Series, start: str, end: str, wedge: float):
    """Daily-rebalanced equal-weight static pool vs the index over one window.

    Deliberately *not* buy-and-hold: requiring a full price history would keep
    only the oldest surviving names and confound the bias with the coverage
    filter.  Daily equal-weight uses every name that trades on a given day, and
    is the same kind of continuously-rebalanced portfolio the index itself is.
    """

    sub = panel.loc[str(start):str(end)]
    if sub.empty or len(sub) < 2:
        return None
    bench_window = bench.loc[str(start):str(end)]
    if bench_window.empty or len(bench_window) < 2:
        return None

    years = (sub.index[-1] - sub.index[0]).days / 365.25
    per_day_names = sub.notna().sum(axis=1)
    daily = sub.pct_change().mean(axis=1, skipna=True).dropna()
    pool_return = float((1.0 + daily).prod() - 1.0)
    bench_return = float(bench_window.iloc[-1] / bench_window.iloc[0] - 1.0)
    raw_gap = pool_return - bench_return
    wedge_total = (1.0 + wedge) ** years - 1.0
    return {
        "start": str(sub.index[0].date()),
        "end": str(sub.index[-1].date()),
        "years": years,
        "names_first_day": int(per_day_names.iloc[0]),
        "names_last_day": int(per_day_names.iloc[-1]),
        "names_min": int(per_day_names.min()),
        "coverage_of_loaded": float(per_day_names.mean() / panel.shape[1]) if panel.shape[1] else np.nan,
        "pool_ew_daily": pool_return,
        "index_000300": bench_return,
        "raw_gap": raw_gap,
        "raw_gap_per_year": raw_gap / years if years else np.nan,
        "survivorship_after_dividend": raw_gap - wedge_total,
        "survivorship_per_year": (raw_gap - wedge_total) / years if years else np.nan,
    }


def run(
    pool_file: str = DEFAULT_POOL,
    price_dir: str = DEFAULT_PRICE_DIR,
    benchmark: str = DEFAULT_BENCHMARK,
    starts: Sequence[str] = ("2015-01-01", "2017-01-01", "2019-07-17", "2022-07-20"),
    end: str = "2024-12-31",
    dividend_wedge: float = DEFAULT_DIVIDEND_WEDGE,
) -> pd.DataFrame:
    symbols = read_pool(Path(pool_file))
    panel = build_panel(symbols, Path(price_dir))
    bench = index_closes(Path(benchmark))
    rows = []
    for start in starts:
        stats = window_stats(panel, bench, start, end, dividend_wedge)
        if stats is not None:
            stats["requested_start"] = start
            rows.append(stats)
    frame = pd.DataFrame(rows)
    columns = [
        "requested_start", "start", "end", "years", "names_first_day", "names_last_day",
        "names_min", "coverage_of_loaded", "pool_ew_daily", "index_000300", "raw_gap",
        "survivorship_after_dividend", "survivorship_per_year",
    ]
    frame = frame[columns]

    print("\n=== static-HS300 pool vs real index, by window start (end {}) ===".format(end))
    print("pool  = today's {} constituents that have cached prices ({} loaded), daily equal-weight, hfq".format(
        len(symbols), panel.shape[1]))
    print("index = 000300 price index; dividend wedge assumption {:.1%}/yr (pool has dividends, index does not)\n".format(
        dividend_wedge))
    display = frame.copy()
    for column in ("coverage_of_loaded", "pool_ew_daily", "index_000300", "raw_gap",
                   "survivorship_after_dividend", "survivorship_per_year"):
        display[column] = display[column].map(lambda v: "{:>8.2%}".format(v))
    display["years"] = display["years"].map(lambda v: "{:>5.1f}".format(v))
    print(display.to_string(index=False))
    missing = sorted(set(symbols) - set(panel.columns))
    if missing:
        print("\n{} pool symbols have no cached hfq price file, e.g. {}".format(len(missing), missing[:8]))
    return frame


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-file", default=DEFAULT_POOL)
    parser.add_argument("--price-dir", default=DEFAULT_PRICE_DIR)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--starts", default="2015-01-01,2017-01-01,2019-07-17,2022-07-20")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--dividend-wedge", type=float, default=DEFAULT_DIVIDEND_WEDGE)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    frame = run(
        args.pool_file, args.price_dir, args.benchmark,
        tuple(s.strip() for s in args.starts.split(",") if s.strip()),
        args.end, args.dividend_wedge,
    )
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "hs300_pool_bias_audit.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
