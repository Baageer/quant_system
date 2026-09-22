"""Fetch tushare ``index_weight`` month-end snapshots -> point-in-time membership.

``data/HS300.txt`` is *today's* 299 constituents applied to the whole history,
which is exactly the survivorship/look-ahead the pipeline has always documented
as an open limitation (``stock_valuation_diagnostic.py:96``,
``zz1000_full_pipeline_summary.md:90``).  This script fetches the monthly
constituent-weight snapshots that turn that static list into a real
point-in-time universe.

Two tushare quirks drive the design:

* a range query only returns the **most recent 24 months**, so long history has
  to go month-end by month-end;
* month-ends are read off the local 000300 benchmark calendar rather than
  guessed, so every requested date is a real trading day.

Cached layout::

    data/raw/tushare/index_weight/000300.SH.csv

Example::

    python -m research.experiments.fetch_tushare_index_weight --limit 6     # smoke
    python -m research.experiments.fetch_tushare_index_weight               # full history
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from research.experiments.fetch_tushare_daily_basic import call_api, read_token

DEFAULT_INDEXES = "000300.SH"
DEFAULT_OUTPUT_DIR = "data/raw/tushare/index_weight"
DEFAULT_BENCHMARK = "data/benchmarks/000300.csv"
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-12-31"
FIELDS = "index_code,con_code,trade_date,weight"


def month_end_trading_days(benchmark_file: str, start: str, end: str) -> List[str]:
    """Last real trading day of every month in [start, end], as ``YYYYMMDD``."""

    frame = pd.read_csv(benchmark_file, low_memory=False)
    frame.columns = [c.strip().lstrip("﻿") for c in frame.columns]
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if dates.empty:
        return []
    months = dates.dt.to_period("M")
    return [d.strftime("%Y%m%d") for d in dates.groupby(months).max().sort_values()]


def cached_dates(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["trade_date"], low_memory=False)
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(existing["trade_date"].astype(str))


def fetch_index(
    index_code: str,
    token: str,
    output_dir: Path,
    dates: Sequence[str],
    force: bool = False,
    sleep: float = 0.6,
) -> pd.DataFrame:
    """Fetch (or reuse) one index's monthly snapshots, appending to its cache."""

    path = output_dir / "{}.csv".format(index_code)
    have = set() if force else cached_dates(path)
    todo = [d for d in dates if d not in have]

    records = []
    fetched_frames = []
    for trade_date in tqdm(todo, desc="index_weight {}".format(index_code), unit="date"):
        try:
            frame = call_api("index_weight", token, {"index_code": index_code, "trade_date": trade_date},
                             fields=FIELDS, retries=2)
        except Exception as exc:  # keep going; one bad date must not kill the run
            records.append({"trade_date": trade_date, "status": "failed", "rows": 0,
                            "error": str(exc)[:160]})
            time.sleep(sleep)
            continue
        if frame.empty:
            records.append({"trade_date": trade_date, "status": "empty", "rows": 0})
        else:
            records.append({"trade_date": trade_date, "status": "fetched", "rows": int(len(frame))})
            fetched_frames.append(frame)
        if sleep:
            time.sleep(sleep)

    if fetched_frames:
        frames = []
        if path.exists() and not force:
            frames.append(pd.read_csv(path, dtype=str, low_memory=False))
        frames.extend(fetched_frames)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["index_code", "con_code", "trade_date"], keep="last")
        combined = combined.sort_values(["trade_date", "con_code"])
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False, encoding="utf-8-sig")

    return pd.DataFrame(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", default=DEFAULT_INDEXES, help="comma-separated tushare index codes")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK, help="trading calendar source")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--sleep", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None, help="only the first N month-ends (smoke test)")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    token = read_token(Path(args.env_file))
    dates = month_end_trading_days(args.benchmark_file, args.start_date, args.end_date)
    if args.limit:
        dates = dates[: args.limit]
    print("{} month-end trading days from {} to {}".format(len(dates), args.start_date, args.end_date))

    failures = 0
    for index_code in [c.strip() for c in args.indexes.split(",") if c.strip()]:
        report = fetch_index(index_code, token, Path(args.output_dir), dates,
                             force=args.force, sleep=args.sleep)
        if not report.empty:
            counts = report["status"].value_counts().to_dict()
            print("{}: {}".format(index_code, counts))
            failures += int(counts.get("failed", 0))
            bad = report[report["status"] == "failed"]
            if not bad.empty:
                print(bad.head(5).to_string(index=False))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
