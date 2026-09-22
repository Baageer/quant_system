"""Fetch 2026 daily bars / adjustment factors / index snapshots for forward tracking.

The local research cache ends **2025-12-31**:

* ``data/raw/tushare/price_history/raw_hfq_pct/`` (per-symbol OHLCV)
* ``data/raw/tushare/index_weight/{000300,000852}.SH.csv`` (point-in-time membership)

so a 2026 forward window needs new data.  This script fetches it into a
**separate** tree, ``data/raw/tushare/forward_2026/``, on purpose: appending to
``raw_hfq_pct/`` would silently extend the *published* test window on any future
re-run of the existing pipeline, and those published numbers are what every
conclusion in ``docs/research/`` rests on.  The forward runner merges the two
trees explicitly instead.

Layout::

    forward_2026/trade_cal.csv                2026 open sessions
    forward_2026/daily/{YYYYMMDD}.csv         whole-market bars, one file per session
    forward_2026/adj_factor/{YYYYMMDD}.csv    adjustment factors, same shape
    forward_2026/index_weight/{code}.csv      2026 month-end membership snapshots
    forward_2026/benchmark/{code}.csv         000300 / 000852 daily index
    forward_2026/fetch_manifest.json          row counts, date ranges, failures

One file per session (rather than one growing CSV) keeps every re-run resumable
by file existence alone.  That matters here: the API host intermittently fails
DNS resolution mid-run, and re-fetching 173 sessions because the process died at
session 170 would be waste.

A broad-market session is ~5500 rows, so a full 2026 run is ~175 calls.

Example::

    python -m research.experiments.fetch_tushare_forward_2026 --limit 3   # smoke
    python -m research.experiments.fetch_tushare_forward_2026             # full
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from research.experiments.fetch_tushare_daily_basic import call_api, read_token

DEFAULT_OUTPUT_DIR = "data/raw/tushare/forward_2026"
DEFAULT_START_DATE = "2026-01-01"

#: ``None`` = run through today.  A pinned date is wrong the next morning in a
#: way that does not announce itself: the run simply stops at the old boundary
#: and looks like a success.  Asking for today is safe because ``trade_cal`` only
#: returns open sessions and today's session may not be published yet -- that
#: yields a header-only file, which ``cached_session_rows`` now refuses to treat
#: as cached, so the next run repairs it.
DEFAULT_END_DATE = None

DEFAULT_INDEXES = "000300.SH,000852.SH"

#: The splice needs each symbol's *raw* close on its last cached day, which the
#: price-history cache cannot supply (it is a rescaled series).  Every cached
#: file ends 2025-12-31, so that one session is fetched as the anchor.
DEFAULT_ANCHOR_SESSIONS = "20251231"

DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
ADJ_FIELDS = "ts_code,trade_date,adj_factor"
WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"
INDEX_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"


def trading_days(token: str, start: str, end: str, retries: int) -> List[str]:
    """Open SSE sessions in ``[start, end]`` as ``YYYYMMDD``, ascending."""

    frame = call_api(
        "trade_cal", token,
        {"exchange": "SSE", "start_date": start.replace("-", ""),
         "end_date": end.replace("-", ""), "is_open": "1"},
        retries=retries,
    )
    if frame.empty:
        raise RuntimeError("trade_cal returned no rows for {}..{}".format(start, end))
    days = sorted(frame.loc[frame["is_open"].astype(str) == "1", "cal_date"].astype(str))
    return days


def cached_session_rows(path: Path) -> int:
    """Data rows in a cached session file; 0 when empty or unreadable.

    Counts lines rather than parsing, because this runs once per session on
    every resumable re-run and the only question asked is "is there anything
    here".

    An unreadable file counts as *not* cached, so the session is re-fetched and
    overwritten.  Both failure modes are caught: ``OSError`` for a locked or
    vanished file, and ``ValueError`` -- which is what a half-written or
    non-UTF-8 file raises as ``UnicodeDecodeError``.  Letting the latter through
    would abort a 173-session run at the one session that needed repairing.
    """

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return max(sum(1 for _ in handle) - 1, 0)  # minus the header
    except (OSError, ValueError):
        return 0


def fetch_by_session(
    api_name: str,
    fields: str,
    token: str,
    days: Sequence[str],
    directory: Path,
    retries: int,
    sleep: float,
    force: bool,
) -> List[dict]:
    """One API call per session, cached as ``directory/{day}.csv``.

    Skipping on file existence is what makes the run resumable.  A **zero-row**
    file, however, is not a cache hit: every ``day`` here comes from
    ``trade_cal`` with ``is_open=1``, so an empty response means the API had not
    published that session yet (or errored), not that the market was shut.  Such
    a file is re-fetched, because otherwise one unlucky run poisons the cache
    permanently -- which is exactly what happened to ``20260917.csv``, fetched
    before publication and then skipped by every later run.
    """

    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for day in tqdm(days, desc=api_name, unit="day"):
        path = directory / "{}.csv".format(day)
        stale = False
        if path.exists() and not force:
            if cached_session_rows(path):
                records.append({"date": day, "status": "cached", "rows": None})
                continue
            stale = True
        try:
            frame = call_api(api_name, token, {"trade_date": day}, fields=fields, retries=retries)
        except Exception as exc:  # one bad session must not kill a 173-call run
            records.append({"date": day, "status": "failed", "rows": 0, "error": str(exc)[:200]})
            if sleep:
                time.sleep(sleep)
            continue
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        records.append({"date": day, "rows": int(len(frame)),
                        "status": "refetched" if stale else "fetched"})
        if sleep:
            time.sleep(sleep)
    return records


def fetch_index_weight(
    index_code: str,
    token: str,
    month_ends: Sequence[str],
    directory: Path,
    retries: int,
    sleep: float,
    force: bool,
) -> List[dict]:
    """Month-end membership snapshots, appended into one file per index."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}.csv".format(index_code)
    have = set()
    if path.exists() and not force:
        try:
            existing = pd.read_csv(path, usecols=["trade_date"], low_memory=False)
            have = set(existing["trade_date"].astype(str))
        except (ValueError, pd.errors.EmptyDataError):
            have = set()

    frames, records = [], []
    todo = [d for d in month_ends if d not in have]
    for day in tqdm(todo, desc="index_weight {}".format(index_code), unit="date"):
        try:
            frame = call_api("index_weight", token,
                             {"index_code": index_code, "trade_date": day},
                             fields=WEIGHT_FIELDS, retries=retries)
        except Exception as exc:
            records.append({"date": day, "status": "failed", "rows": 0, "error": str(exc)[:200]})
            if sleep:
                time.sleep(sleep)
            continue
        records.append({"date": day, "status": "fetched" if not frame.empty else "empty",
                        "rows": int(len(frame))})
        if not frame.empty:
            frames.append(frame)
        if sleep:
            time.sleep(sleep)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if path.exists() and not force:
            combined = pd.concat([pd.read_csv(path, dtype=str, low_memory=False), combined],
                                 ignore_index=True)
        combined = combined.drop_duplicates(["index_code", "con_code", "trade_date"], keep="last")
        combined = combined.sort_values(["trade_date", "con_code"])
        combined.to_csv(path, index=False, encoding="utf-8-sig")
    return records


def fetch_index_daily(
    index_code: str,
    token: str,
    start: str,
    end: str,
    directory: Path,
    retries: int,
) -> dict:
    """Full-range daily index history for one benchmark, cached to one file."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}.csv".format(index_code)
    try:
        frame = call_api("index_daily", token,
                         {"ts_code": index_code, "start_date": start.replace("-", ""),
                          "end_date": end.replace("-", "")},
                         fields=INDEX_DAILY_FIELDS, retries=retries)
    except Exception as exc:
        return {"index": index_code, "status": "failed", "rows": 0, "error": str(exc)[:200]}
    if frame.empty:
        return {"index": index_code, "status": "empty", "rows": 0}
    frame = frame.sort_values("trade_date")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {"index": index_code, "status": "fetched", "rows": int(len(frame)),
            "start": str(frame["trade_date"].iloc[0]), "end": str(frame["trade_date"].iloc[-1])}


def month_end_sessions(days: Sequence[str]) -> List[str]:
    """Last session of each calendar month in ``days``."""

    frame = pd.DataFrame({"day": list(days)})
    frame["month"] = pd.to_datetime(frame["day"], format="%Y%m%d").dt.to_period("M")
    return [str(v) for v in frame.groupby("month")["day"].max().sort_values()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE,
                        help="YYYY-MM-DD inclusive; default = today")
    parser.add_argument("--indexes", default=DEFAULT_INDEXES)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sessions (smoke test)")
    parser.add_argument("--anchor-sessions", default=DEFAULT_ANCHOR_SESSIONS,
                        help="extra pre-window sessions to fetch into daily/ for the splice anchor")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-adj-factor", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    token = read_token(Path(args.env_file))
    output_dir = Path(args.output_dir)
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    days = trading_days(token, args.start_date, end_date, args.retries)
    print("{} open sessions in {}..{}".format(len(days), args.start_date, end_date))
    if args.limit:
        days = days[: args.limit]
        print("limited to the first {} sessions".format(len(days)))

    manifest = {
        "start_date": args.start_date,
        "end_date": end_date,
        "sessions": len(days),
        "first_session": days[0] if days else None,
        "last_session": days[-1] if days else None,
    }

    anchor_days = [d.strip().replace("-", "")
                   for d in str(args.anchor_sessions).split(",") if d.strip()]
    if args.skip_daily:
        daily_records = []
    else:
        daily_records = fetch_by_session("daily", DAILY_FIELDS, token, anchor_days + days,
                                         output_dir / "daily", args.retries, args.sleep, args.force)
    if args.skip_adj_factor:
        adj_records = []
    else:
        adj_records = fetch_by_session("adj_factor", ADJ_FIELDS, token, days,
                                       output_dir / "adj_factor", args.retries, args.sleep, args.force)

    month_ends = month_end_sessions(days)
    weight_records, index_records = [], []
    for index_code in [c.strip() for c in args.indexes.split(",") if c.strip()]:
        weight_records += fetch_index_weight(index_code, token, month_ends,
                                             output_dir / "index_weight",
                                             args.retries, args.sleep, args.force)
        index_records.append(fetch_index_daily(index_code, token, args.start_date,
                                               end_date, output_dir / "benchmark",
                                               args.retries))

    for name, records in [("daily", daily_records), ("adj_factor", adj_records),
                          ("index_weight", weight_records)]:
        if not records:
            manifest[name] = {"calls": 0}
            continue
        report = pd.DataFrame(records)
        counts = report["status"].value_counts().to_dict()
        manifest[name] = {"calls": int(len(report)), "by_status": counts,
                          "dates": sorted(report["date"].astype(str).tolist())}
        print("{}: {}".format(name, counts))
        bad = report[report["status"] == "failed"]
        if not bad.empty:
            print(bad.head(5).to_string(index=False))
    manifest["benchmark"] = index_records

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fetch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest -> {}".format(output_dir / "fetch_manifest.json"))

    failures = sum(1 for records in (daily_records, adj_records, weight_records)
                   for r in records if r["status"] == "failed")
    failures += sum(1 for r in index_records if r["status"] == "failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
