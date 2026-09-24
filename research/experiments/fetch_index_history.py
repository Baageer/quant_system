"""Fetch an index history into the ``data/benchmarks/`` schema.

The research pipelines measure excess against an index read by
``excess_target_dataset.load_hs300_benchmark``, which expects a BOM'd, fully
quoted CSV with the columns
``date,index_code,index_name,open,close,high,low,volume,amount``.

``data/benchmarks/000300.csv`` was assembled that way but only by hand, and the
中证1000 counterpart (000852) did not exist at all -- the only local 000852 data
was a 2018+ snapshot under ``data/raw/akshare/index_history/``.  That matters
more than it sounds: the excess label voids any row whose *benchmark* level is
missing, so the benchmark's start date silently becomes the training set's start
date.  暂时用 2018 起的数据会把训练集从 14 年压到 1 年。

This script bridges that gap: fetch the missing span and merge it with whatever
snapshots already sit on disk.  ``--source`` picks the remote: ``akshare``
(the original default) or ``tushare``, which is the fallback when akshare's
eastmoney endpoint is unreachable from the network at hand.

Only ``close`` is load-bearing downstream (see ``add_window_best_labels``); the
other OHLCV columns are carried for schema compatibility.

Example::

    python -m research.experiments.fetch_index_history --code 000852 \
        --name 中证1000 --start 20141017 --snapshot data/raw/akshare/index_history
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


BENCHMARK_COLUMNS = ["date", "index_code", "index_name", "open", "close",
                     "high", "low", "volume", "amount"]
DEFAULT_OUTPUT_DIR = "data/benchmarks"
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 3.0
# ``index_daily`` returns at most 4000 rows per call, so a long span has to be
# split into chunks smaller than that (4 years ~= 1000 sessions).
TUSHARE_CHUNK_YEARS = 4
TUSHARE_SLEEP_SECONDS = 0.4


def normalize_index_frame(frame: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    """Coerce a raw akshare frame into the benchmark schema.

    Accepts either the ``stock_zh_index_daily_em`` shape (already lowercase
    ``date/open/close/high/low/volume/amount``) or any frame carrying at least
    ``date`` and ``close``.  Missing optional columns become NaN rather than
    being invented, so a merge never fabricates a price.
    """

    if frame is None or len(frame) == 0:
        raise ValueError("index frame is empty")
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("index frame needs date and close columns, got {}".format(list(frame.columns)))
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    for column in ("open", "close", "high", "low", "volume", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = float("nan")
    data = data.dropna(subset=["close"])
    if (data["close"] <= 0).any():
        raise ValueError("index frame has non-positive close levels")
    data["index_code"] = str(code)
    data["index_name"] = str(name)
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return data[BENCHMARK_COLUMNS].sort_values("date").reset_index(drop=True)


def merge_index_snapshots(frames: Sequence[pd.DataFrame], keep: str = "last") -> pd.DataFrame:
    """Concatenate same-schema snapshots and de-duplicate by date.

    Overlaps are expected: snapshots are fetched over overlapping ranges and the
    newer one wins, matching how ``load_local_price_history`` resolves its own
    duplicate dates.
    """

    usable = [frame for frame in frames if frame is not None and len(frame)]
    if not usable:
        raise ValueError("no index snapshots to merge")
    merged = pd.concat(usable, ignore_index=True)
    missing = [column for column in BENCHMARK_COLUMNS if column not in merged.columns]
    if missing:
        raise ValueError("snapshot is missing benchmark columns: {}".format(missing))
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date", "close"]).sort_values("date")
    # ``keep="last"`` after the sort means the later-listed snapshot wins the
    # overlap, so callers pass snapshots oldest-first.
    merged = merged.drop_duplicates("date", keep=keep)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged[BENCHMARK_COLUMNS].sort_values("date").reset_index(drop=True)


def read_snapshot(path: Path) -> pd.DataFrame:
    """Read an existing index snapshot from disk."""

    data = pd.read_csv(path, encoding="utf-8-sig")
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    if "date" not in data.columns or "close" not in data.columns:
        raise ValueError("{} needs date and close columns".format(path))
    return data


def load_local_snapshots(snapshot_dir: Optional[str], code: str) -> List[pd.DataFrame]:
    """Every ``{code}_*.csv`` snapshot under ``snapshot_dir``, oldest range first."""

    if not snapshot_dir:
        return []
    paths = sorted(Path(snapshot_dir).glob("{}_*.csv".format(code)))
    frames = []
    for path in paths:
        try:
            frames.append(read_snapshot(path))
        except ValueError as exc:
            print("[skip] {}: {}".format(path.name, exc), flush=True)
    if frames:
        print("[snapshots] {} file(s) under {}".format(len(frames), snapshot_dir), flush=True)
    return frames


def fetch_remote_range(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch ``[start, end]`` from akshare, retrying transient disconnects."""

    import akshare as ak  # imported lazily: only this script needs the dependency

    symbol = "sh{}".format(code) if code.startswith(("0", "6")) else "sz{}".format(code)
    last_error = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            frame = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)
            if frame is not None and len(frame):
                return frame
            last_error = ValueError("empty response")
        except Exception as exc:  # noqa: BLE001 - akshare raises bare connection errors
            last_error = exc
        if attempt + 1 < FETCH_ATTEMPTS:
            time.sleep(FETCH_BACKOFF_SECONDS)
    raise RuntimeError(
        "akshare fetch for {} [{}..{}] failed after {} attempts: {}".format(
            code, start, end, FETCH_ATTEMPTS, last_error)
    )


def year_spans(start: str, end: str, years: int) -> List[tuple]:
    """Split ``[start, end]`` (``YYYYMMDD``) into ``years``-long chunks."""

    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if last < first:
        raise ValueError("end {} is before start {}".format(end, start))
    spans = []
    cursor = first
    while cursor <= last:
        span_end = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), last)
        spans.append((cursor.strftime("%Y%m%d"), span_end.strftime("%Y%m%d")))
        cursor = span_end + pd.Timedelta(days=1)
    return spans


def fetch_remote_range_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch ``[start, end]`` from tushare ``index_daily`` instead of akshare.

    akshare's eastmoney endpoint (``stock_zh_index_daily_em``) refuses the
    connection from some networks, which matters here because the benchmark's
    start date silently becomes the training set's start date.  ``index_daily``
    goes through the shared authenticated POST in
    :mod:`research.experiments.fetch_tushare_daily_basic`, so it needs nothing
    beyond the ``TS_TOKEN`` already in ``.env``.

    Returns the same column names akshare would, so the frame flows through
    :func:`normalize_index_frame` unchanged.
    """

    from research.experiments.fetch_tushare_daily_basic import call_api, read_token

    ts_code = "{}.SH".format(code) if code.startswith(("0", "6")) else "{}.SZ".format(code)
    token = read_token()
    frames = []
    for span_start, span_end in year_spans(start, end, TUSHARE_CHUNK_YEARS):
        payload = call_api("index_daily", token, {
            "ts_code": ts_code, "start_date": span_start, "end_date": span_end,
        })
        if payload is None or not len(payload):
            print("[fetch] {} empty [{}.{}]".format(ts_code, span_start, span_end), flush=True)
            continue
        frames.append(payload.rename(columns={"trade_date": "date", "vol": "volume"}))
        time.sleep(TUSHARE_SLEEP_SECONDS)
    if not frames:
        raise RuntimeError(
            "tushare index_daily returned nothing for {} [{}..{}]".format(ts_code, start, end)
        )
    merged = pd.concat(frames, ignore_index=True)
    # ``trade_date`` is a YYYYMMDD string; without an explicit format pandas
    # would have to guess, and guessing is how a date column silently shifts.
    merged["date"] = pd.to_datetime(merged["date"], format="%Y%m%d", errors="coerce")
    return merged


def write_benchmark(frame: pd.DataFrame, path: Path) -> Path:
    """Write in ``000300.csv``'s exact style: BOM + every field quoted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="000852")
    parser.add_argument("--name", default="中证1000")
    parser.add_argument("--start", default="20141017", help="fetch start, YYYYMMDD")
    parser.add_argument("--end", default="", help="fetch end, YYYYMMDD (default: today)")
    parser.add_argument("--snapshot", default="data/raw/akshare/index_history",
                        help="directory of existing {code}_*.csv snapshots to merge in")
    parser.add_argument("--source", choices=("akshare", "tushare"), default="akshare",
                        help="remote source for the missing span; akshare is the original default")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-fetch", action="store_true",
                        help="merge local snapshots only; no network")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    code, name = str(args.code), str(args.name)

    frames: List[Dict] = []
    local = [normalize_index_frame(frame, code, name)
             for frame in load_local_snapshots(args.snapshot, code)]
    if args.no_fetch:
        if not local:
            raise SystemExit("--no-fetch given but no local snapshots found")
        merged = merge_index_snapshots(local)
    else:
        # Local first, then the fetched span: merge_index_snapshots keeps the
        # last entry per date, so the freshly fetched rows win any overlap.
        remote_end = args.end or pd.Timestamp.today().strftime("%Y%m%d")
        fetch = fetch_remote_range if args.source == "akshare" else fetch_remote_range_tushare
        remote = normalize_index_frame(fetch(code, args.start, remote_end), code, name)
        print("[fetch] {} rows [{}..{}]".format(len(remote), remote["date"].iloc[0], remote["date"].iloc[-1]),
              flush=True)
        merged = merge_index_snapshots(local + [remote])

    path = write_benchmark(merged, Path(args.output_dir) / "{}.csv".format(code))
    print("[write] {} rows {} .. {} -> {}".format(
        len(merged), merged["date"].iloc[0], merged["date"].iloc[-1], path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
