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

This script bridges that gap: fetch the missing span from akshare and merge it
with whatever snapshots already sit on disk.

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
        remote = normalize_index_frame(fetch_remote_range(code, args.start, remote_end), code, name)
        print("[fetch] {} rows [{}..{}]".format(len(remote), remote["date"].iloc[0], remote["date"].iloc[-1]),
              flush=True)
        merged = merge_index_snapshots(local + [remote])

    path = write_benchmark(merged, Path(args.output_dir) / "{}.csv".format(code))
    print("[write] {} rows {} .. {} -> {}".format(
        len(merged), merged["date"].iloc[0], merged["date"].iloc[-1], path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
