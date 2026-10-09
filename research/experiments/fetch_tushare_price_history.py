"""Fetch raw daily bars for a stock list into the ``raw_hfq_pct`` price cache.

Why this exists
---------------
``data/data_api.py`` is the normal route into the price cache, but its
``tushare`` branch imports the ``tushare`` package, which is not installed in
this environment -- the rest of the repo talks to the API through the
standard-library POST in :mod:`research.experiments.fetch_tushare_daily_basic`.
This script is the same idea for historical bars: one ``daily`` query per
symbol, written straight into the cache directory the market-regime pipeline
reads.

The ``raw_hfq_pct`` convention
------------------------------
Despite the name, the cache holds a per-symbol **constant rescale of raw OHLC**,
not a total-return series: ``verify_price_convention`` found the cached returns
*are* ``daily.close`` close-to-close returns (0 mismatches, max return
difference 5e-5), and ``splice_forward_2026_prices`` established that
``data_api._apply_raw_price_adjustment`` is a no-op on raw input, so the only
difference from raw is a free per-symbol constant near 1.0 (drifting ~4e-4 from
two-decimal rounding).  A constant rescale cancels in every return and ratio the
pipeline computes, which is why a fresh symbol can be written as plain raw OHLC
with ``k_s = 1`` and still be commensurate with the symbols already on disk.

``turnover`` is left all-NaN on purpose: it needs the ``daily_basic`` endpoint,
which is rate-limited to one call per hour and unusable here.  Every existing
file in the cache has it NaN too, so this matches rather than diverges.

Example::

    python -m research.experiments.fetch_tushare_price_history --limit 5
    python -m research.experiments.fetch_tushare_price_history
"""

import argparse
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from data.data_api import DataAPI
from research.experiments.fetch_tushare_daily_basic import call_api, read_token

CACHE_DIR = Path("data/raw/tushare/price_history/raw_hfq_pct")
DEFAULT_STOCK_FILE = "data/ZZ_500.txt"
START_DATE = "20050101"
END_DATE = "20251231"
# ``daily`` caps a call at 6000 rows and truncates silently when a span exceeds
# it, dropping the *earliest* sessions -- which for a price series is the least
# visible and most damaging kind of loss.  The whole 2005-2025 span is ~5000
# sessions, so one chunk covers it and the check in ``fetch_daily`` should
# never fire.
#
# One chunk rather than several because the endpoint's cost tracks *server-side
# latency*, not span: measured back to back on one symbol with the full field
# list, a 10-year span took 0.43s and a 21-year span 7.6s, then 28.6s for the
# identical 21-year query.  The distribution is heavy-tailed, so the cheapest
# lever is to draw from it fewer times.
CHUNK_YEARS = 25
DAILY_ROW_CAP = 6000
SLEEP_SECONDS = 0.35
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
# Matches ``data_api.STANDARD_PRICE_COLUMNS``, including the trailing column
# that stays empty.
CACHE_COLUMNS = ["date", "code", "open", "close", "high", "low", "volume",
                 "amount", "amplitude", "pct_change", "change", "turnover"]


def read_symbols(path: Path) -> List[str]:
    """Six-digit codes from a stock-list file, in file order, de-duplicated."""

    symbols: List[str] = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*(\d{6})", line)
        if not match:
            continue
        code = match.group(1)
        if code not in seen:
            seen.add(code)
            symbols.append(code)
    return symbols


def ts_code_for(symbol: str) -> str:
    """Reuse the production suffix rule so cache filenames stay consistent."""

    return DataAPI._normalize_symbol_for_tushare(symbol)


def cached_ts_codes(cache_dir: Path) -> set:
    """Every ``{ts_code}_{start}_{end}.csv`` already on disk."""

    codes = set()
    for path in Path(cache_dir).glob("*.csv"):
        match = re.match(r"^([0-9]{6}\.[A-Z]{2})_(\d{8})_(\d{8})\.csv$", path.name)
        if match:
            codes.add(match.group(1))
    return codes


def year_spans(start: str, end: str, years: int) -> List[tuple]:
    """Split ``[start, end]`` (``YYYYMMDD``) into ``years``-long chunks."""

    first, last = pd.Timestamp(start), pd.Timestamp(end)
    spans = []
    cursor = first
    while cursor <= last:
        span_end = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), last)
        spans.append((cursor.strftime("%Y%m%d"), span_end.strftime("%Y%m%d")))
        cursor = span_end + pd.Timedelta(days=1)
    return spans


def fetch_daily(ts_code: str, token: str, start: str = START_DATE, end: str = END_DATE,
                sleep: float = SLEEP_SECONDS) -> pd.DataFrame:
    """Raw daily bars for one symbol, oldest row first."""

    frames = []
    for span_start, span_end in year_spans(start, end, CHUNK_YEARS):
        payload = call_api("daily", token,
                           {"ts_code": ts_code, "start_date": span_start, "end_date": span_end},
                           fields=DAILY_FIELDS)
        if payload is not None and len(payload):
            if len(payload) >= DAILY_ROW_CAP:
                print("[warn] {} [{}.{}] returned {} rows, at the {} cap -- "
                      "the earliest sessions may have been dropped. Shorten CHUNK_YEARS.".format(
                          ts_code, span_start, span_end, len(payload), DAILY_ROW_CAP), flush=True)
            frames.append(payload)
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "open", "close", "high", "low", "vol", "amount"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("trade_date", keep="last")
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d", errors="coerce")
    return merged.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def build_cache_frame(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """Shape raw bars into the cached schema, deriving the percentage columns.

    The derived columns are recomputed from close exactly as
    ``data_api._standardize_daily_price_data`` does, so a freshly written file is
    indistinguishable from one that went through the normal route.
    """

    if raw is None or not len(raw):
        raise ValueError("no bars for {}".format(ts_code))
    frame = pd.DataFrame({
        "date": raw["trade_date"],
        "code": ts_code,
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "volume": pd.to_numeric(raw["vol"], errors="coerce"),
        "amount": pd.to_numeric(raw["amount"], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    previous_close = frame["close"].shift(1)
    previous_close = previous_close.where(previous_close != 0, float("nan"))
    frame["amplitude"] = (frame["high"] - frame["low"]) / previous_close * 100.0
    frame["pct_change"] = (frame["close"] / previous_close - 1.0) * 100.0
    frame["change"] = frame["close"].diff()
    frame["turnover"] = np.nan
    return frame[CACHE_COLUMNS]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stock-file", default=DEFAULT_STOCK_FILE)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--limit", type=int, default=None, help="fetch at most N symbols this run")
    parser.add_argument("--force", action="store_true", help="re-fetch symbols already cached")
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols = read_symbols(Path(args.stock_file))
    already = cached_ts_codes(cache_dir)
    todo = [] if args.force else [s for s in symbols if ts_code_for(s) not in already]
    if args.force:
        todo = list(symbols)
    if args.limit is not None:
        todo = todo[:args.limit]

    print("[universe] {} symbols, {} already cached, {} to fetch".format(
        len(symbols), len(symbols) - len(todo), len(todo)), flush=True)
    if not todo:
        print("[done] nothing to fetch", flush=True)
        return 0

    token = read_token()
    started = time.time()
    written, skipped = 0, []
    for index, symbol in enumerate(todo, start=1):
        ts_code = ts_code_for(symbol)
        try:
            raw = fetch_daily(ts_code, token, args.start, args.end, args.sleep)
            if not len(raw):
                skipped.append({"symbol": ts_code, "reason": "empty response"})
                continue
            frame = build_cache_frame(raw, ts_code)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
            skipped.append({"symbol": ts_code, "reason": "{}: {}".format(type(exc).__name__, exc)})
            continue
        # A superseded file (an older end date for the same symbol) is deleted
        # first: ``find_local_price_files`` globs ``{symbol}_*.csv`` and would
        # otherwise load both.
        for stale in cache_dir.glob("{}_*.csv".format(ts_code)):
            if stale.name != "{}_{}_{}.csv".format(ts_code, args.start, args.end):
                stale.unlink()
        frame.to_csv(cache_dir / "{}_{}_{}.csv".format(ts_code, args.start, args.end),
                     index=True, encoding="utf-8-sig")
        written += 1
        if index % 10 == 0 or index == len(todo):
            elapsed = time.time() - started
            rate = elapsed / index
            print("[fetch] {}/{} written={} skipped={} {:.0f}s elapsed, {:.1f}s/symbol, "
                  "eta {:.0f}min".format(index, len(todo), written, len(skipped),
                                         elapsed, rate, rate * (len(todo) - index) / 60.0),
                  flush=True)

    print("[done] written={} skipped={}".format(written, len(skipped)), flush=True)
    for row in skipped[:20]:
        print("[skip] {} {}".format(row["symbol"], row["reason"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
