"""Fetch per-symbol valuation history into ``data/raw/akshare/valuation_v2/``.

Two problems with the legacy cache at ``data/raw/akshare/valuation/`` make it
unusable for a *current* valuation-state diagnostic:

1. It stops in 2025, and stops on **different dates for different symbols**
   (1209 files end 2025-06-23, 673 end 2025-07-24, 326 end 2025-08-04).  Taking
   "the last row" per symbol for a cross-section would compare stocks as of
   different as-of dates without saying so.
2. Its source, ``stock_a_indicator_lg``, was **deleted in akshare 1.18**, so the
   cache cannot be extended or reproduced.

The replacement is ``stock_value_em``: daily, one call per symbol, 2018-01 ->
today, carrying 总市值 / PE(TTM) / PE(静) / 市净率 / 市现率 / 市销率.

It is **not** written over the legacy cache.  ``valuation_v2/`` sits beside it
because the legacy files are inputs to already-closed studies.

**The two sources are not spliced, and that is a measured decision.**  Where the
sources overlap (2018-01 -> the legacy cutoff) the per-symbol comparison is
recorded in ``compare_with_legacy`` and aggregated by ``_summarize_legacy``:

* ``pe_ttm`` agrees to ~1e-6 on the *median* day for 59 of 60 symbols -- the
  vendors are computing the same quantity.
* But the agreement breaks down in **contiguous runs** with a *constant* ratio
  across the run (e.g. exactly 0.603055 every day from 2019-09 to 2021, or
  1.423736 across 2021).  A constant ratio over a stretch is the signature of a
  **stale or pre-restatement earnings base**, not of noise: one vendor carried
  an older TTM figure for that window.
* ``pb`` differs because the vendors re-base quarterly net assets on different
  dates; ``ps`` differs by a fixed few percent on *every* day for some symbols.
  ``total_mv`` is a clean 10000x unit mismatch (legacy 万元, this source 元).

A percentile computed across such a junction measures the vendor, not the
market, so only the single internally-consistent source is used.  The cost is
that per-symbol history starts at 2018 -- stated as a limitation rather than
papered over with a splice.

Deliberately NOT here: the splice itself and the percentile computation.  This
script only fetches and merges raw vendor rows.

Example::

    python -m research.experiments.fetch_valuation_history --pool hs300 --limit-symbols 5
    python -m research.experiments.fetch_valuation_history --pool hs300+zz1000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from research.market_regime.feature_study_core import read_csv_with_fallback


DEFAULT_OUTPUT_DIR = "data/raw/akshare/valuation_v2"
LEGACY_DIR = "data/raw/akshare/valuation"
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 2.0
THROTTLE_SECONDS = 0.15
PROGRESS_EVERY = 50

# Raw vendor column -> our column name.  Explicit rather than positional: the
# vendor has renamed columns before and an extra column appearing upstream must
# not silently shift which number becomes which metric.
VENDOR_COLUMNS = {
    "数据日期": "date",
    "当日收盘价": "close",
    "总市值": "total_mv",
    "PE(TTM)": "pe_ttm",
    "PE(静)": "pe_static",
    "市净率": "pb",
    "市现率": "pcf",
    "市销率": "ps_ttm",
}

# Our column -> the legacy cache's column, for the overlap cross-check.
# ``total_mv`` is checked even though it is never spliced: the legacy cache
# stores 万元 and this source stores 元, a clean 10000x unit mismatch that the
# cross-check reports as a relative difference of exactly 9999.  It is listed
# here so that mismatch stays visible rather than being rediscovered later.
LEGACY_COLUMNS = {
    "pe_ttm": "pe_ttm",
    "pe_static": "pe",
    "pb": "pb",
    "ps_ttm": "ps_ttm",
    "pcf": None,       # the legacy cache has no PCF column
    "total_mv": "total_mv",
    "close": None,     # ... nor a close price
}
VALUE_COLUMNS = ["date", "symbol"] + sorted(set(VENDOR_COLUMNS.values()) - {"date"})

# A metric is candidate for splicing only when the two vendors agree to within
# this median relative difference on the overlap, for essentially every symbol.
# Measured agreement: PE(TTM) matches to ~1e-6, while PS differs by 3-7% on
# *every* day for some symbols and PB differs on 900/1841 days because the
# vendors re-base the quarterly net assets on different dates.  Thresholding per
# metric per symbol turns that from a judgement call into a recorded decision.
SPLICE_TOLERANCE = 0.001
# The tail bound is looser than the median bound but must still be tight.
SPLICE_P90_TOLERANCE = 0.01
# ...and "essentially every" is load-bearing.  Aggregating per-symbol medians by
# taking *their* median lets a badly-behaved minority hide: 600519's PS differs
# by 3% and was invisible behind four well-behaved symbols.
SPLICE_MIN_SHARE_OK = 0.95
MIN_OVERLAP_DAYS = 120

POOLS = {
    "hs300": ["data/HS300.txt"],
    "zz500": ["data/ZZ_500.txt"],
    "zz1000": ["data/ZZ_1000.txt"],
    "sh_all": ["data/SH_ALL.txt"],
    "sz_all": ["data/SZ_ALL.txt"],
}


_LAST_CALL_AT = [0.0]


def _throttle() -> None:
    """Keep the request rate modest; the vendor throttles bursts."""

    elapsed = time.time() - _LAST_CALL_AT[0]
    if 0 < elapsed < THROTTLE_SECONDS:
        time.sleep(THROTTLE_SECONDS - elapsed)
    _LAST_CALL_AT[0] = time.time()


def normalize_symbol(symbol: str) -> str:
    """Six-digit, zero-padded, no exchange prefix.

    Pool files are plain codes but any CSV/Excel round-trip will have read
    ``000001`` as the integer ``1``; padding silently elsewhere would drop the
    leading zero and fetch the wrong company.
    """

    text = str(symbol).strip().upper()
    for prefix in ("SH", "SZ", "BJ"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.lstrip(".").strip()
    if not text.isdigit():
        raise ValueError("symbol {!r} is not numeric".format(symbol))
    return text.zfill(6)


def read_symbols(paths: Sequence[str]) -> List[str]:
    """Union of the symbol codes in ``paths``, de-duplicated and sorted."""

    symbols = set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8-sig", errors="coerce")
        for line in text.splitlines():
            token = line.strip().strip(",").strip('"')
            if not token or token.startswith("#"):
                continue
            try:
                symbols.add(normalize_symbol(token))
            except ValueError:
                continue  # a header row, not a symbol
    if not symbols:
        raise ValueError("no symbols found in {}".format(list(paths)))
    return sorted(symbols)


def normalize_value_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce a raw ``stock_value_em`` frame into our long-lived schema.

    Rows whose *date* is unusable are dropped; a row whose PE is missing is
    kept, because a loss-making company has no PE and dropping the row would
    drop its PB and PS along with it.
    """

    if frame is None or len(frame) == 0:
        raise ValueError("valuation frame for {} is empty".format(symbol))
    data = frame.loc[:, [column for column in VENDOR_COLUMNS if column in frame.columns]].rename(
        columns=VENDOR_COLUMNS
    )
    if "date" not in data.columns:
        raise ValueError("frame for {} has no usable date column: {}".format(symbol, list(frame.columns)))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    for column in data.columns:
        if column != "date":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.drop_duplicates("date", keep="last").sort_values("date")
    data.insert(1, "symbol", symbol)
    for column in VALUE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data.loc[:, VALUE_COLUMNS].reset_index(drop=True)


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """Fetch one symbol's full available history, retrying transient failures."""

    import akshare as ak  # lazy: only this script needs the dependency

    last_error = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            _throttle()
            frame = ak.stock_value_em(symbol=symbol)
            if frame is not None and len(frame):
                return frame
            last_error = ValueError("empty response")
        except Exception as exc:  # noqa: BLE001 - akshare raises bare connection errors
            last_error = exc
        if attempt + 1 < FETCH_ATTEMPTS:
            time.sleep(FETCH_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError("fetch {} failed after {} attempts: {}".format(symbol, FETCH_ATTEMPTS, last_error))


def merge_with_cache(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Stack fresh rows over the cache, newer rows winning per (date, symbol)."""

    frames = [frame for frame in (existing, fresh) if frame is not None and len(frame)]
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).sort_values(["date"])
    merged = merged.drop_duplicates(["date", "symbol"], keep="last")
    for column in VALUE_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged.loc[:, VALUE_COLUMNS].sort_values("date").reset_index(drop=True)


def read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached per-symbol file, or None when absent/unreadable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        frame = read_csv_with_fallback(path, low_memory=False)
    except Exception:  # noqa: BLE001 - a corrupt cache must not block a refresh
        print("[warn] unreadable cache {}, rebuilding".format(path.name), flush=True)
        return None
    return frame if "date" in frame.columns else None


def write_cache(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------- #
# cross-check against the legacy cache
# --------------------------------------------------------------------------- #

def compare_with_legacy(fresh: pd.DataFrame, legacy_path: Path) -> Dict[str, Dict[str, float]]:
    """Per-metric agreement between the new source and the legacy cache.

    Restricted to dates both sources carry (2018-01 -> the legacy cutoff).  The
    result drives the splice decision: a metric that disagrees is *not* spliced,
    however much history that would buy, because a percentile computed across a
    vendor change measures the vendor.
    """

    legacy_path = Path(legacy_path)
    if not legacy_path.exists():
        return {}
    try:
        legacy = pd.read_csv(legacy_path, encoding="utf-8-sig", low_memory=False)
    except Exception:  # noqa: BLE001 - an unreadable legacy file is just no comparison
        return {}
    if "trade_date" not in legacy.columns:
        return {}
    legacy = legacy.copy()
    legacy["date"] = pd.to_datetime(legacy["trade_date"], errors="coerce")
    legacy = legacy.dropna(subset=["date"])

    # Rename the legacy columns explicitly rather than relying on pandas'
    # ``suffixes=``: that only fires on a collision, so ``pe`` (legacy) would
    # arrive bare next to ``pe_static`` while ``pe_ttm`` arrived suffixed.
    renames = {
        legacy_column: legacy_column + "_legacy"
        for legacy_column in LEGACY_COLUMNS.values()
        if legacy_column and legacy_column in legacy.columns
    }
    if not renames:
        return {}
    legacy = legacy.rename(columns=renames)[["date"] + list(renames.values())]

    merged = fresh.merge(legacy, on="date", how="inner")
    if len(merged) < MIN_OVERLAP_DAYS:
        return {}

    report: Dict[str, Dict[str, float]] = {}
    for metric, legacy_column in LEGACY_COLUMNS.items():
        if not legacy_column or metric not in fresh.columns:
            continue
        legacy_name = legacy_column + "_legacy"
        if legacy_name not in merged.columns:
            continue
        new_values = pd.to_numeric(merged[metric], errors="coerce")
        old_values = pd.to_numeric(merged[legacy_name], errors="coerce")
        usable = new_values.notna() & old_values.notna() & (old_values > 0)
        if int(usable.sum()) < MIN_OVERLAP_DAYS:
            continue
        relative = ((new_values[usable] - old_values[usable]).abs() / old_values[usable])
        median = float(relative.median())
        p90 = float(relative.quantile(0.90))
        report[metric] = {
            "n_overlap": int(usable.sum()),
            "median_rel_diff": median,
            "p90_rel_diff": p90,
            # Both moments must be tight, not just the median.  ``pe_static``
            # clears the median by six orders of magnitude while differing by
            # 25% on a tenth of days -- the vendors roll the annual report in on
            # different dates, so it is a *step*, and a step is exactly what a
            # spliced percentile would be measuring.
            "splice_ok": bool(median <= SPLICE_TOLERANCE and p90 <= SPLICE_P90_TOLERANCE),
        }
    return report


def refresh_symbol(
    symbol: str, output_dir: Path, legacy_dir: Path, compare: bool = True
) -> Dict[str, object]:
    """Fetch one symbol, merge it into its cache, and cross-check the overlap."""

    path = Path(output_dir) / "{}_history.csv".format(symbol)
    status: Dict[str, object] = {"symbol": symbol, "path": str(path)}
    try:
        fresh = normalize_value_frame(fetch_symbol(symbol), symbol)
    except Exception as exc:  # noqa: BLE001 - report, keep the batch going
        status.update({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)})
        return status

    merged = merge_with_cache(read_cache(path), fresh)
    write_cache(merged, path)
    status.update({
        "ok": True,
        "fetched_rows": int(len(fresh)),
        "total_rows": int(len(merged)),
        "date_min": str(fresh["date"].min().date()),
        "date_max": str(fresh["date"].max().date()),
        "legacy_rows": None,
    })
    if compare:
        report = compare_with_legacy(fresh, Path(legacy_dir) / "{}_history.csv".format(symbol))
        status["legacy"] = report
        status["legacy_rows"] = len(report)
    return status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="hs300",
                        help="'{}' or a '+'-join like hs300+zz1000".format("/".join(sorted(POOLS))))
    parser.add_argument("--symbols-file", default="", help="extra symbol list file, one code per line")
    parser.add_argument("--symbols", default="", help="explicit comma-separated codes (smoke tests)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--legacy-dir", default=LEGACY_DIR)
    parser.add_argument("--limit-symbols", type=int, default=0, help="fetch only the first N (smoke)")
    parser.add_argument("--no-compare", action="store_true", help="skip the legacy overlap cross-check")
    return parser.parse_args(argv)


def resolve_symbols(args) -> List[str]:
    """Explicit ``--symbols`` wins; otherwise the pool, plus any ``--symbols-file``."""

    if args.symbols.strip():
        symbols = sorted({normalize_symbol(token) for token in args.symbols.split(",") if token.strip()})
    else:
        paths = pool_paths(args.pool)
        if args.symbols_file:
            paths = paths + [args.symbols_file]
        symbols = read_symbols(paths)
    if args.limit_symbols:
        symbols = symbols[: int(args.limit_symbols)]
    return symbols


def pool_paths(pool: str) -> List[str]:
    names = [name.strip() for name in str(pool).split("+") if name.strip()]
    unknown = [name for name in names if name not in POOLS]
    if unknown:
        raise SystemExit("unknown pool {}; expected one of {}".format(unknown, sorted(POOLS)))
    return [path for name in names for path in POOLS[name]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    symbols = resolve_symbols(args)
    if not symbols:
        raise SystemExit("no symbols resolved from --pool/--symbols")

    output_dir = Path(args.output_dir)
    started = time.time()
    results: List[Dict[str, object]] = []
    print("fetching {} symbols -> {}".format(len(symbols), output_dir), flush=True)

    for index, symbol in enumerate(symbols, start=1):
        result = refresh_symbol(symbol, output_dir, Path(args.legacy_dir), compare=not args.no_compare)
        results.append(result)
        if index % PROGRESS_EVERY == 0 or index == len(symbols):
            failed = sum(1 for item in results if not item.get("ok"))
            elapsed = time.time() - started
            print("  {}/{}  failed={}  {:.1f}s  ({:.2f}s/symbol)".format(
                index, len(symbols), failed, elapsed, elapsed / index), flush=True)

    failed = [item["symbol"] for item in results if not item.get("ok")]
    splice_decisions = _summarize_legacy(results)
    manifest = {
        "analysis_id": "fetch_valuation_history",
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "pool": args.pool,
        "symbols_file": args.symbols_file or None,
        "requested": len(symbols),
        "succeeded": len(symbols) - len(failed),
        "failed": failed,
        "columns": VALUE_COLUMNS,
        "splice_tolerance": SPLICE_TOLERANCE,
        "splice_decisions": splice_decisions,
        "duration_seconds": round(time.time() - started, 1),
        "symbols": results,
    }
    manifest_path = output_dir / "fetch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("fetched {}/{}; manifest={}".format(len(symbols) - len(failed), len(symbols), manifest_path))
    if splice_decisions:
        print("splice decisions (relative difference vs legacy cache, per metric):")
        print("  {:<10} {:>6} {:>12} {:>12} {:>8}  {}".format(
            "metric", "n", "median_sym", "worst_sym", "share_ok", "splice_ok"))
        for metric, stats in sorted(splice_decisions.items()):
            print("  {:<10} {:>6} {:>12.6f} {:>12.6f} {:>8.2f}  {}".format(
                metric, stats["n_symbols"], stats["median_rel_diff"],
                stats["worst_symbol_rel_diff"], stats["share_ok"], stats["splice_ok"]))
    if failed:
        print("failed: {}".format(", ".join(failed[:20]) + ("..." if len(failed) > 20 else "")))
    return 0 if not failed else 1


def _summarize_legacy(results: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Aggregate per-symbol overlap statistics into one verdict per metric.

    Deliberately reports the **worst** symbol alongside the median one.  A
    median across symbols is the wrong statistic for a splice decision: it
    answers "does the typical stock agree", when the question is "does every
    stock agree".  600519's PS differs by 3% on every single day and vanished
    behind four well-behaved symbols when the median was the verdict.
    """

    by_metric: Dict[str, List[Dict[str, float]]] = {}
    for result in results:
        for metric, stats in (result.get("legacy") or {}).items():
            by_metric.setdefault(metric, []).append(stats)

    summary: Dict[str, Dict[str, object]] = {}
    for metric, entries in by_metric.items():
        medians = sorted(entry["median_rel_diff"] for entry in entries)
        share_ok = sum(1 for entry in entries if entry["splice_ok"]) / len(entries)
        summary[metric] = {
            "n_symbols": len(entries),
            "median_rel_diff": medians[len(medians) // 2],
            "worst_symbol_rel_diff": medians[-1],
            "share_ok": share_ok,
            "splice_ok_symbols": sum(1 for entry in entries if entry["splice_ok"]),
            "splice_ok": bool(share_ok >= SPLICE_MIN_SHARE_OK),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
