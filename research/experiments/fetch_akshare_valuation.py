"""Fetch index-level and market-level valuation anchors from akshare.

The valuation studies in ``research/market_regime`` so far only ever used
*per-symbol* valuation (``data/raw/akshare/valuation/``), and that cache has two
problems: it stops in 2025, and its source function ``stock_a_indicator_lg`` no
longer exists in akshare 1.18.  Nothing in the repo fetches the *index* and
*market aggregate* valuation series at all -- which is precisely the level at
which "is the market expensive?" is answerable, since per-symbol PE is mostly
noise.

This script fills that gap.  It writes one long-format CSV per source endpoint
(``date, anchor, value``) under ``data/raw/akshare/index_valuation/`` so that a
refresh can *merge* rather than overwrite -- several of these endpoints are
truncated or revised by the vendor, and blowing away history to add a week of
new rows would be destructive.

Anchors fetched (the ``anchor`` values in the written long frames):

* ``{hs300,sse50,csi500,csi1000}_{pe_ttm,pe_lyr,pe_ttm_median,pe_lyr_median,pb,pb_median}``
  monthly, 2005+ (``stock_index_pe_lg`` / ``stock_index_pb_lg``)
* ``market_{pe_ttm_median,pe_ttm_avg,pe_lyr_median,pe_lyr_avg}`` monthly, 2005+
  (``stock_a_ttm_lyr``)
* ``board_{sh,sz,cyb,kcb}_{pe,pe_median,pb,pb_median}`` per-board aggregates
  (``stock_market_pe_lg`` / ``stock_market_pb_lg``)
* ``cn_yield_{2y,5y,10y,30y}`` daily (``bond_zh_us_rate``) -- the risk-free leg
  of the equity risk premium
* ``market_dividend_yield`` (``stock_a_gxl_lg``)
* ``market_mv_gdp`` (``stock_buffett_index_lg``), total market cap over GDP

Deliberately NOT here: per-symbol valuation refresh.  That is a separate,
much slower job and belongs with the per-symbol diagnostic.

Example::

    python -m research.experiments.fetch_akshare_valuation --source all
    python -m research.experiments.fetch_akshare_valuation --source index --force
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from research.market_regime.feature_study_core import read_csv_with_fallback


DEFAULT_OUTPUT_DIR = "data/raw/akshare/index_valuation"
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 4.0
THROTTLE_SECONDS = 2.0
VALUE_COLUMNS = ["date", "anchor", "value"]

# ``stock_index_pe_lg`` / ``stock_index_pb_lg`` take a Chinese index name.
INDEX_SYMBOLS = {
    "hs300": "沪深300",
    "sse50": "上证50",
    "csi500": "中证500",
    "csi1000": "中证1000",
}
# Raw vendor column -> anchor suffix.  Keeping the mapping explicit rather than
# positional means an extra column appearing upstream cannot silently shift
# which number becomes which anchor.
INDEX_PE_COLUMNS = {
    "静态市盈率": "pe_lyr",
    "静态市盈率中位数": "pe_lyr_median",
    "滚动市盈率": "pe_ttm",
    "滚动市盈率中位数": "pe_ttm_median",
}
INDEX_PB_COLUMNS = {
    "市净率": "pb",
    "市净率中位数": "pb_median",
}
MARKET_PE_COLUMNS = {
    "middlePETTM": "market_pe_ttm_median",
    "averagePETTM": "market_pe_ttm_avg",
    "middlePELYR": "market_pe_lyr_median",
    "averagePELYR": "market_pe_lyr_avg",
}
# ``stock_market_{pe,pb}_lg`` are board endpoints; only these four are documented.
BOARD_SYMBOLS = {
    "sh": "上证",
    "sz": "深证",
    "cyb": "创业板",
    "kcb": "科创版",
}
BOND_COLUMNS = {
    "中国国债收益率2年": "cn_yield_2y",
    "中国国债收益率5年": "cn_yield_5y",
    "中国国债收益率10年": "cn_yield_10y",
    "中国国债收益率30年": "cn_yield_30y",
}


_LAST_CALL_AT = [0.0]


def _throttle() -> None:
    """Space out legulegu calls.

    Every ``*_lg`` endpoint shares akshare's cookie/CSRF helper
    (``stock_a_indicator.get_cookie_csrf``), which raises
    ``'NoneType' object has no attribute 'attrs'`` when the vendor serves a page
    without the CSRF meta tag -- i.e. when we are being throttled.  The failures
    look like intermittent network errors but are self-inflicted by request
    density, so pacing the batch is more effective than retrying harder.
    """

    elapsed = time.time() - _LAST_CALL_AT[0]
    if 0 < elapsed < THROTTLE_SECONDS:
        time.sleep(THROTTLE_SECONDS - elapsed)
    _LAST_CALL_AT[0] = time.time()


def _retry(callable_, label: str):
    """Call ``callable_`` up to ``FETCH_ATTEMPTS`` times with a flat backoff."""

    last_error = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            _throttle()
            frame = callable_()
            if frame is not None and len(frame):
                return frame
            last_error = ValueError("empty response")
        except Exception as exc:  # noqa: BLE001 - akshare raises bare connection errors
            last_error = exc
        if attempt + 1 < FETCH_ATTEMPTS:
            time.sleep(FETCH_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError("fetch {} failed after {} attempts: {}".format(label, FETCH_ATTEMPTS, last_error))


def to_long(frame: pd.DataFrame, date_column: str, mapping: Dict[str, str], prefix: str = "") -> pd.DataFrame:
    """Melt a wide vendor frame into the ``date, anchor, value`` schema.

    Columns absent from ``mapping`` are skipped rather than raising: akshare has
    already renamed vendor columns once, and a missing optional column should
    not cost us the ones that are present.
    """

    if frame is None or len(frame) == 0:
        raise ValueError("valuation frame is empty")
    if date_column not in frame.columns:
        raise ValueError("valuation frame needs {}, got {}".format(date_column, list(frame.columns)))
    data = frame.copy()
    data["date"] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=["date"])
    parts = []
    for source_column, anchor in mapping.items():
        if source_column not in data.columns:
            continue
        values = pd.to_numeric(data[source_column], errors="coerce")
        part = pd.DataFrame({
            "date": data["date"],
            "anchor": "{}{}".format(prefix, anchor),
            "value": values,
        })
        parts.append(part.dropna(subset=["value"]))
    if not parts:
        raise ValueError("no mapped columns present; frame had {}".format(list(frame.columns)))
    merged = pd.concat(parts, ignore_index=True)
    return merged.sort_values(["anchor", "date"]).reset_index(drop=True)


def merge_anchors(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Stack fresh rows onto the cached frame, newer rows winning per (date, anchor).

    The merge matters more than it looks: ``stock_a_ttm_lyr`` is a rolling
    10-year window and the vendor revises recent rows, so an overwrite would
    drop history the API no longer serves.
    """

    if fresh is None or len(fresh) == 0:
        raise ValueError("no fresh rows to merge")
    frames = [frame for frame in (existing, fresh) if frame is not None and len(frame)]
    merged = pd.concat(frames, ignore_index=True)
    missing = [column for column in VALUE_COLUMNS if column not in merged.columns]
    if missing:
        raise ValueError("valuation frame is missing columns: {}".format(missing))
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged.dropna(subset=["date", "value"])
    # Oldest first, so ``keep="last"`` lets the freshly fetched rows win.
    merged = merged.sort_values(["anchor", "date"])
    merged = merged.drop_duplicates(["date", "anchor"], keep="last")
    return merged[VALUE_COLUMNS].sort_values(["anchor", "date"]).reset_index(drop=True)


def read_anchor_file(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached long-format anchor file, or None when absent/unreadable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        frame = read_csv_with_fallback(path, low_memory=False)
    except Exception:  # noqa: BLE001 - a corrupt cache should not block a refresh
        print("[warn] unreadable cache {}, refetching from scratch".format(path), flush=True)
        return None
    if "date" not in frame.columns:
        return None
    return frame


def write_anchors(frame: pd.DataFrame, path: Path) -> Path:
    """Write the long-format anchor frame, BOM'd for Excel and UTF-8 consumers."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------- #
# endpoint fetchers -- each returns a raw vendor frame, no normalisation here
# --------------------------------------------------------------------------- #

def fetch_index_pe(symbol: str) -> pd.DataFrame:
    import akshare as ak  # lazy: only this script needs the dependency
    return _retry(lambda: ak.stock_index_pe_lg(symbol=symbol), "stock_index_pe_lg({})".format(symbol))


def fetch_index_pb(symbol: str) -> pd.DataFrame:
    import akshare as ak
    return _retry(lambda: ak.stock_index_pb_lg(symbol=symbol), "stock_index_pb_lg({})".format(symbol))


def fetch_market_pe_ttm_lyr() -> pd.DataFrame:
    import akshare as ak
    return _retry(ak.stock_a_ttm_lyr, "stock_a_ttm_lyr")


def fetch_market_pe_lg(symbol: str) -> pd.DataFrame:
    import akshare as ak
    return _retry(lambda: ak.stock_market_pe_lg(symbol=symbol), "stock_market_pe_lg({})".format(symbol))


def fetch_market_pb_lg(symbol: str) -> pd.DataFrame:
    import akshare as ak
    return _retry(lambda: ak.stock_market_pb_lg(symbol=symbol), "stock_market_pb_lg({})".format(symbol))


def fetch_bond_yield() -> pd.DataFrame:
    import akshare as ak
    return _retry(lambda: ak.bond_zh_us_rate(start_date="20050101"), "bond_zh_us_rate")


def fetch_dividend_yield() -> pd.DataFrame:
    import akshare as ak
    return _retry(ak.stock_a_gxl_lg, "stock_a_gxl_lg")


def fetch_buffett_index() -> pd.DataFrame:
    import akshare as ak
    return _retry(ak.stock_buffett_index_lg, "stock_buffett_index_lg")


# --------------------------------------------------------------------------- #
# source families
# --------------------------------------------------------------------------- #

def build_index_family() -> pd.DataFrame:
    """PE and PB anchors for each configured index."""

    parts = []
    for key, symbol in INDEX_SYMBOLS.items():
        parts.append(("{}_pe".format(key),
                      lambda symbol=symbol, key=key: to_long(
                          fetch_index_pe(symbol), "日期", INDEX_PE_COLUMNS, prefix="{}_".format(key))))
        parts.append(("{}_pb".format(key),
                      lambda symbol=symbol, key=key: to_long(
                          fetch_index_pb(symbol), "日期", INDEX_PB_COLUMNS, prefix="{}_".format(key))))
    return build_optional_parts(parts)


def build_optional_parts(parts: Sequence[tuple]) -> pd.DataFrame:
    """Build a family from several endpoints, tolerating individually dead ones.

    Endpoints go down independently and for long stretches (``stock_a_ttm_lyr``
    returns ``None`` from the vendor often enough that akshare then trips over
    it).  A family must degrade to the endpoints that still answer rather than
    losing every anchor to the first failure -- the anchors are complements, so
    a partially-populated family is still useful.
    """

    frames, failures = [], []
    for label, builder in parts:
        try:
            frames.append(builder())
        except Exception as exc:  # noqa: BLE001 - report, keep going
            failures.append("{}: {}: {}".format(label, type(exc).__name__, exc))
            print("[skip] {} unavailable: {}".format(label, exc), flush=True)
    if not frames:
        raise RuntimeError("all endpoints in this family failed: {}".format("; ".join(failures)))
    return pd.concat(frames, ignore_index=True)


def build_market_family() -> pd.DataFrame:
    """Whole-market PE anchors plus the per-board PE/PB aggregates.

    ``stock_market_pe_lg`` / ``stock_market_pb_lg`` are documented as *board*
    endpoints (上证/深证/创业板/科创版).  Passing the undocumented ``全部A股``
    does return rows, but it also returns a PE of ~117 that is not an all-A
    median, so only documented symbols are used -- an anchor carrying the wrong
    label is worse than a missing one.  The genuine all-A anchor is
    ``stock_a_ttm_lyr``.
    """

    parts = [("stock_a_ttm_lyr", lambda: to_long(fetch_market_pe_ttm_lyr(), "date", MARKET_PE_COLUMNS))]
    for key, symbol in BOARD_SYMBOLS.items():
        # The board PE endpoint exposes a single 平均市盈率 column -- it has no
        # median sibling, unlike the PB endpoint and unlike the index endpoints.
        parts.append(("board_{}_pe".format(key),
                      lambda symbol=symbol, key=key: to_long(
                          fetch_market_pe_lg(symbol), "日期",
                          {"平均市盈率": "board_{}_pe".format(key)})))
        parts.append(("board_{}_pb".format(key),
                      lambda symbol=symbol, key=key: to_long(
                          fetch_market_pb_lg(symbol), "日期",
                          {"市净率": "board_{}_pb".format(key), "市净率中位数": "board_{}_pb_median".format(key)})))
    return build_optional_parts(parts)


def build_bond_family() -> pd.DataFrame:
    return to_long(fetch_bond_yield(), "日期", BOND_COLUMNS)


def build_dividend_family() -> pd.DataFrame:
    return to_long(fetch_dividend_yield(), "日期", {"股息率": "market_dividend_yield"})


def build_buffett_family() -> pd.DataFrame:
    """``stock_buffett_index_lg`` carries levels in 亿元; the ratio is in percent."""

    raw = fetch_buffett_index()
    data = raw.copy()
    data["date"] = pd.to_datetime(data["日期"], errors="coerce")
    total_mv = pd.to_numeric(data["总市值"], errors="coerce")
    gdp = pd.to_numeric(data["GDP"], errors="coerce")
    ratio = (total_mv / gdp.where(gdp > 0)) * 100.0
    frame = pd.DataFrame({"date": data["date"], "anchor": "market_mv_gdp", "value": ratio})
    return frame.dropna(subset=["date", "value"]).reset_index(drop=True)


SOURCE_FAMILIES = {
    "index": ("index_anchors", build_index_family),
    "market": ("market_anchors", build_market_family),
    "bond": ("bond_anchors", build_bond_family),
    "dividend": ("dividend_anchors", build_dividend_family),
    "buffett": ("buffett_anchors", build_buffett_family),
}


def refresh_family(name: str, output_dir: Path, force: bool = False) -> Dict[str, object]:
    """Fetch one source family and merge it into its cache file.

    A network failure is recorded and returned, never raised, so that one dead
    endpoint cannot stop the others.
    """

    stem, builder = SOURCE_FAMILIES[name]
    path = Path(output_dir) / "{}.csv".format(stem)
    status: Dict[str, object] = {"source": name, "path": str(path)}
    try:
        fresh = builder()
    except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
        status.update({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)})
        print("[fail] {}: {}".format(name, status["error"]), flush=True)
        return status

    existing = None if force else read_anchor_file(path)
    merged = merge_anchors(existing, fresh)
    write_anchors(merged, path)
    status.update({
        "ok": True,
        "fetched_rows": int(len(fresh)),
        "total_rows": int(len(merged)),
        "anchor_count": int(merged["anchor"].nunique()),
        "date_min": str(pd.to_datetime(merged["date"]).min().date()),
        "date_max": str(pd.to_datetime(merged["date"]).max().date()),
        "replaced_cache": bool(existing is None),
    })
    print("[ok] {:<9} {:>6} rows -> {:>6} total, {}..{}  {}".format(
        name, status["fetched_rows"], status["total_rows"],
        status["date_min"], status["date_max"], path), flush=True)
    return status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="all",
                        help="one of {} or 'all'".format(",".join(sorted(SOURCE_FAMILIES))))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="ignore the existing cache and rebuild from the full vendor response")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.source == "all":
        names = sorted(SOURCE_FAMILIES)
    elif args.source in SOURCE_FAMILIES:
        names = [args.source]
    else:
        raise SystemExit("unknown --source {!r}; expected one of {}".format(
            args.source, sorted(SOURCE_FAMILIES) + ["all"]))

    output_dir = Path(args.output_dir)
    results: List[Dict[str, object]] = [refresh_family(name, output_dir, args.force) for name in names]

    manifest = {
        "analysis_id": "fetch_akshare_valuation",
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "requested_source": args.source,
        "force": bool(args.force),
        "fetch_attempts": FETCH_ATTEMPTS,
        "sources": results,
    }
    manifest_path = output_dir / "fetch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    failed = [item["source"] for item in results if not item.get("ok")]
    print("Fetched {}/{} source families; manifest={}".format(len(results) - len(failed), len(results), manifest_path))
    if failed:
        print("Failed: {}".format(", ".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
