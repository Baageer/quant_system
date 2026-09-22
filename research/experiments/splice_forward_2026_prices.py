"""Splice the fetched 2026 bars onto the local ``raw_hfq_pct`` cache.

The convention
--------------
``verify_price_convention`` established, on 28 symbols x 6751 symbol-days, that
the cached series' returns *are* tushare ``daily.close`` close-to-close returns:
the cache tracks raw close with **0 mismatches** and max absolute return
difference 5e-5, while the three adjusted candidates each break on exactly the
corporate-action days (30 of them, errors up to 32%).  A second check on the
price *level* shows the cache is a per-symbol **constant** rescale of raw OHLC
(one ratio covering open/high/low/close, drifting only 4e-4 from 2-decimal
rounding accumulation).

So the splice is exact rather than approximate:

    cached_price[s, t] = k_s * raw_price[s, t],   k_s = cached_close[s, d0] / raw_close[s, d0]

where ``d0`` is the symbol's last cached session -- 2025-12-31 for 1291 of 1297
symbols.  ``data_api._apply_raw_price_adjustment`` turns out to be a no-op here
(its input ``pct_change`` was already the raw return, so the rescale factor it
derives is constant), which is why the per-symbol ``k_s`` is a free constant
near 1.0 rather than something the cache itself records.

Why a separate tree
-------------------
Output lands in ``forward_2026/price_history/raw_hfq_pct/`` and the historical
cache is never written.  Appending 2026 rows to ``raw_hfq_pct/`` would silently
extend the *published* test window on any future re-run of the existing
pipeline, and every conclusion in ``docs/research/`` rests on those numbers.
The forward runner merges the two trees explicitly at load time.

Files are named ``{SYMBOL}_{start}_{end}.csv`` so the existing
``data_loader.find_local_price_files`` glob (``{symbol}_*.csv``) still matches.

A corporate action inside 2026 stays a raw price gap, exactly as it is in every
earlier year of the cache.  That is the point -- the forward window must use the
*same* convention as the training window, or the features shift underneath the
frozen model.

Example::

    python -m research.experiments.splice_forward_2026_prices --limit-symbols 50
    python -m research.experiments.splice_forward_2026_prices
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from research.experiments.fetch_tushare_daily_basic import call_api, read_token

CACHE_DIR = Path("data/raw/tushare/price_history/raw_hfq_pct")
FORWARD_DIR = Path("data/raw/tushare/forward_2026")
ANCHOR_SESSION = "20251231"

#: The window's first session.  Fixed: moving it would shift every published
#: feature, because the percentile families are computed inside the panel.
FIRST_SESSION = "20260105"

#: The window's last session.  ``None`` = auto-detect the newest session in
#: ``daily/`` that actually carries rows, which is the only definition that does
#: not need editing every day.  It used to be the constant ``"20260916"`` with
#: 2026-09-17 hard-coded as "the current day, unpublished" -- true when written,
#: false the moment tushare published it, and the failure mode was silent (the
#: new session simply never appeared).  ``--last-session`` pins it when a fixed
#: window is what you want.
DEFAULT_LAST_SESSION = None

#: Columns of the historical cache, in its own order, minus the unnamed index.
CACHE_COLUMNS = ("date", "code", "open", "close", "high", "low", "volume",
                 "amount", "amplitude", "pct_change", "change", "turnover")

ADJ_FIELDS = "ts_code,trade_date,adj_factor"
ANCHOR_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"


def cached_last_close(path: Path) -> Optional[tuple]:
    """``(timestamp, close)`` of the symbol's last cached row with a usable close."""

    frame = pd.read_csv(path, usecols=["date", "close"], low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        return None
    return frame["date"].iloc[-1], float(frame["close"].iloc[-1])


def load_session(path: Path) -> pd.DataFrame:
    """One whole-market session file from ``forward_2026/daily``."""

    frame = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str}, low_memory=False)
    if frame.empty or "close" not in frame.columns:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["close"])


def latest_published_session(forward_dir: Path, first_session: str = FIRST_SESSION) -> Optional[str]:
    """Newest session in ``daily/`` that carries at least one row.

    A session file with only a header is tushare having not published yet (or an
    error swallowed by the fetcher), never a market holiday -- the fetcher only
    asks for sessions ``trade_cal`` marks open.  See
    ``fetch_by_session``'s note on why such a file must not be cached as valid.
    """

    days = [path.stem for path in (forward_dir / "daily").glob("*.csv")
            if re.fullmatch(r"\d{8}", path.stem)
            and path.stem >= first_session
            and len(load_session(path)) > 0]
    return max(days) if days else None


def forward_sessions(forward_dir: Path, last_session: str) -> List[str]:
    """Published 2026 sessions inside the window, ascending."""

    days = []
    for path in sorted((forward_dir / "daily").glob("*.csv")):
        day = path.stem
        if not re.fullmatch(r"\d{8}", day):
            continue
        if FIRST_SESSION <= day <= last_session and len(load_session(path)) > 0:
            days.append(day)
    return days


def anchor_closes(forward_dir: Path, token: str, retries: int) -> Dict[str, float]:
    """Raw close on the anchor session, for every symbol that traded then."""

    frame = load_session(forward_dir / "daily" / "{}.csv".format(ANCHOR_SESSION))
    if frame.empty:
        raise FileNotFoundError(
            "anchor session {} is missing; run fetch_tushare_forward_2026 first".format(ANCHOR_SESSION))
    return dict(zip(frame["ts_code"], frame["close"].astype(float)))


def fetch_symbol_anchor(
    ts_code: str, token: str, day: str, cache_path: Path, retries: int,
) -> Optional[float]:
    """Raw close for a symbol absent from the anchor session (suspended that day).

    Only a handful of names hit this path, so the extra per-symbol call is
    cheaper than rethinking the anchor.
    """

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        frame = pd.read_csv(cache_path, dtype=str, low_memory=False)
    else:
        frame = call_api("daily", token, {"ts_code": ts_code, "trade_date": day},
                         fields=ANCHOR_FIELDS, retries=retries)
        frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
    if frame.empty:
        return None
    hits = frame.loc[frame["trade_date"].astype(str) == day, "close"]
    if hits.empty:
        return None
    return float(pd.to_numeric(hits, errors="coerce").iloc[0])


def build_panel(forward_dir: Path, days: Sequence[str]) -> pd.DataFrame:
    """All 2026 rows in the window as one long frame."""

    frames = [load_session(forward_dir / "daily" / "{}.csv".format(day)) for day in days]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no non-empty 2026 sessions found in {}".format(forward_dir / "daily"))
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "vol", "amount"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel


def resample_cached_symbol(symbol_frame: pd.DataFrame, anchor_close: float, anchor_date: pd.Timestamp) -> pd.DataFrame:
    """Turn one symbol's spliced raw series into the cache's column schema.

    ``pct_change`` / ``change`` / ``amplitude`` are recomputed from the rescaled
    close rather than copied from tushare: that is what
    ``data_api._apply_raw_price_adjustment`` does at lines 329-333, so the
    spliced rows are byte-compatible with how the cache regenerates its own.
    The anchor row is prepended first so the first 2026 session gets a genuine
    previous close instead of a NaN.
    """

    frame = symbol_frame.sort_values("date").copy()
    if not frame.empty:
        first = frame.iloc[0]
        anchor_row = {column: first[column] for column in frame.columns}
        anchor_row.update({"date": anchor_date, "open": anchor_close, "high": anchor_close,
                           "low": anchor_close, "close": anchor_close, "volume": np.nan,
                           "amount": np.nan,
                           "code": first.get("code"), "turnover": np.nan})
        frame = pd.concat([pd.DataFrame([anchor_row]), frame], ignore_index=True)
    previous_close = frame["close"].shift(1)
    previous_close = previous_close.where(previous_close != 0, np.nan)
    frame["change"] = frame["close"].diff()
    frame["pct_change"] = (frame["close"] / previous_close - 1.0) * 100.0
    frame["amplitude"] = (frame["high"] - frame["low"]) / previous_close * 100.0
    return frame.loc[frame["date"] > anchor_date, list(CACHE_COLUMNS)].reset_index(drop=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--forward-dir", default=str(FORWARD_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--last-session", default=DEFAULT_LAST_SESSION,
                        help="YYYYMMDD end of the window; default = newest session in "
                             "daily/ that carries rows")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache_dir = Path(args.cache_dir)
    forward_dir = Path(args.forward_dir)
    output_dir = Path(args.output_dir) if args.output_dir else forward_dir / "price_history" / "raw_hfq_pct"
    output_dir.mkdir(parents=True, exist_ok=True)

    token = read_token(Path(args.env_file))
    last_session = args.last_session or latest_published_session(forward_dir)
    if last_session is None:
        print("no published session found under {}".format(forward_dir / "daily"))
        return 1
    days = forward_sessions(forward_dir, last_session)
    if not days:
        print("no session in {}..{} carries rows".format(FIRST_SESSION, last_session))
        return 1
    pending = [path.stem for path in (forward_dir / "daily").glob("*.csv")
               if re.fullmatch(r"\d{8}", path.stem) and path.stem > last_session]
    print("{} published sessions in {}..{}{}".format(
        len(days), FIRST_SESSION, last_session,
        "  ({} still unpublished: {})".format(len(pending), ", ".join(sorted(pending)))
        if pending else ""))
    anchors = anchor_closes(forward_dir, token, args.retries)
    panel = build_panel(forward_dir, days)
    panel["date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d", errors="coerce")
    print("{} rows, {} symbols in window".format(len(panel), panel["ts_code"].nunique()))

    cached = {}
    for path in sorted(cache_dir.glob("*.csv")):
        match = re.match(r"([0-9]{6}\.[A-Z]{2})_(\d+)_(\d+)\.csv$", path.name)
        if not match:
            continue
        code, end = match.group(1), match.group(3)
        if code not in cached or end > cached[code][1]:
            cached[code] = (path, end)

    symbols = sorted(cached)
    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]

    written, skipped, scales, anchor_fetches = 0, [], [], 0
    for ts_code in tqdm(symbols, desc="splicing", unit="symbol"):
        path = cached[ts_code][0]
        last = cached_last_close(path)
        if last is None:
            skipped.append({"symbol": ts_code, "reason": "no_usable_cached_close"})
            continue
        anchor_date, cached_anchor = last
        raw_anchor = anchors.get(ts_code)
        if raw_anchor is None:
            anchor_fetches += 1
            raw_anchor = fetch_symbol_anchor(
                ts_code, token, anchor_date.strftime("%Y%m%d"),
                forward_dir / "anchor" / "{}.csv".format(ts_code), args.retries)
        if raw_anchor is None or raw_anchor <= 0:
            skipped.append({"symbol": ts_code, "reason": "no_raw_anchor_close",
                            "anchor_date": str(anchor_date.date())})
            continue

        scale = cached_anchor / raw_anchor
        rows = panel[panel["ts_code"] == ts_code]
        if rows.empty:
            skipped.append({"symbol": ts_code, "reason": "no_2026_rows"})
            continue

        frame = rows.loc[:, ["date", "open", "high", "low", "close", "vol", "amount"]].copy()
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] * scale
        frame["volume"] = frame["vol"]
        frame["code"] = ts_code
        frame["turnover"] = np.nan
        frame = resample_cached_symbol(frame, cached_anchor, anchor_date)
        target = output_dir / "{}_{}_{}.csv".format(ts_code, FIRST_SESSION, last_session)
        # The end session is in the filename, so extending the window leaves the
        # previous file for the same symbol behind.  Both would match the
        # loader's ``{symbol}_*.csv`` glob, and the two copies of an overlapping
        # date resolve through ``sort_values("date")`` -- whose tie order is not
        # guaranteed stable -- so which one survives would be luck.  Drop the
        # superseded ones instead of leaving that to chance.
        for stale in output_dir.glob("{}_*.csv".format(ts_code)):
            if stale != target:
                stale.unlink()
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        written += 1
        scales.append({"symbol": ts_code, "scale": scale, "anchor_date": str(anchor_date.date()),
                       "rows": int(len(frame))})

    scale_frame = pd.DataFrame(scales)
    manifest = {
        "first_session": FIRST_SESSION,
        "last_session": last_session,
        "sessions": len(days),
        "anchor_session": ANCHOR_SESSION,
        "anchor_fetched_per_symbol": anchor_fetches,
        "symbols_written": written,
        "symbols_skipped": len(skipped),
        "skipped_detail": skipped[:50],
        "rows_total": int(scale_frame["rows"].sum()) if not scale_frame.empty else 0,
        "scale": {
            "min": float(scale_frame["scale"].min()) if not scale_frame.empty else None,
            "median": float(scale_frame["scale"].median()) if not scale_frame.empty else None,
            "max": float(scale_frame["scale"].max()) if not scale_frame.empty else None,
        },
        "symbols_2026_without_cached_anchor": int(panel["ts_code"].nunique() - written),
        "output_dir": str(output_dir),
    }
    (forward_dir / "splice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nwrote {} symbol files to {}".format(written, output_dir))
    if not scale_frame.empty:
        print("scale k = cached/raw: min {:.6f}  median {:.6f}  max {:.6f}".format(
            scale_frame["scale"].min(), scale_frame["scale"].median(), scale_frame["scale"].max()))
    print("row counts: min {}  median {}  max {}".format(
        int(scale_frame["rows"].min()), int(scale_frame["rows"].median()), int(scale_frame["rows"].max())))
    if skipped:
        print("skipped {}: {}".format(len(skipped), skipped[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
