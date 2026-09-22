"""Build 2026 samples (features + auxiliary labels) for the forward window.

Everything here reuses the production constructors rather than reimplementing
them, because the frozen model was fit on their output and any divergence in the
feature recipe shows up as a silent shift in every score:

* features  -- ``feature_event_study.build_causal_feature_panel``
* labels    -- ``tradability_labels.add_auxiliary_labels`` (T+1 entry, h-day exit,
               ``future_excess_return_{h}d`` against the index, ``future_mae_{h}d``)
* universe  -- ``index_membership.membership_mask`` as a **row filter**, which is
               exactly what ``pit_pool_filter`` did to the published datasets

That last point matters for comparability.  ``pit_pool_filter``'s docstring is
explicit that the PIT rerun was a row filter on a sample set whose universe came
from the static pool file, not a rebuild of the universe.  The forward window
therefore also starts from the static pool symbols and filters rows, so the two
windows answer the same question on the same footing.  One consequence is
recorded rather than papered over: a 2026 index member that is not in the static
pool file cannot appear, which is a limitation the published runs share (see
``docs/research/forward_paper_tracking_2026.md`` §3).

Prices come from the two trees merged explicitly (``data_loader``'s
``load_local_price_history_multi``): the historical cache is read-only and the
spliced 2026 bars live in ``forward_2026/price_history``.  The benchmark level
series is merged the same way -- ``data/benchmarks/000300.csv`` stops at
2025-12-31, so the forward index bars are appended.

The panel is computed over **full history** and sliced afterwards.  Feature
windows reach back 120 sessions, so truncating the input first would corrupt the
first 2026 rows; the slice happens once the causal rolling windows are already
formed.

Example::

    python -m research.experiments.build_forward_2026_samples --limit-symbols 40
    python -m research.experiments.build_forward_2026_samples
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR, load_local_price_history_multi, load_hs300_symbols, to_tushare_symbol
from research.market_regime.feature_event_study import build_causal_feature_panel
from research.market_regime.index_membership import INDEX_CODE_BY_POOL, membership_mask
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.tradability_labels import add_auxiliary_labels

FORWARD_ROOT = Path("data/raw/tushare/forward_2026")
FORWARD_PRICE_ROOT = FORWARD_ROOT / "price_history"
MEMBERSHIP_DIR = FORWARD_ROOT / "index_weight_merged"

POOL_FILES = {"hs300": Path("data/HS300.txt"), "zz1000": Path("data/ZZ_1000.txt")}
BENCHMARK_FILES = {"hs300": "000300", "zz1000": "000852"}

#: A little before the published test end (2025-12-02) so the parity re-score has
#: its rows, and far enough back to warm up the 120-session percentile windows.
PANEL_START = pd.Timestamp("2025-11-01")
HORIZON = 10

#: How much price history to keep ahead of ``PANEL_START``.  ``None`` = all of it.
#:
#: Truncating looked like a free 3x speedup and is *almost* exact: with 300
#: sessions of warm-up, 85 of 89 features reproduce to within 3e-8 and
#: ``bollinger_bandwidth_20_percentile_120`` / ``rsrs_zscore`` / the
#: ``*_percentile_120`` family reproduce **bit-for-bit**.  But two do not, and
#: ``--verify-truncation`` is what caught them:
#:
#: * ``obv`` -- a cumulative sum, off by 3.3e7 (not a selected feature);
#: * ``vwap_distance`` -- off by **0.56**, against values of order 0.1.
#:
#: ``vwap_distance`` is wired to ``signals/indicators.py::vwap``, which is
#: ``(tp * volume).cumsum() / volume.cumsum()`` -- a **since-listing** ratio, not
#: the rolling-20 cousin of the same name in that module (``vwap_distance``, line
#: 432).  Its docstring says so plainly ("cumulative VWAP").  No finite warm-up
#: can reproduce it, and it sits in **both** selected feature lists, so
#: truncation is off the table.
#:
#: Full history costs ~1.2 s/symbol (vs ~0.4 truncated), i.e. ~25 min for the
#: whole universe -- cheap enough that exactness is not worth trading away.
LOOKBACK_SESSIONS = None


def load_symbols(pool: str) -> List[str]:
    """Bare six-digit codes from a pool list file, matching the published runs."""

    if pool == "hs300":
        return [str(code).strip().split(".")[0].zfill(6)
                for code in load_hs300_symbols(POOL_FILES[pool]) if str(code).strip()]
    return [line.strip().split(".")[0].zfill(6)
            for line in POOL_FILES[pool].read_text(encoding="utf-8").splitlines() if line.strip()]


def load_benchmark(pool: str) -> pd.Series:
    """Index close level series, historical file + 2026 bars, date-indexed."""

    code = BENCHMARK_FILES[pool]
    history = pd.read_csv("data/benchmarks/{}.csv".format(code), low_memory=False)
    history = history.loc[:, [c for c in history.columns if c in ("date", "close")]]
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    forward = pd.read_csv(FORWARD_ROOT / "benchmark" / "{}.SH.csv".format(code), low_memory=False)
    forward = forward.loc[:, ["trade_date", "close"]].rename(columns={"trade_date": "date"})
    forward["date"] = pd.to_datetime(forward["date"].astype(str), format="%Y%m%d", errors="coerce")

    combined = pd.concat([history, forward], ignore_index=True)
    combined["close"] = pd.to_numeric(combined["close"], errors="coerce")
    combined = combined.dropna(subset=["date", "close"]).sort_values("date")
    combined = combined.drop_duplicates(subset=["date"], keep="last").set_index("date")["close"]
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def truncate_history(data: pd.DataFrame, lookback: Optional[int],
                     panel_start: pd.Timestamp = PANEL_START) -> pd.DataFrame:
    """Drop rows more than ``lookback`` sessions before ``panel_start``.

    ``lookback=None`` keeps everything (used by ``--verify-truncation`` as the
    reference arm).  The cutoff is expressed in *sessions of that symbol*, not
    calendar days, so a halted name keeps the same number of usable rows.
    """

    if not lookback:
        return data
    cutoff_position = int(data.index.searchsorted(panel_start, side="left")) - int(lookback)
    if cutoff_position <= 0:
        return data
    return data.iloc[cutoff_position:]


def build_batch(
    symbols: Sequence[str], benchmark: pd.Series, price_dirs: Sequence[Path],
    lookback: Optional[int] = LOOKBACK_SESSIONS, panel_start: pd.Timestamp = PANEL_START,
) -> tuple:
    """Features + labels for one batch. Returns ``(samples, skipped)``."""

    stock_data, skipped = {}, []
    for symbol in symbols:
        try:
            data = load_local_price_history_multi(symbol, price_dirs)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)[:160]})
            continue
        stock_data[symbol] = truncate_history(data, lookback, panel_start)
    if not stock_data:
        return pd.DataFrame(), skipped

    panel = build_causal_feature_panel(stock_data, show_progress=False)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel.loc[panel["date"] >= PANEL_START].reset_index(drop=True)
    if panel.empty:
        return pd.DataFrame(), skipped

    labeled = add_auxiliary_labels(panel, stock_data, horizons=(HORIZON,),
                                   benchmark_returns=benchmark)
    # Every forward row is a test-split row, and test has no end boundary, so
    # the split-crossing purge that ``mark_auxiliary_label_eligibility`` applies
    # to train/validation can never remove one.  Availability is the only gate.
    labeled["source_split"] = "test"
    labeled["auxiliary_label_eligible"] = labeled["auxiliary_label_available_date"].notnull()
    return labeled, skipped


def verify_truncation(symbols: Sequence[str], price_dirs: Sequence[Path], lookback: int,
                      panel_start: pd.Timestamp = PANEL_START) -> pd.DataFrame:
    """Features with and without truncation must agree on every retained row.

    This is the check that turns "rolling windows are bounded, so 300 sessions is
    enough" from an argument into a measurement.  Differences are reported per
    feature, so a wrong lookback shows *which* window ran out rather than just
    failing a total.
    """

    stock_data = {}
    for symbol in symbols:
        try:
            stock_data[symbol] = load_local_price_history_multi(symbol, price_dirs)
        except (FileNotFoundError, ValueError):
            continue
    if not stock_data:
        raise SystemExit("no symbols loadable for the truncation check")

    full = build_causal_feature_panel(stock_data, show_progress=False).reset_index()
    full["date"] = pd.to_datetime(full["date"], errors="coerce")

    truncated_data = {s: truncate_history(d, lookback, panel_start) for s, d in stock_data.items()}
    truncated = build_causal_feature_panel(truncated_data, show_progress=False).reset_index()
    truncated["date"] = pd.to_datetime(truncated["date"], errors="coerce")

    keys = ["date", "symbol"]
    feature_columns = [c for c in full.columns if c not in keys and c in truncated.columns]
    merged = full.merge(truncated, on=keys, suffixes=("_full", "_trunc"))
    merged = merged.loc[merged["date"] >= panel_start]
    rows = []
    for column in feature_columns:
        a = pd.to_numeric(merged["{}_full".format(column)], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged["{}_trunc".format(column)], errors="coerce").to_numpy(dtype=float)
        both_nan = ~np.isfinite(a) & ~np.isfinite(b)
        difference = np.where(both_nan, 0.0, np.abs(a - b))
        difference = np.where(np.isfinite(difference), difference, np.inf)
        rows.append({
            "feature": column,
            "compared_rows": int(len(a)),
            "max_abs_difference": float(np.max(difference)) if len(a) else np.nan,
            "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
        })
    return pd.DataFrame(rows).sort_values("max_abs_difference", ascending=False).reset_index(drop=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz1000")
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--lookback-sessions", type=int, default=LOOKBACK_SESSIONS)
    parser.add_argument("--price-history-dir", default=str(DEFAULT_PRICE_HISTORY_DIR))
    parser.add_argument("--forward-price-dir", default=str(FORWARD_PRICE_ROOT))
    parser.add_argument("--membership-dir", default=str(MEMBERSHIP_DIR))
    parser.add_argument("--verify-truncation", action="store_true",
                        help="compare truncated vs full-history features and exit")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    price_dirs = [Path(args.price_history_dir), Path(args.forward_price_dir)]

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]

    if args.verify_truncation:
        # The check is only meaningful against an explicit warm-up length; the
        # shipped default is None precisely because truncation was rejected, so
        # asking for the check without naming a lookback is a usage error rather
        # than something to paper over with a fallback constant.
        if not args.lookback_sessions:
            raise SystemExit("--verify-truncation needs --lookback-sessions N (default is None = no truncation)")
        symbols = load_symbols(pools[0])
        if args.limit_symbols:
            symbols = symbols[: args.limit_symbols]
        table = verify_truncation(symbols, price_dirs, int(args.lookback_sessions))
        print("\n=== truncated (last {} sessions before {}) vs full history ===".format(
            args.lookback_sessions, PANEL_START.date()))
        print(table.to_string(index=False))
        worst = table["max_abs_difference"].max()
        mismatch = int(table["nan_mismatch"].sum())
        print("\nworst feature difference: {:.3e}   NaN-position mismatches: {}".format(worst, mismatch))
        print("verdict: {}".format("EXACT" if (worst < 1e-9 and mismatch == 0) else "NOT EXACT"))
        return 0 if (worst < 1e-9 and mismatch == 0) else 2

    benchmarks = {pool: load_benchmark(pool) for pool in pools}
    for pool, series in benchmarks.items():
        print("[{}] benchmark {} rows {}..{}".format(
            pool, BENCHMARK_FILES[pool], series.index[0].date(), series.index[-1].date()))

    batch_size = max(1, int(args.batch_size))
    report = {}
    for pool in pools:
        symbols = load_symbols(pool)
        if args.limit_symbols:
            symbols = symbols[: args.limit_symbols]
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        print("\n[{}] {} symbols -> {} batches of {}".format(pool, len(symbols), len(batches), batch_size),
              flush=True)

        frames, skipped, started = [], [], time.time()
        for index, batch in enumerate(tqdm(batches, desc="Building {}".format(pool), unit="batch")):
            frame, batch_skipped = build_batch(batch, benchmarks[pool], price_dirs)
            skipped += batch_skipped
            if not frame.empty:
                frames.append(frame)
        if not frames:
            print("[{}] no samples built".format(pool), flush=True)
            continue
        samples = pd.concat(frames, ignore_index=True)
        rows_before = len(samples)

        mask = membership_mask(samples, pool, Path(args.membership_dir))
        kept = samples.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)

        path = output_dir / "forward_2026_samples_{}.csv".format(pool)
        kept.to_csv(path, index=False, encoding="utf-8-sig")
        report[pool] = {
            "path": str(path),
            "symbols_requested": len(symbols),
            "symbols_built": int(samples["symbol"].nunique()),
            "rows_before_pit": int(rows_before),
            "rows_after_pit": int(len(kept)),
            "kept_pct": float(len(kept) / rows_before) if rows_before else None,
            "date_min": str(kept["date"].min().date()) if len(kept) else None,
            "date_max": str(kept["date"].max().date()) if len(kept) else None,
            "rows_2026": int((kept["date"] >= pd.Timestamp("2026-01-01")).sum()),
            "rows_2025_12_02": int((kept["date"] == pd.Timestamp("2025-12-02")).sum()),
            "skipped_symbols": len(skipped),
            "skipped_detail": skipped[:30],
            "seconds": round(time.time() - started, 1),
        }
        print("[{}] rows {} -> {} after PIT ({:.1%}); 2026 rows {}".format(
            pool, rows_before, len(kept), len(kept) / rows_before if rows_before else 0,
            report[pool]["rows_2026"]), flush=True)

    (FORWARD_ROOT / "samples_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmanifest -> {}".format(FORWARD_ROOT / "samples_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
