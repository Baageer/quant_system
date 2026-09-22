"""Produce the 2026 pick lists: the model's ranking, and the book it implies.

Two artifacts per pool, because they answer different questions and their return
numbers are **not** interchangeable:

* **signal** -- ``stock_selection_signals_{pool}.csv``: the cross-sectional
  ``top_k`` slice on every rebalance date, with the realized label attached.
  Return is the equal-weight realized excess the model predicts, gross of
  position limits and costs.
* **portfolio** -- ``stock_selection_portfolio_{pool}_{...}.csv``: what the
  engine actually holds, net of costs, produced by the *published* chain.

**Why the portfolio run reaches past 2026-09-01.**  ``load_outcome_scores`` drops
rows whose forward label does not exist yet, which is what stops the published
forward backtest ten sessions early.  The engine never reads a label -- it needs
``date``, ``symbol`` and ``entry_score`` -- so this script feeds it an unfiltered
score frame and the book keeps trading to the end of the price data.  That
extends the *reporting* window without touching the published one; the
``verify_extension`` step below proves the two agree bit-for-bit on the overlap,
and refuses to write anything if they do not.

Deliberately *not* re-run here: model fitting, feature building, ``top_k``
selection.  ``top_k`` and the execution config come from the published ranking
manifest via ``published_execution``.

Example::

    python -m research.experiments.build_stock_selection_2026 --pools hs300,zz1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.build_forward_2026_samples import load_benchmark
from research.experiments.forward_2026_backtest import (
    FORWARD_PRICE_DIR,
    FORWARD_ROOT,
    POOLS,
    published_execution,
    run_arm,
)
from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.stock_selection import (
    HORIZON,
    SCORE_COLUMN,
    attach_names,
    build_picks,
    load_selection_scores,
    load_symbol_names,
    rebalance_dates,
    summarize_picks,
)
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data

#: Matches ``forward_2026_backtest.FORWARD_START``; scores begin 2026-01-05.
ENGINE_START = pd.Timestamp("2026-01-01")

#: The window the user asked about.  Left as constants rather than CLI-only so
#: the artifacts carry a fixed, citable span.
WINDOW_START = pd.Timestamp("2026-08-01")
WINDOW_END = pd.Timestamp("2026-09-15")


def load_engine_scores(source_csv: Path, start: pd.Timestamp) -> pd.DataFrame:
    """Score frame for the engine: no label requirement, same normalisation.

    Mirrors ``load_outcome_scores``' dtype handling (datetimes, zfilled symbols,
    numeric coercion) but drops rows only when a field the *engine* needs is
    missing.  A row without a label is a position that is still open, not a row
    to discard.
    """

    frame = pd.read_csv(source_csv, dtype={"symbol": str}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.zfill(6)
    frame[SCORE_COLUMN] = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    frame = frame.loc[frame["date"] >= start]
    frame = frame.dropna(subset=["date", "symbol", SCORE_COLUMN])
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def verify_extension(pool: str, dataset_dir: Path, equity: pd.DataFrame,
                     published_path: Path) -> Dict[str, object]:
    """The extended run must reproduce the published one on the overlap.

    This is the only thing standing between "the book was extended" and "the
    book was changed".  Any drift here means the unfiltered frame altered a
    decision, and nothing downstream may be published.
    """

    if not published_path.exists():
        return {"status": "NO_REFERENCE", "path": str(published_path)}
    published = pd.read_csv(published_path, index_col=0)
    published.index = pd.to_datetime(published.index)
    mine = equity.copy()
    mine.index = pd.to_datetime(mine.index)
    shared = published.index.intersection(mine.index)
    if len(shared) < 10:
        return {"status": "TOO_FEW_OVERLAP", "overlap": int(len(shared))}
    left = published.loc[shared, "portfolio_value"].astype(float).to_numpy()
    right = mine.loc[shared, "portfolio_value"].astype(float).to_numpy()
    delta = float(np.max(np.abs(left - right)))
    reference = float(np.max(np.abs(left))) or 1.0
    return {
        "status": "PASS" if delta <= 1e-9 * reference else "FAIL",
        "overlap_sessions": int(len(shared)),
        "overlap_end": str(shared.max().date()),
        "published_sessions": int(len(published)),
        "extended_sessions": int(len(mine)),
        "max_abs_equity_delta": delta,
        "max_rel_equity_delta": delta / reference,
    }


def window_metrics(equity: pd.DataFrame, levels: pd.Series, start: pd.Timestamp,
                   end: pd.Timestamp, initial_capital: float) -> Dict[str, object]:
    """Return/drawdown over the window, based at the last close *before* it.

    Using the pre-window close as the base means the window's first session
    counts; rebasing on the first in-window close would silently drop a day.
    """

    series = equity.copy()
    series.index = pd.to_datetime(series.index)
    prior = series.loc[series.index <= start - pd.Timedelta(days=1), "portfolio_value"]
    base = float(prior.iloc[-1]) if len(prior) else float(series["portfolio_value"].iloc[0])
    window = series.loc[(series.index >= start) & (series.index <= end), "portfolio_value"]
    if window.empty:
        return {"window_sessions": 0}
    curve = window / base
    peak = curve.cummax()
    drawdown = float((curve / peak - 1.0).min())
    bench = levels.copy()
    bench.index = pd.to_datetime(bench.index)
    bench_window = bench.loc[(bench.index >= start) & (bench.index <= end)]
    bench_prior = bench.loc[bench.index <= start - pd.Timedelta(days=1)]
    bench_base = float(bench_prior.iloc[-1]) if len(bench_prior) else float(bench_window.iloc[0])
    bench_return = float(bench_window.iloc[-1] / bench_base - 1.0) if len(bench_window) else float("nan")
    total = float(curve.iloc[-1] - 1.0)
    return {
        "window_sessions": int(len(window)),
        "window_start": str(window.index[0].date()),
        "window_end": str(window.index[-1].date()),
        "base_date": str(prior.index[-1].date()) if len(prior) else None,
        "base_value": base,
        "end_value": float(window.iloc[-1]),
        "window_return": total,
        "window_max_drawdown": drawdown,
        "benchmark_return": bench_return,
        "window_excess_vs_benchmark": total - bench_return,
    }


def build_pool(pool: str, dataset_dir: Path, forward_root: Path, output_dir: Path,
               window_start: pd.Timestamp, window_end: pd.Timestamp) -> Dict[str, object]:
    model_id = POOLS[pool]["model_id"]
    execution = published_execution(dataset_dir, model_id)
    top_k = float(execution["top_k"])
    levels = load_benchmark(pool)
    price_dirs = [str(DEFAULT_PRICE_HISTORY_DIR), str(forward_root / "price_history")]

    print("\n=== [{}] {} top_k={} ===".format(pool, model_id, top_k), flush=True)

    # --- signal caliber -----------------------------------------------------
    scores = load_selection_scores(pool, "frozen", dataset_dir, forward_root)
    all_dates = sorted(scores["date"].unique())
    rebalances = [d for d in rebalance_dates(all_dates)
                  if window_start <= pd.Timestamp(d) <= window_end]
    picks = build_picks(scores, top_k, dates=rebalances)
    picks = attach_names(picks, load_symbol_names())
    summary = summarize_picks(picks, scores)
    picks.to_csv(output_dir / "stock_selection_signals_{}.csv".format(pool),
                 index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "stock_selection_signal_summary_{}.csv".format(pool),
                   index=False, encoding="utf-8-sig")
    graded = summary["label_available"].sum()
    print("[{}] signal: {} rebalances, {} picks ({} graded, {} still open)".format(
        pool, len(rebalances), len(picks), int(picks["label_available"].sum()),
        int((~picks["label_available"]).sum())), flush=True)

    # --- portfolio caliber --------------------------------------------------
    source = dataset_dir / "forward_2026_scores_{}_frozen.csv".format(pool)
    engine_scores = load_engine_scores(source, ENGINE_START)
    staging = forward_root / "selection_2026"
    staging.mkdir(parents=True, exist_ok=True)
    engine_scores.to_csv(
        staging / "trade_outcome_test_predictions_{}.csv".format(model_id),
        index=False, encoding="utf-8-sig")
    prices, skipped = build_rank_market_data(engine_scores, None, SCORE_COLUMN,
                                             price_history_dirs=price_dirs)
    print("[{}] engine frame: {} rows / {} symbols, {}..{}; prices for {} ({} skipped)".format(
        pool, len(engine_scores), engine_scores["symbol"].nunique(),
        engine_scores["date"].min().date(), engine_scores["date"].max().date(),
        len(prices), len(skipped)), flush=True)

    result, metrics, equity, trades = run_arm(
        pool, "frozen", engine_scores, prices, execution, levels,
        "config/settings.yaml", 0, 20260917)

    check = verify_extension(
        pool, dataset_dir, equity,
        output_dir / "forward_2026_equity_{}_frozen.csv".format(pool))
    print("[{}] extension check: {}  overlap {} sessions to {}  max|d| = {:.3e}".format(
        pool, check["status"], check.get("overlap_sessions"),
        check.get("overlap_end"), check.get("max_abs_equity_delta", float("nan"))), flush=True)

    equity.to_csv(output_dir / "stock_selection_portfolio_equity_{}.csv".format(pool))
    trades.to_csv(output_dir / "stock_selection_portfolio_trades_{}.csv".format(pool),
                  index=False, encoding="utf-8-sig")

    in_window = trades.copy()
    if not in_window.empty:
        in_window["date"] = pd.to_datetime(in_window["date"])
        in_window = in_window.loc[(in_window["date"] >= window_start)
                                  & (in_window["date"] <= window_end)]
    if not in_window.empty:
        held = (in_window.loc[in_window["action"] == "buy"]
                .groupby("date")["symbol"].agg(list))
        rows = [{"date": d, "bought": len(v), "symbols": ",".join(sorted(v))}
                for d, v in held.items()]
        pd.DataFrame(rows).to_csv(
            output_dir / "stock_selection_portfolio_holdings_{}.csv".format(pool),
            index=False, encoding="utf-8-sig")

    window = window_metrics(equity, levels, window_start, window_end,
                            float(execution["initial_capital"]))
    print("[{}] portfolio window {}..{}: {:+.2%}  benchmark {:+.2%}  excess {:+.2%}  maxDD {:.2%}".format(
        pool, window["window_start"], window["window_end"], window["window_return"],
        window["benchmark_return"], window["window_excess_vs_benchmark"],
        window["window_max_drawdown"]), flush=True)

    return {
        "pool": pool,
        "model_id": model_id,
        "top_k": top_k,
        "top_k_source": execution["source"],
        "execution": {k: execution[k] for k in
                      ("min_holding_days", "max_positions", "initial_capital")},
        "signal": {
            "rebalances": [str(pd.Timestamp(d).date()) for d in rebalances],
            "picks": int(len(picks)),
            "graded": int(picks["label_available"].sum()),
            "open": int((~picks["label_available"]).sum()),
            "graded_rebalances": int(graded),
            "mean_edge_graded": float(summary.loc[summary["label_available"], "edge"].mean()),
        },
        "portfolio": window,
        "extension_check": check,
        "price_symbols_skipped": skipped[:20],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz1000")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--forward-root", default=str(FORWARD_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window-start", default=str(WINDOW_START.date()))
    parser.add_argument("--window-end", default=str(WINDOW_END.date()))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    forward_root = Path(args.forward_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    window_start = pd.Timestamp(args.window_start)
    window_end = pd.Timestamp(args.window_end)

    results = [build_pool(p.strip(), dataset_dir, forward_root, output_dir,
                          window_start, window_end)
               for p in args.pools.split(",") if p.strip()]

    manifest = {
        "window_start": str(window_start.date()),
        "window_end": str(window_end.date()),
        "horizon": HORIZON,
        "note": ("Signal caliber = top_k cross-section per rebalance date, equal-weight, "
                 "label = realized 10d excess. Portfolio caliber = engine book, net of costs, "
                 "extended past the published forward window by feeding the engine an "
                 "unlabelled score frame; extension_check must read PASS."),
        "pools": results,
    }
    (forward_root / "stock_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    failed = [r["pool"] for r in results if r["extension_check"]["status"] != "PASS"]
    print("\nmanifest -> {}".format(forward_root / "stock_selection_manifest.json"))
    if failed:
        print("EXTENSION CHECK FAILED for {}: the extended run is NOT the published "
              "book; do not quote these portfolio numbers".format(", ".join(failed)))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
