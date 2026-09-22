"""Where inside the HS300 pool does v1's alpha actually live?

Before spending a full rebuild on a 中证1000 (small-cap) test, it is worth
knowing what a universe change would even be testing.  v1 selects the top 10% of
the *whole* HS300 pool by predicted 10-day excess.  If that alpha comes from the
large, liquid end of the pool, moving to a small-cap index tests something the
model has never shown a edge in; if it comes from the small, illiquid end, v1 is
already a size tilt wearing a dispersion costume -- and a 中证1000 run would be a
test of the tilt, not of the model.

So: cut the pool into equal-count size terciles and look at each separately.

**The size proxy is causal.**  For every (date, symbol) the strata are built from
that symbol's *trailing* mean ``amount`` (成交额) ending on the signal date --
information available at that day's close, the same information the score itself
uses.  Nothing here looks at a price after the signal date, and the terciles are
equal-count *within each date*, so bucket membership cannot drift with the
market's overall liquidity.

Two levels of evidence, because they answer different questions:

* **Cross-sectional** (no backtest).  Restrict the universe to one tercile and
  re-apply the same top-10% rule.  ``selected_excess_over_universe`` then says
  whether the *ranking* is any good inside that tercile, with no cost, position
  sizing or capital-deployment confound.  This is the decisive table.
* **Backtest** (confirmatory).  Run v1 restricted to each tercile, at the same
  selection rate and with the capital fully deployed (``max_positions`` = the
  names the rule actually picks).  This adds the friction and equity-path view.

Plus the crux question, answered directly: **where does v1's own top-10% come
from?**  ``selection_mix`` compares the bucket composition of the selected names
against the universe composition.  A mild tilt is expected; a heavy one means v1
is mostly a size bet.

Note this is a *diagnostic*, not a new strategy -- the per-bucket arms are not
tradable candidates (they hold ~10 names on 500k), they exist to localise alpha.

Example::

    python -m research.market_regime.pool_stratification_diagnostic
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.data_loader import load_local_price_history
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    outcome_quality,
)


BUCKETS = ("small", "mid", "large")
LOOKBACK_SESSIONS = 60
TOP_K = 0.10
BASELINE_ARM = "full_pool_v1"


def load_liquidity_proxy(
    scores: pd.DataFrame, price_history_dir: Optional[str], lookback: int
) -> pd.DataFrame:
    """Trailing mean traded value per (date, symbol), in signal-date units.

    ``amount`` is stored per local session; the trailing mean is taken over the
    ``lookback`` sessions up to and including the signal date.  A symbol needs at
    least half a window of history before it gets a value -- short-history names
    drop out of the strata rather than landing in a bucket by accident.

    Rows are emitted per *scored* (date, symbol) pair, not per calendar signal
    date: a symbol that has no score on some date must not contribute a liquidity
    row there, or that date's tercile boundaries would be cut over a different
    universe than the one being ranked.
    """

    rows = []
    skipped = []
    for symbol, block in scores.groupby("symbol", sort=True):
        try:
            prices = load_local_price_history(symbol, price_history_dir=price_history_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        if "amount" not in prices.columns:
            skipped.append({"symbol": symbol, "reason": "no_amount_column"})
            continue
        series = pd.to_numeric(prices["amount"], errors="coerce")
        series.index = pd.DatetimeIndex(prices.index)
        trailing = series.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
        wanted = pd.DatetimeIndex(block["date"])
        frame = pd.DataFrame({"date": wanted, "symbol": symbol,
                              "liquidity": trailing.reindex(wanted).to_numpy()})
        rows.append(frame[frame["liquidity"].notna()])
    if not rows:
        raise ValueError("No symbol produced a liquidity proxy")
    return pd.concat(rows, ignore_index=True), skipped


def assign_buckets(liquidity: pd.DataFrame, bucket_count: int = 3) -> pd.DataFrame:
    """Equal-count, per-date strata of the liquidity proxy (rank based)."""

    if bucket_count != len(BUCKETS):
        raise ValueError("this diagnostic assumes {} buckets".format(len(BUCKETS)))
    frame = liquidity.copy()
    ranks = frame.groupby("date", sort=False)["liquidity"].rank(method="first")
    counts = frame.groupby("date", sort=False)["liquidity"].transform("size")
    fraction = (ranks - 1.0) / counts
    edges = np.arange(1, bucket_count, dtype=float) / bucket_count
    frame["bucket"] = np.asarray(BUCKETS, dtype=object)[np.searchsorted(edges, fraction, side="right")]
    return frame


def _membership_key(frame: pd.DataFrame) -> pd.Series:
    return frame["date"].astype(str) + "|" + frame["symbol"].astype(str)


def bucket_membership(scores: pd.DataFrame, buckets: pd.DataFrame, bucket: str) -> np.ndarray:
    """Boolean mask over ``scores``' own rows, positionally aligned with it.

    Deliberately not a merge: ``merge`` hands back a fresh RangeIndex, so
    assigning a merged selection back onto ``scores`` by label would write to the
    wrong rows whenever the merge dropped or reordered anything.
    """

    keyed = set(_membership_key(buckets[buckets["bucket"] == bucket]))
    return np.asarray(_membership_key(scores).isin(keyed), dtype=bool)


def selection_mix(scores: pd.DataFrame, buckets: pd.DataFrame, top_k: float) -> pd.DataFrame:
    """Bucket composition of the full-pool top-k, against the universe composition."""

    selected = np.asarray(select_top_k(scores, top_k, SCORE_COLUMN), dtype=bool)
    total_universe = len(scores)
    total_selected = int(selected.sum())
    rows = []
    for bucket in BUCKETS:
        in_bucket = bucket_membership(scores, buckets, bucket)
        picked = in_bucket & selected
        rows.append({
            "bucket": bucket,
            "universe_rows": int(in_bucket.sum()),
            "universe_share": float(in_bucket.sum() / total_universe) if total_universe else np.nan,
            "selected_rows": int(picked.sum()),
            "selected_share": float(picked.sum() / total_selected) if total_selected else np.nan,
            "selection_rate": float(picked.sum() / in_bucket.sum()) if in_bucket.sum() else np.nan,
        })
    table = pd.DataFrame(rows)
    table["tilt_vs_universe"] = table["selected_share"] / table["universe_share"]
    return table


def cross_sectional_by_bucket(
    scores: pd.DataFrame, buckets: pd.DataFrame, horizon: int, top_k: float
) -> pd.DataFrame:
    """The same top-k rule applied inside each tercile, scored on forward excess."""

    rows = []
    for bucket in BUCKETS:
        subset = scores[bucket_membership(scores, buckets, bucket)]
        subset = subset.sort_values(["date", "symbol"]).reset_index(drop=True)
        selected = select_top_k(subset, top_k, SCORE_COLUMN)
        quality = outcome_quality(subset, selected, horizon)
        rows.append({"bucket": bucket, "rows": len(subset), **quality})
    full = scores.sort_values(["date", "symbol"]).reset_index(drop=True)
    rows.append({"bucket": "ALL", "rows": len(full),
                 **outcome_quality(full, select_top_k(full, top_k, SCORE_COLUMN), horizon)})
    return pd.DataFrame(rows)


def backtest_bucket_universe(
    scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    buckets: pd.DataFrame,
    bucket: str,
    horizon: int,
    initial_capital: float,
    config_path: str,
    label: str,
    frictionless: bool = False,
) -> Dict[str, object]:
    """v1 restricted to one tercile, at the same selection rate and fully deployed."""

    in_bucket = bucket_membership(scores, buckets, bucket)
    eligible = scores[in_bucket].sort_values(["date", "symbol"]).reset_index(drop=True)
    if eligible.empty:
        raise ValueError("bucket {} has no rows".format(bucket))
    selected_eligible = np.asarray(select_top_k(eligible, TOP_K, SCORE_COLUMN), dtype=bool)
    selected = pd.Series(False, index=scores.index)
    selected.loc[in_bucket] = selected_eligible

    prepared = apply_rank_selection(raw_market_data, scores, selected)
    # Equal-count terciles select ~1/3 of v1's 30 names, so capital is spread over
    # that many slots to keep the two comparable as "fully deployed" books.
    per_date = int(round(len(eligible) / eligible["date"].nunique() * TOP_K))
    max_positions = max(1, per_date)
    target_notional = initial_capital / max_positions
    strategy = make_rank_strategy(max_positions, target_notional, min_holding_days=horizon)
    start_date, end_date = scores["date"].min(), scores["date"].max()
    results, trades = _run_engine(prepared, strategy, initial_capital, config_path,
                                  start_date, end_date, frictionless=frictionless)
    performance = summarize_backtest("{}_bt_{}".format(bucket, label), results, trades, initial_capital)
    quality = outcome_quality(scores, selected, horizon)
    row = {"bucket": bucket, "max_positions": max_positions,
           "selected_rows": int(len(selected_eligible)),
           "selected_excess_over_universe": quality["selected_excess_over_universe"],
           **{key: value for key, value in performance.items() if key != "strategy"}}
    print("    [{}] {} max_positions={} net={:.4f} turnover={:.2f}x".format(
        label, bucket, max_positions, row["total_return"], row["turnover_multiple"]), flush=True)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="pool_strata_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--lookback-sessions", type=int, default=LOOKBACK_SESSIONS)
    parser.add_argument("--skip-backtest", action="store_true",
                        help="cross-sectional tables only; no engine runs")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loads = {split: load_outcome_scores(dataset_dir, args.model_id, split, args.horizon)
             for split in ("validation", "test")}
    raw_market, skipped_prices = {}, {}
    for split in ("validation", "test"):
        raw_market[split], skipped_prices[split] = build_rank_market_data(
            loads[split], args.price_history_dir, SCORE_COLUMN)

    started = time.time()
    liquidity, skipped_proxy = {}, {}
    for split in ("validation", "test"):
        liquidity[split], skipped_proxy[split] = load_liquidity_proxy(
            loads[split], args.price_history_dir, args.lookback_sessions)
        print("[liquidity] {} {} rows from {} symbols ({:.1f}s)".format(
            split, len(liquidity[split]), liquidity[split]["symbol"].nunique(), time.time() - started), flush=True)
    buckets = {split: assign_buckets(liquidity[split]) for split in liquidity}

    strata_rows, mix_rows, cross_rows, backtest_rows = [], [], [], []
    for split in ("validation", "test"):
        scores = loads[split]
        frame = buckets[split]
        coverage = len(frame) / len(scores) if len(scores) else np.nan
        print("\n[{}] strata coverage {:.1%} ({} of {} scored rows)".format(
            split, coverage, len(frame), len(scores)), flush=True)
        summary = frame.groupby("bucket")["liquidity"].agg(
            rows="size", median_amount="median", min_amount="min", max_amount="max").reset_index()
        summary.insert(0, "split", split)
        strata_rows.append(summary)
        print(summary.round(0).to_string(index=False), flush=True)

        mix = selection_mix(scores, frame, TOP_K)
        mix.insert(0, "split", split)
        mix_rows.append(mix)
        print("\n  v1 top-{:.0%} composition by bucket:".format(TOP_K))
        print(mix.round(4).to_string(index=False), flush=True)

        cross = cross_sectional_by_bucket(scores, frame, args.horizon, TOP_K)
        cross.insert(0, "split", split)
        cross_rows.append(cross)
        print("\n  cross-sectional quality (same top-{:.0%} rule inside each tercile):".format(TOP_K))
        print(cross.round(4).to_string(index=False), flush=True)

        if args.skip_backtest:
            continue
        print("\n  backtests:", flush=True)
        parity = run_arm(scores, raw_market[split], args, "full_pool", args.max_positions, split)
        backtest_rows.append(parity)
        for bucket in BUCKETS:
            row = backtest_bucket_universe(
                scores, raw_market[split], frame, bucket, args.horizon,
                args.initial_capital, args.config_path, split)
            row["split"] = split
            backtest_rows.append(row)

    strata = pd.concat(strata_rows, ignore_index=True)
    mix_table = pd.concat(mix_rows, ignore_index=True)
    cross_table = pd.concat(cross_rows, ignore_index=True)
    backtests = pd.DataFrame(backtest_rows) if backtest_rows else pd.DataFrame()

    if not backtests.empty:
        view = ["split", "bucket", "max_positions", "selected_rows", "total_return", "annual_return",
                "sharpe_zero_rate", "max_drawdown", "turnover_multiple", "selected_excess_over_universe"]
        pd.set_option("display.width", 240)
        print("\n=== backtest by stratum (equal capital, fully deployed, same selection rate) ===")
        print(backtests[view].round(4).to_string(index=False))

    artifacts = {"strata": strata, "selection_mix": mix_table, "cross_sectional": cross_table}
    if not backtests.empty:
        artifacts["backtests"] = backtests
    paths = {}
    for name, table in artifacts.items():
        path = output_dir / "{}_{}.csv".format(args.run_id, name)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)

    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "where inside the HS300 pool does v1's alpha live (size terciles)",
        "model_id": args.model_id,
        "horizon": int(args.horizon),
        "top_k": TOP_K,
        "lookback_sessions": int(args.lookback_sessions),
        "size_proxy": "trailing mean of the raw session `amount` field (成交额), cross-sectional rank per date",
        "buckets": list(BUCKETS),
        "note": ("per-bucket backtest arms hold ~1/3 of v1's names on the same capital; they "
                 "localise alpha and are not tradable candidates"),
        "proxy_symbols_skipped": {split: len(value) for split, value in skipped_proxy.items()},
        "price_symbols_skipped": {split: len(value) for split, value in skipped_prices.items()},
        "paths": paths,
    }
    manifest_path = output_dir / "{}_manifest.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nmanifest={}".format(manifest_path))
    return 0


def run_arm(scores, raw_market_data, args, arm, max_positions, label):
    """The unmodified v1 arm, for parity on the same code path."""

    selected = select_top_k(scores, TOP_K, SCORE_COLUMN)
    prepared = apply_rank_selection(raw_market_data, scores, selected)
    target_notional = args.initial_capital / max_positions
    strategy = make_rank_strategy(max_positions, target_notional, min_holding_days=args.horizon)
    start_date, end_date = scores["date"].min(), scores["date"].max()
    results, trades = _run_engine(prepared, strategy, args.initial_capital, args.config_path,
                                  start_date, end_date)
    performance = summarize_backtest(BASELINE_ARM, results, trades, args.initial_capital)
    row = {"split": label, "bucket": BASELINE_ARM, "max_positions": max_positions,
           "selected_rows": int(np.asarray(selected, dtype=bool).sum()),
           "selected_excess_over_universe": outcome_quality(scores, selected, args.horizon)[
               "selected_excess_over_universe"],
           **{key: value for key, value in performance.items() if key != "strategy"}}
    print("    [{}] {} max_positions={} net={:.4f} turnover={:.2f}x".format(
        label, BASELINE_ARM, max_positions, row["total_return"], row["turnover_multiple"]), flush=True)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
