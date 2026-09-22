"""Per-symbol valuation state: own history (L3) and within-pool (L2).

Companion to ``index_valuation_diagnostic``.  That module answers "is the
market expensive?"; this one answers "is *this stock* expensive, and by which
yardstick" -- and it insists on keeping the two yardsticks apart:

* **L3 (own history)** -- where does today's multiple sit in this symbol's own
  trailing range?  A cheap stock in an expensive sector and an expensive stock
  in a cheap sector both look "fair" to a universe percentile; this does not.
* **L2 (within pool)** -- where does it sit against its peers *today*?

Neither is a summary of the other, and :func:`own_vs_peer_cell` is where the gap
becomes readable: ``cheap_self_expensive_peers`` is a stock that fell while its
group did not, which is a different fact from "cheap".

Three conventions carry over from ``valuation_state_core`` and one is new:

* Percentiles are trailing and self-inclusive; ``min_periods == window`` so an
  unfilled window is NaN rather than a mislabelled long-window number.
* Every percentile ships beside its own sample count.
* Every row carries its ``scope``-equivalent (the pool it was ranked in).

The new one: **a non-positive multiple is a fourth state, not a cheap one.**
PE <= 0 means the company lost money; PCF <= 0 means it burned cash.  Ranked
naively they land at the *bottom* of the range and get labelled "extremely
cheap" -- the single most misleading output a valuation screen can produce.
They are excluded from the ranking and counted separately.

Descriptive only -- see ``valuation_state_core.assert_no_forward_columns``, which
refuses any forward-looking column in anything this module emits.

Example::

    python -m research.market_regime.stock_valuation_diagnostic --run-id stock_valuation_v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.valuation_state_core import (
    BAND_CUTS,
    MIN_PEERS,
    assert_no_forward_columns,
    band_from_score,
    cheapness_score,
    composite_state,
    own_vs_peer_cell,
    rel_own_vs_peers,
    trailing_percentile,
    window_label,
)


PREFIX = "stock_valuation"
DEFAULT_INPUT_DIR = "data/raw/akshare/valuation_v2"
# Every metric here is a price multiple: higher = more expensive.  The direction
# convention is still routed through ``cheapness_score`` rather than assumed, so
# adding a yield later cannot silently inherit the wrong sign.
METRICS = ("pe_ttm", "pb", "ps_ttm", "pcf")
DEFAULT_POOLS = {"hs300": "data/HS300.txt", "zz1000": "data/ZZ_1000.txt"}

# Own-history windows for the per-symbol layer.  Deliberately *not* the index
# layer's ``(36, 60, 120, None)``: the per-symbol source only reaches back to
# 2018-01, so on a month-end grid there are ~104 observations in total.  120m
# therefore never fills and would emit an all-NaN column set -- a window that
# cannot exist is noise, not a measurement.  36m fills from 2021-01 and 60m from
# 2023-01; the split is stated in the doc rather than hidden behind a number.
WINDOWS_MONTHS_STOCK = (36, 60, None)
# Own-history window used when pairing against the cross-section: the headline
# window, matching the index layer's summary window.
PAIRING_WINDOW = 60


def read_pool(path: Path) -> List[str]:
    """Six-digit codes from a pool file, de-duplicated and sorted."""

    text = Path(path).read_text(encoding="utf-8-sig", errors="coerce")
    codes = set()
    for line in text.splitlines():
        token = line.strip().strip(",").strip('"')
        if token.isdigit():
            codes.add(token.zfill(6))
    return sorted(codes)


def load_pool_membership(pools: Mapping[str, str] = DEFAULT_POOLS) -> Dict[str, set]:
    """``{pool_name: {symbol, ...}}`` from the current pool snapshots.

    These files are *today's* membership.  Using them as historical membership
    would be survivorship bias -- a stock in today's HS300 was not necessarily
    in it in 2018 -- so they are used only to label the current snapshot, never
    to define who was in the pool at a past date.
    """

    return {name: set(read_pool(path)) for name, path in pools.items() if Path(path).exists()}


def load_symbol_history(symbol: str, input_dir: Path) -> Optional[pd.DataFrame]:
    """One symbol's cached valuation history, or None when absent/empty."""

    path = Path(input_dir) / "{}_history.csv".format(symbol)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:  # noqa: BLE001 - a corrupt cache should skip, not abort
        return None
    if "date" not in frame.columns or frame.empty:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = str(symbol)
    return frame.dropna(subset=["date"])


def to_month_end(frame: pd.DataFrame, metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    """Collapse a daily symbol history onto a month-end grid.

    Both L2 and L3 are computed on this grid.  The alternative -- comparing a
    daily series for one symbol against a differently-sampled series for another
    -- would let sampling density leak into a cross-sectional percentile.
    Month-end also makes the grid identical across symbols, so a "same date"
    cross-section really is the same date.

    Carries ``observed_date`` -- the *real* last observation behind each
    month-end cell.  ``resample("ME")`` labels the in-progress month by its
    future end date, so the newest bucket is called ``2026-09-30`` while the
    newest row the vendor served is ``2026-09-14``.  Reading the label as the
    as-of date would have this module claim data that does not exist yet, and
    the latest snapshot is the one row a reader acts on.
    """

    keep = ["date"] + [metric for metric in metrics if metric in frame.columns]
    data = frame.loc[:, keep].set_index("date").sort_index()
    monthly = data.resample("ME").last()
    observed = pd.Series(data.index.to_numpy(), index=data.index).resample("ME").last()
    keep_rows = monthly.notna().any(axis=1)
    monthly = monthly.loc[keep_rows].copy()
    monthly["observed_date"] = observed.loc[keep_rows]
    return monthly


def build_panel(
    input_dir: Path, symbols: Optional[Sequence[str]] = None, metrics: Sequence[str] = METRICS
) -> pd.DataFrame:
    """Long panel ``date, symbol, <metrics>`` on a month-end grid.

    Ragged by construction: symbols list on different dates, so each series
    starts where its own history starts.  Nothing is forward-filled backwards --
    inventing a pre-IPO valuation would corrupt every percentile that reads it.
    """

    directory = Path(input_dir)
    if symbols is None:
        symbols = sorted(path.name[: -len("_history.csv")] for path in directory.glob("*_history.csv"))
    frames = []
    for symbol in symbols:
        history = load_symbol_history(symbol, directory)
        if history is None or len(history) < 2:
            continue
        monthly = to_month_end(history, metrics)
        if monthly.empty:
            continue
        monthly = monthly.reset_index().assign(symbol=str(symbol))
        frames.append(monthly)
    if not frames:
        raise FileNotFoundError(
            "no per-symbol history in {}; run `python -m research.experiments.fetch_valuation_history` first".format(directory)
        )
    panel = pd.concat(frames, ignore_index=True)
    for metric in metrics:
        if metric not in panel.columns:
            panel[metric] = np.nan
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def flag_undefined(panel: pd.DataFrame, metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    """Split each multiple into "defined and positive" vs "undefined".

    Non-positive multiples are moved out of the ranking rather than left in it.
    A loss-making company with PE = -8 would otherwise rank below every real
    company and be labelled *very cheap* -- which is exactly backwards.  The
    raw value is preserved in ``<metric>_raw`` so the undefined share is
    auditable rather than merely asserted.
    """

    out = panel.copy()
    for metric in metrics:
        if metric not in out.columns:
            continue
        values = pd.to_numeric(out[metric], errors="coerce")
        out["{}_raw".format(metric)] = values
        positive = values.where(values > 0)
        out[metric] = positive
        out["{}_undefined".format(metric)] = values.notna() & (values <= 0)
    return out


def compute_own_state(
    panel: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
    windows: Sequence[Optional[int]] = WINDOWS_MONTHS_STOCK,
) -> pd.DataFrame:
    """L3: trailing percentile of each metric within the symbol's own history.

    Returns a long frame, one row per (date, symbol, metric, window), so a
    reader can see every window rather than a single chosen one -- the index
    layer showed the label genuinely changes between windows.
    """

    rows = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("date")
        for metric in metrics:
            if metric not in group.columns:
                continue
            series = pd.to_numeric(group[metric], errors="coerce")
            if series.notna().sum() < 2:
                continue
            for window in windows:
                percentiles = trailing_percentile(series, window)
                label = window_label(window)
                n_obs = (
                    series.notna().rolling(int(window) if window else len(group), min_periods=1).sum()
                )
                rows.append(pd.DataFrame({
                    "date": group["date"].to_numpy(),
                    "symbol": symbol,
                    "metric": metric,
                    "window": label,
                    "level": series.to_numpy(),
                    "percentile_own": percentiles.to_numpy(),
                    "n_obs": n_obs.to_numpy(),
                    "undefined": group.get(
                        "{}_undefined".format(metric), pd.Series(False, index=group.index)
                    ).to_numpy(),
                    # The real observation date behind the month-end bucket.  An
                    # in-progress month is labelled by its *future* end, and the
                    # latest row is the one a reader acts on -- it must not claim
                    # a day that has not happened.
                    "observed_date": group["observed_date"].to_numpy()
                    if "observed_date" in group.columns else pd.NaT,
                }))
    if not rows:
        raise ValueError("no per-symbol percentile could be computed")
    return pd.concat(rows, ignore_index=True)


def _score_by_metric(frame: pd.DataFrame, column: str, metrics: Sequence[str]) -> pd.Series:
    """Apply :func:`cheapness_score` per metric across a long frame.

    Vectorised per metric rather than row-wise: the long frame for a full pool
    is several hundred thousand rows, and a ``DataFrame.apply`` over that costs
    minutes for what is a branch on the metric name.
    """

    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    for metric in metrics:
        mask = frame["metric"].eq(metric)
        if not mask.any():
            continue
        scores.loc[mask] = cheapness_score(metric, frame.loc[mask, column]).to_numpy()
    return scores


def composite_own_state(
    own: pd.DataFrame, min_metrics: int = 2
) -> pd.DataFrame:
    """Per-stock multi-metric state: median cheapness across the metrics that voted.

    Same construction as the index layer, for the same reason -- a single
    multiple is not a valuation.  A stock can be expensive on PE (depressed
    earnings) and cheap on PB simultaneously; reporting only one of them is a
    choice, and reporting their blend without the count is worse.  ``n_metrics``
    travels with the score, and ``min_metrics`` vetoes a stock that only had one
    measurable multiple to begin with.
    """

    pivot = own.pivot_table(
        index=["date", "symbol", "window"], columns="metric", values="cheapness", dropna=False
    )
    if pivot.empty:
        raise ValueError("no composite state could be computed")
    # ``composite_state`` is the index layer's own blending rule -- reused rather
    # than reimplemented, so the two layers cannot drift apart on what a median
    # across metrics means.
    state = composite_state({metric: pivot[metric] for metric in pivot.columns}, min_metrics=min_metrics)
    out = state.reset_index()
    out["band_composite"] = band_from_score(out["state_score"])
    if "observed_date" in own.columns:
        # observed_date is a property of (symbol, date), not of the metric, so it
        # survives the pivot as a lookup rather than as another voter.
        stamps = own.drop_duplicates(["date", "symbol", "window"])[
            ["date", "symbol", "window", "observed_date"]
        ]
        out = out.merge(stamps, on=["date", "symbol", "window"], how="left")
    return out


def compute_cross_section(
    panel: pd.DataFrame, metrics: Sequence[str] = METRICS, min_peers: int = MIN_PEERS
) -> pd.DataFrame:
    """L2: within-pool cross-sectional percentile for each date and metric.

    ``n_peers`` travels with every rank.  Early years have few symbols with a
    long enough history, and a "pool percentile" over a handful of names is not
    a measurement -- the guard makes that an explicit NaN instead of a number
    that happens to lie between 0 and 1.
    """

    frames = []
    for metric in metrics:
        if metric not in panel.columns:
            continue
        values = pd.to_numeric(panel[metric], errors="coerce")
        if values.notna().sum() < min_peers:
            continue
        grouped = values.groupby(panel["date"], dropna=True)
        count = grouped.transform("count")
        ranked = grouped.rank(pct=True, method="max")
        frames.append(pd.DataFrame({
            "date": panel["date"].to_numpy(),
            "symbol": panel["symbol"].to_numpy(),
            "metric": metric,
            "level": values.to_numpy(),
            "percentile_pool": ranked.where(count >= int(min_peers)).to_numpy(),
            "n_peers": count.to_numpy(),
            # The real observation date behind the month-end bucket, carried so
            # the latest snapshot can report when its data actually exists.
            "observed_date": panel["observed_date"].to_numpy()
            if "observed_date" in panel.columns else pd.NaT,
        }))
    if not frames:
        raise ValueError("no cross-sectional percentile could be computed")
    cross = pd.concat(frames, ignore_index=True)
    cross["cheapness"] = _score_by_metric(cross, "percentile_pool", metrics)
    cross["band_pool"] = band_from_score(cross["cheapness"])
    return cross


def pair_own_and_pool(
    own: pd.DataFrame, cross: pd.DataFrame, window: int = PAIRING_WINDOW
) -> pd.DataFrame:
    """Join L3 and L2 on (date, symbol, metric) and label the 3x3 cells.

    The join is inner: a symbol with no cross-sectional rank (too few peers that
    month) contributes no cell rather than a cell built from half a measurement.
    """

    label = window_label(window)
    left = own.loc[own["window"].eq(label), ["date", "symbol", "metric", "percentile_own", "n_obs", "band_own", "undefined"]]
    right = cross.loc[:, ["date", "symbol", "metric", "percentile_pool", "n_peers", "band_pool"]]
    joined = left.merge(right, on=["date", "symbol", "metric"], how="inner")
    # The merge only guarantees the join *keys* are present, not that both
    # measurements exist: an un-filled own window arrives here as NaN with a
    # perfectly good pool rank beside it.  Drop those rows rather than ship a
    # frame where most ``cell`` values are NaN.
    joined = joined.loc[joined["percentile_own"].notna() & joined["percentile_pool"].notna()].copy()
    joined["window"] = label
    joined["rel_own"] = rel_own_vs_peers(joined["percentile_own"], joined["percentile_pool"])
    joined["cell"] = own_vs_peer_cell(joined["percentile_own"], joined["percentile_pool"])
    return joined


def undefined_share(state: pd.DataFrame) -> pd.DataFrame:
    """Share of symbols whose multiple is non-positive, per date and metric.

    Reported rather than dropped: "20% of the pool has no meaningful PE" is a
    fact about the market, and it is the number that tells a reader how much of
    a PE-based cross-section is actually populated.
    """

    if state.empty:
        return pd.DataFrame(columns=[
            "date", "observed_date", "metric", "n_symbols", "n_undefined", "share_undefined",
        ])
    subset = state.loc[state["window"].eq(state["window"].iloc[0])] if "window" in state.columns else state
    grouped = subset.groupby(["date", "metric"])["undefined"]
    summary = pd.DataFrame({
        "n_symbols": grouped.size(),
        "n_undefined": grouped.sum(),
    }).reset_index()
    summary["share_undefined"] = summary["n_undefined"] / summary["n_symbols"].where(summary["n_symbols"] > 0)
    if "observed_date" in subset.columns:
        stamps = (
            subset.groupby(["date", "metric"])["observed_date"].max()
            .rename("observed_date").reset_index()
        )
        summary = summary.merge(stamps, on=["date", "metric"], how="left")
    return summary


def latest_cross_section(
    cross: pd.DataFrame, membership: Mapping[str, set]
) -> pd.DataFrame:
    """Most recent pool percentile per symbol, tagged with current pool membership.

    Membership is applied as a *label on today's snapshot*, never as a
    historical group definition -- see :func:`load_pool_membership`.
    """

    latest_date = cross["date"].max()
    snapshot = cross.loc[cross["date"].eq(latest_date)].copy()
    snapshot["pool"] = "other"
    for name, members in membership.items():
        snapshot.loc[snapshot["symbol"].isin(members), "pool"] = name
    snapshot["period_end"] = latest_date
    # ``as_of`` is the newest real observation among the symbols in the
    # snapshot, never the month-end bucket label -- see ``to_month_end``.
    if "observed_date" in cross.columns:
        snapshot["as_of"] = cross.loc[cross["date"].eq(latest_date), "observed_date"].max()
    return snapshot.sort_values(["metric", "cheapness"]).reset_index(drop=True)


def band_distribution(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Count of symbols per valuation band, per metric and pool."""

    if snapshot.empty:
        return pd.DataFrame(columns=["metric", "pool", "band", "n_symbols"])
    grouped = snapshot.groupby(["metric", "pool", "band_pool"], dropna=False)
    return grouped.size().rename("n_symbols").reset_index().rename(columns={"band_pool": "band"})


def evaluate(
    input_dir: Path = Path(DEFAULT_INPUT_DIR),
    pools: Mapping[str, str] = DEFAULT_POOLS,
    metrics: Sequence[str] = METRICS,
    windows: Sequence[Optional[int]] = WINDOWS_MONTHS_STOCK,
    cuts: Sequence[float] = BAND_CUTS,
    min_peers: int = MIN_PEERS,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Load, measure, pair.  Pure with respect to the filesystem."""

    membership = load_pool_membership(pools)
    panel = build_panel(Path(input_dir), symbols=symbols, metrics=metrics)
    panel = flag_undefined(panel, metrics)

    own = compute_own_state(panel, metrics=metrics, windows=windows)
    own["cheapness"] = _score_by_metric(own, "percentile_own", metrics)
    own["band_own"] = band_from_score(own["cheapness"])

    cross = compute_cross_section(panel, metrics=metrics, min_peers=min_peers)
    paired = pair_own_and_pool(own, cross, window=PAIRING_WINDOW)
    snapshot = latest_cross_section(cross, membership)
    composite = composite_own_state(own)

    assert_no_forward_columns(panel.columns)
    assert_no_forward_columns(own.columns)
    assert_no_forward_columns(cross.columns)
    assert_no_forward_columns(paired.columns)
    assert_no_forward_columns(composite.columns)

    return {
        "panel": panel,
        "own": own,
        "cross": cross,
        "paired": paired,
        "snapshot": snapshot,
        "composite": composite,
        "bands": band_distribution(snapshot),
        "undefined": undefined_share(own),
        "metadata": {
            "input_dir": str(input_dir),
            "metrics": list(metrics),
            "windows_months": [window_label(window) for window in windows],
            "band_cuts": list(cuts),
            "min_peers": int(min_peers),
            "pairing_window": window_label(PAIRING_WINDOW),
            "pool_files": {name: str(path) for name, path in pools.items()},
            "n_symbols": int(panel["symbol"].nunique()),
            "panel_start": str(panel["date"].min().date()),
            "panel_end": str(panel["date"].max().date()),
            "last_period_end": str(panel["date"].max().date()),
            "last_observation": str(panel["observed_date"].max().date())
            if "observed_date" in panel.columns else None,
            "panel_rows": int(len(panel)),
            "membership": {name: len(members) for name, members in membership.items()},
            "pool_membership_is_snapshot": True,
        },
    }


def export(result: Mapping[str, object], output_dir: Path, run_id: str) -> Dict[str, str]:
    """Write the own-history state, cross-section, pairing, snapshot and manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for key, suffix in (
        ("own", "own_state"),
        ("cross", "cross_section"),
        ("paired", "paired"),
        ("snapshot", "latest_snapshot"),
        ("composite", "composite_state"),
        ("bands", "band_distribution"),
        ("undefined", "undefined_share"),
    ):
        frame = result[key]
        rel = "{}_{}_{}.csv".format(PREFIX, suffix, run_id)
        frame.to_csv(output_dir / rel, index=False, encoding="utf-8-sig")
        paths[key] = str(output_dir / rel)

    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "question": "个股当前估值处于自身历史与池内什么位置（描述型，非收益预测）",
        "descriptive_only": True,
        "predictive_use_closed_by": "docs/research/value_factor_regime_study_summary.md",
        "direction_convention": "price_multiple -> cheapness = 1 - percentile",
        "undefined_handling": "non-positive multiples are excluded from the rank and counted separately",
        "paths": paths,
        **dict(result["metadata"]),
    }
    manifest_path = output_dir / "{}_manifest_{}.json".format(PREFIX, run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--run-id", default="stock_valuation_v1")
    parser.add_argument("--limit-symbols", type=int, default=0, help="use only the first N symbols (smoke)")
    parser.add_argument("--windows", default=",".join(window_label(w) for w in WINDOWS_MONTHS_STOCK))
    parser.add_argument("--min-peers", type=int, default=MIN_PEERS)
    return parser.parse_args(argv)


def _parse_windows(raw: str) -> List[Optional[int]]:
    windows: List[Optional[int]] = []
    for token in str(raw).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            windows.append(None)
        elif token.endswith("m") and token[:-1].isdigit():
            windows.append(int(token[:-1]))
        else:
            raise SystemExit("unrecognised window {!r}; expected e.g. 36m or all".format(token))
    return windows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pd.set_option("display.width", 200)
    windows = _parse_windows(args.windows)

    result = evaluate(
        input_dir=Path(args.input_dir),
        windows=windows,
        min_peers=args.min_peers,
        symbols=None,
    )
    if args.limit_symbols:
        # Re-run on the truncated universe rather than slicing the result: a
        # cross-sectional percentile computed over the full pool and then sliced
        # would report ranks that the truncated run would not reproduce.
        kept = sorted(result["panel"]["symbol"].unique())[: int(args.limit_symbols)]
        result = evaluate(input_dir=Path(args.input_dir), windows=windows, min_peers=args.min_peers, symbols=kept)

    paths = export(result, Path(args.output_dir), args.run_id)

    print("\n=== 个股估值状态（池内分布，最新快照）===")
    bands = result["bands"]
    if not bands.empty:
        pivot = bands.pivot_table(index=["metric", "band"], columns="pool", values="n_symbols", aggfunc="sum")
        print(pivot.fillna(0).astype(int).to_string())
    print("\n=== 非正倍数占比（最近一期）===")
    undefined = result["undefined"]
    if not undefined.empty:
        latest = undefined.loc[undefined["date"].eq(undefined["date"].max())]
        print(latest.round(4).to_string(index=False))
    print("\n=== own vs pool 九宫格（window={}）===".format(window_label(PAIRING_WINDOW)))
    paired = result["paired"]
    if not paired.empty:
        print(paired["cell"].value_counts(dropna=False).to_string())
    print("\nstock valuation diagnostic done; manifest={}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
