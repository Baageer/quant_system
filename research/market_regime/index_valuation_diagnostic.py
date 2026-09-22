"""Describe the current index / whole-market valuation state.

Answers "is the market expensive?" -- descriptively.  It reports where each
valuation anchor sits in its own trailing history and labels the state, and it
makes **no claim about future returns**.  That boundary is deliberate: the
predictive use of valuation was closed by three independent studies (see
``docs/research/value_factor_regime_study_summary.md``), the third of which
showed the value spread itself carries a full-period Sharpe of ~0.03, so even a
perfect timer would earn ~nothing.  ``valuation_state_core`` enforces the
boundary by refusing any forward-looking column.

Why several anchors instead of one PE percentile: they disagree, and the
disagreement is the finding.  At 2026-09-14 the HS300 reads PE 64.7th
percentile but PB 40.3rd -- PE high with PB low means ROE is at a cyclical low,
which no single-multiple dashboard can show.  A composite is therefore reported
as a *median across anchors plus the number of anchors that voted*, never as a
bare blended number.

Anchor comparability: the anchors do not all describe the same population.
``hs300_*`` describe the HS300 index; ``market_*`` describe all A-shares;
``board_*`` describe one exchange board.  Mixing them into one headline is a
population error, so every row carries its ``scope`` and the composite is
computed within scope.

Example::

    python -m research.market_regime.index_valuation_diagnostic --run-id index_valuation_v1
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
    KIND_PRICE_MULTIPLE,
    KIND_RATE,
    KIND_YIELD,
    WINDOWS_MONTHS,
    assert_no_forward_columns,
    band_from_score,
    cheapness_score,
    composite_state,
    trailing_percentile,
    window_label,
)


PREFIX = "index_valuation"
DEFAULT_ANCHOR_DIR = "data/raw/akshare/index_valuation"

# anchor -> (Chinese label, scope, kind).  ``kind`` drives the direction
# convention and nothing else; ``scope`` is the population the anchor describes.
ANCHOR_SPECS: Mapping[str, Mapping[str, str]] = {
    "hs300_pe_ttm": {"label": "沪深300 滚动PE", "scope": "hs300", "kind": KIND_PRICE_MULTIPLE},
    "hs300_pe_ttm_median": {"label": "沪深300 滚动PE中位数", "scope": "hs300", "kind": KIND_PRICE_MULTIPLE},
    "hs300_pe_lyr": {"label": "沪深300 静态PE", "scope": "hs300", "kind": KIND_PRICE_MULTIPLE},
    "hs300_pb": {"label": "沪深300 PB", "scope": "hs300", "kind": KIND_PRICE_MULTIPLE},
    "hs300_pb_median": {"label": "沪深300 PB中位数", "scope": "hs300", "kind": KIND_PRICE_MULTIPLE},
    "sse50_pe_ttm": {"label": "上证50 滚动PE", "scope": "sse50", "kind": KIND_PRICE_MULTIPLE},
    "sse50_pb": {"label": "上证50 PB", "scope": "sse50", "kind": KIND_PRICE_MULTIPLE},
    "csi500_pe_ttm": {"label": "中证500 滚动PE", "scope": "csi500", "kind": KIND_PRICE_MULTIPLE},
    "csi500_pb": {"label": "中证500 PB", "scope": "csi500", "kind": KIND_PRICE_MULTIPLE},
    "csi1000_pe_ttm": {"label": "中证1000 滚动PE", "scope": "csi1000", "kind": KIND_PRICE_MULTIPLE},
    "csi1000_pb": {"label": "中证1000 PB", "scope": "csi1000", "kind": KIND_PRICE_MULTIPLE},
    "market_pe_ttm_median": {"label": "全A PE(TTM)中位数", "scope": "market", "kind": KIND_PRICE_MULTIPLE},
    "market_pe_ttm_avg": {"label": "全A PE(TTM)平均", "scope": "market", "kind": KIND_PRICE_MULTIPLE},
    "market_pe_lyr_median": {"label": "全A PE(LYR)中位数", "scope": "market", "kind": KIND_PRICE_MULTIPLE},
    "market_dividend_yield": {"label": "全A 股息率", "scope": "market", "kind": KIND_YIELD},
    "market_mv_gdp": {"label": "全A 市值/GDP", "scope": "market", "kind": KIND_PRICE_MULTIPLE},
    "board_sh_pe": {"label": "上证 平均PE", "scope": "board_sh", "kind": KIND_PRICE_MULTIPLE},
    "board_sz_pe": {"label": "深证 平均PE", "scope": "board_sz", "kind": KIND_PRICE_MULTIPLE},
    "board_cyb_pe": {"label": "创业板 平均PE", "scope": "board_cyb", "kind": KIND_PRICE_MULTIPLE},
    "board_kcb_pe": {"label": "科创版 平均PE", "scope": "board_kcb", "kind": KIND_PRICE_MULTIPLE},
    "board_sh_pb": {"label": "上证 PB", "scope": "board_sh", "kind": KIND_PRICE_MULTIPLE},
    "board_sz_pb": {"label": "深证 PB", "scope": "board_sz", "kind": KIND_PRICE_MULTIPLE},
    "board_cyb_pb": {"label": "创业板 PB", "scope": "board_cyb", "kind": KIND_PRICE_MULTIPLE},
    "board_kcb_pb": {"label": "科创版 PB", "scope": "board_kcb", "kind": KIND_PRICE_MULTIPLE},
    # Derived below in ``derive_erp``.  A yield, so a *high* ERP means cheap --
    # the opposite direction to every price multiple above.
    "erp_hs300": {"label": "沪深300 ERP(E/P-10Y)", "scope": "hs300", "kind": KIND_YIELD},
    # The risk-free leg: carried for the ERP, never banded on its own.
    "cn_yield_10y": {"label": "中国10年期国债收益率", "scope": "rate", "kind": KIND_RATE},
    "cn_yield_2y": {"label": "中国2年期国债收益率", "scope": "rate", "kind": KIND_RATE},
}
DERIVED_ANCHORS = ("erp_hs300",)
# Anchors whose own scope is the HS300 itself -- the only ones the ERP may use,
# since dividing an all-A median PE into a treasury yield mixes populations.
ERP_PE_ANCHOR = "hs300_pe_ttm"
ERP_SCOPE = "hs300"


def load_anchors(anchor_dir: Path) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Pivot every cached long-format anchor file into one wide monthly panel.

    Resampled to month-end before anything else: the vendors disagree on
    frequency (the index anchors are monthly, the bond yield is daily), and a
    percentile computed across mixed frequencies is comparing windows of
    different lengths without saying so.

    Returns ``(monthly, observed)`` -- two frames with identical shape.  The
    first carries values on the month-end grid used for every percentile; the
    second carries the **real observation date** behind each cell.  They are not
    the same thing: ``resample("ME")`` labels a bucket by its end, so the
    in-progress month is labelled ``2026-09-30`` while the newest row the vendor
    actually served is ``2026-09-14``.  Reporting the label as the as-of date
    would have this module claim a day of data that has not happened yet --
    and the current state is the one row the reader actually acts on.
    """

    frames = []
    for path in sorted(Path(anchor_dir).glob("*_anchors.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if not {"date", "anchor", "value"} <= set(frame.columns):
            print("[skip] {}: unexpected schema".format(path.name), flush=True)
            continue
        frames.append(frame.loc[:, ["date", "anchor", "value"]])
    if not frames:
        raise FileNotFoundError(
            "no anchor files in {}; run `python -m research.experiments.fetch_akshare_valuation` first".format(anchor_dir)
        )
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["date", "value"])

    wide = long.pivot_table(index="date", columns="anchor", values="value", aggfunc="last")
    wide = wide.sort_index()

    dates = pd.Series(wide.index.to_numpy(), index=wide.index)
    observed = pd.DataFrame(
        {column: dates.where(wide[column].notna()) for column in wide.columns},
        index=wide.index,
    ).resample("ME").last()

    monthly = wide.resample("ME").last()
    keep = monthly.notna().any(axis=1)
    return monthly.loc[keep], observed.loc[keep]


def derive_erp(
    panel: pd.DataFrame, observed: Optional[pd.DataFrame] = None
) -> "tuple[pd.DataFrame, Optional[pd.DataFrame]]":
    """Equity risk premium = earnings yield minus the 10y treasury yield.

    ``E/P`` rather than ``P/E`` because the subtraction only makes sense on the
    yield scale.  Both legs are already on the same month-end grid, so the
    subtraction never aligns two different months.

    When ``observed`` is supplied, the derived anchor inherits the *older* of its
    two legs' observation dates -- an ERP is only as fresh as its staler input,
    and both legs are monthly vendor series that do not publish on the same day.
    """

    out = panel.copy()
    stamps = None if observed is None else observed.copy()
    if ERP_PE_ANCHOR not in out.columns or "cn_yield_10y" not in out.columns:
        print("[warn] missing {} or cn_yield_10y; skipping ERP".format(ERP_PE_ANCHOR), flush=True)
        return out, stamps
    pe = pd.to_numeric(out[ERP_PE_ANCHOR], errors="coerce")
    y10 = pd.to_numeric(out["cn_yield_10y"], errors="coerce")
    erp = (100.0 / pe.where(pe > 0)) - y10
    out["erp_hs300"] = erp
    if stamps is not None:
        pe_stamp = stamps[ERP_PE_ANCHOR]
        y_stamp = stamps["cn_yield_10y"]
        # Conservative: the ERP is not knowable before both legs are published.
        stamps["erp_hs300"] = pe_stamp.where(pe_stamp <= y_stamp, y_stamp).where(erp.notna())
    return out, stamps


def evaluate_anchors(
    panel: pd.DataFrame,
    windows: Sequence[Optional[int]] = WINDOWS_MONTHS,
    cuts: Sequence[float] = BAND_CUTS,
    observed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Percentile, band and cheapness score for every anchor x window.

    Returns a long frame: one row per (anchor, window), with the latest value
    and its state.  Anchors missing from the panel are skipped rather than
    raising, because the vendor throttles individual endpoints and a partial
    anchor set is still informative.

    ``as_of`` is taken from ``observed`` (the real last observation date), never
    from the month-end bucket label.  ``period_end`` carries the bucket label
    separately for anything that needs to join on the monthly grid.
    """

    stamps = None if observed is None else observed

    def _as_of(anchor: str, period_end) -> Optional[str]:
        if stamps is None or anchor not in stamps.columns:
            return None
        stamp = stamps.at[period_end, anchor]
        return None if pd.isna(stamp) else str(pd.Timestamp(stamp).date())

    rows: List[Dict[str, object]] = []
    for anchor, spec in ANCHOR_SPECS.items():
        if anchor not in panel.columns:
            continue
        series = pd.to_numeric(panel[anchor], errors="coerce")
        if series.notna().sum() < 2:
            continue
        for window in windows:
            percentile = trailing_percentile(series, window)
            latest_pct = percentile.dropna()
            if latest_pct.empty:
                rows.append({
                    "anchor": anchor, "anchor_label": spec["label"], "scope": spec["scope"],
                    "kind": spec["kind"], "window": window_label(window),
                    "level": np.nan, "as_of": None, "period_end": None, "percentile": np.nan,
                    "n_obs": 0, "cheapness": np.nan, "band": np.nan,
                })
                continue
            period_end = latest_pct.index[-1]
            value = latest_pct.iloc[-1]
            n_obs = int(series.loc[:period_end].tail(int(window) if window else len(series)).notna().sum())
            banded = spec["kind"] != KIND_RATE
            # The band comes from the *score*, not from the raw percentile: for a
            # yield the two are mirrors, and labelling a dividend yield in its
            # lowest decile "very cheap" is the exact inversion this module must
            # not ship.
            cheap = cheapness_score(spec["kind"], pd.Series([value])).iloc[0] if banded else np.nan
            band_label = band_from_score(pd.Series([cheap]), cuts).iloc[0] if banded else np.nan
            rows.append({
                "anchor": anchor, "anchor_label": spec["label"], "scope": spec["scope"],
                "kind": spec["kind"], "window": window_label(window),
                "level": float(series.loc[period_end]), "as_of": _as_of(anchor, period_end),
                "period_end": str(period_end.date()),
                "percentile": float(value), "n_obs": n_obs,
                "cheapness": float(cheap) if pd.notna(cheap) else np.nan,
                "band": band_label,
            })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no anchor produced an evaluable percentile")
    return table.sort_values(["scope", "anchor", "window"]).reset_index(drop=True)


def scope_summary(evaluation: pd.DataFrame, window: str = "60m") -> pd.DataFrame:
    """Median cheapness per scope at one window, with the anchor count.

    Two guards travel with the number by construction: ``n_anchors`` (how many
    voted) and ``spread`` (max - min), because anchors that disagree are the
    interesting case and a median would otherwise hide that.  The endpoints
    themselves are exported alongside the spread so a reader can draw the band
    rather than take the summary's word for it.
    """

    subset = evaluation.loc[evaluation["window"].eq(window) & evaluation["cheapness"].notna()]
    if subset.empty:
        return pd.DataFrame(columns=[
            "scope", "window", "n_anchors", "median_cheapness",
            "min_cheapness", "max_cheapness", "spread", "bands",
        ])
    grouped = subset.groupby("scope")
    summary = pd.DataFrame({
        "n_anchors": grouped["cheapness"].count(),
        "median_cheapness": grouped["cheapness"].median(),
        "min_cheapness": grouped["cheapness"].min(),
        "max_cheapness": grouped["cheapness"].max(),
        "bands": grouped["band"].apply(lambda values: ",".join(sorted(set(values.dropna())))),
    }).reset_index()
    summary["spread"] = summary["max_cheapness"] - summary["min_cheapness"]
    summary["window"] = window
    return summary.loc[:, [
        "scope", "window", "n_anchors", "median_cheapness",
        "min_cheapness", "max_cheapness", "spread", "bands",
    ]]


def evaluate(
    anchor_dir: Path = Path(DEFAULT_ANCHOR_DIR),
    windows: Sequence[Optional[int]] = WINDOWS_MONTHS,
    cuts: Sequence[float] = BAND_CUTS,
    summary_window: str = "60m",
) -> Dict[str, object]:
    """Load, derive, evaluate.  Pure with respect to the filesystem."""

    monthly, observed = load_anchors(anchor_dir)
    panel, observed = derive_erp(monthly, observed)
    evaluation = evaluate_anchors(panel, windows=windows, cuts=cuts, observed=observed)
    summary = scope_summary(evaluation, window=summary_window)
    assert_no_forward_columns(panel.columns)
    assert_no_forward_columns(evaluation.columns)

    latest_as_of = evaluation["as_of"].dropna()
    return {
        "panel": panel,
        "observed": observed,
        "evaluation": evaluation,
        "summary": summary,
        "metadata": {
            "anchor_dir": str(anchor_dir),
            "windows_months": [window_label(window) for window in windows],
            "band_cuts": list(cuts),
            "summary_window": summary_window,
            "anchors_available": sorted(column for column in panel.columns if column in ANCHOR_SPECS),
            "panel_start": str(panel.index.min().date()),
            "panel_end": str(panel.index.max().date()),
            "panel_rows": int(len(panel)),
            # The bucket label and the newest real observation date, side by side
            # -- they differ whenever the newest bucket is the in-progress month.
            "last_period_end": str(panel.index.max().date()),
            "last_observation": str(latest_as_of.max()) if len(latest_as_of) else None,
        },
    }


def export(result: Mapping[str, object], output_dir: Path, run_id: str) -> Dict[str, str]:
    """Write the evaluation, the scope summary and the manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    evaluation = result["evaluation"]
    evaluation.to_csv(output_dir / "{}_anchors_{}.csv".format(PREFIX, run_id),
                      index=False, encoding="utf-8-sig")
    paths["anchors"] = str(output_dir / "{}_anchors_{}.csv".format(PREFIX, run_id))

    summary = result["summary"]
    summary.to_csv(output_dir / "{}_summary_{}.csv".format(PREFIX, run_id),
                   index=False, encoding="utf-8-sig")
    paths["summary"] = str(output_dir / "{}_summary_{}.csv".format(PREFIX, run_id))

    panel = result["panel"]
    history = panel.reset_index().rename(columns={"index": "date"})
    history.to_csv(output_dir / "{}_panel_{}.csv".format(PREFIX, run_id),
                   index=False, encoding="utf-8-sig")
    paths["panel"] = str(output_dir / "{}_panel_{}.csv".format(PREFIX, run_id))

    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "question": "当前指数/全市场估值处于什么状态（描述型，非收益预测）",
        "descriptive_only": True,
        "predictive_use_closed_by": "docs/research/value_factor_regime_study_summary.md",
        "anchor_specs": {key: dict(value) for key, value in ANCHOR_SPECS.items()},
        "direction_convention": "price_multiple -> cheapness = 1 - percentile; yield -> cheapness = percentile",
        "paths": paths,
        **dict(result["metadata"]),
    }
    manifest_path = output_dir / "{}_manifest_{}.json".format(PREFIX, run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--run-id", default="index_valuation_v1")
    parser.add_argument("--windows", default=",".join(window_label(w) for w in WINDOWS_MONTHS),
                        help="comma-separated: month windows like 36m,60m,120m plus 'all'")
    parser.add_argument("--cuts", default=",".join(str(cut) for cut in BAND_CUTS))
    parser.add_argument("--summary-window", default="60m")
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
    cuts = tuple(float(token) for token in str(args.cuts).split(",") if token.strip())

    result = evaluate(Path(args.anchor_dir), windows=windows, cuts=cuts, summary_window=args.summary_window)
    paths = export(result, Path(args.output_dir), args.run_id)

    summary = result["summary"]
    if not summary.empty:
        print("\n=== 估值状态摘要（window={}）===".format(args.summary_window))
        print(summary.round(4).to_string(index=False))
    print("\n=== 各锚点最新状态（window={}）===".format(args.summary_window))
    latest = result["evaluation"].loc[result["evaluation"]["window"].eq(args.summary_window)]
    print(latest.loc[:, ["anchor_label", "scope", "level", "as_of", "percentile", "n_obs", "band"]]
          .round(4).to_string(index=False))
    print("\nindex valuation diagnostic done; manifest={}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
