"""Causal feature panels and offline turning-point event studies.

The feature columns in this module use only OHLCV data available on the
sample date.  ``MarketRegimeLabeler`` is deliberately used only after the
panel is calculated, to define *offline* events and matched controls.
Neither daily regime fields nor segment statistics are model features.

Run a full HS300 study from the repository root with::

    python -m research.market_regime.feature_event_study --output-dir output
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from factors.factor_panel import AVAILABLE_FACTOR_SPECS, calculate_single_stock_factors
from signals.indicators import sma, volatility

from .data_loader import (
    DEFAULT_PRICE_HISTORY_DIR,
    load_hs300_symbols,
    load_local_price_history,
)
from .labeler import MarketRegimeLabeler


DEFAULT_FACTOR_NAMES = tuple(AVAILABLE_FACTOR_SPECS)
DEFAULT_MATCH_FEATURES = ("ret_20d", "volatility_20", "volume_ratio_20")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    def current_percentile(window_values) -> float:
        array = np.asarray(window_values, dtype=float)
        return float(np.sum(array <= array[-1]) / len(array))

    return values.rolling(window=window, min_periods=window).apply(current_percentile)


def _consecutive_improvements(values: pd.Series) -> pd.Series:
    """Count consecutive dates on which a series improved, using past data only."""

    result = pd.Series(0.0, index=values.index)
    changes = values.diff()
    for position in range(1, len(result)):
        if np.isfinite(changes.iloc[position]) and changes.iloc[position] > 0:
            result.iloc[position] = result.iloc[position - 1] + 1
    return result


def _extra_causal_features(data: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    result = pd.DataFrame(index=data.index)

    sma_5 = sma(close, 5)
    sma_20 = sma(close, 20)
    sma_60 = sma(close, 60)
    result["ret_10d"] = close.pct_change(10)
    result["ret_60d"] = close.pct_change(60)
    result["sma_gap_5_20"] = _safe_divide(sma_5, sma_20) - 1
    result["sma_gap_20_60"] = _safe_divide(sma_20, sma_60) - 1
    for window, average in ((5, sma_5), (20, sma_20), (60, sma_60)):
        for lag in (3, 5, 10):
            result["sma_{}_slope_{}d".format(window, lag)] = average.pct_change(lag) / lag

    result["high_distance_60"] = _safe_divide(close, high.rolling(60, min_periods=60).max()) - 1
    result["low_distance_60"] = _safe_divide(close, low.rolling(60, min_periods=60).min()) - 1
    result["di_spread_14"] = base["plus_di_14"] - base["minus_di_14"]
    result["di_cross_up"] = (
        (result["di_spread_14"] > 0) & (result["di_spread_14"].shift(1) <= 0)
    ).astype(float)
    result["di_cross_down"] = (
        (result["di_spread_14"] < 0) & (result["di_spread_14"].shift(1) >= 0)
    ).astype(float)

    for column in ("rsi_14", "macd_hist", "rsrs_score", "bollinger_bandwidth_20"):
        for lag in (3, 5, 10):
            result["{}_change_{}d".format(column, lag)] = base[column].diff(lag)
    result["macd_hist_improving_days"] = _consecutive_improvements(base["macd_hist"])
    result["volatility_10"] = volatility(close, 10)
    result["volatility_60"] = volatility(close, 60)
    for column in ("volatility_20", "bollinger_bandwidth_20", "atr_pct_14"):
        result["{}_percentile_120".format(column)] = _rolling_percentile(base[column], 120)

    result["volume_ratio_5"] = _safe_divide(volume, sma(volume, 5)) - 1
    result["volume_ratio_60"] = _safe_divide(volume, sma(volume, 60)) - 1
    result["price_up_volume_down"] = ((close.pct_change() > 0) & (volume.pct_change() < 0)).astype(float)
    result["price_down_volume_up"] = ((close.pct_change() < 0) & (volume.pct_change() > 0)).astype(float)
    return result


def build_causal_feature_panel(
    stock_data: Mapping[str, pd.DataFrame],
    factor_names: Optional[Iterable[str]] = None,
    feature_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build a ``date, symbol`` panel without future-derived feature fields.

    All calculations are per-symbol and use only the current and preceding
    rows.  The input must contain chronological ``high``, ``low``, ``close``
    and ``volume`` columns.  No regime label output is accepted here.  When
    ``feature_limit`` is set, the first N features in the stable generation
    order are retained.
    """

    if feature_limit is not None and feature_limit <= 0:
        raise ValueError("feature_limit must be a positive integer or None")
    names = tuple(DEFAULT_FACTOR_NAMES if factor_names is None else factor_names)
    frames = []
    required = {"high", "low", "close", "volume"}
    for symbol, raw_data in stock_data.items():
        missing = required.difference(raw_data.columns)
        if missing:
            raise ValueError("{} is missing OHLCV columns: {}".format(symbol, sorted(missing)))
        data = raw_data.sort_index().copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("{} must use a DatetimeIndex".format(symbol))
        if data.index.has_duplicates:
            raise ValueError("{} has duplicate dates".format(symbol))
        base = calculate_single_stock_factors(data, factor_names=names, include_ohlcv=False)
        features = pd.concat([base, _extra_causal_features(data, base)], axis=1)
        if feature_limit is not None:
            features = features.iloc[:, :feature_limit]
        features["symbol"] = str(symbol)
        features["date"] = features.index
        frames.append(features.set_index(["date", "symbol"]))
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
    return pd.concat(frames).sort_index()


def audit_feature_causality(
    data: pd.DataFrame,
    factor_names: Optional[Iterable[str]] = None,
    cutoffs: Optional[Sequence[int]] = None,
    feature_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Verify prefix invariance: appending future rows cannot alter features."""

    if len(data) < 4:
        return pd.DataFrame(columns=["cutoff", "feature", "max_abs_difference"])
    full = build_causal_feature_panel(
        {"audit": data}, factor_names, feature_limit
    ).xs("audit", level="symbol")
    points = cutoffs or sorted({max(2, len(data) // 3), max(2, 2 * len(data) // 3)})
    rows = []
    for cutoff in points:
        if cutoff >= len(data):
            continue
        prefix = build_causal_feature_panel(
            {"audit": data.iloc[:cutoff]}, factor_names, feature_limit
        ).xs("audit", level="symbol")
        comparison = full.loc[prefix.index, prefix.columns] - prefix
        for column in prefix.columns:
            values = np.asarray(comparison[column], dtype=float)
            if np.isfinite(values).any():
                max_difference = float(np.nanmax(np.abs(values)))
                if max_difference > 1e-12:
                    rows.append({"cutoff": cutoff, "feature": column, "max_abs_difference": max_difference})
    return pd.DataFrame(rows, columns=["cutoff", "feature", "max_abs_difference"])


def build_turning_point_events(
    stock_data: Mapping[str, pd.DataFrame], labeler: Optional[MarketRegimeLabeler] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create closed offline state-transition events, separate from features."""

    regime_labeler = labeler or MarketRegimeLabeler()
    event_frames = []
    skipped = []
    for symbol, data in stock_data.items():
        try:
            segments, _, pivots = regime_labeler.label(data.sort_index(), str(symbol))
        except ValueError as exc:
            skipped.append({"symbol": str(symbol), "reason": str(exc)})
            continue
        if pivots.empty:
            continue
        segment_status = segments.loc[:, ["segment_id", "end_date", "open_segment"]].copy()
        segment_status["next_segment_id"] = segment_status["segment_id"] + 1
        next_status = segment_status.loc[:, ["next_segment_id", "open_segment"]].rename(
            columns={"open_segment": "next_open_segment"}
        )
        event_rows = pivots.merge(
            segment_status.loc[:, ["segment_id", "next_segment_id", "end_date"]],
            left_on="pivot_date",
            right_on="end_date",
            how="left",
        ).merge(next_status, on="next_segment_id", how="left")
        event_rows = event_rows[
            event_rows["confirm_date"].notna() & ~event_rows["next_open_segment"].fillna(True)
        ].copy()
        event_rows["event_date"] = pd.to_datetime(event_rows["pivot_date"])
        event_rows["transition"] = event_rows["previous_regime"] + "_to_" + event_rows["next_regime"]
        event_rows["event_id"] = event_rows.apply(
            lambda row: "{}:{}:{}".format(row["symbol"], row["event_date"].date().isoformat(), row["transition"]), axis=1
        )
        event_frames.append(event_rows.drop(columns=["end_date"], errors="ignore"))
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return events, pd.DataFrame(skipped, columns=["symbol", "reason"])


def _event_window_observations(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    feature_columns: Sequence[str],
    pre_window: int,
    post_window: int,
    sample_group: str,
) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        try:
            symbol_panel = panel.xs(str(event.symbol), level="symbol")
        except KeyError:
            continue
        event_date = pd.Timestamp(getattr(event, "control_date", event.event_date))
        location = symbol_panel.index.get_indexer([event_date])[0]
        if location < 0:
            continue
        for relative_day in range(-pre_window, post_window + 1):
            position = location + relative_day
            if position < 0 or position >= len(symbol_panel):
                continue
            values = symbol_panel.iloc[position][list(feature_columns)]
            for feature, value in values.items():
                rows.append(
                    {
                        "event_id": event.event_id,
                        "symbol": str(event.symbol),
                        "transition": event.transition,
                        "sample_group": sample_group,
                        "event_date": pd.Timestamp(event.event_date),
                        "observation_date": symbol_panel.index[position],
                        "relative_day": relative_day,
                        "feature": feature,
                        "value": value,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id", "symbol", "transition", "sample_group", "event_date", "observation_date",
            "relative_day", "feature", "value",
        ],
    )


def match_non_event_controls(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    match_features: Sequence[str] = DEFAULT_MATCH_FEATURES,
    exclusion_days: int = 20,
) -> pd.DataFrame:
    """Match one same-symbol, same-year non-event day to each event.

    Matching uses only feature values observed on the candidate day.  Event
    labels are used solely to exclude turning-point neighbourhoods.
    """

    available = [column for column in match_features if column in panel.columns]
    if not available or events.empty:
        return pd.DataFrame(columns=["event_id", "symbol", "transition", "event_date", "control_date", "match_distance"])
    rows = []
    for event in events.itertuples(index=False):
        try:
            symbol_panel = panel.xs(str(event.symbol), level="symbol")
        except KeyError:
            continue
        event_date = pd.Timestamp(event.event_date)
        if event_date not in symbol_panel.index:
            continue
        candidates = symbol_panel[symbol_panel.index.year == event_date.year].copy()
        for pivot_date in events.loc[events["symbol"].astype(str) == str(event.symbol), "event_date"]:
            candidates = candidates[np.abs((candidates.index - pd.Timestamp(pivot_date)).days) > exclusion_days]
        target = symbol_panel.loc[event_date, available]
        candidates = candidates.dropna(subset=available)
        if candidates.empty or target.isna().any():
            continue
        scale = symbol_panel.loc[:event_date, available].std().replace(0, np.nan)
        distances = ((candidates[available] - target) / scale).pow(2).sum(axis=1, min_count=1)
        distances = distances.replace([np.inf, -np.inf], np.nan).dropna()
        if distances.empty:
            continue
        control_date = distances.idxmin()
        rows.append(
            {
                "event_id": event.event_id,
                "symbol": str(event.symbol),
                "transition": event.transition,
                "event_date": event_date,
                "control_date": control_date,
                "match_distance": float(distances.loc[control_date]),
            }
        )
    return pd.DataFrame(rows, columns=["event_id", "symbol", "transition", "event_date", "control_date", "match_distance"])


def summarize_turning_point_features(observations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return event/control paths and standardized event-minus-control effects."""

    summary_columns = ["transition", "sample_group", "feature", "relative_day", "count", "mean", "median", "q25", "q75", "std"]
    if observations.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame()
    grouped = observations.groupby(["transition", "sample_group", "feature", "relative_day"])["value"]
    summary = pd.DataFrame(
        {
            "count": grouped.count(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
            "std": grouped.std(),
        }
    ).reset_index()
    event_summary = summary[summary["sample_group"] == "event"].set_index(["transition", "feature", "relative_day"])
    control_summary = summary[summary["sample_group"] == "control"].set_index(["transition", "feature", "relative_day"])
    effect = event_summary.join(control_summary, lsuffix="_event", rsuffix="_control", how="inner").reset_index()
    pooled_std = np.sqrt((effect["std_event"].pow(2) + effect["std_control"].pow(2)) / 2).replace(0, np.nan)
    effect["mean_difference"] = effect["mean_event"] - effect["mean_control"]
    effect["median_difference"] = effect["median_event"] - effect["median_control"]
    effect["standardized_effect"] = effect["mean_difference"] / pooled_std
    return summary, effect


def build_turning_point_feature_study(
    stock_data: Mapping[str, pd.DataFrame],
    labeler: Optional[MarketRegimeLabeler] = None,
    factor_names: Optional[Iterable[str]] = None,
    pre_window: int = 60,
    post_window: int = 20,
    feature_limit: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Build the causal panel, offline events, matched controls and summaries."""

    panel = build_causal_feature_panel(stock_data, factor_names, feature_limit)
    events, skipped = build_turning_point_events(stock_data, labeler)
    feature_columns = list(panel.columns)
    event_observations = _event_window_observations(panel, events, feature_columns, pre_window, post_window, "event")
    controls = match_non_event_controls(panel, events)
    control_events = events.merge(controls.loc[:, ["event_id", "control_date", "match_distance"]], on="event_id", how="inner") if not controls.empty else events.iloc[0:0].copy()
    control_observations = _event_window_observations(panel, control_events, feature_columns, pre_window, post_window, "control")
    observations = pd.concat([event_observations, control_observations], ignore_index=True)
    summary, effects = summarize_turning_point_features(observations)
    return {
        "feature_panel": panel.reset_index(),
        "events": events,
        "matched_controls": controls,
        "event_observations": observations,
        "feature_path_summary": summary,
        "feature_effects": effects,
        "skipped_symbols": skipped,
    }


def export_turning_point_feature_study(report: Mapping[str, pd.DataFrame], output_dir: Union[str, Path], run_id: Optional[str] = None) -> Dict[str, Path]:
    """Export reproducible CSV tables and a compact provenance manifest."""

    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "feature_panel": "trend_feature_panel",
        "events": "trend_turning_point_events",
        "matched_controls": "trend_matched_controls",
        "event_observations": "trend_feature_event_observations",
        "feature_path_summary": "trend_feature_event_summary",
        "feature_effects": "trend_feature_event_effects",
        "skipped_symbols": "trend_feature_study_skipped_symbols",
    }
    paths = {}
    for key, prefix in names.items():
        path = destination / "{}_{}.csv".format(prefix, identifier)
        report.get(key, pd.DataFrame()).to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    manifest = {
        "analysis_id": identifier,
        "created_at": datetime.now().isoformat(),
        "feature_rule": "OHLCV values at date t and earlier only; no regime fields in feature panel",
        "label_rule": "MarketRegimeLabeler output is used only for offline event/control definition",
        "feature_count": int(len(report.get("feature_panel", pd.DataFrame()).columns) - 2),
        "event_count": int(len(report.get("events", pd.DataFrame()))),
        "matched_control_count": int(len(report.get("matched_controls", pd.DataFrame()))),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest_path = destination / "trend_feature_study_manifest_{}.json".format(identifier)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _load_stock_data(symbols: Sequence[str], price_history_dir: Optional[str], adjustment: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    loaded = {}
    skipped = []
    for symbol in symbols:
        try:
            loaded[str(symbol)] = load_local_price_history(symbol, price_history_dir, adjustment)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": str(symbol), "reason": str(exc)})
    return loaded, pd.DataFrame(skipped, columns=["symbol", "reason"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build causal factor panels and turning-point feature statistics.")
    parser.add_argument("--symbols", nargs="*", help="Six-digit symbols; default is data/HS300.txt")
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for a quick research run")
    parser.add_argument("--price-history-dir", default=str(DEFAULT_PRICE_HISTORY_DIR))
    parser.add_argument("--adjustment", default="raw_hfq_pct")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pre-window", type=int, default=60)
    parser.add_argument("--post-window", type=int, default=20)
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=None,
        help="Keep the first N generated features; omit to use all features",
    )
    args = parser.parse_args(argv)
    symbols = args.symbols or load_hs300_symbols()
    if args.limit is not None:
        symbols = symbols[: args.limit]
    stock_data, loading_skips = _load_stock_data(symbols, args.price_history_dir, args.adjustment)
    report = build_turning_point_feature_study(
        stock_data,
        feature_limit=args.feature_limit,
        pre_window=args.pre_window,
        post_window=args.post_window,
    )
    report["skipped_symbols"] = pd.concat([loading_skips, report["skipped_symbols"]], ignore_index=True)
    paths = export_turning_point_feature_study(report, args.output_dir, args.run_id)
    print("Built causal panel for {} symbols and {} events.".format(len(stock_data), len(report["events"])))
    print("Manifest: {}".format(paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
