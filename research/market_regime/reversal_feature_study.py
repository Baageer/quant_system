"""Factor-discovery study for K-line reversal features vs HS300-relative excess.

The frozen 28 features of ``hs300_logistic_v2`` are built from
``high/low/close/volume`` only -- ``open`` is never used -- so there is no
candlestick body/wick geometry and only two distance-from-extreme proxies
(``high_distance_60`` / ``low_distance_60``).  This module fills that gap: it
computes a battery of causal reversal/candlestick features at the signal date
``t`` (using only data <= t) and measures their cross-sectional predictive power
for the forward HS300-relative excess ``future_excess_return_{h}d`` from the
``excess_hs300_v1`` dataset.

Evaluation (IC / ICIR / win rate, quantile groups, stability, redundancy) lives in
``feature_study_core``; this module only supplies the feature panel.

Example::

    python -m research.market_regime.reversal_feature_study \
        --dataset-id excess_hs300_v1 --run-id reversal_study_v1 --horizon 10
"""

import argparse
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from research.market_regime.feature_study_core import (
    LABEL_COLUMN,
    MAE_COLUMN,
    SPLITS,
    evaluate_feature_panel,
    export_study,
    load_ohlcv_histories,
    load_study_panels,
    surviving_candidates,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols
from signals.indicators import adx, kdj, macd


PREFIX = "reversal_feature"

# Existing reversal-related columns kept for redundancy comparison.  These are
# the nearest thing the frozen 28 have to "reversal": distance from 60-day
# extreme + overbought/oversold oscillators.
EXISTING_REVERSAL_PROXIES = (
    "high_distance_60",
    "low_distance_60",
    "williams_r_14",
    "kdj_d",
    "rsi_14",
    "mfi_14",
)

DISTANCE_WINDOWS = (5, 10, 20, 60)

# name -> category for the manifest.  Feature computation lives in
# ``compute_reversal_features`` and returns exactly these columns.
REVERSAL_FEATURE_SPECS = (
    # ---- single-day candle geometry --------------------------------------
    {"name": "body_pct", "category": "candle_geometry"},
    {"name": "body_to_range", "category": "candle_geometry"},
    {"name": "upper_wick", "category": "candle_geometry"},
    {"name": "lower_wick", "category": "candle_geometry"},
    {"name": "close_position", "category": "candle_geometry"},
    {"name": "doji", "category": "candle_geometry"},
    {"name": "hammer", "category": "candle_geometry"},
    {"name": "shooting_star", "category": "candle_geometry"},
    {"name": "engulfing_bullish", "category": "candle_geometry"},
    {"name": "engulfing_bearish", "category": "candle_geometry"},
    # ---- distance from extreme / reversal position -----------------------
    {"name": "dist_from_high_5", "category": "distance_extreme"},
    {"name": "dist_from_low_5", "category": "distance_extreme"},
    {"name": "dist_from_high_10", "category": "distance_extreme"},
    {"name": "dist_from_low_10", "category": "distance_extreme"},
    {"name": "dist_from_high_20", "category": "distance_extreme"},
    {"name": "dist_from_low_20", "category": "distance_extreme"},
    {"name": "days_since_high_20", "category": "distance_extreme"},
    {"name": "days_since_low_20", "category": "distance_extreme"},
    # ---- oversold-bounce / reversal triggers -----------------------------
    {"name": "di_cross_up", "category": "reversal_trigger"},
    {"name": "di_cross_down", "category": "reversal_trigger"},
    {"name": "macd_hist_improving_days", "category": "reversal_trigger"},
    {"name": "stoch_cross_up", "category": "reversal_trigger"},
    {"name": "stoch_cross_down", "category": "reversal_trigger"},
    # ---- volume divergence / gap -----------------------------------------
    {"name": "price_up_volume_down", "category": "volume_divergence"},
    {"name": "price_down_volume_up", "category": "volume_divergence"},
    {"name": "gap_open", "category": "gap"},
    {"name": "gap_up_filled", "category": "gap"},
    {"name": "gap_down_filled", "category": "gap"},
)

FEATURE_NAMES = [spec["name"] for spec in REVERSAL_FEATURE_SPECS]


def _days_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    """Bars since the rolling-window max (mode='high') or min (mode='low')."""

    values = np.asarray(series.ffill().to_numpy(dtype=float))
    count = len(values)
    result = np.full(count, np.nan, dtype=float)
    if count >= window:
        windows = sliding_window_view(values, window)
        arg = np.argmax(windows, axis=1) if mode == "high" else np.argmin(windows, axis=1)
        result[window - 1:] = (window - 1) - arg
    return pd.Series(result, index=series.index)


def compute_reversal_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute all causal reversal features from one symbol's OHLCV series.

    ``prices`` is date-indexed with at least ``open/high/low/close`` (volume is
    optional and only needed for volume/gap features).  Everything here uses only
    data at or before each timestamp, so the result is causal at the signal date.
    """

    open_ = pd.to_numeric(prices["open"], errors="coerce").ffill()
    high = pd.to_numeric(prices["high"], errors="coerce").ffill()
    low = pd.to_numeric(prices["low"], errors="coerce").ffill()
    close = pd.to_numeric(prices["close"], errors="coerce").ffill()
    volume = pd.to_numeric(prices.get("volume", pd.Series(np.nan, index=prices.index)), errors="coerce")

    features = pd.DataFrame(index=prices.index)
    rng = (high - low).replace(0, np.nan)
    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)
    body = close - open_
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # ---- candle geometry ---------------------------------------------------
    features["body_pct"] = body / open_.replace(0, np.nan)
    features["body_to_range"] = body.abs() / rng
    features["upper_wick"] = (high - body_top) / rng
    features["lower_wick"] = (body_bot - low) / rng
    features["close_position"] = (close - low) / rng
    small_body = features["body_to_range"] <= 0.1
    features["doji"] = small_body.astype(float)
    features["hammer"] = ((features["lower_wick"] >= 0.6) & small_body & (features["close_position"] >= 0.5)).astype(float)
    features["shooting_star"] = ((features["upper_wick"] >= 0.6) & small_body & (features["close_position"] <= 0.5)).astype(float)
    prev_body_top = np.maximum(prev_open, prev_close)
    prev_body_bot = np.minimum(prev_open, prev_close)
    features["engulfing_bullish"] = (
        (close > open_) & (prev_close < prev_open) & (open_ <= prev_body_bot) & (close >= prev_body_top)
    ).astype(float)
    features["engulfing_bearish"] = (
        (close < open_) & (prev_close > prev_open) & (open_ >= prev_body_top) & (close <= prev_body_bot)
    ).astype(float)

    # ---- distance from extreme --------------------------------------------
    for window in DISTANCE_WINDOWS:
        roll_max = high.rolling(window, min_periods=window).max()
        roll_min = low.rolling(window, min_periods=window).min()
        features["dist_from_high_{}".format(window)] = close / roll_max - 1.0
        features["dist_from_low_{}".format(window)] = close / roll_min - 1.0
    features["days_since_high_20"] = _days_since_extreme(high, 20, "high")
    features["days_since_low_20"] = _days_since_extreme(low, 20, "low")

    # ---- reversal triggers -------------------------------------------------
    _, plus_di, minus_di = adx(high, low, close)
    di_spread = plus_di - minus_di
    features["di_cross_up"] = ((di_spread > 0) & (di_spread.shift(1) <= 0)).astype(float)
    features["di_cross_down"] = ((di_spread < 0) & (di_spread.shift(1) >= 0)).astype(float)
    _, _, macd_hist = macd(close)
    improving = (macd_hist.diff() > 0).fillna(False)
    features["macd_hist_improving_days"] = improving.astype(int).groupby((~improving).cumsum()).cumsum()
    k_line, d_line, _ = kdj(high, low, close)
    features["stoch_cross_up"] = ((k_line > d_line) & (k_line.shift(1) <= d_line.shift(1))).astype(float)
    features["stoch_cross_down"] = ((k_line < d_line) & (k_line.shift(1) >= d_line.shift(1))).astype(float)

    # ---- volume divergence / gap -------------------------------------------
    features["price_up_volume_down"] = ((close > prev_close) & (volume < volume.shift(1))).astype(float)
    features["price_down_volume_up"] = ((close < prev_close) & (volume > volume.shift(1))).astype(float)
    features["gap_open"] = open_ / prev_close - 1.0
    features["gap_up_filled"] = ((open_ > prev_high) & (close < open_)).astype(float)
    features["gap_down_filled"] = ((open_ < prev_low) & (close > open_)).astype(float)

    return features.loc[:, FEATURE_NAMES]


def attach_reversal_features(samples: pd.DataFrame, histories: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join the causal reversal features onto the frozen sample rows."""

    parts = []
    for symbol, group in tqdm(samples.groupby("symbol", sort=False), desc="Computing reversal features", unit="symbol"):
        prices = histories.get(symbol)
        if prices is None or prices.empty:
            continue
        group = group.sort_values("date")
        features = compute_reversal_features(prices).reindex(pd.DatetimeIndex(group["date"]))
        features = features.reset_index(drop=True)
        features["date"] = group["date"].to_numpy()
        features["symbol"] = symbol
        parts.append(features)
    if not parts:
        raise ValueError("No symbols produced reversal features")
    return samples.merge(pd.concat(parts, ignore_index=True), on=["date", "symbol"], how="left")


def build_study(
    dataset_dir: Path,
    dataset_id: str,
    benchmark_file: Path,
    horizon: int,
    price_history_dir: Optional[str] = None,
    limit_symbols: Optional[int] = None,
) -> dict:
    """Compute features and run the full factor evaluation."""

    label = LABEL_COLUMN.format(int(horizon))
    samples, splits = load_study_panels(dataset_dir, dataset_id, horizon, limit_symbols=limit_symbols)
    histories, skipped = load_ohlcv_histories(samples["symbol"].unique(), price_history_dir)

    panel = attach_reversal_features(samples, histories)
    keep = ["date", "symbol", "source_split", label, MAE_COLUMN.format(int(horizon)), *EXISTING_REVERSAL_PROXIES]
    panel = panel.loc[:, [column for column in keep if column in panel.columns] + FEATURE_NAMES]
    panel[FEATURE_NAMES] = panel[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

    tables = evaluate_feature_panel(panel, FEATURE_NAMES, label, EXISTING_REVERSAL_PROXIES, Path(benchmark_file))
    return {
        "question": "are causal K-line reversal features predictive of forward HS300-relative excess",
        "label": label,
        "horizon": int(horizon),
        "feature_names": FEATURE_NAMES,
        "feature_specs": list(REVERSAL_FEATURE_SPECS),
        "existing_proxies": list(EXISTING_REVERSAL_PROXIES),
        "row_counts": {name: int(len(splits[name])) for name in SPLITS},
        "symbol_count": int(panel["symbol"].nunique()),
        "skipped_prices": skipped,
        "missing_rate": float(panel[FEATURE_NAMES].isnull().mean().mean()),
        "extra_manifest": {"skipped_price_symbols": skipped.to_dict("records")},
        **tables,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="excess_hs300_v1")
    parser.add_argument("--benchmark-file", default="data/benchmarks/000300.csv")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="reversal_study_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--limit-symbols", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_study(
        Path(args.dataset_dir), args.dataset_id, Path(args.benchmark_file), args.horizon,
        price_history_dir=args.price_history_dir, limit_symbols=args.limit_symbols,
    )
    paths = export_study(
        report, Path(args.output_dir), args.run_id, PREFIX,
        {"dataset_id": args.dataset_id, "horizon": args.horizon, "limit_symbols": args.limit_symbols},
    )
    survivors = surviving_candidates(report["ic_table"], report["group_table"], FEATURE_NAMES)
    count = int(survivors["survives"].sum()) if not survivors.empty else 0
    print("Reversal feature study done. {} features, {} survive screening; missing_rate={:.2%}; manifest={}".format(
        len(FEATURE_NAMES), count, report["missing_rate"], paths["manifest"]))
    if count:
        print("Survivors:\n{}".format(
            survivors[survivors["survives"]][["feature", "mean_ic", "icir", "win_rate", "top_bottom_spread"]].to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
