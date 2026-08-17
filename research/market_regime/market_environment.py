"""Causal benchmark features and market-state-stratified probability reports."""

from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from signals.indicators import adx, bollinger_bandwidth, sma, volatility


MARKET_STATE_COLUMN = "market_state"
MARKET_STATE_ORDER = ("risk_on", "neutral", "risk_off")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def load_benchmark_history(path: Union[str, Path]) -> pd.DataFrame:
    """Read a benchmark CSV with date, close and optional high/low columns."""

    data = pd.read_csv(path, low_memory=False)
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    required = {"date", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Benchmark CSV is missing columns: {}".format(sorted(missing)))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("close", "high", "low"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if data.empty:
        raise ValueError("Benchmark CSV has no valid date/close rows")
    if "high" not in data:
        data["high"] = data["close"]
    if "low" not in data:
        data["low"] = data["close"]
    return data.set_index("date").sort_index()


def build_causal_market_features(benchmark_data: pd.DataFrame, prefix: str = "mkt_") -> pd.DataFrame:
    """Calculate market features and a current-close-only three-state regime."""

    required = {"close", "high", "low"}
    missing = required.difference(benchmark_data.columns)
    if missing:
        raise ValueError("Benchmark data is missing columns: {}".format(sorted(missing)))
    data = benchmark_data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Benchmark data must use a DatetimeIndex")
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    sma_5, sma_20, sma_60 = sma(close, 5), sma(close, 20), sma(close, 60)
    adx_value, plus_di, minus_di = adx(high, low, close)
    features = pd.DataFrame(index=data.index)
    for window in (5, 20, 60):
        features["{}ret_{}d".format(prefix, window)] = close.pct_change(window)
    features["{}sma_ratio_20".format(prefix)] = _safe_divide(close, sma_20) - 1
    features["{}sma_ratio_60".format(prefix)] = _safe_divide(close, sma_60) - 1
    features["{}sma_gap_5_20".format(prefix)] = _safe_divide(sma_5, sma_20) - 1
    features["{}sma_gap_20_60".format(prefix)] = _safe_divide(sma_20, sma_60) - 1
    features["{}volatility_20".format(prefix)] = volatility(close, 20)
    features["{}adx_14".format(prefix)] = adx_value
    features["{}di_spread_14".format(prefix)] = plus_di - minus_di
    bandwidth = bollinger_bandwidth(close, 20, 2.0)
    features["{}bollinger_bandwidth_20".format(prefix)] = bandwidth
    features["{}bollinger_bandwidth_percentile_120".format(prefix)] = bandwidth.rolling(
        120, min_periods=120
    ).apply(lambda values: float(np.sum(values <= values[-1]) / len(values)))
    risk_on = (features["{}ret_20d".format(prefix)] > 0) & (close > sma_60) & (sma_20 > sma_60)
    risk_off = (features["{}ret_20d".format(prefix)] < 0) & (close < sma_60) & (sma_20 < sma_60)
    features[MARKET_STATE_COLUMN] = np.select([risk_on, risk_off], ["risk_on", "risk_off"], default="neutral")
    features.loc[sma_60.isnull(), MARKET_STATE_COLUMN] = np.nan
    return features


def merge_market_features(panel: pd.DataFrame, market_features: pd.DataFrame) -> pd.DataFrame:
    """Join date-level causal market context onto a date × symbol panel."""

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        raise ValueError("Feature panel must have a date, symbol MultiIndex")
    overlap = sorted(set(panel.columns).intersection(market_features.columns))
    if overlap:
        raise ValueError("Market feature columns already exist in panel: {}".format(overlap))
    market_frame = market_features.copy()
    market_frame.index.name = "date"
    market_frame = market_frame.reset_index()
    if market_frame["date"].duplicated().any():
        raise ValueError("Market features have duplicate dates")
    joined = panel.reset_index().merge(market_frame, on="date", how="left")
    return joined.set_index(["date", "symbol"]).sort_index()


def stratify_probability_predictions(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
    evaluate_function,
    split: str = "test",
) -> pd.DataFrame:
    """Evaluate probability predictions separately for each causal market state."""

    required = {"date", "symbol", MARKET_STATE_COLUMN, "target"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError("Samples are missing market-state fields: {}".format(sorted(missing)))
    prediction_columns = ["date", "symbol", "p_down", "p_none", "p_up"]
    missing_prediction = set(prediction_columns).difference(predictions.columns)
    if missing_prediction:
        raise ValueError("Predictions are missing columns: {}".format(sorted(missing_prediction)))
    source_columns = [column for column in ("date", "symbol", "target", "sample_weight", MARKET_STATE_COLUMN) if column in samples]
    source = samples.loc[:, source_columns]
    prediction_frame = predictions.loc[:, prediction_columns]
    if source.duplicated(["date", "symbol"]).any() or prediction_frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Samples and predictions must each have unique date, symbol rows")
    merged = source.merge(prediction_frame, on=["date", "symbol"], how="inner")
    rows = []
    probabilities = ["p_down", "p_none", "p_up"]
    for state in MARKET_STATE_ORDER:
        subset = merged.loc[merged[MARKET_STATE_COLUMN].eq(state)]
        if subset.empty or subset["target"].nunique() < 2:
            continue
        targets = np.asarray(subset["target"].astype(str), dtype=str)
        weights = np.asarray(pd.to_numeric(subset.get("sample_weight", 1.0), errors="coerce"), dtype=float)
        weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
        metrics = evaluate_function(targets, np.asarray(subset[probabilities], dtype=float), weights, split, "market_state_stratified")
        metrics[MARKET_STATE_COLUMN] = state
        rows.append(metrics)
    return pd.DataFrame(rows)
