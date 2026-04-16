from typing import Any, Dict, Tuple

import pandas as pd

from signals.indicators import sma


COMMON_TIMING_FILTER_DEFAULTS: Dict[str, Any] = {
    "use_trend_filter": False,
    "trend_window": 60,
    "trend_slope_window": 3,
    "trend_price_col": "close",
    "use_volume_filter": False,
    "volume_window": 20,
    "volume_multiplier": 1.2,
    "volume_col": "volume",
}


def extract_common_timing_filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read common timing-filter params from strategy YAML params."""
    return {
        key: params.get(key, default)
        for key, default in COMMON_TIMING_FILTER_DEFAULTS.items()
    }


def has_common_timing_filters_enabled(filter_params: Dict[str, Any]) -> bool:
    return bool(filter_params.get("use_trend_filter")) or bool(
        filter_params.get("use_volume_filter")
    )


def _validate_filter_columns(data: pd.DataFrame, filter_params: Dict[str, Any]) -> None:
    required_cols = []
    if filter_params["use_trend_filter"]:
        required_cols.append(filter_params["trend_price_col"])
    if filter_params["use_volume_filter"]:
        required_cols.append(filter_params["volume_col"])

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for common filters: {', '.join(missing_cols)}"
        )


def _build_direction_masks(
    data: pd.DataFrame, filter_params: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series]:
    allow_long = pd.Series(True, index=data.index, dtype=bool)
    allow_short = pd.Series(True, index=data.index, dtype=bool)

    if filter_params["use_trend_filter"]:
        trend_price_col = filter_params["trend_price_col"]
        prices = data[trend_price_col]
        trend_ma = sma(prices, int(filter_params["trend_window"]))
        slope_window = max(int(filter_params["trend_slope_window"]), 1)
        trend_rising = trend_ma > trend_ma.shift(slope_window)
        trend_falling = trend_ma < trend_ma.shift(slope_window)

        allow_long &= ((prices > trend_ma) & trend_rising).fillna(False)
        allow_short &= ((prices < trend_ma) & trend_falling).fillna(False)

    if filter_params["use_volume_filter"]:
        volume_col = filter_params["volume_col"]
        volume = data[volume_col]
        volume_ma = sma(volume, int(filter_params["volume_window"]))
        volume_multiplier = float(filter_params["volume_multiplier"])

        volume_ok = (
            (volume > volume_ma * volume_multiplier) & (volume > volume.shift(1))
        ).fillna(False)
        allow_long &= volume_ok
        allow_short &= volume_ok

    return allow_long, allow_short


def apply_common_timing_filters(
    signal: pd.Series, data: pd.DataFrame, filter_params: Dict[str, Any]
) -> pd.Series:
    """
    Apply reusable trend/volume filters to a timing signal.

    The input signal is treated as a target position series in {-1, 0, 1}.
    Filters only gate *position transitions* into long/short, so existing
    holdings are not force-closed when filters temporarily turn off.
    """
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal, index=data.index)

    signal = signal.reindex(data.index).fillna(0).clip(-1, 1).astype(int)
    if not has_common_timing_filters_enabled(filter_params):
        return signal

    _validate_filter_columns(data, filter_params)
    allow_long, allow_short = _build_direction_masks(data, filter_params)

    filtered = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(len(data)):
        target_pos = int(signal.iloc[i])

        if target_pos == current_pos:
            filtered.iloc[i] = current_pos
            continue

        if target_pos == 0:
            current_pos = 0
        elif target_pos > 0:
            if allow_long.iloc[i]:
                current_pos = 1
        else:
            if allow_short.iloc[i]:
                current_pos = -1

        filtered.iloc[i] = current_pos

    return filtered


class FilteredTimingStrategy:
    """
    Strategy proxy that applies common timing filters outside strategy logic.
    """

    def __init__(self, strategy: object, filter_params: Dict[str, Any]):
        self._strategy = strategy
        self._filter_params = filter_params

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        raw_signal = self._strategy.generate_signal(data)
        return apply_common_timing_filters(raw_signal, data, self._filter_params)

    def __getattr__(self, name: str):
        return getattr(self._strategy, name)
