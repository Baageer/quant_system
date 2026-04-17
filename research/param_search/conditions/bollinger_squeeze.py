import numpy as np
import pandas as pd

from signals.indicators import bollinger_bands, sma, supertrend


def _is_missing(value):
    return value is None or value != value


def build_bollinger_squeeze_condition(df, params):
    prices = df["close"]
    window = int(params.get("window", 20))
    num_std = float(params.get("num_std", 2.0))
    upper_band, middle_band, lower_band = bollinger_bands(prices, window=window, num_std=num_std)
    bandwidth = (upper_band - lower_band) / middle_band.replace(0, np.nan)

    squeeze_threshold = params.get("squeeze_threshold")
    if squeeze_threshold is None:
        threshold = bandwidth.rolling(
            window=int(params.get("squeeze_lookback", 60)),
            min_periods=max(window, 5),
        ).quantile(float(params.get("squeeze_quantile", 0.1)))
    else:
        threshold = pd.Series(float(squeeze_threshold), index=df.index, dtype=float)

    squeeze_condition = (bandwidth <= threshold).fillna(False)
    require_breakout_confirmation = params.get("require_breakout_confirmation", False)
    breakout_direction = params.get("breakout_direction", "both")
    if breakout_direction not in {"up", "down", "both"}:
        raise ValueError("breakout_direction must be one of: up, down, both")

    breakout_buffer = float(params.get("breakout_buffer", 0.0))
    breakout_max_wait = max(int(params.get("breakout_max_wait", 10)), 1)
    breakout_confirm_bars = max(int(params.get("breakout_confirm_bars", 1)), 1)

    volume_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_volume_filter", False):
        if "volume" not in df.columns:
            raise ValueError("volume column is required when use_volume_filter=True")
        volume_window = int(params.get("volume_window", 20))
        volume_multiplier = float(params.get("volume_multiplier", 1.5))
        volume_ma = sma(df["volume"], volume_window)
        volume_confirmation = ((df["volume"] >= volume_ma * volume_multiplier) & (df["volume"] > df["volume"].shift(1))).fillna(False)

    trend_long_confirmation = pd.Series(True, index=df.index, dtype=bool)
    trend_short_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_trend_filter", False):
        trend_window = int(params.get("trend_window", 60))
        trend_slope_window = int(params.get("trend_slope_window", 3))
        trend_ma = sma(prices, trend_window)
        trend_long_confirmation = ((prices > trend_ma) & (trend_ma > trend_ma.shift(trend_slope_window))).fillna(False)
        trend_short_confirmation = ((prices < trend_ma) & (trend_ma < trend_ma.shift(trend_slope_window))).fillna(False)

    supertrend_long_confirmation = pd.Series(True, index=df.index, dtype=bool)
    supertrend_short_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_supertrend_filter", False):
        required_columns = {"high", "low", "close"}
        if not required_columns.issubset(df.columns):
            raise ValueError("high, low, close columns are required when use_supertrend_filter=True")
        _, trend_direction = supertrend(
            df["high"],
            df["low"],
            prices,
            atr_period=int(params.get("supertrend_atr_period", 10)),
            multiplier=float(params.get("supertrend_multiplier", 3.0)),
        )
        supertrend_long_confirmation = (trend_direction > 0).fillna(False)
        supertrend_short_confirmation = (trend_direction < 0).fillna(False)

    band_expansion_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_band_expansion_filter", False):
        band_expansion_lookback = max(int(params.get("band_expansion_lookback", 1)), 1)
        band_expansion_confirmation = (bandwidth > bandwidth.shift(band_expansion_lookback)).fillna(False)

    return_up_confirmation = pd.Series(True, index=df.index, dtype=bool)
    return_down_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_return_filter", False):
        min_breakout_return = float(params.get("min_breakout_return", 0.0))
        daily_return = prices.pct_change()
        return_up_confirmation = (daily_return >= min_breakout_return).fillna(False)
        return_down_confirmation = (daily_return <= -min_breakout_return).fillna(False)

    breakout_up = pd.Series(False, index=df.index, dtype=bool)
    breakout_down = pd.Series(False, index=df.index, dtype=bool)
    breakout_valid = pd.Series(False, index=df.index, dtype=bool)
    event_direction = pd.Series(index=df.index, dtype=object)
    condition = squeeze_condition.copy() if not require_breakout_confirmation else pd.Series(False, index=df.index, dtype=bool)

    if require_breakout_confirmation:
        in_squeeze = False
        bars_since_squeeze = 0
        up_streak = 0
        down_streak = 0

        for i in range(len(df)):
            if any(
                _is_missing(series.iloc[i])
                for series in (bandwidth, threshold, upper_band, lower_band)
            ):
                continue

            if squeeze_condition.iloc[i]:
                in_squeeze = True
                bars_since_squeeze = 0
                up_streak = 0
                down_streak = 0
                continue

            if not in_squeeze:
                continue

            bars_since_squeeze += 1
            if bars_since_squeeze > breakout_max_wait:
                in_squeeze = False
                up_streak = 0
                down_streak = 0
                continue

            close_above_upper = prices.iloc[i] > upper_band.iloc[i] * (1 + breakout_buffer)
            close_below_lower = prices.iloc[i] < lower_band.iloc[i] * (1 - breakout_buffer)
            up_streak = up_streak + 1 if close_above_upper else 0
            down_streak = down_streak + 1 if close_below_lower else 0

            breakout_up.iloc[i] = up_streak >= breakout_confirm_bars
            breakout_down.iloc[i] = down_streak >= breakout_confirm_bars

            long_filters_ok = (
                volume_confirmation.iloc[i]
                and trend_long_confirmation.iloc[i]
                and supertrend_long_confirmation.iloc[i]
                and band_expansion_confirmation.iloc[i]
                and return_up_confirmation.iloc[i]
            )
            short_filters_ok = (
                volume_confirmation.iloc[i]
                and trend_short_confirmation.iloc[i]
                and supertrend_short_confirmation.iloc[i]
                and band_expansion_confirmation.iloc[i]
                and return_down_confirmation.iloc[i]
            )

            if breakout_direction in {"up", "both"} and breakout_up.iloc[i] and long_filters_ok:
                breakout_valid.iloc[i] = True
                event_direction.iloc[i] = "up"
                condition.iloc[i] = True
                in_squeeze = False
                up_streak = 0
                down_streak = 0
            elif breakout_direction in {"down", "both"} and breakout_down.iloc[i] and short_filters_ok:
                breakout_valid.iloc[i] = True
                event_direction.iloc[i] = "down"
                condition.iloc[i] = True
                in_squeeze = False
                up_streak = 0
                down_streak = 0

    return pd.DataFrame(
        {
            "upper_band": upper_band,
            "middle_band": middle_band,
            "lower_band": lower_band,
            "bandwidth": bandwidth,
            "condition_threshold": threshold,
            "squeeze_condition": squeeze_condition,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "breakout_valid": breakout_valid,
            "event_direction": event_direction,
            "volume_confirmation": volume_confirmation,
            "trend_long_confirmation": trend_long_confirmation,
            "trend_short_confirmation": trend_short_confirmation,
            "supertrend_long_confirmation": supertrend_long_confirmation,
            "supertrend_short_confirmation": supertrend_short_confirmation,
            "band_expansion_confirmation": band_expansion_confirmation,
            "return_up_confirmation": return_up_confirmation,
            "return_down_confirmation": return_down_confirmation,
            "condition": condition,
        },
        index=df.index,
    )
