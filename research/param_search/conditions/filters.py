import pandas as pd

from signals.indicators import sma, supertrend


def _true_series(index):
    return pd.Series(True, index=index, dtype=bool)


def build_volume_confirmation(df, params):
    if not params.get("use_volume_filter", False):
        return _true_series(df.index)

    if "volume" not in df.columns:
        raise ValueError("volume column is required when use_volume_filter=True")

    volume_window = max(int(params.get("volume_window", 20)), 2)
    volume_multiplier = float(params.get("volume_multiplier", 1.5))
    volume_ma = sma(df["volume"], volume_window)
    return (
        (df["volume"] >= volume_ma * volume_multiplier)
        & (df["volume"] > df["volume"].shift(1))
    ).fillna(False)


def build_trend_confirmation(df, params, prices=None):
    if not params.get("use_trend_filter", False):
        true_series = _true_series(df.index)
        return true_series, true_series

    prices = df["close"] if prices is None else prices
    trend_window = max(int(params.get("trend_window", 60)), 2)
    trend_slope_window = max(int(params.get("trend_slope_window", 3)), 1)
    trend_ma = sma(prices, trend_window)
    long_confirmation = ((prices > trend_ma) & (trend_ma > trend_ma.shift(trend_slope_window))).fillna(False)
    short_confirmation = ((prices < trend_ma) & (trend_ma < trend_ma.shift(trend_slope_window))).fillna(False)
    return long_confirmation, short_confirmation


def build_supertrend_confirmation(df, params, prices=None):
    if not params.get("use_supertrend_filter", False):
        true_series = _true_series(df.index)
        return true_series, true_series

    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(df.columns):
        raise ValueError("high, low, close columns are required when use_supertrend_filter=True")

    prices = df["close"] if prices is None else prices
    _, trend_direction = supertrend(
        df["high"],
        df["low"],
        prices,
        atr_period=int(params.get("supertrend_atr_period", 10)),
        multiplier=float(params.get("supertrend_multiplier", 3.0)),
    )
    long_confirmation = (trend_direction > 0).fillna(False)
    short_confirmation = (trend_direction < 0).fillna(False)
    return long_confirmation, short_confirmation


def build_band_expansion_confirmation(df, params, bandwidth=None):
    if not params.get("use_band_expansion_filter", False):
        return _true_series(df.index)

    if bandwidth is None:
        raise ValueError("bandwidth is required when use_band_expansion_filter=True")

    band_expansion_lookback = max(int(params.get("band_expansion_lookback", 1)), 1)
    return (bandwidth > bandwidth.shift(band_expansion_lookback)).fillna(False)


def build_return_confirmation(df, params, prices=None):
    if not params.get("use_return_filter", False):
        true_series = _true_series(df.index)
        return true_series, true_series

    prices = df["close"] if prices is None else prices
    min_breakout_return = float(params.get("min_breakout_return", 0.0))
    daily_return = prices.pct_change()
    up_confirmation = (daily_return >= min_breakout_return).fillna(False)
    down_confirmation = (daily_return <= -min_breakout_return).fillna(False)
    return up_confirmation, down_confirmation


def build_common_filter_context(df, params, prices=None, bandwidth=None):
    trend_long_confirmation, trend_short_confirmation = build_trend_confirmation(
        df=df,
        params=params,
        prices=prices,
    )
    supertrend_long_confirmation, supertrend_short_confirmation = build_supertrend_confirmation(
        df=df,
        params=params,
        prices=prices,
    )
    return_up_confirmation, return_down_confirmation = build_return_confirmation(
        df=df,
        params=params,
        prices=prices,
    )

    return {
        "volume_confirmation": build_volume_confirmation(df, params),
        "trend_long_confirmation": trend_long_confirmation,
        "trend_short_confirmation": trend_short_confirmation,
        "supertrend_long_confirmation": supertrend_long_confirmation,
        "supertrend_short_confirmation": supertrend_short_confirmation,
        "band_expansion_confirmation": build_band_expansion_confirmation(
            df=df,
            params=params,
            bandwidth=bandwidth,
        ),
        "return_up_confirmation": return_up_confirmation,
        "return_down_confirmation": return_down_confirmation,
    }


def direction_filters_ok(filter_context, row_index, direction):
    if direction == "up":
        direction_columns = (
            "trend_long_confirmation",
            "supertrend_long_confirmation",
            "return_up_confirmation",
        )
    elif direction == "down":
        direction_columns = (
            "trend_short_confirmation",
            "supertrend_short_confirmation",
            "return_down_confirmation",
        )
    else:
        raise ValueError("direction must be one of: up, down")

    common_columns = (
        "volume_confirmation",
        "band_expansion_confirmation",
    )
    return all(
        bool(filter_context[column].iloc[row_index])
        for column in (*common_columns, *direction_columns)
    )
