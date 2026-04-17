import pandas as pd

from signals.indicators import rsrs, sma


def build_rsrs_breakout_condition(df, params):
    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(df.columns):
        raise ValueError("high, low, close columns are required for rsrs_breakout")

    window = max(int(params.get("window", 18)), 5)
    zscore_window = max(int(params.get("zscore_window", 120)), 20)
    min_valid_window = max(int(params.get("min_valid_window", window // 2)), 5)

    breakout_direction = params.get("breakout_direction", "up")
    if breakout_direction not in {"up", "down", "both"}:
        raise ValueError("breakout_direction must be one of: up, down, both")

    entry_zscore = float(params.get("entry_zscore", 0.7))
    use_r2_weight = bool(params.get("use_r2_weight", True))
    use_beta_adjustment = bool(params.get("use_beta_adjustment", False))

    beta, r2, rsrs_zscore, rsrs_score = rsrs(
        high=df["high"],
        low=df["low"],
        window=window,
        zscore_window=zscore_window,
        min_valid_window=min(min_valid_window, window),
        use_r2_weight=use_r2_weight,
        use_beta_adjustment=use_beta_adjustment,
    )

    threshold = pd.Series(entry_zscore, index=df.index, dtype=float)
    breakout_up = (rsrs_score >= threshold).fillna(False)
    breakout_down = (rsrs_score <= -threshold).fillna(False)

    volume_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_volume_filter", False):
        if "volume" not in df.columns:
            raise ValueError("volume column is required when use_volume_filter=True")
        volume_window = max(int(params.get("volume_window", 20)), 2)
        volume_multiplier = float(params.get("volume_multiplier", 1.2))
        volume_ma = sma(df["volume"], volume_window)
        volume_confirmation = (
            (df["volume"] >= volume_ma * volume_multiplier)
            & (df["volume"] > df["volume"].shift(1))
        ).fillna(False)

    trend_long_confirmation = pd.Series(True, index=df.index, dtype=bool)
    trend_short_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_trend_filter", False):
        trend_window = max(int(params.get("trend_window", 60)), 2)
        trend_slope_window = max(int(params.get("trend_slope_window", 3)), 1)
        trend_ma = sma(df["close"], trend_window)
        trend_long_confirmation = ((df["close"] > trend_ma) & (trend_ma > trend_ma.shift(trend_slope_window))).fillna(False)
        trend_short_confirmation = ((df["close"] < trend_ma) & (trend_ma < trend_ma.shift(trend_slope_window))).fillna(False)

    return_up_confirmation = pd.Series(True, index=df.index, dtype=bool)
    return_down_confirmation = pd.Series(True, index=df.index, dtype=bool)
    if params.get("use_return_filter", False):
        min_breakout_return = float(params.get("min_breakout_return", 0.0))
        daily_return = df["close"].pct_change()
        return_up_confirmation = (daily_return >= min_breakout_return).fillna(False)
        return_down_confirmation = (daily_return <= -min_breakout_return).fillna(False)

    long_signal = (
        breakout_up
        & volume_confirmation
        & trend_long_confirmation
        & return_up_confirmation
    )
    short_signal = (
        breakout_down
        & volume_confirmation
        & trend_short_confirmation
        & return_down_confirmation
    )

    if breakout_direction == "up":
        condition = long_signal
    elif breakout_direction == "down":
        condition = short_signal
    else:
        condition = long_signal | short_signal

    event_direction = pd.Series(index=df.index, dtype=object)
    event_direction.loc[long_signal] = "up"
    event_direction.loc[short_signal] = "down"

    breakout_valid = condition.fillna(False)
    return pd.DataFrame(
        {
            "rsrs_beta": beta,
            "rsrs_r2": r2,
            "rsrs_zscore": rsrs_zscore,
            "rsrs_score": rsrs_score,
            "condition_threshold": threshold,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "breakout_valid": breakout_valid,
            "event_direction": event_direction,
            "volume_confirmation": volume_confirmation,
            "trend_long_confirmation": trend_long_confirmation,
            "trend_short_confirmation": trend_short_confirmation,
            "return_up_confirmation": return_up_confirmation,
            "return_down_confirmation": return_down_confirmation,
            "condition": breakout_valid,
        },
        index=df.index,
    )
