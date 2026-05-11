import pandas as pd

from research.param_search.conditions.filters import build_common_filter_context
from signals.indicators import rsrs


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

    filter_context = build_common_filter_context(
        df=df,
        params=params,
        prices=df["close"],
    )

    long_signal = (
        breakout_up
        & filter_context["volume_confirmation"]
        & filter_context["trend_long_confirmation"]
        & filter_context["supertrend_long_confirmation"]
        & filter_context["return_up_confirmation"]
    )
    short_signal = (
        breakout_down
        & filter_context["volume_confirmation"]
        & filter_context["trend_short_confirmation"]
        & filter_context["supertrend_short_confirmation"]
        & filter_context["return_down_confirmation"]
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
            "volume_confirmation": filter_context["volume_confirmation"],
            "trend_long_confirmation": filter_context["trend_long_confirmation"],
            "trend_short_confirmation": filter_context["trend_short_confirmation"],
            "supertrend_long_confirmation": filter_context["supertrend_long_confirmation"],
            "supertrend_short_confirmation": filter_context["supertrend_short_confirmation"],
            "band_expansion_confirmation": filter_context["band_expansion_confirmation"],
            "return_up_confirmation": filter_context["return_up_confirmation"],
            "return_down_confirmation": filter_context["return_down_confirmation"],
            "condition": breakout_valid,
        },
        index=df.index,
    )
