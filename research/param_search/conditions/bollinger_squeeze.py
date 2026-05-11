import numpy as np
import pandas as pd

from research.param_search.conditions.filters import (
    build_common_filter_context,
    direction_filters_ok,
)
from signals.indicators import bollinger_bands


def _is_missing(value):
    return value is None or value != value


def _prepare_bollinger_context(df, params):
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

    filter_context = build_common_filter_context(
        df=df,
        params=params,
        prices=prices,
        bandwidth=bandwidth,
    )

    context = {
        "prices": prices,
        "upper_band": upper_band,
        "middle_band": middle_band,
        "lower_band": lower_band,
        "bandwidth": bandwidth,
        "threshold": threshold,
        "squeeze_condition": squeeze_condition,
        "require_breakout_confirmation": require_breakout_confirmation,
        "breakout_direction": breakout_direction,
        "breakout_buffer": breakout_buffer,
        "breakout_max_wait": breakout_max_wait,
        "breakout_confirm_bars": breakout_confirm_bars,
    }
    context.update(filter_context)
    return context


def _build_result_frame(
    df,
    context,
    breakout_up,
    breakout_down,
    breakout_valid,
    event_direction,
    condition,
    extra_columns=None,
):
    result = {
        "upper_band": context["upper_band"],
        "middle_band": context["middle_band"],
        "lower_band": context["lower_band"],
        "bandwidth": context["bandwidth"],
        "condition_threshold": context["threshold"],
        "squeeze_condition": context["squeeze_condition"],
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "breakout_valid": breakout_valid,
        "event_direction": event_direction,
        "volume_confirmation": context["volume_confirmation"],
        "trend_long_confirmation": context["trend_long_confirmation"],
        "trend_short_confirmation": context["trend_short_confirmation"],
        "supertrend_long_confirmation": context["supertrend_long_confirmation"],
        "supertrend_short_confirmation": context["supertrend_short_confirmation"],
        "band_expansion_confirmation": context["band_expansion_confirmation"],
        "return_up_confirmation": context["return_up_confirmation"],
        "return_down_confirmation": context["return_down_confirmation"],
        "condition": condition,
    }
    if extra_columns:
        result.update(extra_columns)
    return pd.DataFrame(result, index=df.index)


def build_bollinger_squeeze_condition(df, params):
    context = _prepare_bollinger_context(df, params)
    prices = context["prices"]
    upper_band = context["upper_band"]
    lower_band = context["lower_band"]

    breakout_up = pd.Series(False, index=df.index, dtype=bool)
    breakout_down = pd.Series(False, index=df.index, dtype=bool)
    breakout_valid = pd.Series(False, index=df.index, dtype=bool)
    event_direction = pd.Series(index=df.index, dtype=object)
    condition = context["squeeze_condition"].copy() if not context["require_breakout_confirmation"] else pd.Series(False, index=df.index, dtype=bool)

    if context["require_breakout_confirmation"]:
        in_squeeze = False
        bars_since_squeeze = 0
        up_streak = 0
        down_streak = 0

        for i in range(len(df)):
            if any(
                _is_missing(series.iloc[i])
                for series in (context["bandwidth"], context["threshold"], upper_band, lower_band)
            ):
                continue

            if context["squeeze_condition"].iloc[i]:
                in_squeeze = True
                bars_since_squeeze = 0
                up_streak = 0
                down_streak = 0
                continue

            if not in_squeeze:
                continue

            bars_since_squeeze += 1
            if bars_since_squeeze > context["breakout_max_wait"]:
                in_squeeze = False
                up_streak = 0
                down_streak = 0
                continue

            close_above_upper = prices.iloc[i] > upper_band.iloc[i] * (1 + context["breakout_buffer"])
            close_below_lower = prices.iloc[i] < lower_band.iloc[i] * (1 - context["breakout_buffer"])
            up_streak = up_streak + 1 if close_above_upper else 0
            down_streak = down_streak + 1 if close_below_lower else 0

            breakout_up.iloc[i] = up_streak >= context["breakout_confirm_bars"]
            breakout_down.iloc[i] = down_streak >= context["breakout_confirm_bars"]

            long_filters_ok = direction_filters_ok(context, i, "up")
            short_filters_ok = direction_filters_ok(context, i, "down")

            if context["breakout_direction"] in {"up", "both"} and breakout_up.iloc[i] and long_filters_ok:
                breakout_valid.iloc[i] = True
                event_direction.iloc[i] = "up"
                condition.iloc[i] = True
                in_squeeze = False
                up_streak = 0
                down_streak = 0
            elif context["breakout_direction"] in {"down", "both"} and breakout_down.iloc[i] and short_filters_ok:
                breakout_valid.iloc[i] = True
                event_direction.iloc[i] = "down"
                condition.iloc[i] = True
                in_squeeze = False
                up_streak = 0
                down_streak = 0

    return _build_result_frame(
        df=df,
        context=context,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        breakout_valid=breakout_valid,
        event_direction=event_direction,
        condition=condition,
    )


def build_bollinger_squeeze_pullback_condition(df, params):
    context = _prepare_bollinger_context(df, params)
    prices = context["prices"]
    upper_band = context["upper_band"]
    middle_band = context["middle_band"]
    lower_band = context["lower_band"]

    pullback_reference = params.get("pullback_reference", "breakout_band")
    if pullback_reference not in {"breakout_band", "middle_band", "breakout_close"}:
        raise ValueError("pullback_reference must be one of: breakout_band, middle_band, breakout_close")

    pullback_max_wait = max(int(params.get("pullback_max_wait", 10)), 1)
    pullback_tolerance = max(float(params.get("pullback_tolerance", 0.01)), 0.0)
    require_rebound_confirmation = bool(params.get("require_rebound_confirmation", True))
    invalidate_on_middle_cross = bool(params.get("invalidate_on_middle_cross", True))

    breakout_up = pd.Series(False, index=df.index, dtype=bool)
    breakout_down = pd.Series(False, index=df.index, dtype=bool)
    breakout_valid = pd.Series(False, index=df.index, dtype=bool)
    pullback_valid = pd.Series(False, index=df.index, dtype=bool)
    event_direction = pd.Series(index=df.index, dtype=object)
    condition = pd.Series(False, index=df.index, dtype=bool)
    pullback_reference_price = pd.Series(np.nan, index=df.index, dtype=float)
    bars_since_breakout = pd.Series(np.nan, index=df.index, dtype=float)

    in_squeeze = False
    squeeze_wait_bars = 0
    up_streak = 0
    down_streak = 0
    awaiting_pullback = False
    pullback_direction_state = None
    breakout_anchor_price = np.nan
    pullback_wait_bars = 0

    for i in range(len(df)):
        if any(
            _is_missing(series.iloc[i])
            for series in (context["bandwidth"], context["threshold"], upper_band, middle_band, lower_band)
        ):
            continue

        if awaiting_pullback:
            pullback_wait_bars += 1
            if pullback_wait_bars > pullback_max_wait:
                awaiting_pullback = False
                pullback_direction_state = None
                breakout_anchor_price = np.nan
                pullback_wait_bars = 0
            else:
                reference_price = breakout_anchor_price
                if pullback_reference == "middle_band":
                    reference_price = middle_band.iloc[i]

                if not _is_missing(reference_price):
                    bars_since_breakout.iloc[i] = float(pullback_wait_bars)

                    if invalidate_on_middle_cross:
                        if pullback_direction_state == "up" and prices.iloc[i] < middle_band.iloc[i] * (1 - pullback_tolerance):
                            awaiting_pullback = False
                            pullback_direction_state = None
                            breakout_anchor_price = np.nan
                            pullback_wait_bars = 0
                            continue
                        if pullback_direction_state == "down" and prices.iloc[i] > middle_band.iloc[i] * (1 + pullback_tolerance):
                            awaiting_pullback = False
                            pullback_direction_state = None
                            breakout_anchor_price = np.nan
                            pullback_wait_bars = 0
                            continue

                    if pullback_direction_state == "up":
                        touched_reference = (
                            df["low"].iloc[i] >= reference_price * (1 - pullback_tolerance)
                            and df["low"].iloc[i] <= reference_price * (1 + pullback_tolerance)
                        )
                        close_holds = prices.iloc[i] >= reference_price * (1 - pullback_tolerance)
                        rebound_ok = prices.iloc[i] >= prices.iloc[i - 1] if require_rebound_confirmation and i > 0 else True
                    else:
                        touched_reference = (
                            df["high"].iloc[i] >= reference_price * (1 - pullback_tolerance)
                            and df["high"].iloc[i] <= reference_price * (1 + pullback_tolerance)
                        )
                        close_holds = prices.iloc[i] <= reference_price * (1 + pullback_tolerance)
                        rebound_ok = prices.iloc[i] <= prices.iloc[i - 1] if require_rebound_confirmation and i > 0 else True

                    if touched_reference and close_holds and rebound_ok:
                        pullback_valid.iloc[i] = True
                        event_direction.iloc[i] = pullback_direction_state
                        condition.iloc[i] = True
                        pullback_reference_price.iloc[i] = float(reference_price)
                        awaiting_pullback = False
                        pullback_direction_state = None
                        breakout_anchor_price = np.nan
                        pullback_wait_bars = 0
                        continue

        if context["squeeze_condition"].iloc[i]:
            in_squeeze = True
            squeeze_wait_bars = 0
            up_streak = 0
            down_streak = 0
            continue

        if not in_squeeze:
            continue

        squeeze_wait_bars += 1
        if squeeze_wait_bars > context["breakout_max_wait"]:
            in_squeeze = False
            up_streak = 0
            down_streak = 0
            continue

        close_above_upper = prices.iloc[i] > upper_band.iloc[i] * (1 + context["breakout_buffer"])
        close_below_lower = prices.iloc[i] < lower_band.iloc[i] * (1 - context["breakout_buffer"])
        up_streak = up_streak + 1 if close_above_upper else 0
        down_streak = down_streak + 1 if close_below_lower else 0

        breakout_up.iloc[i] = up_streak >= context["breakout_confirm_bars"]
        breakout_down.iloc[i] = down_streak >= context["breakout_confirm_bars"]

        long_filters_ok = direction_filters_ok(context, i, "up")
        short_filters_ok = direction_filters_ok(context, i, "down")

        if context["breakout_direction"] in {"up", "both"} and breakout_up.iloc[i] and long_filters_ok:
            breakout_valid.iloc[i] = True
            event_direction.iloc[i] = "up"
            awaiting_pullback = True
            pullback_direction_state = "up"
            breakout_anchor_price = float(upper_band.iloc[i] if pullback_reference == "breakout_band" else prices.iloc[i])
            pullback_wait_bars = 0
            in_squeeze = False
            up_streak = 0
            down_streak = 0
        elif context["breakout_direction"] in {"down", "both"} and breakout_down.iloc[i] and short_filters_ok:
            breakout_valid.iloc[i] = True
            event_direction.iloc[i] = "down"
            awaiting_pullback = True
            pullback_direction_state = "down"
            breakout_anchor_price = float(lower_band.iloc[i] if pullback_reference == "breakout_band" else prices.iloc[i])
            pullback_wait_bars = 0
            in_squeeze = False
            up_streak = 0
            down_streak = 0

    return _build_result_frame(
        df=df,
        context=context,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        breakout_valid=breakout_valid,
        event_direction=event_direction,
        condition=condition,
        extra_columns={
            "pullback_valid": pullback_valid,
            "pullback_reference_price": pullback_reference_price,
            "bars_since_breakout": bars_since_breakout,
        },
    )
