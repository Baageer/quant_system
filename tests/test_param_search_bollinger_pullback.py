import pandas as pd

from research.param_search.conditions.registry import build_condition_frame


def _base_params():
    return {
        "window": 5,
        "num_std": 1.5,
        "squeeze_threshold": 0.01,
        "require_breakout_confirmation": True,
        "breakout_direction": "up",
        "breakout_buffer": 0.0,
        "breakout_max_wait": 5,
        "breakout_confirm_bars": 1,
        "use_volume_filter": False,
        "use_trend_filter": False,
        "pullback_reference": "breakout_band",
        "pullback_max_wait": 5,
        "pullback_tolerance": 0.01,
        "require_rebound_confirmation": False,
        "invalidate_on_middle_cross": True,
    }


def test_bollinger_squeeze_pullback_emits_signal_on_retest():
    idx = pd.date_range("2024-01-01", periods=25, freq="D")
    close = [100] * 12 + [100.2, 100.1, 100.3, 100.2, 100.1, 104, 106, 105, 104.2, 105.5, 106, 107, 108]
    high = [value + 0.6 for value in close]
    low = [value - 0.6 for value in close]
    volume = [1000] * len(close)
    df = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)

    result = build_condition_frame(df, "bollinger_squeeze_pullback", _base_params())

    breakout_date = pd.Timestamp("2024-01-18")
    pullback_date = pd.Timestamp("2024-01-20")

    assert bool(result.loc[breakout_date, "breakout_valid"]) is True
    assert bool(result.loc[breakout_date, "condition"]) is False
    assert bool(result.loc[pullback_date, "pullback_valid"]) is True
    assert bool(result.loc[pullback_date, "condition"]) is True
    assert result.loc[pullback_date, "event_direction"] == "up"
    assert result.loc[pullback_date, "bars_since_breakout"] == 2.0


def test_bollinger_squeeze_pullback_respects_wait_window():
    idx = pd.date_range("2024-01-01", periods=23, freq="D")
    close = [100] * 12 + [100.2, 100.1, 100.3, 100.2, 100.1, 104, 106, 108, 109, 110, 111]
    high = [value + 0.5 for value in close]
    low = [value - 0.5 for value in close]
    volume = [1000] * len(close)
    df = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)

    params = _base_params()
    params["pullback_max_wait"] = 3

    result = build_condition_frame(df, "bollinger_squeeze_pullback", params)

    assert bool(result["breakout_valid"].any()) is True
    assert bool(result["pullback_valid"].any()) is False
    assert bool(result["condition"].any()) is False
