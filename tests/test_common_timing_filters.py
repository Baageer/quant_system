import pandas as pd

from signals.timing.common_filters import apply_common_timing_filters


def test_common_trend_filter_blocks_entry_but_not_existing_position():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    data = pd.DataFrame(
        {
            "close": [10.0, 9.0, 8.0, 9.0, 10.0, 9.8],
            "volume": [100, 100, 100, 100, 100, 100],
        },
        index=idx,
    )
    raw_signal = pd.Series([0, 0, 0, 1, 1, 1], index=idx)

    filtered = apply_common_timing_filters(
        raw_signal,
        data,
        {
            "use_trend_filter": True,
            "trend_window": 2,
            "trend_slope_window": 1,
            "trend_price_col": "close",
            "use_volume_filter": False,
            "volume_window": 20,
            "volume_multiplier": 1.2,
            "volume_col": "volume",
        },
    )

    expected = pd.Series([0, 0, 0, 0, 1, 1], index=idx)
    pd.testing.assert_series_equal(filtered, expected, check_dtype=False)
