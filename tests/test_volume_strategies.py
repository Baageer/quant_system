import numpy as np
import pandas as pd
import pytest

from signals.indicators import mfi, obv, sma
from signals.timing.volume import (
    MFIStrategy,
    OBVStrategy,
    VolumeBreakoutStrategy,
    VolumePriceDivergenceStrategy,
    mfi_signal,
    obv_signal,
    volume_breakout_signal,
    volume_price_divergence_signal,
)


@pytest.fixture
def breakout_long_data():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "close": [10, 10, 10, 10, 11],
            "volume": [100, 100, 100, 100, 1000],
        },
        index=index,
    )


@pytest.fixture
def breakout_short_data():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "close": [10, 10, 10, 10, 9],
            "volume": [100, 100, 100, 100, 1000],
        },
        index=index,
    )


@pytest.fixture
def obv_uptrend_data():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "close": [10, 11, 12, 13],
            "volume": [100, 100, 100, 100],
        },
        index=index,
    )


@pytest.fixture
def obv_downtrend_data():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "close": [13, 12, 11, 10],
            "volume": [100, 100, 100, 100],
        },
        index=index,
    )


@pytest.fixture
def mfi_long_data():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "high": [10.0, 9.8, 9.6, 9.4, 9.8, 10.5],
            "low": [9.5, 9.3, 9.1, 8.9, 9.4, 10.0],
            "close": [9.8, 9.5, 9.2, 9.0, 9.7, 10.3],
            "volume": [100, 100, 100, 100, 500, 600],
        },
        index=index,
    )


@pytest.fixture
def mfi_short_data():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "high": [10.0, 10.5, 11.0, 11.5, 10.8, 10.2],
            "low": [9.6, 10.0, 10.5, 11.0, 10.1, 9.8],
            "close": [10.0, 10.4, 10.9, 11.3, 10.2, 9.9],
            "volume": [100, 120, 140, 160, 500, 600],
        },
        index=index,
    )


@pytest.fixture
def bearish_divergence_data():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "close": [10, 11, 12, 13, 14],
            "volume": [100, 120, 140, 150, 130],
        },
        index=index,
    )


@pytest.fixture
def bullish_divergence_data():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "close": [14, 13, 12, 11, 10],
            "volume": [100, 90, 80, 85, 95],
        },
        index=index,
    )


class TestVolumeBreakoutSignal:
    def test_long_breakout_signal(self, breakout_long_data):
        signal = volume_breakout_signal(
            breakout_long_data,
            volume_window=3,
            z_thresh=1.0,
            trend_window=3,
        )

        expected = pd.Series([0, 0, 0, 0, 1], index=breakout_long_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_short_breakout_signal(self, breakout_short_data):
        signal = volume_breakout_signal(
            breakout_short_data,
            volume_window=3,
            z_thresh=1.0,
            trend_window=3,
        )

        expected = pd.Series([0, 0, 0, 0, -1], index=breakout_short_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_strategy_matches_function(self, breakout_long_data):
        strategy = VolumeBreakoutStrategy(volume_window=3, z_thresh=1.0, trend_window=3)
        strategy_signal = strategy.generate_signal(breakout_long_data)
        direct_signal = volume_breakout_signal(
            breakout_long_data,
            volume_window=3,
            z_thresh=1.0,
            trend_window=3,
        )

        pd.testing.assert_series_equal(strategy_signal, direct_signal)
        pd.testing.assert_series_equal(
            strategy.get_volume_ma(breakout_long_data),
            sma(breakout_long_data["volume"], 3),
        )


class TestOBVSignal:
    def test_obv_long_signal(self, obv_uptrend_data):
        signal = obv_signal(obv_uptrend_data, window=2)

        expected = pd.Series([0, 1, 1, 1], index=obv_uptrend_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_obv_short_signal(self, obv_downtrend_data):
        signal = obv_signal(obv_downtrend_data, window=2)

        expected = pd.Series([0, -1, -1, -1], index=obv_downtrend_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_strategy_returns_obv_values(self, obv_uptrend_data):
        strategy = OBVStrategy(obv_window=2)
        values = strategy.get_obv_values(obv_uptrend_data)

        assert "obv" in values
        assert "obv_ma" in values
        pd.testing.assert_series_equal(values["obv"], obv(obv_uptrend_data["close"], obv_uptrend_data["volume"]))
        pd.testing.assert_series_equal(values["obv_ma"], sma(values["obv"], 2))


class TestMFISignal:
    def test_mfi_long_signal(self, mfi_long_data):
        signal = mfi_signal(
            mfi_long_data,
            window=2,
            oversold=40,
            overbought=60,
            trend_window=2,
        )

        expected = pd.Series([0, 0, 0, 0, 1, 0], index=mfi_long_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_mfi_short_signal(self, mfi_short_data):
        signal = mfi_signal(
            mfi_short_data,
            window=2,
            oversold=40,
            overbought=60,
            trend_window=2,
        )

        expected = pd.Series([0, 0, 0, 0, -1, 0], index=mfi_short_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_strategy_returns_indicator_values(self, mfi_long_data):
        strategy = MFIStrategy(
            mfi_window=2,
            overbought=60,
            oversold=40,
            trend_window=2,
        )
        indicator = strategy.get_mfi_value(mfi_long_data)
        expected = mfi(
            mfi_long_data["high"],
            mfi_long_data["low"],
            mfi_long_data["close"],
            mfi_long_data["volume"],
            2,
        )

        pd.testing.assert_series_equal(indicator, expected)


class TestVolumePriceDivergenceSignal:
    def test_bearish_divergence_signal(self, bearish_divergence_data):
        signal = volume_price_divergence_signal(bearish_divergence_data, window=3)

        expected = pd.Series([0, 0, 0, 0, -1], index=bearish_divergence_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_bullish_divergence_signal(self, bullish_divergence_data):
        signal = volume_price_divergence_signal(bullish_divergence_data, window=3)

        expected = pd.Series([0, 0, 0, 1, 1], index=bullish_divergence_data.index)
        pd.testing.assert_series_equal(signal, expected)

    def test_strategy_returns_divergence_values(self, bullish_divergence_data):
        strategy = VolumePriceDivergenceStrategy(window=3)
        values = strategy.get_divergence_values(bullish_divergence_data)

        assert "price_change" in values
        assert "volume_change" in values
        pd.testing.assert_series_equal(
            values["price_change"],
            bullish_divergence_data["close"].pct_change(3),
        )
        pd.testing.assert_series_equal(
            values["volume_change"],
            bullish_divergence_data["volume"].pct_change(3),
        )
