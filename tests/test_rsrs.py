"""
RSRS strategy tests.
"""
import numpy as np
import pandas as pd
import pytest

from signals.timing.rsrs import RSRSStrategy, rsrs_breakout_signal


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 220
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

    base = np.linspace(10, 22, n)
    noise = rng.normal(0.0, 0.4, n)
    close = base + noise
    spread = np.abs(rng.normal(0.35, 0.12, n))
    high = close + spread
    low = close - spread

    return pd.DataFrame(
        {
            "open": close + rng.normal(0.0, 0.1, n),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000000, 4000000, n),
        },
        index=dates,
    )


class TestRSRSSignal:

    def test_signal_output_shape(self, sample_data):
        signal = rsrs_breakout_signal(
            sample_data,
            window=18,
            zscore_window=90,
            entry_zscore=0.7,
            breakout_direction="both",
        )
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)

    def test_signal_values(self, sample_data):
        signal = rsrs_breakout_signal(
            sample_data,
            window=18,
            zscore_window=90,
            entry_zscore=0.7,
            breakout_direction="both",
        )
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_signal_delay(self, sample_data):
        signal_no_delay = rsrs_breakout_signal(
            sample_data,
            window=18,
            zscore_window=90,
            entry_zscore=0.7,
            signal_delay=0,
        )
        signal_delay_1 = rsrs_breakout_signal(
            sample_data,
            window=18,
            zscore_window=90,
            entry_zscore=0.7,
            signal_delay=1,
        )
        shifted = signal_no_delay.shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(signal_delay_1, shifted)


class TestRSRSStrategy:

    def test_strategy_generate_signal(self, sample_data):
        strategy = RSRSStrategy(window=18, zscore_window=90, entry_zscore=0.7)
        signal = strategy.generate_signal(sample_data)
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)

    def test_strategy_get_rsrs_values(self, sample_data):
        strategy = RSRSStrategy(window=18, zscore_window=90, entry_zscore=0.7)
        values = strategy.get_rsrs_values(sample_data)

        assert "rsrs_beta" in values
        assert "rsrs_r2" in values
        assert "rsrs_zscore" in values
        assert "rsrs_score" in values
        assert isinstance(values["rsrs_beta"], pd.Series)

    def test_strategy_invalid_direction(self):
        with pytest.raises(ValueError, match="breakout_direction"):
            RSRSStrategy(breakout_direction="invalid")
