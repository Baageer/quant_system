import numpy as np
import pandas as pd
import pytest

from signals.timing.bollinger_bands import (
    BollingerBandsStrategy,
    bollinger_squeeze_signal,
)


def _sample_ohlc(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.2, size=n)), index=idx)
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


class TestBollingerBandsStrategyValidation:
    def test_init_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            BollingerBandsStrategy(window=1)
        with pytest.raises(ValueError):
            BollingerBandsStrategy(num_std=0)
        with pytest.raises(ValueError):
            BollingerBandsStrategy(mode="invalid")
        with pytest.raises(ValueError):
            BollingerBandsStrategy(confirm_bars=0)
        with pytest.raises(ValueError):
            BollingerBandsStrategy(signal_delay=-1)

    def test_generate_signal_rejects_missing_columns(self):
        df = _sample_ohlc()[["close"]]
        strategy = BollingerBandsStrategy(mode="squeeze")
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.generate_signal(df)


class TestBollingerBandsBehavior:
    def test_signal_delay_behaves_as_shift(self):
        df = _sample_ohlc()
        base = BollingerBandsStrategy(
            mode="breakout",
            window=10,
            num_std=1.2,
            signal_delay=0,
        ).generate_signal(df)
        delayed = BollingerBandsStrategy(
            mode="breakout",
            window=10,
            num_std=1.2,
            signal_delay=1,
        ).generate_signal(df)
        expected = base.shift(1).fillna(0).astype(int)
        pd.testing.assert_series_equal(delayed, expected)

    def test_squeeze_adaptive_threshold_runs(self):
        df = _sample_ohlc()
        signal = bollinger_squeeze_signal(
            data=df,
            window=20,
            num_std=2.0,
            squeeze_threshold=None,
            squeeze_quantile=0.2,
            squeeze_lookback=60,
            use_supertrend_filter=True,
            signal_delay=1,
        )
        assert len(signal) == len(df)
        assert set(signal.dropna().unique()).issubset({-1, 0, 1})

    def test_squeeze_filter_toggle_runs(self):
        df = _sample_ohlc()
        signal_on = bollinger_squeeze_signal(
            data=df,
            window=20,
            num_std=2.0,
            squeeze_threshold=None,
            squeeze_quantile=0.2,
            squeeze_lookback=60,
            use_supertrend_filter=True,
            signal_delay=0,
        )
        signal_off = bollinger_squeeze_signal(
            data=df,
            window=20,
            num_std=2.0,
            squeeze_threshold=None,
            squeeze_quantile=0.2,
            squeeze_lookback=60,
            use_supertrend_filter=False,
            signal_delay=0,
        )
        assert len(signal_on) == len(signal_off)
