import numpy as np
import pandas as pd
import pytest

from data.indicator_cache import IndicatorCache
from run_backtest import apply_signals_to_dataframe
from signals.signal_engine import SignalEngine
from signals.timing.bollinger_bands import (
    BollingerBandsStrategy,
    bollinger_breakout_signal,
    bollinger_squeeze_signal,
)
from signals.timing.common_filters import COMMON_TIMING_FILTER_DEFAULTS, FilteredTimingStrategy


def _sample_ohlc(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.2, size=n)), index=idx)
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    volume = pd.Series(rng.integers(800, 1600, size=n), index=idx)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)


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

    def test_breakout_volume_filter_blocks_low_volume_breakout(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "close": [10.0, 10.0, 10.0, 10.0, 10.0, 10.2, 10.4, 12.0],
                "high": [10.1, 10.1, 10.1, 10.1, 10.1, 10.3, 10.5, 12.2],
                "low": [9.9, 9.9, 9.9, 9.9, 9.9, 10.0, 10.2, 11.8],
                "volume": [100, 100, 100, 100, 100, 100, 100, 100],
            },
            index=idx,
        )

        signal_without_filter = bollinger_breakout_signal(
            data=df,
            window=3,
            num_std=1.0,
            use_volume_filter=False,
        )
        signal_with_filter = bollinger_breakout_signal(
            data=df,
            window=3,
            num_std=1.0,
            use_volume_filter=True,
            volume_window=3,
            volume_multiplier=1.5,
        )

        assert signal_without_filter.iloc[-1] == 1
        assert signal_with_filter.iloc[-1] == 0

    def test_breakout_trend_filter_blocks_countertrend_breakout(self):
        idx = pd.date_range("2024-02-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "close": [15.0, 14.0, 13.0, 12.0, 11.0, 10.8, 11.2, 11.8],
                "high": [15.2, 14.2, 13.2, 12.2, 11.2, 11.0, 11.4, 12.0],
                "low": [14.8, 13.8, 12.8, 11.8, 10.8, 10.6, 11.0, 11.6],
                "volume": [100, 100, 100, 100, 100, 120, 180, 220],
            },
            index=idx,
        )

        signal_without_filter = bollinger_breakout_signal(
            data=df,
            window=3,
            num_std=1.0,
            use_trend_filter=False,
            use_volume_filter=False,
        )
        signal_with_filter = bollinger_breakout_signal(
            data=df,
            window=3,
            num_std=1.0,
            use_trend_filter=True,
            trend_window=7,
            trend_slope_window=1,
            use_volume_filter=False,
        )

        assert signal_without_filter.iloc[-1] == 1
        assert signal_with_filter.iloc[-1] == 0


class TestBollingerBandsCacheIntegration:
    def test_breakout_signal_reuses_indicator_cache(self, tmpdir):
        df = _sample_ohlc()
        cache = IndicatorCache(cache_dir=str(tmpdir))

        first = bollinger_breakout_signal(
            data=df,
            window=20,
            num_std=2.0,
            indicator_cache=cache,
            cache_symbol="000001.SZ",
        )
        assert cache.last_cache_status == "miss"

        second = bollinger_breakout_signal(
            data=df,
            window=20,
            num_std=2.0,
            indicator_cache=cache,
            cache_symbol="000001.SZ",
        )
        assert cache.last_cache_status == "hit"
        pd.testing.assert_series_equal(first, second)

    def test_strategy_breakout_uses_indicator_cache(self, tmpdir):
        df = _sample_ohlc()
        cache = IndicatorCache(cache_dir=str(tmpdir))
        strategy = BollingerBandsStrategy(
            mode="breakout",
            window=20,
            num_std=2.0,
            indicator_cache=cache,
            cache_symbol="000001.SZ",
        )

        strategy.generate_signal(df)
        assert cache.last_cache_status == "miss"

        values = strategy.get_bollinger_values(df)
        assert cache.last_cache_status == "hit"
        assert set(values) == {"upper_band", "middle_band", "lower_band"}

    def test_backtest_signal_application_injects_cache_into_wrapped_strategy(self, tmpdir):
        df = _sample_ohlc()
        cache = IndicatorCache(cache_dir=str(tmpdir))
        base_strategy = BollingerBandsStrategy(
            mode="breakout",
            window=20,
            num_std=2.0,
        )
        wrapped_strategy = FilteredTimingStrategy(
            base_strategy,
            dict(COMMON_TIMING_FILTER_DEFAULTS),
        )

        result = apply_signals_to_dataframe(
            df=df.copy(),
            strategies=[wrapped_strategy],
            signal_engine=SignalEngine(),
            signal_weights=[1.0],
            signal_combination="weighted",
            signal_threshold=0.5,
            symbol="000001.SZ",
            indicator_cache=cache,
        )

        assert "signal" in result.columns
        assert base_strategy.indicator_cache is cache
        assert base_strategy.cache_symbol == "000001.SZ"
        assert cache.last_cache_status == "miss"
