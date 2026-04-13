"""
双均线交叉策略测试
"""
import pytest
import pandas as pd
import numpy as np
from signals.timing.ma_cross import (
    calculate_ma,
    ma_cross_signal,
    ma_cross_with_filter,
    MACrossStrategy
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    trend = np.linspace(10, 20, n)
    noise = np.random.randn(n) * 0.5
    close = trend + noise
    
    return pd.DataFrame({
        'open': close + np.random.randn(n) * 0.2,
        'high': close + np.abs(np.random.randn(n) * 0.3),
        'low': close - np.abs(np.random.randn(n) * 0.3),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)


@pytest.fixture
def oscillation_data():
    np.random.seed(123)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    base = 15
    amplitude = 3
    t = np.linspace(0, 4 * np.pi, n)
    close = base + amplitude * np.sin(t) + np.random.randn(n) * 0.3
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


class TestCalculateMA:
    
    def test_calculate_ma_basic(self, sample_data):
        ma = calculate_ma(sample_data['close'], window=5)
        
        assert isinstance(ma, pd.Series)
        assert len(ma) == len(sample_data)
        
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[3])
        assert not pd.isna(ma.iloc[4])
    
    def test_calculate_ma_values(self, sample_data):
        ma = calculate_ma(sample_data['close'], window=3)
        
        expected = sample_data['close'].iloc[0:3].mean()
        assert np.isclose(ma.iloc[2], expected)
    
    def test_calculate_ma_different_windows(self, sample_data):
        ma5 = calculate_ma(sample_data['close'], window=5)
        ma20 = calculate_ma(sample_data['close'], window=20)
        assert pd.isna(ma20.iloc[18])
        assert not pd.isna(ma20.iloc[19])

        assert ma5.iloc[50] != ma20.iloc[50]


class TestMACrossSignal:
    
    def test_signal_values(self, sample_data):
        signal = ma_cross_signal(sample_data, short_window=5, long_window=20)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    # def test_signal_with_uptrend(self, sample_data):
    #     signal = ma_cross_signal(sample_data, short_window=5, long_window=20)
        
    #     assert signal.iloc[-1] == 1
    
    def test_signal_with_oscillation(self, oscillation_data):
        signal = ma_cross_signal(oscillation_data, short_window=5, long_window=20)
        
        signals = signal.dropna()
        assert len(signals) > 0
    
    def test_signal_custom_price_col(self, sample_data):
        signal = ma_cross_signal(sample_data, short_window=5, long_window=20, price_col='close')
        
        assert len(signal) == len(sample_data)


class TestMACrossWithFilter:
    
    def test_filter_signal_values(self, sample_data):
        signal = ma_cross_with_filter(
            sample_data,
            short_window=5,
            long_window=20,
            ma_filter_window=60
        )
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
    
    def test_filter_more_conservative(self, sample_data):
        signal_basic = ma_cross_signal(sample_data, short_window=5, long_window=20)
        signal_filtered = ma_cross_with_filter(
            sample_data,
            short_window=5,
            long_window=20,
            ma_filter_window=60
        )
        
        basic_trades = (signal_basic.diff().abs() > 0).sum()
        filtered_trades = (signal_filtered.diff().abs() > 0).sum()
        
        assert filtered_trades <= basic_trades


class TestMACrossStrategy:
    
    def test_strategy_initialization(self):
        strategy = MACrossStrategy(short_window=5, long_window=20)
        
        assert strategy.short_window == 5
        assert strategy.long_window == 20
        assert strategy.use_filter is False
    
    def test_strategy_generate_signal(self, sample_data):
        strategy = MACrossStrategy(short_window=5, long_window=20)
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
    
    def test_strategy_with_filter(self, sample_data):
        strategy = MACrossStrategy(
            short_window=5,
            long_window=20,
            use_filter=True,
            ma_filter_window=60
        )
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert strategy.use_filter is True
    
    def test_get_ma_values(self, sample_data):
        strategy = MACrossStrategy(short_window=5, long_window=20)
        ma_values = strategy.get_ma_values(sample_data)
        
        assert 'short_ma' in ma_values
        assert 'long_ma' in ma_values
        assert isinstance(ma_values['short_ma'], pd.Series)
        assert isinstance(ma_values['long_ma'], pd.Series)
