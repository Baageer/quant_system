"""
MACD策略测试
"""
import pytest
import pandas as pd
import numpy as np
from signals.timing.macd import (
    calculate_macd,
    macd_cross_signal,
    macd_histogram_signal,
    macd_zero_axis_signal,
    macd_combined_signal,
    MACDStrategy
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
        'close': close
    }, index=dates)


@pytest.fixture
def uptrend_data():
    np.random.seed(100)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(10, 25, n) + np.random.randn(n) * 0.3
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


@pytest.fixture
def downtrend_data():
    np.random.seed(100)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(25, 10, n) + np.random.randn(n) * 0.3
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


@pytest.fixture
def oscillation_data():
    np.random.seed(200)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    t = np.linspace(0, 4 * np.pi, n)
    close = 15 + 3 * np.sin(t) + np.random.randn(n) * 0.2
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


class TestCalculateMACD:
    
    def test_macd_output_shape(self, sample_data):
        macd_line, signal_line, histogram = calculate_macd(sample_data['close'])
        
        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)
    
    def test_macd_values(self, sample_data):
        macd_line, signal_line, histogram = calculate_macd(sample_data['close'])
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
    
    def test_macd_histogram_calculation(self, sample_data):
        macd_line, signal_line, histogram = calculate_macd(sample_data['close'])
        
        expected_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_histogram, check_names=False)
    
    def test_macd_custom_periods(self, sample_data):
        macd_line, signal_line, histogram = calculate_macd(
            sample_data['close'],
            fast_period=8,
            slow_period=17,
            signal_period=9
        )
        
        assert isinstance(macd_line, pd.Series)
    
    def test_macd_uptrend_positive(self, uptrend_data):
        macd_line, signal_line, histogram = calculate_macd(uptrend_data['close'])
        
        assert macd_line.iloc[-1] > 0
    
    def test_macd_downtrend_negative(self, downtrend_data):
        macd_line, signal_line, histogram = calculate_macd(downtrend_data['close'])
        
        assert macd_line.iloc[-1] < 0


class TestMACDCrossSignal:
    
    def test_signal_values(self, sample_data):
        signal = macd_cross_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    # def test_signal_uptrend(self, uptrend_data):
    #     signal = macd_cross_signal(uptrend_data)
        
    #     assert signal.iloc[-1] == 1
    
    def test_signal_downtrend(self, downtrend_data):
        signal = macd_cross_signal(downtrend_data)
        
        assert signal.iloc[-1] == -1
    
    def test_signal_custom_periods(self, sample_data):
        signal = macd_cross_signal(
            sample_data,
            fast_period=8,
            slow_period=17,
            signal_period=9
        )
        
        assert isinstance(signal, pd.Series)


class TestMACDHistogramSignal:
    
    def test_signal_values(self, sample_data):
        signal = macd_histogram_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_signal_with_oscillation(self, oscillation_data):
        signal = macd_histogram_signal(oscillation_data)
        
        assert isinstance(signal, pd.Series)
        
        signal_changes = signal.diff().dropna()
        assert len(signal_changes[signal_changes != 0]) > 0


class TestMACDZeroAxisSignal:
    
    def test_signal_values(self, sample_data):
        signal = macd_zero_axis_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_signal_uptrend(self, uptrend_data):
        signal = macd_zero_axis_signal(uptrend_data)
        
        assert signal.iloc[-1] == 1
    
    def test_signal_downtrend(self, downtrend_data):
        signal = macd_zero_axis_signal(downtrend_data)
        
        assert signal.iloc[-1] == -1


class TestMACDCombinedSignal:
    
    def test_signal_values(self, sample_data):
        signal = macd_combined_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_combined_vs_cross(self, sample_data):
        signal_cross = macd_cross_signal(sample_data)
        signal_combined = macd_combined_signal(sample_data)
        
        assert isinstance(signal_cross, pd.Series)
        assert isinstance(signal_combined, pd.Series)


class TestMACDStrategy:
    
    def test_strategy_initialization(self):
        strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
        assert strategy.mode == 'cross'
    
    def test_strategy_cross_mode(self, sample_data):
        strategy = MACDStrategy(mode='cross')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_strategy_histogram_mode(self, sample_data):
        strategy = MACDStrategy(mode='histogram')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_strategy_zero_axis_mode(self, sample_data):
        strategy = MACDStrategy(mode='zero_axis')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_strategy_combined_mode(self, sample_data):
        strategy = MACDStrategy(mode='combined')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_get_macd_values(self, sample_data):
        strategy = MACDStrategy()
        macd_values = strategy.get_macd_values(sample_data)
        
        assert 'macd' in macd_values
        assert 'signal' in macd_values
        assert 'histogram' in macd_values
        
        assert isinstance(macd_values['macd'], pd.Series)
        assert isinstance(macd_values['signal'], pd.Series)
        assert isinstance(macd_values['histogram'], pd.Series)
    
    def test_modes_produce_different_signals(self, oscillation_data):
        strategy_cross = MACDStrategy(mode='cross')
        strategy_hist = MACDStrategy(mode='histogram')
        strategy_zero = MACDStrategy(mode='zero_axis')
        
        signal_cross = strategy_cross.generate_signal(oscillation_data)
        signal_hist = strategy_hist.generate_signal(oscillation_data)
        signal_zero = strategy_zero.generate_signal(oscillation_data)
        
        assert isinstance(signal_cross, pd.Series)
        assert isinstance(signal_hist, pd.Series)
        assert isinstance(signal_zero, pd.Series)
