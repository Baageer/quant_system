"""
RSI策略测试
"""
import pytest
import pandas as pd
import numpy as np
from signals.timing.rsi import (
    calculate_rsi,
    rsi_signal,
    rsi_cross_signal,
    rsi_divergence_signal,
    RSIStrategy
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
def oversold_data():
    np.random.seed(100)
    n = 50
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(20, 10, n) + np.random.randn(n) * 0.2
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


@pytest.fixture
def overbought_data():
    np.random.seed(100)
    n = 50
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(10, 20, n) + np.random.randn(n) * 0.2
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


@pytest.fixture
def divergence_data():
    np.random.seed(200)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    t = np.linspace(0, 4 * np.pi, n)
    close = 15 + 3 * np.sin(t) + np.linspace(0, -2, n)
    
    return pd.DataFrame({
        'close': close
    }, index=dates)


class TestCalculateRSI:
    
    def test_rsi_range(self, sample_data):
        rsi = calculate_rsi(sample_data['close'], window=14)
        
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_rsi_values(self, sample_data):
        rsi = calculate_rsi(sample_data['close'], window=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        
        assert pd.isna(rsi.iloc[0])
        assert pd.isna(rsi.iloc[12])
        assert not pd.isna(rsi.iloc[13])
    
    def test_rsi_uptrend(self, overbought_data):
        rsi = calculate_rsi(overbought_data['close'], window=14)
        
        assert rsi.iloc[-1] > 50
    
    def test_rsi_downtrend(self, oversold_data):
        rsi = calculate_rsi(oversold_data['close'], window=14)
        
        assert rsi.iloc[-1] < 50
    
    def test_rsi_different_windows(self, sample_data):
        rsi7 = calculate_rsi(sample_data['close'], window=7)
        rsi14 = calculate_rsi(sample_data['close'], window=14)
        
        assert pd.isna(rsi14.iloc[12])
        assert not pd.isna(rsi14.iloc[13])


class TestRSISignal:
    
    def test_signal_values(self, sample_data):
        signal = rsi_signal(sample_data, window=14, oversold=30, overbought=70)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_signal_with_oversold(self, oversold_data):
        signal = rsi_signal(oversold_data, window=14, oversold=30, overbought=70)
        
        assert signal.iloc[-1] == 1
    
    def test_signal_with_overbought(self, overbought_data):
        signal = rsi_signal(overbought_data, window=14, oversold=30, overbought=70)
        
        assert signal.iloc[-1] == -1
    
    def test_signal_custom_thresholds(self, sample_data):
        signal = rsi_signal(sample_data, window=14, oversold=20, overbought=80)
        
        assert isinstance(signal, pd.Series)


class TestRSICrossSignal:
    
    def test_cross_signal_values(self, sample_data):
        signal = rsi_cross_signal(sample_data, window=14, oversold=30, overbought=70)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_cross_signal_triggers_on_cross(self, oversold_data):
        signal = rsi_cross_signal(oversold_data, window=14, oversold=30, overbought=70)
        
        assert isinstance(signal, pd.Series)


class TestRSIDivergenceSignal:
    
    def test_divergence_signal_values(self, divergence_data):
        signal = rsi_divergence_signal(divergence_data, window=14, lookback=5)
        
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(divergence_data)
        
        unique_values = set(signal.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})
    
    def test_divergence_signal_detects_patterns(self, divergence_data):
        signal = rsi_divergence_signal(divergence_data, window=14, lookback=5)
        
        assert isinstance(signal, pd.Series)


class TestRSIStrategy:
    
    def test_strategy_initialization(self):
        strategy = RSIStrategy(window=14, oversold=30, overbought=70)
        
        assert strategy.window == 14
        assert strategy.oversold == 30
        assert strategy.overbought == 70
        assert strategy.mode == 'standard'
    
    def test_strategy_standard_mode(self, sample_data):
        strategy = RSIStrategy(window=14, mode='standard')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_strategy_cross_mode(self, sample_data):
        strategy = RSIStrategy(window=14, mode='cross')
        signal = strategy.generate_signal(sample_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_strategy_divergence_mode(self, divergence_data):
        strategy = RSIStrategy(window=14, mode='divergence')
        signal = strategy.generate_signal(divergence_data)
        
        assert isinstance(signal, pd.Series)
    
    def test_get_rsi_values(self, sample_data):
        strategy = RSIStrategy(window=14)
        rsi = strategy.get_rsi_values(sample_data)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
