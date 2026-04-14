"""
技术指标库测试
"""
import pytest
import pandas as pd
import numpy as np
from signals.indicators import (
    sma, ema, wma,
    bollinger_bands, bollinger_bandwidth,
    rsi, macd, atr, kdj,
    stochastic_oscillator, cci, williams_r,
    obv, adx, mfi, roc, momentum,
    vwap, donchian_channel, keltner_channel,
    trix, dmi, typical_price, pivot_points,
    z_score, volatility, supertrend
)


@pytest.fixture
def price_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    trend = np.linspace(10, 20, n)
    noise = np.random.randn(n) * 0.5
    close = trend + noise
    
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 5000, n)
    
    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def uptrend_data():
    np.random.seed(100)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(10, 25, n) + np.random.randn(n) * 0.3
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 5000, n)
    
    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def downtrend_data():
    np.random.seed(100)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    close = np.linspace(25, 10, n) + np.random.randn(n) * 0.3
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 5000, n)
    
    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestSMA:
    
    def test_sma_output_shape(self, price_data):
        result = sma(price_data['close'], window=20)
        assert len(result) == len(price_data)
    
    def test_sma_values(self, price_data):
        result = sma(price_data['close'], window=5)
        
        assert isinstance(result, pd.Series)
        
        expected = price_data['close'].iloc[0:5].mean()
        assert not pd.isna(result.iloc[4])
        np.testing.assert_almost_equal(result.iloc[4], expected, decimal=10)
    
    def test_sma_nan_at_start(self, price_data):
        result = sma(price_data['close'], window=20)
        
        assert pd.isna(result.iloc[18])
        assert not pd.isna(result.iloc[19])
    
    def test_sma_different_windows(self, price_data):
        sma5 = sma(price_data['close'], window=5)
        sma20 = sma(price_data['close'], window=20)
        
        assert pd.isna(sma20.iloc[18])
        assert not pd.isna(sma5.iloc[4])


class TestEMA:
    
    def test_ema_output_shape(self, price_data):
        result = ema(price_data['close'], span=20)
        assert len(result) == len(price_data)
    
    def test_ema_values(self, price_data):
        result = ema(price_data['close'], span=20)
        
        assert isinstance(result, pd.Series)
        assert not pd.isna(result.iloc[0])
    
    def test_ema_responsive(self, price_data):
        ema_short = ema(price_data['close'], span=5)
        ema_long = ema(price_data['close'], span=50)
        
        assert isinstance(ema_short, pd.Series)
        assert isinstance(ema_long, pd.Series)


class TestWMA:
    
    def test_wma_output_shape(self, price_data):
        result = wma(price_data['close'], window=10)
        assert len(result) == len(price_data)
    
    def test_wma_nan_at_start(self, price_data):
        result = wma(price_data['close'], window=10)
        
        assert pd.isna(result.iloc[8])
        assert not pd.isna(result.iloc[9])
    
    def test_wma_weights(self, price_data):
        result = wma(price_data['close'], window=3)
        
        weights = np.array([1, 2, 3])
        data = price_data['close'].iloc[0:3].values
        expected = np.dot(data, weights) / weights.sum()
        
        np.testing.assert_almost_equal(result.iloc[2], expected, decimal=10)


class TestBollingerBands:
    
    def test_bollinger_output_shape(self, price_data):
        upper, middle, lower = bollinger_bands(price_data['close'], window=20)
        
        assert len(upper) == len(price_data)
        assert len(middle) == len(price_data)
        assert len(lower) == len(price_data)
    
    def test_bollinger_bands_order(self, price_data):
        upper, middle, lower = bollinger_bands(price_data['close'], window=20)
        
        valid_mask = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()
    
    def test_bollinger_middle_is_sma(self, price_data):
        _, middle, _ = bollinger_bands(price_data['close'], window=20)
        expected_middle = sma(price_data['close'], window=20)
        
        pd.testing.assert_series_equal(middle, expected_middle)
    
    def test_bollinger_custom_std(self, price_data):
        upper2, _, lower2 = bollinger_bands(price_data['close'], window=20, num_std=2.0)
        upper3, _, lower3 = bollinger_bands(price_data['close'], window=20, num_std=3.0)
        
        valid_mask = ~(pd.isna(upper2) | pd.isna(upper3))
        assert (upper3[valid_mask] > upper2[valid_mask]).all()
        assert (lower3[valid_mask] < lower2[valid_mask]).all()


class TestBollingerBandwidth:
    
    def test_bandwidth_positive(self, price_data):
        bandwidth = bollinger_bandwidth(price_data['close'], window=20)
        
        valid_bandwidth = bandwidth.dropna()
        assert (valid_bandwidth > 0).all()
    
    def test_bandwidth_shape(self, price_data):
        bandwidth = bollinger_bandwidth(price_data['close'], window=20)
        assert len(bandwidth) == len(price_data)


class TestRSI:
    
    def test_rsi_range(self, price_data):
        result = rsi(price_data['close'], window=14)
        
        valid_rsi = result.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_rsi_uptrend(self, uptrend_data):
        result = rsi(uptrend_data['close'], window=14)
        
        assert result.iloc[-1] > 50
    
    def test_rsi_downtrend(self, downtrend_data):
        result = rsi(downtrend_data['close'], window=14)
        
        assert result.iloc[-1] < 50
    
    def test_rsi_nan_at_start(self, price_data):
        result = rsi(price_data['close'], window=14)
        
        assert pd.isna(result.iloc[12])
        assert not pd.isna(result.iloc[13])


class TestMACD:
    
    def test_macd_output_shape(self, price_data):
        macd_line, signal_line, histogram = macd(price_data['close'])
        
        assert len(macd_line) == len(price_data)
        assert len(signal_line) == len(price_data)
        assert len(histogram) == len(price_data)
    
    def test_macd_histogram_calculation(self, price_data):
        macd_line, signal_line, histogram = macd(price_data['close'])
        
        expected_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_histogram, check_names=False)
    
    def test_macd_uptrend_positive(self, uptrend_data):
        macd_line, _, _ = macd(uptrend_data['close'])
        
        assert macd_line.iloc[-1] > 0
    
    def test_macd_downtrend_negative(self, downtrend_data):
        macd_line, _, _ = macd(downtrend_data['close'])
        
        assert macd_line.iloc[-1] < 0


class TestATR:
    
    def test_atr_output_shape(self, price_data):
        result = atr(price_data['high'], price_data['low'], price_data['close'], window=14)
        assert len(result) == len(price_data)
    
    def test_atr_positive(self, price_data):
        result = atr(price_data['high'], price_data['low'], price_data['close'], window=14)
        
        valid_atr = result.dropna()
        assert (valid_atr > 0).all()
    
    def test_atr_nan_at_start(self, price_data):
        result = atr(price_data['high'], price_data['low'], price_data['close'], window=14)
        
        assert pd.isna(result.iloc[12])
        assert not pd.isna(result.iloc[13])


class TestKDJ:
    
    def test_kdj_output_shape(self, price_data):
        k, d, j = kdj(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(k) == len(price_data)
        assert len(d) == len(price_data)
        assert len(j) == len(price_data)
    
    def test_kdj_range(self, price_data):
        k, d, j = kdj(price_data['high'], price_data['low'], price_data['close'])
        
        valid_k = k.dropna()
        valid_d = d.dropna()
        
        assert valid_k.min() >= 0
        assert valid_k.max() <= 100
        assert valid_d.min() >= 0
        assert valid_d.max() <= 100
    
    def test_kdj_j_formula(self, price_data):
        k, d, j = kdj(price_data['high'], price_data['low'], price_data['close'])
        
        valid_mask = ~(pd.isna(k) | pd.isna(d) | pd.isna(j))
        expected_j = 3 * k[valid_mask] - 2 * d[valid_mask]
        pd.testing.assert_series_equal(j[valid_mask], expected_j, check_names=False)


class TestStochasticOscillator:
    
    def test_stochastic_output_shape(self, price_data):
        k, d = stochastic_oscillator(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(k) == len(price_data)
        assert len(d) == len(price_data)
    
    def test_stochastic_range(self, price_data):
        k, d = stochastic_oscillator(price_data['high'], price_data['low'], price_data['close'])
        
        valid_k = k.dropna()
        valid_d = d.dropna()
        
        assert valid_k.min() >= 0
        assert valid_k.max() <= 100
        assert valid_d.min() >= 0
        assert valid_d.max() <= 100


class TestCCI:
    
    def test_cci_output_shape(self, price_data):
        result = cci(price_data['high'], price_data['low'], price_data['close'])
        assert len(result) == len(price_data)
    
    def test_cci_nan_at_start(self, price_data):
        result = cci(price_data['high'], price_data['low'], price_data['close'], window=20)
        
        assert pd.isna(result.iloc[18])
        assert not pd.isna(result.iloc[19])


class TestWilliamsR:
    
    def test_williams_r_output_shape(self, price_data):
        result = williams_r(price_data['high'], price_data['low'], price_data['close'])
        assert len(result) == len(price_data)
    
    def test_williams_r_range(self, price_data):
        result = williams_r(price_data['high'], price_data['low'], price_data['close'])
        
        valid_result = result.dropna()
        assert valid_result.min() >= -100
        assert valid_result.max() <= 0


class TestOBV:
    
    def test_obv_output_shape(self, price_data):
        result = obv(price_data['close'], price_data['volume'])
        assert len(result) == len(price_data)
    
    def test_obv_cumulative(self, price_data):
        result = obv(price_data['close'], price_data['volume'])
        
        assert isinstance(result, pd.Series)
    
    def test_obv_first_value(self, price_data):
        result = obv(price_data['close'], price_data['volume'])
        
        assert result.iloc[0] == 0


class TestADX:
    
    def test_adx_output_shape(self, price_data):
        adx_val, plus_di, minus_di = adx(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(adx_val) == len(price_data)
        assert len(plus_di) == len(price_data)
        assert len(minus_di) == len(price_data)
    
    def test_adx_range(self, price_data):
        adx_val, plus_di, minus_di = adx(price_data['high'], price_data['low'], price_data['close'])
        
        valid_adx = adx_val.dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 100


class TestMFI:
    
    def test_mfi_output_shape(self, price_data):
        result = mfi(price_data['high'], price_data['low'], price_data['close'], price_data['volume'])
        assert len(result) == len(price_data)
    
    def test_mfi_range(self, price_data):
        result = mfi(price_data['high'], price_data['low'], price_data['close'], price_data['volume'])
        
        valid_mfi = result.dropna()
        assert valid_mfi.min() >= 0
        assert valid_mfi.max() <= 100


class TestROC:
    
    def test_roc_output_shape(self, price_data):
        result = roc(price_data['close'], window=12)
        assert len(result) == len(price_data)
    
    def test_roc_nan_at_start(self, price_data):
        result = roc(price_data['close'], window=12)
        
        assert pd.isna(result.iloc[11])


class TestMomentum:
    
    def test_momentum_output_shape(self, price_data):
        result = momentum(price_data['close'], window=10)
        assert len(result) == len(price_data)
    
    def test_momentum_nan_at_start(self, price_data):
        result = momentum(price_data['close'], window=10)
        
        assert pd.isna(result.iloc[9])


class TestVWAP:
    
    def test_vwap_output_shape(self, price_data):
        result = vwap(price_data['high'], price_data['low'], price_data['close'], price_data['volume'])
        assert len(result) == len(price_data)
    
    def test_vwap_positive(self, price_data):
        result = vwap(price_data['high'], price_data['low'], price_data['close'], price_data['volume'])
        
        valid_result = result.dropna()
        assert (valid_result > 0).all()


class TestDonchianChannel:
    
    def test_donchian_output_shape(self, price_data):
        upper, middle, lower = donchian_channel(price_data['high'], price_data['low'], window=20)
        
        assert len(upper) == len(price_data)
        assert len(middle) == len(price_data)
        assert len(lower) == len(price_data)
    
    def test_donchian_bands_order(self, price_data):
        upper, middle, lower = donchian_channel(price_data['high'], price_data['low'], window=20)
        
        valid_mask = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()
    
    def test_donchian_middle_is_average(self, price_data):
        upper, middle, lower = donchian_channel(price_data['high'], price_data['low'], window=20)
        
        valid_mask = ~(pd.isna(upper) | pd.isna(lower))
        expected_middle = (upper[valid_mask] + lower[valid_mask]) / 2
        pd.testing.assert_series_equal(middle[valid_mask], expected_middle, check_names=False)


class TestKeltnerChannel:
    
    def test_keltner_output_shape(self, price_data):
        upper, middle, lower = keltner_channel(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(upper) == len(price_data)
        assert len(middle) == len(price_data)
        assert len(lower) == len(price_data)
    
    def test_keltner_bands_order(self, price_data):
        upper, middle, lower = keltner_channel(price_data['high'], price_data['low'], price_data['close'])
        
        valid_mask = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()


class TestTRIX:
    
    def test_trix_output_shape(self, price_data):
        result = trix(price_data['close'], window=14)
        assert len(result) == len(price_data)
    
    def test_trix_values(self, price_data):
        result = trix(price_data['close'], window=14)
        
        assert isinstance(result, pd.Series)


class TestDMI:
    
    def test_dmi_output_shape(self, price_data):
        pdi, mdi, dx, adx_val = dmi(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(pdi) == len(price_data)
        assert len(mdi) == len(price_data)
        assert len(dx) == len(price_data)
        assert len(adx_val) == len(price_data)


class TestTypicalPrice:
    
    def test_typical_price_output_shape(self, price_data):
        result = typical_price(price_data['high'], price_data['low'], price_data['close'])
        assert len(result) == len(price_data)
    
    def test_typical_price_calculation(self, price_data):
        result = typical_price(price_data['high'], price_data['low'], price_data['close'])
        
        expected = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        pd.testing.assert_series_equal(result, expected)


class TestPivotPoints:
    
    def test_pivot_points_output(self, price_data):
        result = pivot_points(price_data['high'], price_data['low'], price_data['close'])
        
        assert 'pivot' in result
        assert 'r1' in result
        assert 's1' in result
        assert 'r2' in result
        assert 's2' in result
        assert 'r3' in result
        assert 's3' in result
    
    def test_pivot_points_order(self, price_data):
        result = pivot_points(price_data['high'], price_data['low'], price_data['close'])
        
        valid_mask = ~(pd.isna(result['pivot']) | pd.isna(result['r1']) | pd.isna(result['s1']))
        assert (result['r1'][valid_mask] > result['pivot'][valid_mask]).all()
        assert (result['s1'][valid_mask] < result['pivot'][valid_mask]).all()


class TestZScore:
    
    def test_zscore_output_shape(self, price_data):
        result = z_score(price_data['close'], window=20)
        assert len(result) == len(price_data)
    
    def test_zscore_nan_at_start(self, price_data):
        result = z_score(price_data['close'], window=20)
        
        assert pd.isna(result.iloc[18])
        assert not pd.isna(result.iloc[19])


class TestVolatility:
    
    def test_volatility_output_shape(self, price_data):
        result = volatility(price_data['close'], window=20)
        assert len(result) == len(price_data)
    
    def test_volatility_positive(self, price_data):
        result = volatility(price_data['close'], window=20)
        
        valid_vol = result.dropna()
        assert (valid_vol > 0).all()
    
    def test_volatility_annualized(self, price_data):
        vol_annual = volatility(price_data['close'], window=20, annualize=True)
        vol_daily = volatility(price_data['close'], window=20, annualize=False)
        
        valid_mask = ~(pd.isna(vol_annual) | pd.isna(vol_daily))
        assert (vol_annual[valid_mask] > vol_daily[valid_mask]).all()


class TestSuperTrend:
    
    def test_supertrend_output_shape(self, price_data):
        st_val, direction = supertrend(price_data['high'], price_data['low'], price_data['close'])
        
        assert len(st_val) == len(price_data)
        assert len(direction) == len(price_data)
    
    def test_supertrend_direction_values(self, price_data):
        _, direction = supertrend(price_data['high'], price_data['low'], price_data['close'])
        
        valid_direction = direction.dropna()
        assert set(valid_direction.unique()).issubset({-1, 1})
    
    def test_supertrend_uptrend(self, uptrend_data):
        st_val, direction = supertrend(uptrend_data['high'], uptrend_data['low'], uptrend_data['close'])
        
        assert direction.iloc[-1] == 1
    
    def test_supertrend_downtrend(self, downtrend_data):
        st_val, direction = supertrend(downtrend_data['high'], downtrend_data['low'], downtrend_data['close'])
        
        assert direction.iloc[-1] == -1


class TestIndicatorReferenceValues:

    def test_obv_matches_reference_values(self):
        close = pd.Series([10.0, 11.0, 10.0, 12.0])
        volume = pd.Series([100.0, 200.0, 300.0, 400.0])

        result = obv(close, volume)
        expected = pd.Series([0.0, 200.0, -100.0, 300.0])
        pd.testing.assert_series_equal(result, expected)

    def test_vwap_matches_reference_values(self):
        high = pd.Series([11.0, 12.0, 13.0])
        low = pd.Series([9.0, 10.0, 11.0])
        close = pd.Series([10.0, 11.0, 12.0])
        volume = pd.Series([100.0, 200.0, 300.0])

        result = vwap(high, low, close, volume)
        expected = pd.Series([10.0, 10.666666666666666, 11.333333333333334])
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10, atol=1e-10)

    def test_mfi_matches_reference_values(self):
        high = pd.Series([10.0, 12.0, 11.0, 13.0])
        low = pd.Series([8.0, 9.0, 9.0, 10.0])
        close = pd.Series([9.0, 11.0, 10.0, 12.0])
        volume = pd.Series([100.0, 120.0, 80.0, 150.0])

        result = mfi(high, low, close, volume, window=2)
        expected = pd.Series([np.nan, 100.0, 61.53846153846154, 68.62745098039215])
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_trix_matches_reference_values(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = trix(close, window=2)
        expected = pd.Series([
            np.nan,
            29.629629629629626,
            45.714285714285715,
            41.830065359477125,
            33.58934971838196,
        ])
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_adx_matches_reference_values(self):
        high = pd.Series([10.0, 12.0, 13.0, 15.0, 16.0])
        low = pd.Series([8.0, 9.0, 11.0, 12.0, 14.0])
        close = pd.Series([9.0, 11.0, 12.0, 14.0, 15.0])

        adx_val, plus_di, minus_di = adx(high, low, close, window=2)

        expected_adx = pd.Series([np.nan, np.nan, 100.0, 100.0, 100.0])
        expected_plus_di = pd.Series([np.nan, 40.0, 60.0, 60.0, 60.0])
        expected_minus_di = pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(adx_val.values, expected_adx.values, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(plus_di.values, expected_plus_di.values, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(minus_di.values, expected_minus_di.values, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_dmi_matches_reference_values(self):
        high = pd.Series([10.0, 12.0, 13.0, 15.0, 16.0])
        low = pd.Series([8.0, 9.0, 11.0, 12.0, 14.0])
        close = pd.Series([9.0, 11.0, 12.0, 14.0, 15.0])

        pdi, mdi, dx, adx_val = dmi(high, low, close, window=2)

        expected_pdi = pd.Series([np.nan, 40.0, 60.0, 60.0, 60.0])
        expected_mdi = pd.Series([np.nan, 0.0, 0.0, 0.0, 0.0])
        expected_dx = pd.Series([np.nan, 100.0, 100.0, 100.0, 100.0])
        expected_adx = pd.Series([np.nan, np.nan, 100.0, 100.0, 100.0])

        np.testing.assert_allclose(pdi.values, expected_pdi.values, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(mdi.values, expected_mdi.values, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(dx.values, expected_dx.values, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(adx_val.values, expected_adx.values, rtol=1e-10, atol=1e-10, equal_nan=True)
