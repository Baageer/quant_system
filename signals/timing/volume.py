"""
成交量相关指标的买卖策略
"""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import obv, mfi, sma, ema


def volume_breakout_signal(data: pd.DataFrame,
                           volume_window: int = 20,
                           z_thresh: float = 2.0,
                           trend_window: int = 50,
                           price_col: str = 'close',
                           volume_col: str = 'volume') -> pd.Series:
    """Vectorized volume breakout with trend filter"""

    price = data[price_col]
    volume = data[volume_col]

    vol_ma = sma(volume, volume_window)
    vol_std = volume.rolling(volume_window).std()
    zscore = (volume - vol_ma) / vol_std

    trend = ema(price, trend_window)

    long_signal = (zscore > z_thresh) & (price > trend)
    short_signal = (zscore > z_thresh) & (price < trend)

    signal = pd.Series(0, index=data.index)
    signal[long_signal] = 1
    signal[short_signal] = -1

    return signal


def obv_signal(data: pd.DataFrame,
               window: int = 20,
               price_col: str = 'close',
               volume_col: str = 'volume') -> pd.Series:
    """OBV slope-based signal"""

    price = data[price_col]
    volume = data[volume_col]

    obv_val = obv(price, volume)
    obv_slope = obv_val.diff()

    price_trend = price > sma(price, window)

    long_signal = (obv_slope > 0) & price_trend
    short_signal = (obv_slope < 0) & (~price_trend)

    signal = pd.Series(0, index=data.index)
    signal[long_signal] = 1
    signal[short_signal] = -1

    return signal


def mfi_signal(data: pd.DataFrame,
               window: int = 14,
               overbought: int = 80,
               oversold: int = 20,
               trend_window: int = 50,
               high_col: str = 'high',
               low_col: str = 'low',
               close_col: str = 'close',
               volume_col: str = 'volume') -> pd.Series:
    """MFI reversal with confirmation"""

    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    volume = data[volume_col]
    
    mfi_val = mfi(high, low, close, volume, window)
    price = data[close_col]
    trend = ema(price, trend_window)

    long_signal = (mfi_val.shift(1) < oversold) & (mfi_val > oversold) & (price > trend)
    short_signal = (mfi_val.shift(1) > overbought) & (mfi_val < overbought) & (price < trend)

    signal = pd.Series(0, index=data.index)
    signal[long_signal] = 1
    signal[short_signal] = -1

    return signal


def volume_price_divergence_signal(data: pd.DataFrame,
                                   window: int = 20,
                                   price_col: str = 'close',
                                   volume_col: str = 'volume') -> pd.Series:
    """True divergence detection"""

    price = data[price_col]
    volume = data[volume_col]

    price_high = price.rolling(window).max()
    volume_high = volume.rolling(window).max()

    price_low = price.rolling(window).min()
    volume_low = volume.rolling(window).min()

    bearish_div = (price >= price_high) & (volume < volume_high.shift(1))
    bullish_div = (price <= price_low) & (volume > volume_low.shift(1))

    signal = pd.Series(0, index=data.index)
    signal[bullish_div] = 1
    signal[bearish_div] = -1

    return signal


class VolumeBreakoutStrategy:
    """成交量突破策略类"""
    
    def __init__(
        self,
        volume_window: int = 20,
        z_thresh: float = 2.0,
        trend_window: int = 50,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.volume_window = volume_window
        self.z_thresh = z_thresh
        self.trend_window = trend_window
        self.price_col = price_col
        self.volume_col = volume_col
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        return volume_breakout_signal(
            data,
            self.volume_window,
            self.z_thresh,
            self.trend_window,
            self.price_col,
            self.volume_col
        )
    
    def get_volume_ma(self, data: pd.DataFrame) -> pd.Series:
        """获取成交量移动平均值用于可视化"""
        volume = data[self.volume_col]
        return sma(volume, self.volume_window)


class OBVStrategy:
    """OBV指标策略类"""
    
    def __init__(
        self,
        obv_window: int = 20,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.obv_window = obv_window
        self.price_col = price_col
        self.volume_col = volume_col
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        return obv_signal(
            data,
            self.obv_window,
            self.price_col,
            self.volume_col
        )
    
    def get_obv_values(self, data: pd.DataFrame) -> dict:
        """获取OBV值用于可视化"""
        prices = data[self.price_col]
        volume = data[self.volume_col]
        obv_value = obv(prices, volume)
        return {
            'obv': obv_value,
            'obv_ma': sma(obv_value, self.obv_window)
        }


class MFIStrategy:
    """MFI资金流量指标策略类"""
    
    def __init__(
        self,
        mfi_window: int = 14,
        overbought: int = 80,
        oversold: int = 20,
        trend_window: int = 50,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.mfi_window = mfi_window
        self.overbought = overbought
        self.oversold = oversold
        self.trend_window = trend_window
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        return mfi_signal(
            data,
            self.mfi_window,
            self.overbought,
            self.oversold,
            self.trend_window,
            self.high_col,
            self.low_col,
            self.close_col,
            self.volume_col
        )
    
    def get_mfi_value(self, data: pd.DataFrame) -> pd.Series:
        """获取MFI值用于可视化"""
        high = data[self.high_col]
        low = data[self.low_col]
        close = data[self.close_col]
        volume = data[self.volume_col]
        return mfi(high, low, close, volume, self.mfi_window)


class VolumePriceDivergenceStrategy:
    """量价背离策略类"""
    
    def __init__(
        self,
        window: int = 20,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.window = window
        self.price_col = price_col
        self.volume_col = volume_col
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        return volume_price_divergence_signal(
            data,
            self.window,
            self.price_col,
            self.volume_col
        )
    
    def get_divergence_values(self, data: pd.DataFrame) -> dict:
        """获取量价背离值用于可视化"""
        prices = data[self.price_col]
        volume = data[self.volume_col]
        return {
            'price_change': prices.pct_change(self.window),
            'volume_change': volume.pct_change(self.window)
        }
