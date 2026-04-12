"""
成交量相关指标的买卖策略
"""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import obv, mfi, sma, ema


def volume_breakout_signal(
    data: pd.DataFrame,
    volume_window: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.Series:
    """
    成交量突破策略
    
    当成交量突破近期平均成交量时产生信号
    
    参数:
        data: 包含价格和成交量数据的DataFrame
        volume_window: 成交量平均周期，默认20
        price_col: 价格列名，默认'close'
        volume_col: 成交量列名，默认'volume'
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    volume = data[volume_col]
    
    # 计算成交量移动平均
    volume_ma = sma(volume, volume_window)
    
    # 计算成交量突破
    volume_breakout = volume > 1.5 * volume_ma
    volume_breakdown = volume < 0.5 * volume_ma
    
    signal = pd.Series(0, index=data.index)
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(volume_ma.iloc[i]):
            continue
        
        # 成交量突破且价格上涨，做多
        if volume_breakout.iloc[i] and prices.iloc[i] > prices.iloc[i-1] if i > 0 else False:
            current_pos = 1
        # 成交量突破且价格下跌，做空
        elif volume_breakout.iloc[i] and prices.iloc[i] < prices.iloc[i-1] if i > 0 else False:
            current_pos = -1
        # 成交量极度萎缩，平仓
        elif volume_breakdown.iloc[i]:
            current_pos = 0
        
        position.iloc[i] = current_pos
    
    return position


def obv_signal(
    data: pd.DataFrame,
    obv_window: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.Series:
    """
    OBV指标策略
    
    当OBV与价格背离时产生信号
    
    参数:
        data: 包含价格和成交量数据的DataFrame
        obv_window: OBV移动平均周期，默认20
        price_col: 价格列名，默认'close'
        volume_col: 成交量列名，默认'volume'
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    volume = data[volume_col]
    
    # 计算OBV
    obv_value = obv(prices, volume)
    # 计算OBV移动平均
    obv_ma = sma(obv_value, obv_window)
    
    signal = pd.Series(0, index=data.index)
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(obv_ma.iloc[i]) or i < 1:
            continue
        
        # OBV金叉
        obv_golden_cross = obv_value.iloc[i] > obv_ma.iloc[i] and obv_value.iloc[i-1] <= obv_ma.iloc[i-1]
        # OBV死叉
        obv_death_cross = obv_value.iloc[i] < obv_ma.iloc[i] and obv_value.iloc[i-1] >= obv_ma.iloc[i-1]
        
        if obv_golden_cross:
            current_pos = 1
        elif obv_death_cross:
            current_pos = -1
        
        position.iloc[i] = current_pos
    
    return position


def mfi_signal(
    data: pd.DataFrame,
    mfi_window: int = 14,
    overbought: int = 80,
    oversold: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.Series:
    """
    MFI资金流量指标策略
    
    基于MFI超买超卖产生信号
    
    参数:
        data: 包含价格和成交量数据的DataFrame
        mfi_window: MFI计算周期，默认14
        overbought: 超买阈值，默认80
        oversold: 超卖阈值，默认20
        high_col: 最高价列名，默认'high'
        low_col: 最低价列名，默认'low'
        close_col: 收盘价列名，默认'close'
        volume_col: 成交量列名，默认'volume'
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    volume = data[volume_col]
    
    # 计算MFI
    mfi_value = mfi(high, low, close, volume, mfi_window)
    
    signal = pd.Series(0, index=data.index)
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(mfi_value.iloc[i]):
            continue
        
        # MFI超卖，做多
        if mfi_value.iloc[i] < oversold:
            current_pos = 1
        # MFI超买，做空
        elif mfi_value.iloc[i] > overbought:
            current_pos = -1
        
        position.iloc[i] = current_pos
    
    return position


def volume_price_divergence_signal(
    data: pd.DataFrame,
    window: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.Series:
    """
    量价背离策略
    
    当价格与成交量走势背离时产生信号
    
    参数:
        data: 包含价格和成交量数据的DataFrame
        window: 计算周期，默认20
        price_col: 价格列名，默认'close'
        volume_col: 成交量列名，默认'volume'
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    volume = data[volume_col]
    
    # 计算价格和成交量的变化率
    price_change = prices.pct_change(window)
    volume_change = volume.pct_change(window)
    
    signal = pd.Series(0, index=data.index)
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(price_change.iloc[i]) or pd.isna(volume_change.iloc[i]):
            continue
        
        # 价格上涨，成交量放大，做多
        if price_change.iloc[i] > 0 and volume_change.iloc[i] > 0.2:
            current_pos = 1
        # 价格下跌，成交量放大，做空
        elif price_change.iloc[i] < 0 and volume_change.iloc[i] > 0.2:
            current_pos = -1
        # 量价背离，平仓
        elif (price_change.iloc[i] > 0 and volume_change.iloc[i] < -0.2) or \
             (price_change.iloc[i] < 0 and volume_change.iloc[i] < -0.2):
            current_pos = 0
        
        position.iloc[i] = current_pos
    
    return position


class VolumeBreakoutStrategy:
    """成交量突破策略类"""
    
    def __init__(
        self,
        volume_window: int = 20,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.volume_window = volume_window
        self.price_col = price_col
        self.volume_col = volume_col
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        return volume_breakout_signal(
            data,
            self.volume_window,
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
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ):
        self.mfi_window = mfi_window
        self.overbought = overbought
        self.oversold = oversold
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
