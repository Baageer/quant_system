"""
双均线交叉择时策略
"""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_ma(data: pd.Series, window: int) -> pd.Series:
    """计算移动平均线"""
    return data.rolling(window=window, min_periods=window).mean()


def ma_cross_signal(
    data: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    price_col: str = 'close'
) -> pd.Series:
    """
    双均线交叉策略
    
    参数:
        data: 包含价格数据的DataFrame
        short_window: 短期均线周期，默认5
        long_window: 长期均线周期，默认20
        price_col: 价格列名，默认'close'
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    
    short_ma = calculate_ma(prices, short_window)
    long_ma = calculate_ma(prices, long_window)
    
    signal = pd.Series(0, index=data.index)
    
    golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if golden_cross.iloc[i]:
            current_pos = 1
        elif death_cross.iloc[i]:
            current_pos = -1
        position.iloc[i] = current_pos
    
    return position


def ma_cross_with_filter(
    data: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    ma_filter_window: int = 60,
    price_col: str = 'close'
) -> pd.Series:
    """
    带趋势过滤的双均线交叉策略
    
    仅在长期趋势向上时做多，趋势向下时做空
    
    参数:
        data: 包含价格数据的DataFrame
        short_window: 短期均线周期
        long_window: 长期均线周期
        ma_filter_window: 趋势过滤均线周期
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    
    short_ma = calculate_ma(prices, short_window)
    long_ma = calculate_ma(prices, long_window)
    filter_ma = calculate_ma(prices, ma_filter_window)
    
    signal = pd.Series(0, index=data.index)
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(filter_ma.iloc[i]):
            continue
            
        trend_up = prices.iloc[i] > filter_ma.iloc[i]
        trend_down = prices.iloc[i] < filter_ma.iloc[i]
        
        golden_cross = (short_ma.iloc[i] > long_ma.iloc[i]) and \
                       (short_ma.iloc[i-1] <= long_ma.iloc[i-1]) if i > 0 else False
        death_cross = (short_ma.iloc[i] < long_ma.iloc[i]) and \
                      (short_ma.iloc[i-1] >= long_ma.iloc[i-1]) if i > 0 else False
        
        if golden_cross and trend_up:
            current_pos = 1
        elif death_cross and trend_down:
            current_pos = -1
        elif death_cross and current_pos == 1:
            current_pos = 0
        elif golden_cross and current_pos == -1:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


class MACrossStrategy:
    """双均线交叉策略类"""
    
    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        use_filter: bool = False,
        ma_filter_window: int = 60
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.use_filter = use_filter
        self.ma_filter_window = ma_filter_window
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if self.use_filter:
            return ma_cross_with_filter(
                data,
                self.short_window,
                self.long_window,
                self.ma_filter_window
            )
        return ma_cross_signal(
            data,
            self.short_window,
            self.long_window
        )
    
    def get_ma_values(self, data: pd.DataFrame, price_col: str = 'close') -> dict:
        """获取均线值用于可视化"""
        prices = data[price_col]
        return {
            'short_ma': calculate_ma(prices, self.short_window),
            'long_ma': calculate_ma(prices, self.long_window)
        }
