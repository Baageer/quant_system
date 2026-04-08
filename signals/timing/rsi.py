"""
RSI择时策略
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    参数:
        data: 价格序列
        window: RSI计算周期，默认14
    
    返回:
        RSI值序列 (0-100)
    """
    delta = data.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    avg_gain = avg_gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def rsi_signal(
    data: pd.DataFrame,
    window: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    price_col: str = 'close'
) -> pd.Series:
    """
    RSI超买超卖策略
    
    参数:
        data: 包含价格数据的DataFrame
        window: RSI计算周期
        oversold: 超卖阈值，默认30
        overbought: 超买阈值，默认70
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    rsi = calculate_rsi(prices, window)
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(len(data)):
        if pd.isna(rsi.iloc[i]):
            continue
        
        if rsi.iloc[i] < oversold:
            current_pos = 1
        elif rsi.iloc[i] > overbought:
            current_pos = -1
        elif current_pos == 1 and rsi.iloc[i] > 50:
            current_pos = 0
        elif current_pos == -1 and rsi.iloc[i] < 50:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


def rsi_cross_signal(
    data: pd.DataFrame,
    window: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    price_col: str = 'close'
) -> pd.Series:
    """
    RSI穿越策略
    
    RSI从超卖区域上穿30时做多，从超买区域下穿70时做空
    
    参数:
        data: 包含价格数据的DataFrame
        window: RSI计算周期
        oversold: 超卖阈值
        overbought: 超买阈值
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    rsi = calculate_rsi(prices, window)
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
            continue
        
        cross_up_oversold = rsi.iloc[i] > oversold and rsi.iloc[i-1] <= oversold
        cross_down_overbought = rsi.iloc[i] < overbought and rsi.iloc[i-1] >= overbought
        cross_up_middle = rsi.iloc[i] > 50 and rsi.iloc[i-1] <= 50
        cross_down_middle = rsi.iloc[i] < 50 and rsi.iloc[i-1] >= 50
        
        if cross_up_oversold:
            current_pos = 1
        elif cross_down_overbought:
            current_pos = -1
        elif current_pos == 1 and cross_down_middle:
            current_pos = 0
        elif current_pos == -1 and cross_up_middle:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


def rsi_divergence_signal(
    data: pd.DataFrame,
    window: int = 14,
    lookback: int = 5,
    price_col: str = 'close'
) -> pd.Series:
    """
    RSI背离策略
    
    底背离: 价格创新低但RSI未创新低 -> 做多
    顶背离: 价格创新高但RSI未创新高 -> 做空
    
    参数:
        data: 包含价格数据的DataFrame
        window: RSI计算周期
        lookback: 回溯周期用于判断高低点
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    rsi = calculate_rsi(prices, window)
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(lookback + 1, len(data)):
        if pd.isna(rsi.iloc[i]):
            continue
        
        price_low = prices.iloc[i-lookback:i+1].min()
        price_high = prices.iloc[i-lookback:i+1].max()
        rsi_low = rsi.iloc[i-lookback:i+1].min()
        rsi_high = rsi.iloc[i-lookback:i+1].max()
        
        prev_price_low = prices.iloc[i-2*lookback:i-lookback+1].min()
        prev_price_high = prices.iloc[i-2*lookback:i-lookback+1].max()
        prev_rsi_low = rsi.iloc[i-2*lookback:i-lookback+1].min()
        prev_rsi_high = rsi.iloc[i-2*lookback:i-lookback+1].max()
        
        if pd.isna(prev_rsi_low) or pd.isna(prev_rsi_high):
            continue
        
        bull_div = (price_low < prev_price_low) and (rsi_low > prev_rsi_low)
        bear_div = (price_high > prev_price_high) and (rsi_high < prev_rsi_high)
        
        if bull_div:
            current_pos = 1
        elif bear_div:
            current_pos = -1
        elif current_pos == 1 and rsi.iloc[i] > 70:
            current_pos = 0
        elif current_pos == -1 and rsi.iloc[i] < 30:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


class RSIStrategy:
    """RSI策略类"""
    
    def __init__(
        self,
        window: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        mode: str = 'standard'
    ):
        """
        参数:
            window: RSI计算周期
            oversold: 超卖阈值
            overbought: 超买阈值
            mode: 策略模式 'standard'(标准), 'cross'(穿越), 'divergence'(背离)
        """
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.mode = mode
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if self.mode == 'cross':
            return rsi_cross_signal(
                data,
                self.window,
                self.oversold,
                self.overbought
            )
        elif self.mode == 'divergence':
            return rsi_divergence_signal(data, self.window)
        return rsi_signal(
            data,
            self.window,
            self.oversold,
            self.overbought
        )
    
    def get_rsi_values(self, data: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """获取RSI值用于可视化"""
        return calculate_rsi(data[price_col], self.window)
