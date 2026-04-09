"""
布林带通道突破策略
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_bollinger_bands(
    data: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带指标
    
    参数:
        data: 价格序列
        window: 移动平均周期，默认20
        num_std: 标准差倍数，默认2.0
    
    返回:
        (上轨, 中轨, 下轨)
    """
    middle_band = data.rolling(window=window, min_periods=window).mean()
    std = data.rolling(window=window, min_periods=window).std()
    
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std
    
    return upper_band, middle_band, lower_band


def bollinger_breakout_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = 'close'
) -> pd.Series:
    """
    布林带突破策略
    
    价格突破上轨 -> 做多
    价格跌破下轨 -> 做空
    价格回归中轨 -> 平仓
    
    参数:
        data: 包含价格数据的DataFrame
        window: 移动平均周期
        num_std: 标准差倍数
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = calculate_bollinger_bands(
        prices, window, num_std
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        
        price = prices.iloc[i]
        prev_price = prices.iloc[i-1]
        
        break_upper = price > upper_band.iloc[i] and prev_price <= upper_band.iloc[i-1]
        break_lower = price < lower_band.iloc[i] and prev_price >= lower_band.iloc[i-1]
        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i-1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i-1]
        
        if break_upper:
            current_pos = 1
        elif break_lower:
            current_pos = -1
        elif current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


def bollinger_mean_reversion_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = 'close'
) -> pd.Series:
    """
    布林带均值回归策略
    
    价格触及上轨后回落 -> 做空
    价格触及下轨后反弹 -> 做多
    价格回归中轨 -> 平仓
    
    参数:
        data: 包含价格数据的DataFrame
        window: 移动平均周期
        num_std: 标准差倍数
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = calculate_bollinger_bands(
        prices, window, num_std
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        
        price = prices.iloc[i]
        prev_price = prices.iloc[i-1]
        
        touch_upper = price >= upper_band.iloc[i]
        touch_lower = price <= lower_band.iloc[i]
        pullback_from_upper = prev_price >= upper_band.iloc[i-1] and price < upper_band.iloc[i]
        bounce_from_lower = prev_price <= lower_band.iloc[i-1] and price > lower_band.iloc[i]
        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i-1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i-1]
        
        if bounce_from_lower:
            current_pos = 1
        elif pullback_from_upper:
            current_pos = -1
        elif current_pos == 1 and cross_middle_up:
            current_pos = 0
        elif current_pos == -1 and cross_middle_down:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


def bollinger_squeeze_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    squeeze_threshold: float = 0.02,
    price_col: str = 'close'
) -> pd.Series:
    """
    布林带收窄突破策略
    
    当布林带带宽收窄到阈值以下时，等待突破方向
    突破上轨 -> 做多
    突破下轨 -> 做空
    
    参数:
        data: 包含价格数据的DataFrame
        window: 移动平均周期
        num_std: 标准差倍数
        squeeze_threshold: 带宽收窄阈值（相对于中轨的比例）
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = calculate_bollinger_bands(
        prices, window, num_std
    )
    
    bandwidth = (upper_band - lower_band) / middle_band
    # print(bandwidth.describe())
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    in_squeeze = False
    
    for i in range(1, len(data)):
        if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        
        price = prices.iloc[i]
        prev_price = prices.iloc[i-1]
        
        if bandwidth.iloc[i] < squeeze_threshold:
            in_squeeze = True
        else:
            if in_squeeze:
                break_upper = price > upper_band.iloc[i] and prev_price <= upper_band.iloc[i-1]
                break_lower = price < lower_band.iloc[i] and prev_price >= lower_band.iloc[i-1]
                
                if break_upper:
                    current_pos = 1
                elif break_lower:
                    current_pos = -1
                    
                in_squeeze = False
        
        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i-1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i-1]
        
        if current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


def bollinger_double_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    confirm_bars: int = 2,
    price_col: str = 'close'
) -> pd.Series:
    """
    布林带双确认突破策略
    
    价格连续N根K线收于上轨上方 -> 做多
    价格连续N根K线收于下轨下方 -> 做空
    价格回归中轨 -> 平仓
    
    参数:
        data: 包含价格数据的DataFrame
        window: 移动平均周期
        num_std: 标准差倍数
        confirm_bars: 确认K线数量
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = calculate_bollinger_bands(
        prices, window, num_std
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    above_upper_count = 0
    below_lower_count = 0
    
    for i in range(1, len(data)):
        if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        
        price = prices.iloc[i]
        
        if price > upper_band.iloc[i]:
            above_upper_count += 1
            below_lower_count = 0
        elif price < lower_band.iloc[i]:
            below_lower_count += 1
            above_upper_count = 0
        else:
            above_upper_count = 0
            below_lower_count = 0
        
        if above_upper_count >= confirm_bars:
            current_pos = 1
        elif below_lower_count >= confirm_bars:
            current_pos = -1
        
        cross_middle_up = price > middle_band.iloc[i] and prices.iloc[i-1] <= middle_band.iloc[i-1]
        cross_middle_down = price < middle_band.iloc[i] and prices.iloc[i-1] >= middle_band.iloc[i-1]
        
        if current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0
            
        position.iloc[i] = current_pos
    
    return position


class BollingerBandsStrategy:
    """布林带策略类"""
    
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        mode: str = 'breakout',
        squeeze_threshold: float = 0.02,
        confirm_bars: int = 2
    ):
        """
        参数:
            window: 移动平均周期
            num_std: 标准差倍数
            mode: 策略模式 'breakout'(突破), 'mean_reversion'(均值回归),
                  'squeeze'(收窄突破), 'double'(双确认突破)
            squeeze_threshold: 带宽收窄阈值（仅squeeze模式使用）
            confirm_bars: 确认K线数量（仅double模式使用）
        """
        self.window = window
        self.num_std = num_std
        self.mode = mode
        self.squeeze_threshold = squeeze_threshold
        self.confirm_bars = confirm_bars
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if self.mode == 'mean_reversion':
            return bollinger_mean_reversion_signal(
                data,
                self.window,
                self.num_std
            )
        elif self.mode == 'squeeze':
            return bollinger_squeeze_signal(
                data,
                self.window,
                self.num_std,
                self.squeeze_threshold
            )
        elif self.mode == 'double':
            return bollinger_double_signal(
                data,
                self.window,
                self.num_std,
                self.confirm_bars
            )
        return bollinger_breakout_signal(
            data,
            self.window,
            self.num_std
        )
    
    def get_bollinger_values(
        self, 
        data: pd.DataFrame, 
        price_col: str = 'close'
    ) -> dict:
        """获取布林带值用于可视化"""
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            data[price_col],
            self.window,
            self.num_std
        )
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }
