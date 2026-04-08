"""
MACD择时策略
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标
    
    参数:
        data: 价格序列
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26
        signal_period: 信号线周期，默认9
    
    返回:
        (MACD线, 信号线, MACD柱状图)
    """
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def macd_cross_signal(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = 'close'
) -> pd.Series:
    """
    MACD金叉死叉策略
    
    金叉: MACD线上穿信号线 -> 做多
    死叉: MACD线下穿信号线 -> 做空
    
    参数:
        data: 包含价格数据的DataFrame
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    macd_line, signal_line, histogram = calculate_macd(
        prices, fast_period, slow_period, signal_period
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        
        golden_cross = (macd_line.iloc[i] > signal_line.iloc[i]) and \
                       (macd_line.iloc[i-1] <= signal_line.iloc[i-1])
        death_cross = (macd_line.iloc[i] < signal_line.iloc[i]) and \
                      (macd_line.iloc[i-1] >= signal_line.iloc[i-1])
        
        if golden_cross:
            current_pos = 1
        elif death_cross:
            current_pos = -1
            
        position.iloc[i] = current_pos
    
    return position


def macd_histogram_signal(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = 'close'
) -> pd.Series:
    """
    MACD柱状图策略
    
    柱状图由负转正 -> 做多
    柱状图由正转负 -> 做空
    
    参数:
        data: 包含价格数据的DataFrame
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    macd_line, signal_line, histogram = calculate_macd(
        prices, fast_period, slow_period, signal_period
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(histogram.iloc[i]):
            continue
        
        cross_up = histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0
        cross_down = histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0
        
        if cross_up:
            current_pos = 1
        elif cross_down:
            current_pos = -1
            
        position.iloc[i] = current_pos
    
    return position


def macd_zero_axis_signal(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = 'close'
) -> pd.Series:
    """
    MACD零轴策略
    
    MACD线由下上穿零轴 -> 做多
    MACD线由上下穿零轴 -> 做空
    
    参数:
        data: 包含价格数据的DataFrame
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    macd_line, signal_line, histogram = calculate_macd(
        prices, fast_period, slow_period, signal_period
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(macd_line.iloc[i]):
            continue
        
        cross_up_zero = macd_line.iloc[i] > 0 and macd_line.iloc[i-1] <= 0
        cross_down_zero = macd_line.iloc[i] < 0 and macd_line.iloc[i-1] >= 0
        
        if cross_up_zero:
            current_pos = 1
        elif cross_down_zero:
            current_pos = -1
            
        position.iloc[i] = current_pos
    
    return position


def macd_combined_signal(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = 'close'
) -> pd.Series:
    """
    MACD组合策略
    
    结合金叉和零轴位置:
    - 零轴上方金叉: 强做多信号
    - 零轴下方金叉: 弱做多信号
    - 零轴上方死叉: 弱做空信号
    - 零轴下方死叉: 强做空信号
    
    参数:
        data: 包含价格数据的DataFrame
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        price_col: 价格列名
    
    返回:
        信号序列: 1=做多, -1=做空, 0=空仓
    """
    prices = data[price_col]
    macd_line, signal_line, histogram = calculate_macd(
        prices, fast_period, slow_period, signal_period
    )
    
    position = pd.Series(0, index=data.index)
    current_pos = 0
    
    for i in range(1, len(data)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        
        golden_cross = (macd_line.iloc[i] > signal_line.iloc[i]) and \
                       (macd_line.iloc[i-1] <= signal_line.iloc[i-1])
        death_cross = (macd_line.iloc[i] < signal_line.iloc[i]) and \
                      (macd_line.iloc[i-1] >= signal_line.iloc[i-1])
        
        above_zero = macd_line.iloc[i] > 0
        
        if golden_cross:
            current_pos = 1
        elif death_cross:
            current_pos = -1
            
        position.iloc[i] = current_pos
    
    return position


class MACDStrategy:
    """MACD策略类"""
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        mode: str = 'cross'
    ):
        """
        参数:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            mode: 策略模式 'cross'(金叉死叉), 'histogram'(柱状图), 
                  'zero_axis'(零轴), 'combined'(组合)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.mode = mode
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if self.mode == 'histogram':
            return macd_histogram_signal(
                data,
                self.fast_period,
                self.slow_period,
                self.signal_period
            )
        elif self.mode == 'zero_axis':
            return macd_zero_axis_signal(
                data,
                self.fast_period,
                self.slow_period,
                self.signal_period
            )
        elif self.mode == 'combined':
            return macd_combined_signal(
                data,
                self.fast_period,
                self.slow_period,
                self.signal_period
            )
        return macd_cross_signal(
            data,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
    
    def get_macd_values(
        self, 
        data: pd.DataFrame, 
        price_col: str = 'close'
    ) -> dict:
        """获取MACD值用于可视化"""
        macd_line, signal_line, histogram = calculate_macd(
            data[price_col],
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
