"""
止盈策略
当价格超过止盈价格时触发止盈信号
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum


class ProfitType(Enum):
    ABSOLUTE = "absolute"
    TRAILING = "trailing"
    ATR_BASED = "atr"


def calculate_stop_profit_price(
    entry_price: float,
    profit_pct: float,
    profit_type: str = "absolute",
    lowest_price: Optional[float] = None,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0
) -> float:
    """
    计算止盈价格
    
    参数:
        entry_price: 入场价格
        profit_pct: 止盈比例 (如 0.06 表示 6%)
        profit_type: 止盈类型
            - "absolute": 固定止盈，从入场价上涨一定比例
            - "trailing": 移动止盈，从最低价上涨一定比例（用于做空）
            - "atr": ATR止盈，从入场价上涨ATR的倍数
        lowest_price: 最低价（移动止盈时使用）
        atr: ATR值（ATR止盈时使用）
        atr_multiplier: ATR倍数
    
    返回:
        止盈价格
    """
    if profit_type == "absolute":
        return entry_price * (1 + profit_pct)
    elif profit_type == "trailing":
        base_price = lowest_price if lowest_price is not None else entry_price
        return base_price * (1 + profit_pct)
    elif profit_type == "atr":
        if atr is None:
            return entry_price * (1 + profit_pct)
        return entry_price + atr * atr_multiplier
    else:
        return entry_price * (1 + profit_pct)


def stop_profit_signal(
    data: pd.DataFrame,
    entry_price: float,
    profit_pct: float = 0.06,
    profit_type: str = "absolute",
    price_col: str = 'close',
    low_col: str = 'low'
) -> pd.Series:
    """
    生成止盈信号
    
    参数:
        data: 包含价格数据的DataFrame
        entry_price: 入场价格
        profit_pct: 止盈比例
        profit_type: 止盈类型
        price_col: 收盘价列名
        low_col: 最低价列名
    
    返回:
        止盈信号序列: 1=触发止盈, 0=未触发
    """
    prices = data[price_col]
    signals = pd.Series(0, index=data.index)
    
    if profit_type == "trailing":
        lowest_price = entry_price
        for i in range(len(data)):
            current_low = data[low_col].iloc[i]
            lowest_price = min(lowest_price, current_low)
            profit_price = calculate_stop_profit_price(
                entry_price, profit_pct, profit_type, lowest_price
            )
            if prices.iloc[i] >= profit_price:
                signals.iloc[i] = 1
    else:
        profit_price = calculate_stop_profit_price(entry_price, profit_pct, profit_type)
        signals = (prices >= profit_price).astype(int)
    
    return signals


def calculate_atr(
    data: pd.DataFrame,
    window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    计算ATR (Average True Range)
    
    参数:
        data: 包含价格数据的DataFrame
        window: ATR计算周期
        high_col: 最高价列名
        low_col: 最低价列名
        close_col: 收盘价列名
    
    返回:
        ATR序列
    """
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    
    return atr


class StopProfitStrategy:
    """止盈策略类"""
    
    def __init__(
        self,
        profit_pct: float = 0.06,
        profit_type: str = "absolute",
        atr_window: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        参数:
            profit_pct: 止盈比例，默认6%
            profit_type: 止盈类型
                - "absolute": 固定止盈
                - "trailing": 移动止盈（回撤止盈）
                - "atr": ATR止盈
            atr_window: ATR计算周期
            atr_multiplier: ATR倍数
        """
        self.profit_pct = profit_pct
        self.profit_type = profit_type
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        
        self.entry_price: Optional[float] = None
        self.highest_price: Optional[float] = None
        self.profit_price: Optional[float] = None
        self.position_active: bool = False
        self.profit_reached: bool = False
    
    def set_position(self, entry_price: float):
        """
        设置持仓信息
        
        参数:
            entry_price: 入场价格
        """
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.position_active = True
        self.profit_reached = False
        
        if self.profit_type == "absolute":
            self.profit_price = entry_price * (1 + self.profit_pct)
        elif self.profit_type == "atr":
            self.profit_price = None
    
    def clear_position(self):
        """清除持仓信息"""
        self.entry_price = None
        self.highest_price = None
        self.profit_price = None
        self.position_active = False
        self.profit_reached = False
    
    def update_profit_price(self, current_price: float, current_high: float, atr: Optional[float] = None):
        """
        更新止盈价格
        
        参数:
            current_price: 当前价格
            current_high: 当前最高价
            atr: 当前ATR值
        """
        if not self.position_active or self.entry_price is None:
            return
        
        if current_high > self.highest_price:
            self.highest_price = current_high
        
        if self.profit_type == "trailing":
            if self.highest_price >= self.entry_price * (1 + self.profit_pct):
                self.profit_reached = True
                self.profit_price = self.highest_price * (1 - self.profit_pct)
        elif self.profit_type == "atr" and atr is not None:
            self.profit_price = self.entry_price + atr * self.atr_multiplier
    
    def check_stop_profit(self, current_price: float) -> bool:
        """
        检查是否触发止盈
        
        参数:
            current_price: 当前价格
        
        返回:
            是否触发止盈
        """
        if not self.position_active:
            return False
        
        if self.profit_type == "trailing":
            if self.profit_reached and self.profit_price is not None:
                return current_price <= self.profit_price
            return False
        else:
            if self.profit_price is None:
                return False
            return current_price >= self.profit_price
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成止盈信号
        
        参数:
            data: 包含价格数据的DataFrame
        
        返回:
            止盈信号序列: -1=止盈卖出, 0=无操作
        """
        signals = pd.Series(0, index=data.index)
        
        if self.entry_price is None:
            return signals
        
        prices = data['close']
        highs = data['high'] if 'high' in data.columns else prices
        
        if self.profit_type == "atr":
            atr = calculate_atr(data, self.atr_window)
        else:
            atr = None
        
        for i in range(len(data)):
            if not self.position_active:
                break
            
            current_price = prices.iloc[i]
            current_high = highs.iloc[i]
            current_atr = atr.iloc[i] if atr is not None else None
            
            self.update_profit_price(current_price, current_high, current_atr)
            
            if self.check_stop_profit(current_price):
                signals.iloc[i] = -1
                break
        
        return signals
    
    def get_profit_info(self) -> Dict[str, Any]:
        """
        获取止盈信息
        
        返回:
            止盈信息字典
        """
        return {
            'entry_price': self.entry_price,
            'highest_price': self.highest_price,
            'profit_price': self.profit_price,
            'profit_pct': self.profit_pct,
            'profit_type': self.profit_type,
            'position_active': self.position_active,
            'profit_reached': self.profit_reached,
            'current_profit_pct': (
                (self.highest_price - self.entry_price) / self.entry_price
                if self.entry_price and self.highest_price else None
            )
        }
