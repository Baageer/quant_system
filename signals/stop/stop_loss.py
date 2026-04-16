"""
止损策略
当价格跌破止损价格时触发止损信号
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from enum import Enum


class LossType(Enum):
    ABSOLUTE = "absolute"
    TRAILING = "trailing"
    ATR_BASED = "atr"
    HOLDING_DAY = "holding_day"


def _is_holding_day_type(loss_type: str) -> bool:
    return loss_type in {LossType.HOLDING_DAY.value, "holding_days"}


def calculate_stop_loss_price(
    entry_price: float,
    loss_pct: float,
    loss_type: str = "absolute",
    highest_price: Optional[float] = None,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0
) -> float:
    """
    计算止损价格
    
    参数:
        entry_price: 入场价格
        loss_pct: 止损比例 (如 0.06 表示 6%)
        loss_type: 止损类型
            - "absolute": 固定止损，从入场价下跌一定比例
            - "trailing": 移动止损，从最高价下跌一定比例
            - "atr": ATR止损，从入场价下跌ATR的倍数
        highest_price: 最高价（移动止损时使用）
        atr: ATR值（ATR止损时使用）
        atr_multiplier: ATR倍数
    
    返回:
        止损价格
    """
    if loss_type == "absolute":
        return entry_price * (1 - loss_pct)
    elif loss_type == "trailing":
        base_price = highest_price if highest_price is not None else entry_price
        return base_price * (1 - loss_pct)
    elif loss_type == "atr":
        if atr is None:
            return entry_price * (1 - loss_pct)
        return entry_price - atr * atr_multiplier
    elif _is_holding_day_type(loss_type):
        return np.nan
    else:
        return entry_price * (1 - loss_pct)


def stop_loss_signal(
    data: pd.DataFrame,
    entry_price: float,
    loss_pct: float = 0.06,
    loss_type: str = "absolute",
    holding_day: Optional[int] = None,
    price_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
) -> pd.Series:
    """
    生成止损信号
    
    参数:
        data: 包含价格数据的DataFrame
        entry_price: 入场价格
        loss_pct: 止损比例
        loss_type: 止损类型
        price_col: 收盘价列名
        high_col: 最高价列名
    
    返回:
        止损信号序列: 1=触发止损, 0=未触发
    """
    prices = data[price_col]
    signals = pd.Series(0, index=data.index)
    
    if _is_holding_day_type(loss_type):
        if holding_day is None or int(holding_day) <= 0:
            raise ValueError("holding_day must be a positive integer when loss_type='holding_day'.")
        signals.iloc[int(holding_day):] = 1
    elif loss_type == "trailing":
        highest_price = entry_price
        for i in range(len(data)):
            current_high = data[high_col].iloc[i]
            highest_price = max(highest_price, current_high)
            stop_price = calculate_stop_loss_price(
                entry_price, loss_pct, loss_type, highest_price
            )
            if prices.iloc[i] <= stop_price:
                signals.iloc[i] = 1
    elif loss_type == "atr":
        atr_series = calculate_atr(
            data,
            window=atr_window,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        )
        valid_atr = atr_series == atr_series
        stop_prices = entry_price - atr_series * atr_multiplier
        signals.loc[valid_atr] = (prices.loc[valid_atr] <= stop_prices.loc[valid_atr]).astype(int)
    else:
        stop_price = calculate_stop_loss_price(entry_price, loss_pct, loss_type)
        signals = (prices <= stop_price).astype(int)
    
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


class StopLossStrategy:
    """止损策略类"""
    
    def __init__(
        self,
        loss_pct: float = 0.06,
        loss_type: str = "absolute",
        holding_day: Optional[int] = None,
        atr_window: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        参数:
            loss_pct: 止损比例，默认6%
            loss_type: 止损类型
                - "absolute": 固定止损
                - "trailing": 移动止损
                - "atr": ATR止损
            atr_window: ATR计算周期
            atr_multiplier: ATR倍数
        """
        self.loss_pct = loss_pct
        self.loss_type = loss_type
        self.holding_day = int(holding_day) if holding_day is not None else None
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier

        if _is_holding_day_type(self.loss_type):
            if self.holding_day is None or self.holding_day <= 0:
                raise ValueError(
                    "holding_day must be a positive integer when loss_type='holding_day'."
                )
        
        self.entry_price: Optional[float] = None
        self.entry_date: Optional[pd.Timestamp] = None
        self.highest_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.position_active: bool = False
    
    def set_position(
        self,
        entry_price: float,
        entry_date: Optional[pd.Timestamp] = None,
    ):
        """
        设置持仓信息
        
        参数:
            entry_price: 入场价格
        """
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.highest_price = entry_price
        self.position_active = True
        
        if self.loss_type == "absolute":
            self.stop_price = entry_price * (1 - self.loss_pct)
        elif _is_holding_day_type(self.loss_type):
            self.stop_price = None
        elif self.loss_type == "atr":
            self.stop_price = None
    
    def clear_position(self):
        """清除持仓信息"""
        self.entry_price = None
        self.entry_date = None
        self.highest_price = None
        self.stop_price = None
        self.position_active = False
    
    def update_stop_price(
        self,
        current_high: float,
        atr: Optional[float] = None,
        holding_days: Optional[int] = None,
    ):
        """
        更新止损价格
        
        参数:
            current_high: 当前最高价
            atr: 当前ATR值
        """
        if not self.position_active or self.entry_price is None:
            return
        
        if _is_holding_day_type(self.loss_type):
            return
        if self.loss_type == "trailing":
            if current_high > self.highest_price:
                self.highest_price = current_high
                self.stop_price = self.highest_price * (1 - self.loss_pct)
        elif self.loss_type == "atr" and atr is not None:
            self.stop_price = self.entry_price - atr * self.atr_multiplier
    
    def check_stop_loss(
        self,
        current_price: float,
        holding_days: Optional[int] = None,
    ) -> bool:
        """
        检查是否触发止损
        
        参数:
            current_price: 当前价格
        
        返回:
            是否触发止损
        """
        if not self.position_active:
            return False
        if _is_holding_day_type(self.loss_type):
            return holding_days is not None and holding_days >= self.holding_day
        if self.stop_price is None:
            return False
        return current_price <= self.stop_price
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成止损信号
        
        参数:
            data: 包含价格数据的DataFrame
        
        返回:
            止损信号序列: -1=止损卖出, 0=无操作
        """
        signals = pd.Series(0, index=data.index)
        
        if self.entry_price is None:
            return signals
        
        prices = data['close']
        highs = data['high'] if 'high' in data.columns else prices

        if _is_holding_day_type(self.loss_type):
            if self.holding_day < len(data):
                signals.iloc[self.holding_day] = -1
            return signals
        
        if self.loss_type == "atr":
            atr = calculate_atr(data, self.atr_window)
        else:
            atr = None
        
        for i in range(len(data)):
            if not self.position_active:
                break
            
            current_price = prices.iloc[i]
            current_high = highs.iloc[i]
            current_atr = atr.iloc[i] if atr is not None else None
            
            self.update_stop_price(current_high, current_atr)
            
            if self.check_stop_loss(current_price):
                signals.iloc[i] = -1
                break
        
        return signals
    
    def get_stop_info(self) -> Dict[str, Any]:
        """
        获取止损信息
        
        返回:
            止损信息字典
        """
        return {
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'highest_price': self.highest_price,
            'stop_price': self.stop_price,
            'loss_pct': self.loss_pct,
            'loss_type': self.loss_type,
            'holding_day': self.holding_day,
            'position_active': self.position_active
        }
