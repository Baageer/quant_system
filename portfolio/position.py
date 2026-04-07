"""
仓位管理模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class PositionManager:
    def __init__(
        self,
        initial_capital: float = 1000000,
        max_position_pct: float = 0.05,
        min_position_pct: float = 0.01
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.positions = {}
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        target_weight: float,
        current_position: Optional[int] = None
    ) -> int:
        """计算目标持仓数量"""
        target_value = self.current_capital * target_weight
        target_value = np.clip(
            target_value,
            self.current_capital * self.min_position_pct,
            self.current_capital * self.max_position_pct
        )
        target_shares = int(target_value / price)
        return target_shares
    
    def update_positions(self, positions: Dict[str, int]):
        """更新持仓"""
        self.positions = positions
    
    def get_position_value(self, symbol: str, price: float) -> float:
        """获取持仓市值"""
        if symbol in self.positions:
            return self.positions[symbol] * price
        return 0.0
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """获取总市值"""
        total = self.current_capital
        for symbol, shares in self.positions.items():
            if symbol in prices:
                total += shares * prices[symbol]
        return total
    
    def apply_risk_management(
        self,
        target_positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict[str, int]:
        """应用风险管理规则"""
        adjusted_positions = {}
        
        for symbol, weight in target_positions.items():
            if symbol in prices:
                shares = self.calculate_position_size(
                    symbol,
                    prices[symbol],
                    weight
                )
                if shares > 0:
                    adjusted_positions[symbol] = shares
        
        return adjusted_positions
