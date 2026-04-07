"""
调仓逻辑模块
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class Rebalancer:
    def __init__(
        self,
        rebalance_freq: str = 'monthly',
        threshold: float = 0.05
    ):
        self.rebalance_freq = rebalance_freq
        self.threshold = threshold
        self.last_rebalance_date = None
    
    def need_rebalance(self, current_date: datetime) -> bool:
        """判断是否需要调仓"""
        if self.last_rebalance_date is None:
            return True
        
        if self.rebalance_freq == 'daily':
            return (current_date - self.last_rebalance_date).days >= 1
        elif self.rebalance_freq == 'weekly':
            return (current_date - self.last_rebalance_date).days >= 7
        elif self.rebalance_freq == 'monthly':
            if current_date.month != self.last_rebalance_date.month:
                return True
            return False
        elif self.rebalance_freq == 'quarterly':
            quarter_diff = (
                (current_date.year - self.last_rebalance_date.year) * 4 +
                (current_date.month - self.last_rebalance_date.month) // 3
            )
            return quarter_diff >= 1
        else:
            return False
    
    def calculate_trades(
        self,
        current_positions: Dict[str, int],
        target_positions: Dict[str, int],
        prices: Dict[str, float]
    ) -> Dict[str, Dict]:
        """计算需要交易的股票"""
        trades = {}
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())
        
        for symbol in all_symbols:
            current = current_positions.get(symbol, 0)
            target = target_positions.get(symbol, 0)
            diff = target - current
            
            if diff != 0:
                trades[symbol] = {
                    'action': 'buy' if diff > 0 else 'sell',
                    'shares': abs(diff),
                    'price': prices.get(symbol, 0),
                    'value': abs(diff) * prices.get(symbol, 0)
                }
        
        return trades
    
    def apply_threshold(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """应用调仓阈值"""
        adjusted_weights = {}
        
        for symbol in target_weights:
            current = current_weights.get(symbol, 0)
            target = target_weights[symbol]
            
            if abs(target - current) >= self.threshold:
                adjusted_weights[symbol] = target
            else:
                adjusted_weights[symbol] = current
        
        return adjusted_weights
    
    def execute_rebalance(
        self,
        current_date: datetime,
        current_positions: Dict[str, int],
        target_positions: Dict[str, int],
        prices: Dict[str, float]
    ) -> Optional[Dict[str, Dict]]:
        """执行调仓"""
        if not self.need_rebalance(current_date):
            return None
        
        trades = self.calculate_trades(current_positions, target_positions, prices)
        self.last_rebalance_date = current_date
        
        return trades
