"""
模拟撮合模块
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class Broker:
    def __init__(
        self,
        commission_rate: float = 0.0003,
        slippage: float = 0.0001,
        min_commission: float = 5.0
    ):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_commission = min_commission
    
    def execute_order(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        date: datetime
    ) -> Dict:
        """执行订单"""
        if action == 'buy':
            execution_price = price * (1 + self.slippage)
            commission = max(shares * execution_price * self.commission_rate, self.min_commission)
            total_cost = shares * execution_price + commission
            
            return {
                'symbol': symbol,
                'action': 'buy',
                'shares': shares,
                'price': execution_price,
                'commission': commission,
                'total_cost': total_cost,
                'date': date,
                'status': 'filled'
            }
        
        elif action == 'sell':
            execution_price = price * (1 - self.slippage)
            commission = max(shares * execution_price * self.commission_rate, self.min_commission)
            total_revenue = shares * execution_price - commission
            
            return {
                'symbol': symbol,
                'action': 'sell',
                'shares': shares,
                'price': execution_price,
                'commission': commission,
                'total_revenue': total_revenue,
                'date': date,
                'status': 'filled'
            }
        
        else:
            raise ValueError(f"无效的交易动作: {action}")
    
    def calculate_commission(self, value: float) -> float:
        """计算手续费"""
        return max(value * self.commission_rate, self.min_commission)
    
    def calculate_slippage(self, price: float, action: str) -> float:
        """计算滑点"""
        if action == 'buy':
            return price * self.slippage
        elif action == 'sell':
            return -price * self.slippage
        return 0.0
