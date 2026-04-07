"""
回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import yaml


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        slippage: float = 0.0001,
        config_path: str = "./config/settings.yaml"
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.config = self._load_config(config_path)
        
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """运行回测"""
        all_dates = sorted(set(
            date for df in data.values() 
            for date in df.index
        ))
        
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]
        
        for date in all_dates:
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = df.loc[date, 'close']
            
            signals = strategy_func(date, data, self.positions)
            
            self._execute_trades(signals, prices, date)
            
            portfolio_value = self._calculate_portfolio_value(prices)
            self.daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy()
            })
        
        return pd.DataFrame(self.daily_values).set_index('date')
    
    def _execute_trades(
        self,
        signals: Dict[str, Dict],
        prices: Dict[str, float],
        date: datetime
    ):
        """执行交易"""
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue
            
            action = signal.get('action')
            shares = signal.get('shares', 0)
            price = prices[symbol]
            
            if action == 'buy':
                cost = shares * price * (1 + self.commission_rate + self.slippage)
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'cost': cost
                    })
            
            elif action == 'sell':
                if symbol in self.positions and self.positions[symbol] >= shares:
                    revenue = shares * price * (1 - self.commission_rate - self.slippage)
                    self.cash += revenue
                    self.positions[symbol] -= shares
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares,
                        'price': price,
                        'revenue': revenue
                    })
    
    def _calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """计算组合市值"""
        value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in prices:
                value += shares * prices[symbol]
        return value
    
    def get_trades(self) -> pd.DataFrame:
        """获取交易记录"""
        return pd.DataFrame(self.trades)
