"""
回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import yaml
from tqdm import tqdm
from time import time


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
        self.position_costs = {}
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
        end_date: Optional[str] = None,
        show_progress: bool = True
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
        
        total_days = len(all_dates)
        date_iterator = tqdm(all_dates, desc="回测进度", unit="天", disable=not show_progress)
        
        time_dic = {"prices":0, "signals":0, "trades":0, "portfolio":0, "daily_values":0}
        for date in date_iterator:
            start_time = time()
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = df.loc[date, 'close']
            time_dic["prices"] += time() - start_time
            
            start_time = time()
            signals = strategy_func(date, data, self.positions)
            time_dic["signals"] += time() - start_time
            
            start_time = time()
            self._execute_trades(signals, prices, date)
            time_dic["trades"] += time() - start_time
            
            start_time = time()
            portfolio_value = self._calculate_portfolio_value(prices)
            time_dic["portfolio"] += time() - start_time
            
            start_time = time()
            self.daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy()
            })
            time_dic["daily_values"] += time() - start_time
            if show_progress:
                date_iterator.set_postfix({
                    '市值': f'{portfolio_value:,.0f}',
                    '持仓': len(self.positions)
                })
        
        print(time_dic)
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
                    old_shares = self.positions.get(symbol, 0)
                    old_cost = self.position_costs.get(symbol, 0)
                    
                    self.positions[symbol] = old_shares + shares
                    
                    if old_shares > 0:
                        total_cost = old_cost + cost
                        avg_cost = total_cost / self.positions[symbol]
                        self.position_costs[symbol] = total_cost
                    else:
                        self.position_costs[symbol] = cost
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'cost': cost,
                        'avg_cost': self.position_costs[symbol] / self.positions[symbol]
                    })
            
            elif action == 'sell':
                if symbol in self.positions and self.positions[symbol] >= shares:
                    revenue = shares * price * (1 - self.commission_rate - self.slippage)
                    
                    buy_cost = self.position_costs.get(symbol, 0)
                    cost_per_share = buy_cost / self.positions[symbol] if self.positions[symbol] > 0 else 0
                    sell_cost = cost_per_share * shares
                    
                    profit = revenue - sell_cost
                    profit_pct = (profit / sell_cost * 100) if sell_cost > 0 else 0
                    
                    self.cash += revenue
                    self.positions[symbol] -= shares
                    
                    remaining_ratio = self.positions[symbol] / (self.positions[symbol] + shares) if shares > 0 else 0
                    self.position_costs[symbol] = buy_cost * remaining_ratio
                    
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                        if symbol in self.position_costs:
                            del self.position_costs[symbol]
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares,
                        'price': price,
                        'revenue': revenue,
                        'cost': sell_cost,
                        'profit': profit,
                        'profit_pct': profit_pct
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
