"""
回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import yaml
from tqdm import tqdm
from time import time

from signals.stop.stop_loss import calculate_atr as calculate_stop_loss_atr
from signals.stop.stop_profit import calculate_atr as calculate_stop_profit_atr


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
        self.position_entry_prices = {}
        self.position_high_prices = {}
        self.trades = []
        self.daily_values = []
        
        self.stop_loss_strategy = None
        self.stop_profit_strategy = None
        self.stop_strategies_config = {}
        self.stop_indicator_columns = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def set_stop_strategies(
        self,
        stop_loss_strategy: Any = None,
        stop_profit_strategy: Any = None,
        stop_strategies_config: Dict = None
    ):
        """
        设置止盈止损策略
        
        参数:
            stop_loss_strategy: 止损策略实例
            stop_profit_strategy: 止盈策略实例
            stop_strategies_config: 止盈止损策略配置
        """
        self.stop_loss_strategy = stop_loss_strategy
        self.stop_profit_strategy = stop_profit_strategy
        self.stop_strategies_config = stop_strategies_config or {}
        self.stop_indicator_columns = {}

    def _prepare_stop_indicators(self, data: Dict[str, pd.DataFrame]):
        """Precompute stop-strategy indicators once per run."""
        self.stop_indicator_columns = {}

        if self.stop_loss_strategy and getattr(self.stop_loss_strategy, "loss_type", None) == "atr":
            column_name = f"stop_loss_atr_{self.stop_loss_strategy.atr_window}"
            self.stop_indicator_columns["stop_loss"] = column_name
            for df in data.values():
                if column_name not in df.columns:
                    df[column_name] = calculate_stop_loss_atr(
                        df,
                        window=self.stop_loss_strategy.atr_window,
                    )

        if self.stop_profit_strategy and getattr(self.stop_profit_strategy, "profit_type", None) == "atr":
            column_name = f"stop_profit_atr_{self.stop_profit_strategy.atr_window}"
            self.stop_indicator_columns["stop_profit"] = column_name
            for df in data.values():
                if column_name not in df.columns:
                    df[column_name] = calculate_stop_profit_atr(
                        df,
                        window=self.stop_profit_strategy.atr_window,
                    )
    
    def _check_stop_signals(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, Dict]:
        """
        检查持仓是否触发止盈止损
        
        参数:
            data: 价格数据
            date: 当前日期
        
        返回:
            止盈止损信号字典
        """
        stop_signals = {}
        
        if not self.positions:
            return stop_signals
        
        for symbol in list(self.positions.keys()):
            if symbol not in data or date not in data[symbol].index:
                continue
            
            df = data[symbol]
            current_price = df.loc[date, 'close']
            current_high = df.loc[date, 'high'] if 'high' in df.columns else current_price
            entry_price = self.position_entry_prices.get(symbol)
            highest_price = self.position_high_prices.get(symbol, entry_price)
            
            if entry_price is None:
                continue
            
            self.position_high_prices[symbol] = max(highest_price, current_high)
            
            if self.stop_loss_strategy:
                stop_loss_atr = None
                stop_loss_atr_col = self.stop_indicator_columns.get("stop_loss")
                if stop_loss_atr_col:
                    stop_loss_atr = df.loc[date, stop_loss_atr_col]
                    if stop_loss_atr != stop_loss_atr:
                        stop_loss_atr = None

                self.stop_loss_strategy.set_position(entry_price)
                self.stop_loss_strategy.update_stop_price(
                    self.position_high_prices[symbol],
                    stop_loss_atr,
                )
                if self.stop_loss_strategy.check_stop_loss(current_price):
                    stop_signals[symbol] = {
                        'action': 'sell',
                        'shares': self.positions[symbol],
                        'reason': 'stop_loss',
                        'entry_price': entry_price,
                        'stop_price': self.stop_loss_strategy.stop_price
                    }
                    continue
            
            if self.stop_profit_strategy:
                stop_profit_atr = None
                stop_profit_atr_col = self.stop_indicator_columns.get("stop_profit")
                if stop_profit_atr_col:
                    stop_profit_atr = df.loc[date, stop_profit_atr_col]
                    if stop_profit_atr != stop_profit_atr:
                        stop_profit_atr = None

                self.stop_profit_strategy.set_position(entry_price)
                self.stop_profit_strategy.update_profit_price(
                    current_price,
                    self.position_high_prices[symbol],
                    stop_profit_atr,
                )
                if self.stop_profit_strategy.check_stop_profit(current_price):
                    stop_signals[symbol] = {
                        'action': 'sell',
                        'shares': self.positions[symbol],
                        'reason': 'stop_profit',
                        'entry_price': entry_price,
                        'profit_price': self.stop_profit_strategy.profit_price
                    }
        
        return stop_signals
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """运行回测"""
        self._prepare_stop_indicators(data)

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
        
        time_dic = {"prices":0, "signals":0, "stop_check":0, "trades":0, "portfolio":0, "daily_values":0}
        for date in date_iterator:
            start_time = time()
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = df.loc[date, 'close']
            time_dic["prices"] += time() - start_time
            
            start_time = time()
            stop_signals = self._check_stop_signals(data, date)
            time_dic["stop_check"] += time() - start_time
            
            start_time = time()
            signals = strategy_func(date, data, self.positions)
            time_dic["signals"] += time() - start_time
            
            start_time = time()
            self._execute_trades(stop_signals, prices, date)
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
            reason = signal.get('reason', 'strategy')
            
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
                        self.position_entry_prices[symbol] = price
                        self.position_high_prices[symbol] = price
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': price,
                        'cost': cost,
                        'avg_cost': self.position_costs[symbol] / self.positions[symbol],
                        'reason': reason
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
                        if symbol in self.position_entry_prices:
                            del self.position_entry_prices[symbol]
                        if symbol in self.position_high_prices:
                            del self.position_high_prices[symbol]
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares,
                        'price': price,
                        'revenue': revenue,
                        'cost': sell_cost,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'reason': reason
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
