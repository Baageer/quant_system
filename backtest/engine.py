"""
Backtest engine.
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
        backtest_config = self.config.get("backtest", {})
        self.min_commission = float(backtest_config.get("min_commission", 5.0))
        self.stamp_duty_rate = float(backtest_config.get("stamp_duty_rate", 0.0005))
        
        self.cash = initial_capital
        self.positions = {}
        self.position_costs = {}
        self.position_entry_prices = {}
        self.position_high_prices = {}
        self.last_buy_dates = {}
        self.trades = []
        self.daily_values = []

        default_constraints = {
            "enforce_lot_size": True,
            "lot_size": 100,
            "enable_t1": True,
            "block_suspended": True,
            "block_limit_up": True,
            "block_limit_down": True,
            "limit_pct": 0.10,
            "limit_pct_by_prefix": {
                "688": 0.20,
                "301": 0.20,
                "300": 0.20,
                "8": 0.30,
                "4": 0.30,
            },
            "st_limit_pct": 0.05,
        }
        config_constraints = backtest_config.get("trade_constraints", {})
        self.trade_constraints = {**default_constraints, **config_constraints}
        default_limit_pct_map = default_constraints.get("limit_pct_by_prefix", {})
        custom_limit_pct_map = config_constraints.get("limit_pct_by_prefix", {})
        if isinstance(default_limit_pct_map, dict):
            if not isinstance(custom_limit_pct_map, dict):
                custom_limit_pct_map = {}
            self.trade_constraints["limit_pct_by_prefix"] = {
                **default_limit_pct_map,
                **custom_limit_pct_map,
            }
        
        self.stop_loss_strategy = None
        self.stop_profit_strategy = None
        self.stop_strategies_config = {}
        self.stop_indicator_columns = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML config."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _is_na(value: Any) -> bool:
        pd_isna = getattr(pd, "isna", None)
        if callable(pd_isna):
            return bool(pd_isna(value))
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except TypeError:
            return False

    def _is_not_na(self, value: Any) -> bool:
        return not self._is_na(value)
    
    def set_stop_strategies(
        self,
        stop_loss_strategy: Any = None,
        stop_profit_strategy: Any = None,
        stop_strategies_config: Dict = None
    ):
        """
        Configure stop-loss / stop-profit strategy instances.
        
        Args:
            stop_loss_strategy: stop-loss strategy instance.
            stop_profit_strategy: stop-profit strategy instance.
            stop_strategies_config: optional stop strategy config.
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

    def _get_trade_row(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str,
        date: datetime,
    ) -> Optional[pd.Series]:
        df = data.get(symbol)
        if df is None or date not in df.index:
            return None
        return df.loc[date]

    def _get_previous_close(
        self,
        df: pd.DataFrame,
        date: datetime,
        row: pd.Series,
    ) -> Optional[float]:
        loc = df.index.get_loc(date)
        if isinstance(loc, slice):
            loc = loc.start
        elif isinstance(loc, np.ndarray):
            loc = int(loc[0])

        if isinstance(loc, int) and loc > 0:
            prev_close = df["close"].iloc[loc - 1]
            if self._is_not_na(prev_close):
                return float(prev_close)

        pct_change = row.get("pct_change")
        close_price = row.get("close")
        if self._is_not_na(pct_change) and self._is_not_na(close_price):
            denominator = 1 + float(pct_change) / 100
            if abs(denominator) > 1e-8:
                return float(close_price) / denominator

        return None

    def _get_holding_days(
        self,
        df: pd.DataFrame,
        symbol: str,
        date: datetime,
    ) -> Optional[int]:
        entry_date = self.last_buy_dates.get(symbol)
        if entry_date is None or entry_date not in df.index or date not in df.index:
            return None

        entry_loc = df.index.get_loc(entry_date)
        current_loc = df.index.get_loc(date)

        if isinstance(entry_loc, slice):
            entry_loc = entry_loc.start
        elif isinstance(entry_loc, np.ndarray):
            entry_loc = int(entry_loc[0])

        if isinstance(current_loc, slice):
            current_loc = current_loc.start
        elif isinstance(current_loc, np.ndarray):
            current_loc = int(current_loc[0])

        if not isinstance(entry_loc, int) or not isinstance(current_loc, int):
            return None

        return max(current_loc - entry_loc, 0)

    @staticmethod
    def _is_holding_day_stop_triggered(
        stop_strategy: Any,
        holding_days: Optional[int],
    ) -> bool:
        if stop_strategy is None or holding_days is None:
            return False

        holding_day_limit = getattr(stop_strategy, "holding_day", None)
        if holding_day_limit is None:
            return False

        try:
            return holding_days >= int(holding_day_limit)
        except (TypeError, ValueError):
            return False

    def _is_suspended(self, row: pd.Series) -> bool:
        for column in ["open", "close", "high", "low"]:
            if column not in row or self._is_na(row[column]):
                return True

        if "volume" in row and self._is_not_na(row["volume"]) and float(row["volume"]) <= 0:
            return True

        if "amount" in row and self._is_not_na(row["amount"]) and float(row["amount"]) <= 0:
            return True

        return False

    def _is_limit_move(
        self,
        action: str,
        symbol: str,
        df: pd.DataFrame,
        date: datetime,
        row: pd.Series,
    ) -> bool:
        prev_close = self._get_previous_close(df, date, row)
        if prev_close is None or prev_close <= 0:
            return False

        limit_pct = self._get_limit_pct(symbol, row)
        upper_limit = prev_close * (1 + limit_pct)
        lower_limit = prev_close * (1 - limit_pct)
        tolerance = max(abs(prev_close) * 1e-4, 1e-6)

        close_price = float(row["close"])
        high_price = float(row["high"]) if "high" in row and self._is_not_na(row["high"]) else close_price
        low_price = float(row["low"]) if "low" in row and self._is_not_na(row["low"]) else close_price

        if action == "buy":
            return close_price >= upper_limit - tolerance and high_price >= upper_limit - tolerance

        if action == "sell":
            return close_price <= lower_limit + tolerance and low_price <= lower_limit + tolerance

        return False

    @staticmethod
    def _extract_symbol_digits(symbol: str) -> str:
        return "".join(ch for ch in str(symbol) if ch.isdigit())

    def _get_limit_pct(self, symbol: str, row: Optional[pd.Series] = None) -> float:
        default_limit_pct = float(self.trade_constraints.get("limit_pct", 0.10))

        st_limit_pct = self.trade_constraints.get("st_limit_pct")
        if st_limit_pct is not None and row is not None and "name" in row:
            name_val = row.get("name")
            if self._is_not_na(name_val) and "ST" in str(name_val).upper():
                return float(st_limit_pct)

        symbol_raw = str(symbol)
        symbol_digits = self._extract_symbol_digits(symbol_raw)
        limit_pct_by_prefix = self.trade_constraints.get("limit_pct_by_prefix", {})
        if not isinstance(limit_pct_by_prefix, dict):
            return default_limit_pct

        sorted_limits = sorted(
            ((str(prefix), pct) for prefix, pct in limit_pct_by_prefix.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for prefix, pct in sorted_limits:
            if symbol_raw.startswith(prefix) or symbol_digits.startswith(prefix):
                return float(pct)

        return default_limit_pct

    def _normalize_shares(self, symbol: str, action: str, shares: int) -> int:
        shares = int(shares)
        if shares <= 0:
            return 0

        if not self.trade_constraints.get("enforce_lot_size", True):
            if action == "sell":
                return min(shares, self.positions.get(symbol, 0))
            return shares

        lot_size = max(int(self.trade_constraints.get("lot_size", 100)), 1)
        if action == "buy":
            return (shares // lot_size) * lot_size

        current_pos = self.positions.get(symbol, 0)
        shares = min(shares, current_pos)
        if shares == current_pos:
            return shares
        return (shares // lot_size) * lot_size

    def _violates_t1(self, symbol: str, action: str, date: datetime) -> bool:
        if action != "sell" or not self.trade_constraints.get("enable_t1", True):
            return False

        last_buy_date = self.last_buy_dates.get(symbol)
        if last_buy_date is None:
            return False

        return pd.Timestamp(date).normalize() <= pd.Timestamp(last_buy_date).normalize()

    def _get_trade_rejection_reason(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str,
        action: str,
        date: datetime,
    ) -> Optional[str]:
        row = self._get_trade_row(data, symbol, date)
        if row is None:
            return "missing_trade_row"

        if self.trade_constraints.get("block_suspended", True) and self._is_suspended(row):
            return "suspended"

        df = data[symbol]
        if action == "buy" and self.trade_constraints.get("block_limit_up", True):
            if self._is_limit_move("buy", symbol, df, date, row):
                return "limit_up"

        if action == "sell" and self.trade_constraints.get("block_limit_down", True):
            if self._is_limit_move("sell", symbol, df, date, row):
                return "limit_down"

        if self._violates_t1(symbol, action, date):
            return "t1_restriction"

        return None

    def _can_execute_trade(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str,
        action: str,
        date: datetime,
    ) -> bool:
        return self._get_trade_rejection_reason(data, symbol, action, date) is None

    def _calculate_commission(self, trade_value: float) -> float:
        if trade_value <= 0:
            return 0.0
        return max(trade_value * self.commission_rate, self.min_commission)

    def _calculate_stamp_duty(self, trade_value: float, action: str) -> float:
        if trade_value <= 0 or action != "sell":
            return 0.0
        return trade_value * self.stamp_duty_rate
    
    def _check_stop_signals(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, Dict]:
        """
        Check whether current positions trigger stop exit signals.
        
        Args:
            data: price data.
            date: current date.
        
        Returns:
            stop signal dictionary.
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

            holding_days = self._get_holding_days(df, symbol, date)
            
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
                    holding_days,
                )
                if self.stop_loss_strategy.check_stop_loss(
                    current_price,
                    holding_days=holding_days,
                ):
                    exit_reason = (
                        "stop_holding"
                        if self._is_holding_day_stop_triggered(self.stop_loss_strategy, holding_days)
                        else "stop_loss"
                    )
                    stop_signals[symbol] = {
                        'action': 'sell',
                        'shares': self.positions[symbol],
                        'reason': exit_reason,
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
                    holding_days,
                )
                if self.stop_profit_strategy.check_stop_profit(
                    current_price,
                    holding_days=holding_days,
                ):
                    exit_reason = (
                        "stop_holding"
                        if self._is_holding_day_stop_triggered(self.stop_profit_strategy, holding_days)
                        else "stop_profit"
                    )
                    stop_signals[symbol] = {
                        'action': 'sell',
                        'shares': self.positions[symbol],
                        'reason': exit_reason,
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
        """Run backtest."""
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
        date_iterator = tqdm(all_dates, desc="Backtest progress", unit="day", disable=not show_progress)
        
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
            self._execute_trades(stop_signals, prices, data, date)
            self._execute_trades(signals, prices, data, date)
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
                    "value": f"{portfolio_value:,.0f}",
                    "positions": len(self.positions),
                })
        
        print(time_dic)
        return pd.DataFrame(self.daily_values).set_index('date')
    
    def _execute_trades(
        self,
        signals: Dict[str, Dict],
        prices: Dict[str, float],
        data: Dict[str, pd.DataFrame],
        date: datetime
    ):
        """Execute generated trades."""
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue
            
            action = signal.get("action")
            if action not in {"buy", "sell"}:
                continue

            rejection_reason = self._get_trade_rejection_reason(data, symbol, action, date)
            if rejection_reason is not None:
                if action == "buy":
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "buy",
                        "status": "rejected",
                        "rejection_reason": rejection_reason,
                        "requested_shares": int(signal.get("shares", 0)),
                        "shares": 0,
                        "price": prices[symbol],
                        "trade_value": 0.0,
                        "commission": 0.0,
                        "stamp_duty": 0.0,
                        "cost": 0.0,
                        "reason": signal.get("reason", "strategy"),
                    })
                continue

            requested_shares = int(signal.get("shares", 0))
            shares = self._normalize_shares(symbol, action, requested_shares)
            if shares <= 0:
                if action == "buy":
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "buy",
                        "status": "rejected",
                        "rejection_reason": "lot_size",
                        "requested_shares": requested_shares,
                        "shares": 0,
                        "price": prices[symbol],
                        "trade_value": 0.0,
                        "commission": 0.0,
                        "stamp_duty": 0.0,
                        "cost": 0.0,
                        "reason": signal.get("reason", "strategy"),
                    })
                continue

            price = prices[symbol]
            reason = signal.get("reason", "strategy")
            if self._is_na(price) or price <= 0:
                continue
            
            if action == "buy":
                execution_price = price * (1 + self.slippage)
                lot_size = max(int(self.trade_constraints.get("lot_size", 100)), 1)
                step = lot_size if self.trade_constraints.get("enforce_lot_size", True) else 1

                # Keep shrinking the buy order until it is cash-affordable.
                while shares > 0:
                    trade_value = shares * execution_price
                    commission = self._calculate_commission(trade_value)
                    total_cost = trade_value + commission
                    if total_cost <= self.cash:
                        break
                    shares -= step

                if shares <= 0:
                    continue

                trade_value = shares * execution_price
                commission = self._calculate_commission(trade_value)
                total_cost = trade_value + commission

                self.cash -= total_cost
                old_shares = self.positions.get(symbol, 0)
                old_cost = self.position_costs.get(symbol, 0)

                self.positions[symbol] = old_shares + shares
                self.last_buy_dates[symbol] = date

                if old_shares > 0:
                    holding_cost = old_cost + total_cost
                    self.position_costs[symbol] = holding_cost
                else:
                    self.position_costs[symbol] = total_cost
                    self.position_entry_prices[symbol] = execution_price
                    self.position_high_prices[symbol] = execution_price

                self.trades.append({
                    "date": date,
                    "symbol": symbol,
                    "action": "buy",
                    "status": "filled",
                    "shares": shares,
                    "price": execution_price,
                    "trade_value": trade_value,
                    "commission": commission,
                    "stamp_duty": 0.0,
                    "cost": total_cost,
                    "avg_cost": self.position_costs[symbol] / self.positions[symbol],
                    "reason": reason,
                })
            
            elif action == "sell":
                if symbol in self.positions and self.positions[symbol] >= shares:
                    execution_price = price * (1 - self.slippage)
                    gross_revenue = shares * execution_price
                    commission = self._calculate_commission(gross_revenue)
                    stamp_duty = self._calculate_stamp_duty(gross_revenue, action)
                    revenue = gross_revenue - commission - stamp_duty
                    
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
                        if symbol in self.last_buy_dates:
                            del self.last_buy_dates[symbol]
                    
                    self.trades.append({
                        "date": date,
                        "symbol": symbol,
                        "action": "sell",
                        "status": "filled",
                        "shares": shares,
                        "price": execution_price,
                        "trade_value": gross_revenue,
                        "commission": commission,
                        "stamp_duty": stamp_duty,
                        "revenue": revenue,
                        "cost": sell_cost,
                        "profit": profit,
                        "profit_pct": profit_pct,
                        "reason": reason,
                    })
    
    def _calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value using latest close prices."""
        value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in prices:
                value += shares * prices[symbol]
        return value
    
    def get_trades(self) -> pd.DataFrame:
        """Return all executed trades as a DataFrame."""
        return pd.DataFrame(self.trades)


