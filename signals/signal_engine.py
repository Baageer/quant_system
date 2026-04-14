"""
信号生成引擎
"""
import pandas as pd
from typing import Dict, List, Optional


class SignalEngine:
    def __init__(self):
        self.timing_strategies = {}
        self.ranking_strategies = {}

    def register_timing_strategy(self, name: str, strategy_func):
        """注册择时策略。"""
        self.timing_strategies[name] = strategy_func

    def register_ranking_strategy(self, name: str, strategy_func):
        """注册排序策略。"""
        self.ranking_strategies[name] = strategy_func

    def generate_timing_signals(
        self,
        data: pd.DataFrame,
        strategy_name: str
    ) -> pd.Series:
        """根据策略名称生成择时信号。"""
        if strategy_name not in self.timing_strategies:
            raise ValueError(f"未找到择时策略: {strategy_name}")
        return self.timing_strategies[strategy_name](data)

    def generate_ranking_signals(
        self,
        factor_data: Dict[str, pd.DataFrame],
        strategy_name: str
    ) -> pd.DataFrame:
        """根据策略名称生成排序信号。"""
        if strategy_name not in self.ranking_strategies:
            raise ValueError(f"未找到排序策略: {strategy_name}")
        return self.ranking_strategies[strategy_name](factor_data)

    def combine_signals(
        self,
        signals: List[pd.Series],
        weights: Optional[List[float]] = None
    ) -> pd.Series:
        """按权重合并多个信号序列。"""
        if not signals:
            raise ValueError("signals must not be empty")

        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        elif len(weights) != len(signals):
            raise ValueError("weights length must match signals length")

        combined = pd.Series(0, index=signals[0].index)
        for signal, weight in zip(signals, weights):
            combined += signal * weight
        return combined
