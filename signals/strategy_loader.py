import importlib
from typing import Iterable, List, Optional, Tuple

import yaml


class StrategyLoader:
    """Load strategy metadata and instantiate strategy classes from YAML."""

    def __init__(self, strategy_config_path: str = "./config/strategies.yaml"):
        with open(strategy_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _get_strategy_class(self, section: str, strategy_name: str, optional: bool = False):
        strategies = self.config.get(section, {})

        if strategy_name not in strategies:
            if optional:
                return None, None
            raise ValueError(
                f"Strategy '{strategy_name}' not found in '{section}'. "
                f"Available strategies: {list(strategies.keys())}"
            )

        strategy_info = strategies[strategy_name]
        module = importlib.import_module(strategy_info["module"])
        strategy_class = getattr(module, strategy_info["class"])
        return strategy_class, strategy_info

    def get_strategy(self, strategy_name: str):
        return self._get_strategy_class("timing_strategies", strategy_name)

    def get_stop_strategy(self, strategy_name: str):
        return self._get_strategy_class("stop_strategies", strategy_name, optional=True)

    def build_timing_strategies(self, strategy_names: Iterable[str]) -> Tuple[List[object], List[dict]]:
        strategies = []
        strategy_infos = []

        for strategy_name in strategy_names:
            strategy_class, strategy_info = self.get_strategy(strategy_name)
            strategies.append(strategy_class(**strategy_info["params"]))
            strategy_infos.append(strategy_info)

        return strategies, strategy_infos

    def build_stop_strategies(
        self,
        enable_stop_loss: bool = True,
        enable_stop_profit: bool = True,
        stop_loss_name: str = "stop_loss",
        stop_profit_name: str = "stop_profit",
    ) -> Tuple[Optional[object], Optional[dict], Optional[object], Optional[dict]]:
        stop_loss_strategy = None
        stop_loss_info = None
        stop_profit_strategy = None
        stop_profit_info = None

        if enable_stop_loss:
            stop_loss_class, stop_loss_info = self.get_stop_strategy(stop_loss_name)
            if stop_loss_class and stop_loss_info:
                stop_loss_strategy = stop_loss_class(**stop_loss_info["params"])

        if enable_stop_profit:
            stop_profit_class, stop_profit_info = self.get_stop_strategy(stop_profit_name)
            if stop_profit_class and stop_profit_info:
                stop_profit_strategy = stop_profit_class(**stop_profit_info["params"])

        return stop_loss_strategy, stop_loss_info, stop_profit_strategy, stop_profit_info

    def list_strategies(self):
        strategies = self.config.get("timing_strategies", {})
        print("\nAvailable timing strategies:")
        print("-" * 60)
        for name, info in strategies.items():
            print(f"  {name}: {info['name']}")
            print(f"    Description: {info['description']}")
            print(f"    Params: {info['params']}")
            print()

        stop_strategies = self.config.get("stop_strategies", {})
        if stop_strategies:
            print("\nAvailable stop strategies:")
            print("-" * 60)
            for name, info in stop_strategies.items():
                print(f"  {name}: {info['name']}")
                print(f"    Description: {info['description']}")
                print(f"    Params: {info['params']}")
                print()
