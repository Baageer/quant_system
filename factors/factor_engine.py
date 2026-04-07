"""
因子计算调度引擎
"""
import pandas as pd
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import yaml


class FactorEngine:
    def __init__(self, config_path: str = "./config/factors.yaml"):
        self.config = self._load_config(config_path)
        self.technical_factors = {}
        self.fundamental_factors = {}
        self.custom_factors = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """加载因子配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def register_technical_factor(self, name: str, factor_func):
        """注册技术因子"""
        self.technical_factors[name] = factor_func
    
    def register_fundamental_factor(self, name: str, factor_func):
        """注册基本面因子"""
        self.fundamental_factors[name] = factor_func
    
    def register_custom_factor(self, name: str, factor_func):
        """注册自定义因子"""
        self.custom_factors[name] = factor_func
    
    def calculate_factors(
        self, 
        data: pd.DataFrame, 
        factor_types: List[str] = ['technical', 'fundamental', 'custom']
    ) -> pd.DataFrame:
        """计算所有因子"""
        result = data.copy()
        
        if 'technical' in factor_types:
            for name, func in self.technical_factors.items():
                result[f'{name}_factor'] = func(data)
        
        if 'fundamental' in factor_types:
            for name, func in self.fundamental_factors.items():
                result[f'{name}_factor'] = func(data)
        
        if 'custom' in factor_types:
            for name, func in self.custom_factors.items():
                result[f'{name}_factor'] = func(data)
        
        return result
    
    def calculate_factor_for_multiple_stocks(
        self, 
        stock_data: Dict[str, pd.DataFrame],
        factor_types: List[str] = ['technical', 'fundamental', 'custom']
    ) -> Dict[str, pd.DataFrame]:
        """批量计算多只股票的因子"""
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                symbol: executor.submit(self.calculate_factors, data, factor_types)
                for symbol, data in stock_data.items()
            }
            for symbol, future in futures.items():
                results[symbol] = future.result()
        return results
