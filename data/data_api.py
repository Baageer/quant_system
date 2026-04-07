"""
数据接口封装模块
提供统一的数据访问接口
"""
import pandas as pd
from typing import List, Optional
from datetime import datetime
import os
import json


class DataAPI:
    def __init__(self, cache_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        pass
    
    def get_price_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取股票价格数据"""
        pass
    
    def get_financial_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取财务数据"""
        pass
    
    def save_to_cache(self, data: pd.DataFrame, filename: str):
        """保存数据到缓存"""
        filepath = os.path.join(self.cache_dir, filename)
        data.to_csv(filepath, index=True)
    
    def load_from_cache(self, filename: str) -> pd.DataFrame:
        """从缓存加载数据"""
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return None
    
    def save_processed(self, data: pd.DataFrame, filename: str):
        """保存处理后的数据"""
        filepath = os.path.join(self.processed_dir, filename)
        data.to_csv(filepath, index=True)
    
    def load_processed(self, filename: str) -> pd.DataFrame:
        """加载处理后的数据"""
        filepath = os.path.join(self.processed_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return None
