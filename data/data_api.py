"""
数据接口封装模块
提供统一的数据访问接口
"""
import pandas as pd
from typing import List, Optional
from datetime import datetime
import os
import json

import akshare as ak


class DataAPI:
    def __init__(self, source: str, 
                 stock_file:str = ".test1.txt", 
                 cache_dir: str = "./data/raw", 
                 processed_dir: str = "./data/processed"):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        self.stock_file = stock_file
        self.source = source
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        stock_list = []
        with open(self.stock_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if self.source == 'tushare':
                    if(line[0]=='6'):
                        line = line + ".SH"
                    else:
                        line = line + ".SZ"
                stock_list.append(line)
        return stock_list
    
    def get_price_history_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取股票价格历史数据"""
        d_start_date = "20050101"
        d_end_date = "20241231"
        # 从缓存加载数据
        data = self.load_from_cache(f"{symbol}_{d_start_date}_{d_end_date}.csv", "price_history")
        if data is None:
            # 从数据源获取数据
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=d_start_date, end_date=d_end_date, adjust="qfq")
            # 保存到缓存
            self.save_to_cache(data, f"{symbol}_{d_start_date}_{d_end_date}.csv", "price_history")
            
        # 筛选日期范围
        data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
        
        
        return data
    
    def get_price_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取股票价格数据"""
        # 从缓存加载数据
        data = self.load_from_cache(f"{symbol}_{start_date}_{end_date}.csv", "price")
        if data is None:
            # 从数据源获取数据
            data = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            # 保存到缓存
            self.save_to_cache(data, f"{symbol}_{start_date}_{end_date}.csv", "price")
        return data
    
    def get_financial_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取财务数据"""
        pass
    
    def save_to_cache(self, data: pd.DataFrame, filename: str, data_type: str = "price"):
        """保存数据到缓存"""
        filepath = os.path.join(self.cache_dir, elf.source, data_type, filename)
        data.to_csv(filepath, index=True)
    
    def load_from_cache(self, filename: str, data_type: str = "price") -> pd.DataFrame:
        """从缓存加载数据"""
        filepath = os.path.join(self.cache_dir, self.source, data_type, filename)
        print(filepath)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        return None
    
    def save_processed(self, data: pd.DataFrame, filename: str, data_type: str = "price"):
        """保存处理后的数据"""
        filepath = os.path.join(self.processed_dir, self.source, data_type, filename)
        data.to_csv(filepath, index=True)
    
    def load_processed(self, filename: str, data_type: str = "price") -> pd.DataFrame:
        """加载处理后的数据"""
        filepath = os.path.join(self.processed_dir, self.source, data_type, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        return None
