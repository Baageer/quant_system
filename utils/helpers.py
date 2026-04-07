"""
辅助工具函数模块
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime, timedelta


def date_range(
    start_date: str,
    end_date: str,
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """生成日期范围"""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def trading_days(
    start_date: str,
    end_date: str,
    exchange: str = 'SSE'
) -> pd.DatetimeIndex:
    """获取交易日"""
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    trading_dates = all_dates[
        ~all_dates.to_series().dt.dayofweek.isin([5, 6])
    ]
    
    return trading_dates


def resample_data(
    data: pd.DataFrame,
    freq: str = 'W',
    agg_dict: Optional[dict] = None
) -> pd.DataFrame:
    """重采样数据"""
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    return data.resample(freq).agg(agg_dict)


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple'
) -> pd.Series:
    """计算收益率"""
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"不支持的收益率计算方法: {method}")


def align_dataframes(
    dataframes: List[pd.DataFrame],
    join: str = 'inner'
) -> List[pd.DataFrame]:
    """对齐多个DataFrame"""
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        if join == 'inner':
            common_index = common_index.intersection(df.index)
        else:
            common_index = common_index.union(df.index)
    
    return [df.reindex(common_index) for df in dataframes]


def fill_missing_values(
    data: pd.DataFrame,
    method: str = 'ffill',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """填充缺失值"""
    if method == 'ffill':
        return data.fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        return data.fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        return data.interpolate(limit=limit)
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    else:
        raise ValueError(f"不支持的填充方法: {method}")


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> tuple:
    """分割数据集"""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data


def format_number(
    value: float,
    decimal_places: int = 2,
    percentage: bool = False
) -> str:
    """格式化数字"""
    if percentage:
        return f"{value * 100:.{decimal_places}f}%"
    else:
        return f"{value:.{decimal_places}f}"


def ensure_directory(path: str):
    """确保目录存在"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def save_dataframe(
    data: pd.DataFrame,
    filepath: str,
    format: str = 'csv',
    **kwargs
):
    """保存DataFrame"""
    if format == 'csv':
        data.to_csv(filepath, **kwargs)
    elif format == 'parquet':
        data.to_parquet(filepath, **kwargs)
    elif format == 'pickle':
        data.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的格式: {format}")


def load_dataframe(
    filepath: str,
    format: str = 'csv',
    **kwargs
) -> pd.DataFrame:
    """加载DataFrame"""
    if format == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif format == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif format == 'pickle':
        return pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的格式: {format}")
