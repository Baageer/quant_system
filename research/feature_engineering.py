"""
特征工程模块
用于因子挖掘和特征处理
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class FeatureEngineer:
    def __init__(self):
        self.feature_transformers = {}
    
    def normalize(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """标准化"""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def winsorize(
        self, 
        data: pd.Series, 
        lower: float = 0.025, 
        upper: float = 0.975
    ) -> pd.Series:
        """去极值"""
        lower_bound = data.quantile(lower)
        upper_bound = data.quantile(upper)
        return data.clip(lower_bound, upper_bound)
    
    def neutralize(
        self,
        factor_data: pd.DataFrame,
        industry_data: pd.DataFrame,
        market_cap_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """中性化处理"""
        from sklearn.linear_model import LinearRegression
        
        neutralized = factor_data.copy()
        
        for col in factor_data.columns:
            factor_values = factor_data[col].dropna()
            
            if len(factor_values) == 0:
                continue
            
            X = industry_data.loc[factor_values.index].values
            
            if market_cap_data is not None:
                market_cap = market_cap_data.loc[factor_values.index].values
                X = np.column_stack([X, market_cap])
            
            model = LinearRegression()
            model.fit(X, factor_values)
            residuals = factor_values - model.predict(X)
            neutralized.loc[factor_values.index, col] = residuals
        
        return neutralized
    
    def calculate_ic(
        self,
        factor_data: pd.Series,
        return_data: pd.Series,
        method: str = 'spearman'
    ) -> float:
        """计算IC值"""
        if method == 'spearman':
            return factor_data.corr(return_data, method='spearman')
        elif method == 'pearson':
            return factor_data.corr(return_data, method='pearson')
        else:
            raise ValueError(f"不支持的IC计算方法: {method}")
    
    def calculate_ic_series(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.Series:
        """计算IC时间序列"""
        ic_series = pd.Series(index=factor_data.index)
        
        for date in factor_data.index:
            if date in return_data.index:
                ic = self.calculate_ic(
                    factor_data.loc[date],
                    return_data.loc[date],
                    method
                )
                ic_series[date] = ic
        
        return ic_series
    
    def create_lagged_features(
        self,
        data: pd.DataFrame,
        lags: List[int]
    ) -> pd.DataFrame:
        """创建滞后特征"""
        lagged = pd.DataFrame(index=data.index)
        
        for lag in lags:
            lagged_data = data.shift(lag)
            lagged_data.columns = [f'{col}_lag{lag}' for col in data.columns]
            lagged = pd.concat([lagged, lagged_data], axis=1)
        
        return lagged
    
    def create_rolling_features(
        self,
        data: pd.DataFrame,
        windows: List[int],
        agg_funcs: List[str] = ['mean', 'std', 'max', 'min']
    ) -> pd.DataFrame:
        """创建滚动特征"""
        rolling_features = pd.DataFrame(index=data.index)
        
        for window in windows:
            for func in agg_funcs:
                if func == 'mean':
                    rolling_data = data.rolling(window).mean()
                elif func == 'std':
                    rolling_data = data.rolling(window).std()
                elif func == 'max':
                    rolling_data = data.rolling(window).max()
                elif func == 'min':
                    rolling_data = data.rolling(window).min()
                else:
                    continue
                
                rolling_data.columns = [f'{col}_rolling{window}_{func}' for col in data.columns]
                rolling_features = pd.concat([rolling_features, rolling_data], axis=1)
        
        return rolling_features
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'ic',
        threshold: float = 0.05
    ) -> List[str]:
        """特征选择"""
        if method == 'ic':
            ic_values = X.apply(lambda col: self.calculate_ic(col, y))
            selected_features = ic_values[abs(ic_values) > threshold].index.tolist()
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        return selected_features
