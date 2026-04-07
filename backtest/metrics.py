"""
绩效指标计算模块
"""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> float:
    """计算夏普比率"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> float:
    """计算索提诺比率"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    
    return excess_returns.mean() / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """计算最大回撤"""
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """计算卡尔玛比率"""
    portfolio_values = (1 + returns).cumprod()
    max_dd = calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0:
        return 0.0
    
    annual_return = returns.mean() * periods_per_year
    return annual_return / abs(max_dd)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """计算信息比率"""
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() * periods_per_year / tracking_error


def calculate_treynor_ratio(
    returns: pd.Series,
    beta: float,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> float:
    """计算特雷诺比率"""
    excess_returns = returns.mean() * periods_per_year - risk_free_rate
    if beta == 0:
        return 0.0
    return excess_returns / beta


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> float:
    """计算Alpha"""
    excess_returns = returns - risk_free_rate / periods_per_year
    excess_benchmark = benchmark_returns - risk_free_rate / periods_per_year
    
    covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
    variance = np.var(excess_benchmark)
    
    if variance == 0:
        return 0.0
    
    beta = covariance / variance
    alpha = (excess_returns.mean() - beta * excess_benchmark.mean()) * periods_per_year
    
    return alpha


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """计算Beta"""
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    
    if variance == 0:
        return 0.0
    
    return covariance / variance


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """计算VaR (Value at Risk)"""
    return returns.quantile(1 - confidence_level)


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """计算CVaR (Conditional Value at Risk)"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()
