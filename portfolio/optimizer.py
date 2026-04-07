"""
组合优化器
均值方差模型、风险平价模型等
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, method: str = 'mean_variance'):
        self.method = method
    
    def mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[dict] = None
    ) -> np.ndarray:
        """均值方差优化"""
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -portfolio_return + risk_aversion * portfolio_variance
        
        initial_weights = np.ones(n_assets) / n_assets
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        constraint_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints:
            if 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
            if 'min_weight' in constraints:
                bounds = tuple((constraints['min_weight'], 1) for _ in range(n_assets))
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list
        )
        
        return result.x
    
    def risk_parity_optimization(
        self,
        cov_matrix: np.ndarray,
        target_risk: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """风险平价优化"""
        n_assets = cov_matrix.shape[0]
        
        if target_risk is None:
            target_risk = np.ones(n_assets) / n_assets
        
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
            risk_contribution = weights * marginal_risk
            risk_contribution = risk_contribution / np.sum(risk_contribution)
            return np.sum((risk_contribution - target_risk) ** 2)
        
        initial_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def optimize(
        self,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: np.ndarray = None,
        **kwargs
    ) -> np.ndarray:
        """执行优化"""
        if self.method == 'mean_variance':
            if expected_returns is None:
                raise ValueError("均值方差优化需要预期收益率")
            return self.mean_variance_optimization(expected_returns, cov_matrix, **kwargs)
        elif self.method == 'risk_parity':
            return self.risk_parity_optimization(cov_matrix, **kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")
