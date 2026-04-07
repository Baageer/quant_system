"""
收益分析模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """计算收益率序列"""
        return portfolio_values.pct_change().dropna()
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """计算累计收益率"""
        return (1 + returns).cumprod() - 1
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """计算回撤"""
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """计算最大回撤"""
        drawdown = self.calculate_drawdown(portfolio_values)
        return drawdown.min()
    
    def analyze(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None
    ) -> Dict:
        """综合分析"""
        returns = self.calculate_returns(portfolio_values)
        cumulative_returns = self.calculate_cumulative_returns(returns)
        drawdown = self.calculate_drawdown(portfolio_values)
        
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        analysis = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown
        }
        
        if benchmark_values is not None:
            benchmark_returns = self.calculate_returns(benchmark_values)
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            analysis['tracking_error'] = tracking_error
            analysis['information_ratio'] = information_ratio
            analysis['excess_returns'] = excess_returns
        
        return analysis
    
    def generate_report(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """生成分析报告"""
        analysis = self.analyze(portfolio_values, benchmark_values)
        
        report_data = {
            '指标': [
                '总收益率',
                '年化收益率',
                '年化波动率',
                '夏普比率',
                '最大回撤',
                '卡尔玛比率',
                '胜率',
                '平均盈利',
                '平均亏损',
                '盈亏比'
            ],
            '数值': [
                f"{analysis['total_return']:.2%}",
                f"{analysis['annual_return']:.2%}",
                f"{analysis['annual_volatility']:.2%}",
                f"{analysis['sharpe_ratio']:.4f}",
                f"{analysis['max_drawdown']:.2%}",
                f"{analysis['calmar_ratio']:.4f}",
                f"{analysis['win_rate']:.2%}",
                f"{analysis['avg_win']:.4f}",
                f"{analysis['avg_loss']:.4f}",
                f"{analysis['profit_loss_ratio']:.4f}"
            ]
        }
        
        if 'tracking_error' in analysis:
            report_data['指标'].extend(['跟踪误差', '信息比率'])
            report_data['数值'].extend([
                f"{analysis['tracking_error']:.2%}",
                f"{analysis['information_ratio']:.4f}"
            ])
        
        return pd.DataFrame(report_data)
