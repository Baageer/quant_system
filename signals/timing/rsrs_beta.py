"""
RSRS Beta timing strategy.
基于 RSRS beta 值的择时策略，负向因子（beta 值越小，预期收益越高）
"""
from typing import Dict

import pandas as pd
import numpy as np

from signals.indicators import rsrs
from signals.timing.common_filters import apply_trend_filter, apply_volume_filter


def _is_missing(value: float) -> bool:
    """检查值是否为缺失值"""
    return value != value


def _validate_required_columns(data: pd.DataFrame, required_columns) -> None:
    """验证数据框是否包含必需的列"""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def rsrs_beta_signal(
    data: pd.DataFrame,
    window: int = 20,
    zscore_window: int = 90,
    min_valid_window: int = 12,
    entry_zscore: float = 0.5,
    exit_zscore: float = 0.0,
    use_negative: bool = True,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """
    RSRS beta 择时信号

    信号语义:
    - 1: 做多
    - -1: 做空/平多
    - 0: 空仓

    参数:
        data: 包含价格数据的DataFrame
        window: RSRS 计算窗口
        zscore_window: Z-score 标准化窗口
        min_valid_window: 最小有效窗口
        entry_zscore: 入场 Z-score 阈值
        exit_zscore: 出场 Z-score 阈值
        use_negative: 是否使用负向逻辑（beta 值越小=预期收益越高）
        high_col: 最高价列名
        low_col: 最低价列名

    返回:
        信号序列
    """
    _validate_required_columns(data, [high_col, low_col])

    # 计算 RSRS beta 和 zscore
    beta, r2, zscore, score = rsrs(
        high=data[high_col],
        low=data[low_col],
        window=window,
        zscore_window=zscore_window,
        min_valid_window=min_valid_window,
        use_r2_weight=True,
        use_beta_adjustment=False,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(len(data)):
        current_zscore = zscore.iloc[i]
        if _is_missing(current_zscore):
            position.iloc[i] = current_pos
            continue

        if use_negative:
            # 负向逻辑：beta 值越小（zscore 为负），预期收益越高
            if current_pos == 0:
                if current_zscore <= -entry_zscore:
                    current_pos = 1
                elif current_zscore >= entry_zscore:
                    current_pos = -1
            elif current_pos == 1:
                if current_zscore >= exit_zscore:
                    current_pos = 0
                elif current_zscore >= entry_zscore:
                    current_pos = -1
            else:  # current_pos == -1
                if current_zscore <= -exit_zscore:
                    current_pos = 0
                elif current_zscore <= -entry_zscore:
                    current_pos = 1
        else:
            # 正向逻辑：beta 值越大（zscore 为正），预期收益越高
            if current_pos == 0:
                if current_zscore >= entry_zscore:
                    current_pos = 1
                elif current_zscore <= -entry_zscore:
                    current_pos = -1
            elif current_pos == 1:
                if current_zscore <= exit_zscore:
                    current_pos = 0
                elif current_zscore <= -entry_zscore:
                    current_pos = -1
            else:  # current_pos == -1
                if current_zscore >= -exit_zscore:
                    current_pos = 0
                elif current_zscore >= entry_zscore:
                    current_pos = 1

        position.iloc[i] = current_pos

    return position


class RSRSBetaStrategy:
    """RSRS beta 策略类"""

    def __init__(
        self,
        window: int = 20,
        zscore_window: int = 90,
        min_valid_window: int = 12,
        entry_zscore: float = 0.5,
        exit_zscore: float = 0.0,
        use_negative: bool = True,
        use_trend_filter: bool = True,
        trend_window: int = 60,
        trend_slope_window: int = 3,
        use_volume_filter: bool = True,
        volume_window: int = 20,
        volume_multiplier: float = 1.5,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ):
        """
        初始化 RSRS beta 策略

        参数:
            window: RSRS 计算窗口
            zscore_window: Z-score 标准化窗口
            min_valid_window: 最小有效窗口
            entry_zscore: 入场 Z-score 阈值
            exit_zscore: 出场 Z-score 阈值
            use_negative: 是否使用负向逻辑
            use_trend_filter: 是否使用趋势过滤
            trend_window: 趋势过滤窗口
            trend_slope_window: 趋势斜率窗口
            use_volume_filter: 是否使用成交量过滤
            volume_window: 成交量过滤窗口
            volume_multiplier: 成交量倍数
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
        """
        if window < 2:
            raise ValueError("window must be at least 2")
        if zscore_window < 10:
            raise ValueError("zscore_window must be at least 10")
        if min_valid_window < 2:
            raise ValueError("min_valid_window must be at least 2")
        if entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if exit_zscore < 0:
            raise ValueError("exit_zscore must be non-negative")
        if trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive")

        self.window = window
        self.zscore_window = zscore_window
        self.min_valid_window = min_valid_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.use_negative = use_negative
        self.use_trend_filter = use_trend_filter
        self.trend_window = trend_window
        self.trend_slope_window = trend_slope_window
        self.use_volume_filter = use_volume_filter
        self.volume_window = volume_window
        self.volume_multiplier = volume_multiplier
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        # 生成基础信号
        signal = rsrs_beta_signal(
            data=data,
            window=self.window,
            zscore_window=self.zscore_window,
            min_valid_window=self.min_valid_window,
            entry_zscore=self.entry_zscore,
            exit_zscore=self.exit_zscore,
            use_negative=self.use_negative,
            high_col=self.high_col,
            low_col=self.low_col,
        )

        # 应用趋势过滤
        if self.use_trend_filter:
            signal = apply_trend_filter(
                data=data,
                signal=signal,
                window=self.trend_window,
                slope_window=self.trend_slope_window,
                price_col=self.close_col,
            )

        # 应用成交量过滤
        if self.use_volume_filter:
            signal = apply_volume_filter(
                data=data,
                signal=signal,
                window=self.volume_window,
                multiplier=self.volume_multiplier,
                volume_col=self.volume_col,
            )

        return signal

    def get_rsrs_beta_values(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """获取 RSRS beta 值用于可视化"""
        _validate_required_columns(data, [self.high_col, self.low_col])

        beta, r2, zscore, score = rsrs(
            high=data[self.high_col],
            low=data[self.low_col],
            window=self.window,
            zscore_window=self.zscore_window,
            min_valid_window=self.min_valid_window,
            use_r2_weight=True,
            use_beta_adjustment=False,
        )

        return {
            "rsrs_beta": beta,
            "rsrs_r2": r2,
            "rsrs_zscore": zscore,
            "rsrs_score": score,
            "entry_zscore": pd.Series(self.entry_zscore, index=data.index),
            "exit_zscore": pd.Series(self.exit_zscore, index=data.index),
            "negative_entry_zscore": pd.Series(-self.entry_zscore, index=data.index),
            "negative_exit_zscore": pd.Series(-self.exit_zscore, index=data.index),
        }