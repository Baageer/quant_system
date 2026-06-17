"""
VWAP Distance timing strategy.
基于 VWAP 偏离度的择时策略，负向因子（VWAP 距离越小，预期收益越高）
"""
from typing import Dict, Optional

import pandas as pd
import numpy as np

from signals.indicators import vwap_distance as calc_vwap_distance
from signals.timing.common_filters import apply_trend_filter, apply_volume_filter


def _is_missing(value: float) -> bool:
    """检查值是否为缺失值"""
    return value != value


def _validate_required_columns(data: pd.DataFrame, required_columns) -> None:
    """验证数据框是否包含必需的列"""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def vwap_distance_signal(
    data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.02,
    use_negative: bool = True,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """
    VWAP 距离择时信号

    信号语义:
    - 1: 做多
    - -1: 做空/平多
    - 0: 空仓

    参数:
        data: 包含价格和成交量的DataFrame
        window: VWAP 计算窗口
        threshold: 入场阈值
        use_negative: 是否使用负向逻辑（VWAP距离越小=预期收益越高）
        high_col: 最高价列名
        low_col: 最低价列名
        close_col: 收盘价列名
        volume_col: 成交量列名

    返回:
        信号序列
    """
    _validate_required_columns(data, [high_col, low_col, close_col, volume_col])

    # 计算 VWAP 距离
    vwap_dist = calc_vwap_distance(
        high=data[high_col],
        low=data[low_col],
        close=data[close_col],
        volume=data[volume_col],
        window=window,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(len(data)):
        current_dist = vwap_dist.iloc[i]
        if _is_missing(current_dist):
            position.iloc[i] = current_pos
            continue

        if use_negative:
            # 负向逻辑：VWAP 距离越小，预期收益越高
            if current_pos == 0:
                if current_dist <= -threshold:
                    current_pos = 1
                elif current_dist >= threshold:
                    current_pos = -1
            elif current_pos == 1:
                if current_dist >= 0:
                    current_pos = 0
                elif current_dist >= threshold:
                    current_pos = -1
            else:  # current_pos == -1
                if current_dist <= 0:
                    current_pos = 0
                elif current_dist <= -threshold:
                    current_pos = 1
        else:
            # 正向逻辑：VWAP 距离越大，预期收益越高
            if current_pos == 0:
                if current_dist >= threshold:
                    current_pos = 1
                elif current_dist <= -threshold:
                    current_pos = -1
            elif current_pos == 1:
                if current_dist <= 0:
                    current_pos = 0
                elif current_dist <= -threshold:
                    current_pos = -1
            else:  # current_pos == -1
                if current_dist >= 0:
                    current_pos = 0
                elif current_dist >= threshold:
                    current_pos = 1

        position.iloc[i] = current_pos

    return position


class VWAPDistanceStrategy:
    """VWAP 距离策略类"""

    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.02,
        use_negative: bool = True,
        use_trend_filter: bool = False,
        trend_window: int = 60,
        trend_slope_window: int = 3,
        use_volume_filter: bool = True,
        volume_window: int = 20,
        volume_multiplier: float = 1.2,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ):
        """
        初始化 VWAP 距离策略

        参数:
            window: VWAP 计算窗口
            threshold: 入场阈值
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
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive")

        self.window = window
        self.threshold = threshold
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
        signal = vwap_distance_signal(
            data=data,
            window=self.window,
            threshold=self.threshold,
            use_negative=self.use_negative,
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
            volume_col=self.volume_col,
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

    def get_vwap_distance_values(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """获取 VWAP 距离值用于可视化"""
        _validate_required_columns(data, [self.high_col, self.low_col, self.close_col, self.volume_col])

        vwap_dist = calc_vwap_distance(
            high=data[self.high_col],
            low=data[self.low_col],
            close=data[self.close_col],
            volume=data[self.volume_col],
            window=self.window,
        )

        return {
            "vwap_distance": vwap_dist,
            "vwap_distance_threshold": pd.Series(self.threshold, index=data.index),
            "vwap_distance_negative_threshold": pd.Series(-self.threshold, index=data.index),
        }