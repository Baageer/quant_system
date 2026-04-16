"""
布林带策略
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from signals.indicators import bollinger_bands, sma, supertrend


def _is_missing(value: float) -> bool:
    """兼容当前环境的 NaN 判断。"""
    return bool(np.isnan(value))


def _validate_required_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _create_trend_filters(
    data: pd.DataFrame,
    price_col: str,
    use_trend_filter: bool,
    trend_window: int,
    trend_slope_window: int,
) -> Tuple[pd.Series, pd.Series]:
    if not use_trend_filter:
        allow_long = pd.Series(True, index=data.index, dtype=bool)
        allow_short = pd.Series(True, index=data.index, dtype=bool)
        return allow_long, allow_short

    prices = data[price_col]
    trend_ma = sma(prices, trend_window)
    trend_rising = trend_ma > trend_ma.shift(trend_slope_window)
    trend_falling = trend_ma < trend_ma.shift(trend_slope_window)

    allow_long = ((prices > trend_ma) & trend_rising).fillna(False)
    allow_short = ((prices < trend_ma) & trend_falling).fillna(False)
    return allow_long, allow_short


def _create_volume_filter(
    data: pd.DataFrame,
    volume_col: str,
    use_volume_filter: bool,
    volume_window: int,
    volume_multiplier: float,
) -> pd.Series:
    if not use_volume_filter:
        return pd.Series(True, index=data.index, dtype=bool)

    volume = data[volume_col]
    volume_ma = sma(volume, volume_window)
    volume_expansion = ((volume > volume_ma * volume_multiplier) & (volume > volume.shift(1))).fillna(False)
    return volume_expansion


def _apply_signal_delay(signal: pd.Series, signal_delay: int) -> pd.Series:
    if signal_delay <= 0:
        return signal.astype(int)
    return signal.shift(signal_delay).fillna(0).astype(int)


def bollinger_breakout_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = 'close',
    use_trend_filter: bool = False,
    trend_window: int = 60,
    trend_slope_window: int = 3,
    use_volume_filter: bool = False,
    volume_window: int = 20,
    volume_multiplier: float = 1.2,
    volume_col: str = 'volume',
    signal_delay: int = 0,
) -> pd.Series:
    """
    布林带突破策略

    价格突破上轨 -> 做多
    价格跌破下轨 -> 做空
    价格回归中轨 -> 平仓

    可选增加价格趋势与放量确认，避免无趋势或缩量假突破。
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = bollinger_bands(prices, window, num_std)

    allow_long, allow_short = _create_trend_filters(
        data=data,
        price_col=price_col,
        use_trend_filter=use_trend_filter,
        trend_window=trend_window,
        trend_slope_window=trend_slope_window,
    )
    volume_expansion = _create_volume_filter(
        data=data,
        volume_col=volume_col,
        use_volume_filter=use_volume_filter,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(1, len(data)):
        if _is_missing(upper_band.iloc[i]) or _is_missing(lower_band.iloc[i]):
            continue

        price = prices.iloc[i]
        prev_price = prices.iloc[i - 1]

        break_upper = price > upper_band.iloc[i] and prev_price <= upper_band.iloc[i - 1]
        break_lower = price < lower_band.iloc[i] and prev_price >= lower_band.iloc[i - 1]
        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i - 1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i - 1]

        long_confirmed = allow_long.iloc[i] and volume_expansion.iloc[i]
        short_confirmed = allow_short.iloc[i] and volume_expansion.iloc[i]

        if break_upper and long_confirmed:
            current_pos = 1
        elif break_lower and short_confirmed:
            current_pos = -1
        elif current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0

        position.iloc[i] = current_pos

    return _apply_signal_delay(position, signal_delay)


def bollinger_mean_reversion_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = 'close',
    signal_delay: int = 0,
) -> pd.Series:
    """
    布林带均值回归策略

    价格触及上轨后回落 -> 做空
    价格触及下轨后反弹 -> 做多
    价格回归中轨 -> 平仓
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = bollinger_bands(prices, window, num_std)

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(1, len(data)):
        if _is_missing(upper_band.iloc[i]) or _is_missing(lower_band.iloc[i]):
            continue

        price = prices.iloc[i]
        prev_price = prices.iloc[i - 1]

        pullback_from_upper = prev_price >= upper_band.iloc[i - 1] and price < upper_band.iloc[i]
        bounce_from_lower = prev_price <= lower_band.iloc[i - 1] and price > lower_band.iloc[i]
        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i - 1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i - 1]

        if bounce_from_lower:
            current_pos = 1
        elif pullback_from_upper:
            current_pos = -1
        elif current_pos == 1 and cross_middle_up:
            current_pos = 0
        elif current_pos == -1 and cross_middle_down:
            current_pos = 0

        position.iloc[i] = current_pos

    return _apply_signal_delay(position, signal_delay)


def bollinger_squeeze_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    squeeze_threshold: Optional[float] = 0.02,
    price_col: str = 'close',
    squeeze_quantile: float = 0.1,
    squeeze_lookback: int = 60,
    use_supertrend_filter: bool = False,
    use_trend_filter: bool = False,
    trend_window: int = 60,
    trend_slope_window: int = 3,
    use_volume_filter: bool = False,
    volume_window: int = 20,
    volume_multiplier: float = 1.2,
    volume_col: str = 'volume',
    signal_delay: int = 0,
) -> pd.Series:
    """
    布林带收窄突破策略

    带宽收窄后等待方向性突破，可选叠加 SuperTrend、价格趋势和放量确认。
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = bollinger_bands(prices, window, num_std)
    bandwidth = (upper_band - lower_band) / middle_band.replace(0, np.nan)
    # print("squeeze_threshold", squeeze_threshold, squeeze_threshold is None)
    if squeeze_threshold is None:
        adaptive_threshold = bandwidth.rolling(
            window=squeeze_lookback,
            min_periods=max(window, 5),
        ).quantile(squeeze_quantile)
    else:
        adaptive_threshold = pd.Series(squeeze_threshold, index=data.index, dtype=float)

    if use_supertrend_filter:
        _, trend_direction = supertrend(data["high"], data["low"], prices)
        supertrend_allow_long = (trend_direction > 0).fillna(False)
        supertrend_allow_short = (trend_direction < 0).fillna(False)
    else:
        supertrend_allow_long = pd.Series(True, index=data.index, dtype=bool)
        supertrend_allow_short = pd.Series(True, index=data.index, dtype=bool)

    allow_long, allow_short = _create_trend_filters(
        data=data,
        price_col=price_col,
        use_trend_filter=use_trend_filter,
        trend_window=trend_window,
        trend_slope_window=trend_slope_window,
    )
    volume_expansion = _create_volume_filter(
        data=data,
        volume_col=volume_col,
        use_volume_filter=use_volume_filter,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0
    in_squeeze = False

    for i in range(1, len(data)):
        if _is_missing(upper_band.iloc[i]) or _is_missing(lower_band.iloc[i]):
            continue
        if _is_missing(adaptive_threshold.iloc[i]):
            continue

        price = prices.iloc[i]
        prev_price = prices.iloc[i - 1]

        if bandwidth.iloc[i] <= adaptive_threshold.iloc[i]:
            in_squeeze = True
        elif in_squeeze:
            break_upper = price > upper_band.iloc[i] and prev_price <= upper_band.iloc[i - 1]
            break_lower = price < lower_band.iloc[i] and prev_price >= lower_band.iloc[i - 1]

            long_confirmed = allow_long.iloc[i] and volume_expansion.iloc[i] and supertrend_allow_long.iloc[i]
            short_confirmed = allow_short.iloc[i] and volume_expansion.iloc[i] and supertrend_allow_short.iloc[i]

            if break_upper and long_confirmed:
                current_pos = 1
            elif break_lower and short_confirmed:
                current_pos = -1

            in_squeeze = False

        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i - 1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i - 1]

        if current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0

        position.iloc[i] = current_pos

    return _apply_signal_delay(position, signal_delay)


def bollinger_double_signal(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    confirm_bars: int = 2,
    price_col: str = 'close',
    use_trend_filter: bool = False,
    trend_window: int = 60,
    trend_slope_window: int = 3,
    use_volume_filter: bool = False,
    volume_window: int = 20,
    volume_multiplier: float = 1.2,
    volume_col: str = 'volume',
    signal_delay: int = 0,
) -> pd.Series:
    """
    布林带双确认突破策略

    价格连续 N 根 K 线收于轨道外后入场，可选叠加趋势和放量确认。
    """
    prices = data[price_col]
    upper_band, middle_band, lower_band = bollinger_bands(prices, window, num_std)

    allow_long, allow_short = _create_trend_filters(
        data=data,
        price_col=price_col,
        use_trend_filter=use_trend_filter,
        trend_window=trend_window,
        trend_slope_window=trend_slope_window,
    )
    volume_expansion = _create_volume_filter(
        data=data,
        volume_col=volume_col,
        use_volume_filter=use_volume_filter,
        volume_window=volume_window,
        volume_multiplier=volume_multiplier,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0
    above_upper_count = 0
    below_lower_count = 0

    for i in range(1, len(data)):
        if _is_missing(upper_band.iloc[i]) or _is_missing(lower_band.iloc[i]):
            continue

        price = prices.iloc[i]
        prev_price = prices.iloc[i - 1]

        if price > upper_band.iloc[i]:
            above_upper_count += 1
            below_lower_count = 0
        elif price < lower_band.iloc[i]:
            below_lower_count += 1
            above_upper_count = 0
        else:
            above_upper_count = 0
            below_lower_count = 0

        long_confirmed = allow_long.iloc[i] and volume_expansion.iloc[i]
        short_confirmed = allow_short.iloc[i] and volume_expansion.iloc[i]

        if above_upper_count >= confirm_bars and long_confirmed:
            current_pos = 1
        elif below_lower_count >= confirm_bars and short_confirmed:
            current_pos = -1

        cross_middle_up = price > middle_band.iloc[i] and prev_price <= middle_band.iloc[i - 1]
        cross_middle_down = price < middle_band.iloc[i] and prev_price >= middle_band.iloc[i - 1]

        if current_pos == 1 and cross_middle_down:
            current_pos = 0
        elif current_pos == -1 and cross_middle_up:
            current_pos = 0

        position.iloc[i] = current_pos

    return _apply_signal_delay(position, signal_delay)


class BollingerBandsStrategy:
    """布林带策略类"""

    VALID_MODES = {'breakout', 'mean_reversion', 'squeeze', 'double'}

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        mode: str = 'breakout',
        squeeze_threshold: Optional[float] = 0.02,
        confirm_bars: int = 2,
        squeeze_quantile: float = 0.2,
        squeeze_lookback: int = 60,
        use_supertrend_filter: bool = False,
        use_trend_filter: bool = False,
        trend_window: int = 60,
        trend_slope_window: int = 3,
        use_volume_filter: bool = False,
        volume_window: int = 20,
        volume_multiplier: float = 1.2,
        signal_delay: int = 0,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: str = 'volume',
    ):
        """
        参数:
            window: 移动平均周期
            num_std: 标准差倍数
            mode: 策略模式
            squeeze_threshold: 固定收窄阈值，为 None 时启用自适应阈值
            confirm_bars: 双确认模式所需的轨道外 K 线数量
            squeeze_quantile: 自适应收窄阈值分位数
            squeeze_lookback: 自适应收窄阈值回看窗口
            use_supertrend_filter: squeeze 模式是否叠加 SuperTrend 方向确认
            use_trend_filter: 是否叠加价格趋势确认
            trend_window: 趋势均线周期
            trend_slope_window: 趋势斜率回看周期
            use_volume_filter: 是否叠加成交量放大确认
            volume_window: 成交量均线周期
            volume_multiplier: 放量阈值，当前量需大于均量 * 该倍数
            signal_delay: 信号延迟
        """
        if window <= 1:
            raise ValueError("window must be greater than 1")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be positive")
        if signal_delay < 0:
            raise ValueError("signal_delay must be non-negative")
        if trend_window <= 1:
            raise ValueError("trend_window must be greater than 1")
        if trend_slope_window <= 0:
            raise ValueError("trend_slope_window must be positive")
        if volume_window <= 1:
            raise ValueError("volume_window must be greater than 1")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive")
        if not 0 < squeeze_quantile < 1:
            raise ValueError("squeeze_quantile must be between 0 and 1")
        if squeeze_lookback <= 1:
            raise ValueError("squeeze_lookback must be greater than 1")

        if squeeze_threshold == "None":
            squeeze_threshold = None
        self.window = window
        self.num_std = num_std
        self.mode = mode
        self.squeeze_threshold = squeeze_threshold
        self.confirm_bars = confirm_bars
        self.squeeze_quantile = squeeze_quantile
        self.squeeze_lookback = squeeze_lookback
        self.use_supertrend_filter = use_supertrend_filter
        self.use_trend_filter = use_trend_filter
        self.trend_window = trend_window
        self.trend_slope_window = trend_slope_window
        self.use_volume_filter = use_volume_filter
        self.volume_window = volume_window
        self.volume_multiplier = volume_multiplier
        self.signal_delay = signal_delay
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col

    def _required_columns(self) -> List[str]:
        required_columns = [self.price_col]
        if self.mode == 'squeeze':
            required_columns.extend([self.high_col, self.low_col])
        return list(dict.fromkeys(required_columns))

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        _validate_required_columns(data, self._required_columns())

        if self.mode == 'mean_reversion':
            return bollinger_mean_reversion_signal(
                data=data,
                window=self.window,
                num_std=self.num_std,
                price_col=self.price_col,
                signal_delay=self.signal_delay,
            )

        if self.mode == 'squeeze':
            squeeze_data = data.rename(
                columns={
                    self.high_col: 'high',
                    self.low_col: 'low',
                    self.price_col: self.price_col,
                    self.volume_col: self.volume_col,
                }
            )
            return bollinger_squeeze_signal(
                data=squeeze_data,
                window=self.window,
                num_std=self.num_std,
                squeeze_threshold=self.squeeze_threshold,
                price_col=self.price_col,
                squeeze_quantile=self.squeeze_quantile,
                squeeze_lookback=self.squeeze_lookback,
                use_supertrend_filter=self.use_supertrend_filter,
                signal_delay=self.signal_delay,
            )

        if self.mode == 'double':
            return bollinger_double_signal(
                data=data,
                window=self.window,
                num_std=self.num_std,
                confirm_bars=self.confirm_bars,
                price_col=self.price_col,
                signal_delay=self.signal_delay,
            )

        return bollinger_breakout_signal(
            data=data,
            window=self.window,
            num_std=self.num_std,
            price_col=self.price_col,
            signal_delay=self.signal_delay,
        )

    def get_bollinger_values(self, data: pd.DataFrame) -> dict:
        """获取布林带值用于可视化"""
        upper_band, middle_band, lower_band = bollinger_bands(
            data[self.price_col],
            self.window,
            self.num_std,
        )
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
        }
