"""
技术指标库
提供常用技术指标的计算函数，供策略使用
"""
import pandas as pd
import numpy as np
from typing import Tuple, Union


def sma(data: pd.Series, window: int) -> pd.Series:
    """
    简单移动平均线 (Simple Moving Average)
    
    参数:
        data: 价格序列
        window: 周期
    
    返回:
        SMA序列
    """
    return data.rolling(window=window, min_periods=window).mean()


def ema(data: pd.Series, span: int) -> pd.Series:
    """
    指数移动平均线 (Exponential Moving Average)
    
    参数:
        data: 价格序列
        span: 周期
    
    返回:
        EMA序列
    """
    return data.ewm(span=span, adjust=False).mean()


def wma(data: pd.Series, window: int) -> pd.Series:
    """
    加权移动平均线 (Weighted Moving Average)
    
    参数:
        data: 价格序列
        window: 周期
    
    返回:
        WMA序列
    """
    weights = np.arange(1, window + 1)
    return data.rolling(window=window, min_periods=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def bollinger_bands(
    data: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带指标 (Bollinger Bands)
    
    参数:
        data: 价格序列
        window: 移动平均周期，默认20
        num_std: 标准差倍数，默认2.0
    
    返回:
        (上轨, 中轨, 下轨)
    """
    middle = sma(data, window)
    std = data.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bollinger_bandwidth(
    data: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    布林带带宽 (Bollinger Bandwidth)
    
    参数:
        data: 价格序列
        window: 移动平均周期
        num_std: 标准差倍数
    
    返回:
        带宽序列
    """
    upper, middle, lower = bollinger_bands(data, window, num_std)
    return (upper - lower) / middle


def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    相对强弱指标 (Relative Strength Index)
    
    参数:
        data: 价格序列
        window: RSI计算周期，默认14
    
    返回:
        RSI值序列 (0-100)
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    avg_gain = avg_gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD指标 (Moving Average Convergence Divergence)
    
    参数:
        data: 价格序列
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26
        signal_period: 信号线周期，默认9
    
    返回:
        (MACD线, 信号线, MACD柱状图)
    """
    ema_fast = ema(data, fast_period)
    ema_slow = ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    平均真实波幅 (Average True Range)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认14
    
    返回:
        ATR序列
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(window=window, min_periods=window).mean()


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    KDJ随机指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV计算周期，默认9
        m1: K值平滑周期，默认3
        m2: D值平滑周期，默认3
    
    返回:
        (K值, D值, J值)
    """
    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    随机指标 (Stochastic Oscillator)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: K值周期，默认14
        d_period: D值周期，默认3
    
    返回:
        (K值, D值)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    k = (close - lowest_low) / (highest_high - lowest_low) * 100
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    
    return k, d


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    商品通道指标 (Commodity Channel Index)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认20
    
    返回:
        CCI序列
    """
    tp = (high + low + close) / 3
    ma = tp.rolling(window=window, min_periods=window).mean()
    md = tp.rolling(window=window, min_periods=window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    return (tp - ma) / (0.015 * md)


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    威廉指标 (Williams %R)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认14
    
    返回:
        Williams %R序列 (-100 到 0)
    """
    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    
    return (highest_high - close) / (highest_high - lowest_low) * -100


def obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    能量潮指标 (On-Balance Volume)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
    
    返回:
        OBV序列
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    平均趋向指标 (Average Directional Index)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认14
    
    返回:
        (ADX, +DI, -DI)
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = atr(high, low, close, 1)
    atr_val = tr.rolling(window=window, min_periods=window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / atr_val)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=window, min_periods=window).mean()
    
    return adx_val, plus_di, minus_di


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    资金流量指标 (Money Flow Index)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        window: 计算周期，默认14
    
    返回:
        MFI序列 (0-100)
    """
    tp = (high + low + close) / 3
    mf = tp * volume
    
    positive_mf = mf.where(tp > tp.shift(), 0)
    negative_mf = mf.where(tp < tp.shift(), 0)
    
    positive_sum = positive_mf.rolling(window=window, min_periods=window).sum()
    negative_sum = negative_mf.rolling(window=window, min_periods=window).sum()
    
    mfi_val = 100 - (100 / (1 + positive_sum / negative_sum))
    
    return mfi_val


def roc(data: pd.Series, window: int = 12) -> pd.Series:
    """
    变动率指标 (Rate of Change)
    
    参数:
        data: 价格序列
        window: 计算周期，默认12
    
    返回:
        ROC序列
    """
    return (data - data.shift(window)) / data.shift(window) * 100


def momentum(data: pd.Series, window: int = 10) -> pd.Series:
    """
    动量指标 (Momentum)
    
    参数:
        data: 价格序列
        window: 计算周期，默认10
    
    返回:
        动量序列
    """
    return data - data.shift(window)


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    成交量加权平均价格 (Volume Weighted Average Price)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
    
    返回:
        VWAP序列
    """
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    window: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    唐奇安通道 (Donchian Channel)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        window: 计算周期，默认20
    
    返回:
        (上轨, 中轨, 下轨)
    """
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    肯特纳通道 (Keltner Channel)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认20
        atr_mult: ATR倍数，默认2.0
    
    返回:
        (上轨, 中轨, 下轨)
    """
    middle = ema(close, window)
    atr_val = atr(high, low, close, window)
    
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    
    return upper, middle, lower


def trix(data: pd.Series, window: int = 14) -> pd.Series:
    """
    TRIX指标
    
    参数:
        data: 价格序列
        window: 计算周期，默认14
    
    返回:
        TRIX序列
    """
    ema1 = ema(data, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    
    return (ema3 - ema3.shift(1)) / ema3.shift(1) * 100


def dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    动向指标 (Directional Movement Index)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期，默认14
    
    返回:
        (PDI, MDI, DX, ADX)
    """
    adx_val, plus_di, minus_di = adx(high, low, close, window)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    return plus_di, minus_di, dx, adx_val


def typical_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    典型价格 (Typical Price)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    返回:
        典型价格序列
    """
    return (high + low + close) / 3


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> dict:
    """
    枢轴点 (Pivot Points)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    返回:
        包含枢轴点各层级的字典
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def z_score(data: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-Score标准化
    
    参数:
        data: 价格序列
        window: 计算周期，默认20
    
    返回:
        Z-Score序列
    """
    mean = data.rolling(window=window, min_periods=window).mean()
    std = data.rolling(window=window, min_periods=window).std()
    
    return (data - mean) / std


def volatility(
    data: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    历史波动率
    
    参数:
        data: 价格序列
        window: 计算周期，默认20
        annualize: 是否年化，默认True
    
    返回:
        波动率序列
    """
    returns = data.pct_change()
    vol = returns.rolling(window=window, min_periods=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        atr_period: ATR周期，默认10
        multiplier: ATR倍数，默认3.0
    
    返回:
        (SuperTrend值, 趋势方向: 1=上涨, -1=下跌)
    """
    atr_val = atr(high, low, close, atr_period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    supertrend_val = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    supertrend_val.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend_val.iloc[i-1]:
            direction.iloc[i] = 1
            supertrend_val.iloc[i] = max(lower_band.iloc[i], supertrend_val.iloc[i-1])
        else:
            direction.iloc[i] = -1
            supertrend_val.iloc[i] = min(upper_band.iloc[i], supertrend_val.iloc[i-1])
    
    return supertrend_val, direction
