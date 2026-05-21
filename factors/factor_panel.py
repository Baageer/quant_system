"""
Unified factor panel builder.

This module converts existing technical indicators into research-friendly
factor columns and can merge per-symbol data into a MultiIndex panel:
``date, symbol -> factor values``.
"""

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from signals.indicators import (
    adx,
    atr,
    bollinger_bands,
    bollinger_bandwidth,
    cci,
    donchian_channel,
    kdj,
    keltner_channel,
    macd,
    mfi,
    momentum,
    obv,
    roc,
    rsi,
    rsrs,
    sma,
    supertrend,
    trix,
    volatility,
    vwap,
    williams_r,
    z_score,
)


FactorFunc = Callable[[pd.DataFrame], pd.Series]


class FactorSpec:
    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        required_columns: Tuple[str, ...],
        compute: FactorFunc,
        params: Optional[Dict[str, object]] = None,
    ):
        self.name = name
        self.category = category
        self.description = description
        self.required_columns = required_columns
        self.compute = compute
        self.params = {} if params is None else dict(params)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for factor calculation: {missing}")


def _close(data: pd.DataFrame) -> pd.Series:
    return data["close"].astype(float)


def _high(data: pd.DataFrame) -> pd.Series:
    return data["high"].astype(float)


def _low(data: pd.DataFrame) -> pd.Series:
    return data["low"].astype(float)


def _volume(data: pd.DataFrame) -> pd.Series:
    return data["volume"].astype(float)


def _return(window: int) -> FactorFunc:
    return lambda data: _close(data).pct_change(window)


def _sma_ratio(window: int) -> FactorFunc:
    return lambda data: _safe_divide(_close(data), sma(_close(data), window)) - 1


def _volume_ratio(window: int) -> FactorFunc:
    return lambda data: _safe_divide(_volume(data), sma(_volume(data), window)) - 1


def _bollinger_percent_b(window: int = 20, num_std: float = 2.0) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        upper, _, lower = bollinger_bands(_close(data), window=window, num_std=num_std)
        return _safe_divide(_close(data) - lower, upper - lower)

    return compute


def _macd_part(part: str) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, histogram = macd(_close(data))
        values = {
            "line": macd_line,
            "signal": signal_line,
            "hist": histogram,
        }
        return values[part]

    return compute


def _kdj_part(part: str) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        k_value, d_value, j_value = kdj(_high(data), _low(data), _close(data))
        values = {
            "k": k_value,
            "d": d_value,
            "j": j_value,
        }
        return values[part]

    return compute


def _adx_part(part: str) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        adx_value, plus_di, minus_di = adx(_high(data), _low(data), _close(data))
        values = {
            "adx": adx_value,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
        return values[part]

    return compute


def _donchian_position(window: int = 20) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        upper, _, lower = donchian_channel(_high(data), _low(data), window=window)
        return _safe_divide(_close(data) - lower, upper - lower)

    return compute


def _keltner_position(window: int = 20, atr_mult: float = 2.0) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        upper, _, lower = keltner_channel(
            _high(data), _low(data), _close(data), window=window, atr_mult=atr_mult
        )
        return _safe_divide(_close(data) - lower, upper - lower)

    return compute


def _rsrs_part(part: str) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        beta, r2, zscore, score = rsrs(
            _high(data),
            _low(data),
            window=18,
            zscore_window=90,
            min_valid_window=12,
            use_r2_weight=True,
            use_beta_adjustment=False,
        )
        values = {
            "beta": beta,
            "r2": r2,
            "zscore": zscore,
            "score": score,
        }
        return values[part]

    return compute


def _supertrend_part(part: str) -> FactorFunc:
    def compute(data: pd.DataFrame) -> pd.Series:
        value, direction = supertrend(_high(data), _low(data), _close(data))
        values = {
            "value": value,
            "direction": direction,
        }
        return values[part]

    return compute


AVAILABLE_FACTOR_SPECS: Dict[str, FactorSpec] = {
    "ret_1d": FactorSpec("ret_1d", "price", "1-day close return", ("close",), _return(1), {"window": 1}),
    "ret_5d": FactorSpec("ret_5d", "price", "5-day close return", ("close",), _return(5), {"window": 5}),
    "ret_20d": FactorSpec("ret_20d", "price", "20-day close return", ("close",), _return(20), {"window": 20}),
    "momentum_10": FactorSpec(
        "momentum_10", "momentum", "10-day price momentum", ("close",), lambda data: momentum(_close(data), 10), {"window": 10}
    ),
    "roc_12": FactorSpec("roc_12", "momentum", "12-day rate of change", ("close",), lambda data: roc(_close(data), 12), {"window": 12}),
    "rsi_14": FactorSpec("rsi_14", "momentum", "14-day RSI", ("close",), lambda data: rsi(_close(data), 14), {"window": 14}),
    "macd_line": FactorSpec("macd_line", "momentum", "MACD line", ("close",), _macd_part("line")),
    "macd_signal": FactorSpec("macd_signal", "momentum", "MACD signal line", ("close",), _macd_part("signal")),
    "macd_hist": FactorSpec("macd_hist", "momentum", "MACD histogram", ("close",), _macd_part("hist")),
    "trix_14": FactorSpec("trix_14", "momentum", "14-day TRIX", ("close",), lambda data: trix(_close(data), 14), {"window": 14}),
    "sma_ratio_5": FactorSpec("sma_ratio_5", "trend", "Close relative to 5-day SMA", ("close",), _sma_ratio(5), {"window": 5}),
    "sma_ratio_20": FactorSpec("sma_ratio_20", "trend", "Close relative to 20-day SMA", ("close",), _sma_ratio(20), {"window": 20}),
    "sma_ratio_60": FactorSpec("sma_ratio_60", "trend", "Close relative to 60-day SMA", ("close",), _sma_ratio(60), {"window": 60}),
    "adx_14": FactorSpec("adx_14", "trend", "14-day ADX", ("high", "low", "close"), _adx_part("adx"), {"window": 14}),
    "plus_di_14": FactorSpec("plus_di_14", "trend", "14-day positive DI", ("high", "low", "close"), _adx_part("plus_di"), {"window": 14}),
    "minus_di_14": FactorSpec("minus_di_14", "trend", "14-day negative DI", ("high", "low", "close"), _adx_part("minus_di"), {"window": 14}),
    "supertrend_direction": FactorSpec(
        "supertrend_direction", "trend", "SuperTrend direction", ("high", "low", "close"), _supertrend_part("direction")
    ),
    "bollinger_bandwidth_20": FactorSpec(
        "bollinger_bandwidth_20",
        "volatility",
        "20-day Bollinger bandwidth",
        ("close",),
        lambda data: bollinger_bandwidth(_close(data), 20, 2.0),
        {"window": 20, "num_std": 2.0},
    ),
    "bollinger_percent_b_20": FactorSpec(
        "bollinger_percent_b_20",
        "volatility",
        "20-day Bollinger percent-b",
        ("close",),
        _bollinger_percent_b(20, 2.0),
        {"window": 20, "num_std": 2.0},
    ),
    "atr_pct_14": FactorSpec(
        "atr_pct_14",
        "volatility",
        "14-day ATR divided by close",
        ("high", "low", "close"),
        lambda data: _safe_divide(atr(_high(data), _low(data), _close(data), 14), _close(data)),
        {"window": 14},
    ),
    "volatility_20": FactorSpec(
        "volatility_20", "volatility", "20-day annualized volatility", ("close",), lambda data: volatility(_close(data), 20), {"window": 20}
    ),
    "zscore_20": FactorSpec("zscore_20", "reversion", "20-day close z-score", ("close",), lambda data: z_score(_close(data), 20), {"window": 20}),
    "cci_20": FactorSpec("cci_20", "reversion", "20-day CCI", ("high", "low", "close"), lambda data: cci(_high(data), _low(data), _close(data), 20), {"window": 20}),
    "williams_r_14": FactorSpec(
        "williams_r_14", "reversion", "14-day Williams %R", ("high", "low", "close"), lambda data: williams_r(_high(data), _low(data), _close(data), 14), {"window": 14}
    ),
    "kdj_k": FactorSpec("kdj_k", "reversion", "KDJ K value", ("high", "low", "close"), _kdj_part("k")),
    "kdj_d": FactorSpec("kdj_d", "reversion", "KDJ D value", ("high", "low", "close"), _kdj_part("d")),
    "kdj_j": FactorSpec("kdj_j", "reversion", "KDJ J value", ("high", "low", "close"), _kdj_part("j")),
    "mfi_14": FactorSpec(
        "mfi_14", "volume", "14-day Money Flow Index", ("high", "low", "close", "volume"), lambda data: mfi(_high(data), _low(data), _close(data), _volume(data), 14), {"window": 14}
    ),
    "obv": FactorSpec("obv", "volume", "On-balance volume", ("close", "volume"), lambda data: obv(_close(data), _volume(data))),
    "obv_slope_5": FactorSpec(
        "obv_slope_5",
        "volume",
        "5-day OBV change",
        ("close", "volume"),
        lambda data: obv(_close(data), _volume(data)).diff(5),
        {"window": 5},
    ),
    "volume_ratio_20": FactorSpec(
        "volume_ratio_20", "volume", "Volume relative to 20-day SMA", ("volume",), _volume_ratio(20), {"window": 20}
    ),
    "vwap_distance": FactorSpec(
        "vwap_distance",
        "volume",
        "Close relative to cumulative VWAP",
        ("high", "low", "close", "volume"),
        lambda data: _safe_divide(_close(data), vwap(_high(data), _low(data), _close(data), _volume(data))) - 1,
    ),
    "donchian_position_20": FactorSpec(
        "donchian_position_20", "channel", "Close position in 20-day Donchian channel", ("high", "low", "close"), _donchian_position(20), {"window": 20}
    ),
    "keltner_position_20": FactorSpec(
        "keltner_position_20", "channel", "Close position in 20-day Keltner channel", ("high", "low", "close"), _keltner_position(20, 2.0), {"window": 20, "atr_mult": 2.0}
    ),
    "rsrs_beta": FactorSpec("rsrs_beta", "rsrs", "RSRS regression beta", ("high", "low"), _rsrs_part("beta")),
    "rsrs_r2": FactorSpec("rsrs_r2", "rsrs", "RSRS regression R-squared", ("high", "low"), _rsrs_part("r2")),
    "rsrs_zscore": FactorSpec("rsrs_zscore", "rsrs", "RSRS beta z-score", ("high", "low"), _rsrs_part("zscore")),
    "rsrs_score": FactorSpec("rsrs_score", "rsrs", "RSRS weighted score", ("high", "low"), _rsrs_part("score")),
}


DEFAULT_FACTOR_NAMES: Tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_hist",
    "sma_ratio_20",
    "bollinger_bandwidth_20",
    "bollinger_percent_b_20",
    "atr_pct_14",
    "volatility_20",
    "zscore_20",
    "mfi_14",
    "volume_ratio_20",
    "rsrs_score",
)


def list_available_factors() -> pd.DataFrame:
    rows = []
    for spec in AVAILABLE_FACTOR_SPECS.values():
        rows.append(
            {
                "name": spec.name,
                "category": spec.category,
                "required_columns": ",".join(spec.required_columns),
                "description": spec.description,
                "params": dict(spec.params),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)


def calculate_single_stock_factors(
    data: pd.DataFrame,
    factor_names: Optional[Iterable[str]] = None,
    include_ohlcv: bool = False,
) -> pd.DataFrame:
    names = list(DEFAULT_FACTOR_NAMES if factor_names is None else factor_names)
    unknown = [name for name in names if name not in AVAILABLE_FACTOR_SPECS]
    if unknown:
        raise ValueError(f"Unknown factors: {unknown}")

    result = pd.DataFrame(index=data.index)
    if include_ohlcv:
        base_columns = [column for column in ("open", "high", "low", "close", "volume", "amount") if column in data.columns]
        result = data.loc[:, base_columns].copy()

    for name in names:
        spec = AVAILABLE_FACTOR_SPECS[name]
        _require_columns(data, spec.required_columns)
        series = spec.compute(data)
        result[name] = pd.Series(series, index=data.index, dtype=float)

    return result


def winsorize_panel(
    panel: pd.DataFrame,
    factor_columns: Optional[Sequence[str]] = None,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.DataFrame:
    columns = list(panel.columns if factor_columns is None else factor_columns)
    result = panel.copy()
    for column in columns:
        grouped = result[column].groupby(level="date")
        lower_bound = grouped.transform(lambda values: values.quantile(lower))
        upper_bound = grouped.transform(lambda values: values.quantile(upper))
        result[column] = result[column].clip(lower_bound, upper_bound)
    return result


def standardize_panel(
    panel: pd.DataFrame,
    factor_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    columns = list(panel.columns if factor_columns is None else factor_columns)
    result = panel.copy()
    for column in columns:
        grouped = result[column].groupby(level="date")
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        result[column] = (result[column] - mean) / std
    return result


def build_factor_panel(
    stock_data: Mapping[str, pd.DataFrame],
    factor_names: Optional[Iterable[str]] = None,
    include_ohlcv: bool = False,
    winsorize: bool = False,
    standardize: bool = False,
    winsorize_limits: Tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    names = list(DEFAULT_FACTOR_NAMES if factor_names is None else factor_names)
    frames: List[pd.DataFrame] = []

    for symbol, data in stock_data.items():
        factors = calculate_single_stock_factors(
            data=data,
            factor_names=names,
            include_ohlcv=include_ohlcv,
        )
        frame = factors.copy()
        frame["symbol"] = str(symbol)
        frame["date"] = frame.index
        frame = frame.set_index(["date", "symbol"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))

    panel = pd.concat(frames).sort_index()
    if winsorize:
        panel = winsorize_panel(
            panel,
            factor_columns=names,
            lower=winsorize_limits[0],
            upper=winsorize_limits[1],
        )
    if standardize:
        panel = standardize_panel(panel, factor_columns=names)
    return panel
