import numpy as np
import pandas as pd
import pytest

from factors.factor_panel import (
    AVAILABLE_FACTOR_SPECS,
    build_factor_panel,
    calculate_single_stock_factors,
    list_available_factors,
    standardize_panel,
)


def make_price_data(offset=0.0):
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(10 + offset, 20 + offset, len(dates)), index=dates)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(np.linspace(1000 + offset, 2000 + offset, len(dates)), index=dates)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_list_available_factors_contains_existing_indicator_factors():
    factors = list_available_factors()

    assert {"name", "category", "required_columns", "description", "params"}.issubset(factors.columns)
    assert "rsi_14" in set(factors["name"])
    assert "macd_hist" in set(factors["name"])
    assert "rsrs_score" in set(factors["name"])
    assert set(factors["name"]) == set(AVAILABLE_FACTOR_SPECS)


def test_calculate_single_stock_factors_returns_requested_columns():
    data = make_price_data()

    result = calculate_single_stock_factors(
        data,
        factor_names=["ret_1d", "sma_ratio_20", "rsi_14", "volume_ratio_20"],
    )

    assert list(result.columns) == ["ret_1d", "sma_ratio_20", "rsi_14", "volume_ratio_20"]
    assert result.index.equals(data.index)
    assert np.isclose(result["ret_1d"].iloc[-1], data["close"].pct_change().iloc[-1])


def test_calculate_single_stock_factors_validates_factor_names():
    with pytest.raises(ValueError, match="Unknown factors"):
        calculate_single_stock_factors(make_price_data(), factor_names=["missing_factor"])


def test_calculate_single_stock_factors_validates_required_columns():
    data = make_price_data().drop("volume", axis=1)

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_single_stock_factors(data, factor_names=["mfi_14"])


def test_build_factor_panel_uses_date_symbol_multiindex():
    stock_data = {
        "AAA": make_price_data(),
        "BBB": make_price_data(offset=5.0),
    }

    panel = build_factor_panel(stock_data, factor_names=["ret_1d", "rsi_14"])

    assert panel.index.names == ["date", "symbol"]
    assert list(panel.columns) == ["ret_1d", "rsi_14"]
    assert ("AAA" in panel.index.get_level_values("symbol"))
    assert ("BBB" in panel.index.get_level_values("symbol"))
    assert len(panel) == 240


def test_standardize_panel_cross_section_by_date():
    stock_data = {
        "AAA": make_price_data(),
        "BBB": make_price_data(offset=5.0),
        "CCC": make_price_data(offset=10.0),
    }
    panel = build_factor_panel(stock_data, factor_names=["sma_ratio_20"])

    standardized = standardize_panel(panel, factor_columns=["sma_ratio_20"])
    last_date = standardized.index.get_level_values("date").max()
    values = standardized.loc[last_date, "sma_ratio_20"].dropna()

    assert np.isclose(values.mean(), 0.0)
    assert np.isclose(values.std(), 1.0)
