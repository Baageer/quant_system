import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.indicator_cache import IndicatorCache
from signals.indicators import atr, bollinger_bands, kdj, macd, rsi


def _price_data(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(10, 20, n), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(np.arange(n) + 1000, index=idx)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_bollinger_cache_returns_expected_columns_and_values(tmpdir):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path, source="tushare", adjust_mode="hfq")

    result = cache.get_bollinger_bands(
        data=data,
        symbol="000001.SZ",
        window=20,
        num_std=2.0,
    )

    upper, middle, lower = bollinger_bands(data["close"], window=20, num_std=2.0)
    expected = pd.DataFrame(
        {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": (upper - lower) / middle,
        },
        index=data.index,
    )[["upper", "middle", "lower", "bandwidth"]]

    assert list(result.columns) == ["upper", "middle", "lower", "bandwidth"]
    pd.testing.assert_frame_equal(result, expected)
    assert len(list(tmp_path.rglob("*.csv"))) == 1
    assert len(list(tmp_path.rglob("*.meta.json"))) == 1


def test_bollinger_cache_reuses_valid_cache(tmpdir, monkeypatch):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)
    first = cache.get_indicator(
        "bollinger_bands",
        data=data,
        symbol="000001.SZ",
        params={"window": 20, "num_std": 2.0},
    )

    def fail_compute(*args, **kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(cache, "_compute_bollinger_bands", fail_compute)
    second = cache.get_indicator(
        "bollinger_bands",
        data=data,
        symbol="000001.SZ",
        params={"window": 20, "num_std": 2.0},
    )

    pd.testing.assert_frame_equal(first, second)


def test_bollinger_cache_separates_different_params(tmpdir):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)

    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20, num_std=2.0)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=30, num_std=2.0)

    meta_files = list(tmp_path.rglob("*.meta.json"))
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    indicator_entries = list(meta["indicators"].values())
    params = [entry["params"] for entry in indicator_entries]

    assert len(meta_files) == 1
    assert len(list(tmp_path.rglob("*.csv"))) == 1
    assert len(indicator_entries) == 2
    assert {"window": 20, "num_std": 2.0, "price_col": "close"} in params
    assert {"window": 30, "num_std": 2.0, "price_col": "close"} in params
    assert all(entry["params_hash"] for entry in indicator_entries)


def test_params_hash_is_stable_for_same_params_order(tmpdir):
    cache = IndicatorCache(cache_dir=Path(str(tmpdir)))

    first = cache._params_hash({"window": 20, "num_std": 2.0, "price_col": "close"})
    second = cache._params_hash({"price_col": "close", "num_std": 2.0, "window": 20})

    assert first == second


def test_cache_recomputes_when_meta_hash_mismatches(tmpdir, monkeypatch):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20)

    meta_path = list(tmp_path.rglob("*.meta.json"))[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    indicator_key = next(iter(meta["indicators"]))
    meta["indicators"][indicator_key]["params_hash"] = "bad_hash"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    cache = IndicatorCache(cache_dir=tmp_path)
    called = {"value": False}
    original_compute = cache._compute_bollinger_bands

    def wrapped_compute(*args, **kwargs):
        called["value"] = True
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(cache, "_compute_bollinger_bands", wrapped_compute)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20)

    assert called["value"] is True


def test_bollinger_cache_invalidates_when_data_changes(tmpdir, monkeypatch):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] = 99.0
    cache = IndicatorCache(cache_dir=tmp_path)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20)

    called = {"value": False}
    original_compute = cache._compute_bollinger_bands

    def wrapped_compute(*args, **kwargs):
        called["value"] = True
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(cache, "_compute_bollinger_bands", wrapped_compute)
    cache.get_bollinger_bands(changed, symbol="000001.SZ", window=20)

    assert called["value"] is True


def test_cache_recomputes_when_cached_columns_are_invalid(tmpdir, monkeypatch):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20)

    csv_path = list(tmp_path.rglob("*.csv"))[0]
    pd.DataFrame({"upper": [1.0]}, index=[data.index[0]]).to_csv(csv_path)

    cache = IndicatorCache(cache_dir=tmp_path)
    called = {"value": False}
    original_compute = cache._compute_bollinger_bands

    def wrapped_compute(*args, **kwargs):
        called["value"] = True
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(cache, "_compute_bollinger_bands", wrapped_compute)
    cache.get_bollinger_bands(data, symbol="000001.SZ", window=20)

    assert called["value"] is True


def test_indicator_cache_rejects_unsupported_indicator(tmpdir):
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsupported indicator"):
        cache.get_indicator("not_an_indicator", data=_price_data(), symbol="000001.SZ")


def test_macd_rsi_atr_kdj_values_share_one_wide_cache_file(tmpdir):
    data = _price_data()
    tmp_path = Path(str(tmpdir))
    cache = IndicatorCache(cache_dir=tmp_path)

    macd_result = cache.get_macd(data, symbol="000001.SZ")
    rsi_result = cache.get_rsi(data, symbol="000001.SZ")
    atr_result = cache.get_atr(data, symbol="000001.SZ")
    kdj_result = cache.get_kdj(data, symbol="000001.SZ")

    expected_macd = pd.DataFrame(
        dict(zip(["macd", "signal", "hist"], macd(data["close"]))),
        index=data.index,
    )[["macd", "signal", "hist"]]
    expected_rsi = pd.DataFrame({"rsi": rsi(data["close"])}, index=data.index)
    expected_atr = pd.DataFrame(
        {"atr": atr(data["high"], data["low"], data["close"])},
        index=data.index,
    )
    k, d, j = kdj(data["high"], data["low"], data["close"])
    expected_kdj = pd.DataFrame({"k": k, "d": d, "j": j}, index=data.index)[["k", "d", "j"]]

    pd.testing.assert_frame_equal(macd_result, expected_macd)
    pd.testing.assert_frame_equal(rsi_result, expected_rsi)
    pd.testing.assert_frame_equal(atr_result, expected_atr)
    pd.testing.assert_frame_equal(kdj_result, expected_kdj)

    assert len(list(tmp_path.rglob("*.csv"))) == 1
    assert len(list(tmp_path.rglob("*.meta.json"))) == 1

    meta_path = list(tmp_path.rglob("*.meta.json"))[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    indicators = [entry["indicator"] for entry in meta["indicators"].values()]
    assert sorted(indicators) == ["atr", "kdj", "macd", "rsi"]


def test_get_indicator_dispatches_supported_indicators(tmpdir):
    data = _price_data()
    cache = IndicatorCache(cache_dir=Path(str(tmpdir)))

    result = cache.get_indicator(
        "macd",
        data=data,
        symbol="000001.SZ",
        params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
    )

    assert list(result.columns) == ["macd", "signal", "hist"]
