from pathlib import Path

import pandas as pd

from data.data_api import DataAPI


def _build_tmp_dirs(tmpdir):
    tmp_path = Path(str(tmpdir))
    return str(tmp_path / "raw"), str(tmp_path / "processed")


def test_tushare_stock_list_normalization(tmpdir, monkeypatch):
    raw_dir, processed_dir = _build_tmp_dirs(tmpdir)
    universe_file = Path(str(tmpdir)) / "stocks.txt"
    universe_file.write_text("600000\n000001.SZ\n300001\n", encoding="utf-8")

    monkeypatch.setattr(DataAPI, "_init_tushare", lambda self: setattr(self, "pro", object()))

    api = DataAPI(
        source="tushare",
        stock_file=str(universe_file),
        cache_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert api.get_stock_list() == ["600000.SH", "000001.SZ", "300001.SZ"]


def test_standardize_tushare_daily_payload(tmpdir):
    raw_dir, processed_dir = _build_tmp_dirs(tmpdir)
    api = DataAPI(source="akshare", cache_dir=raw_dir, processed_dir=processed_dir)

    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240103", "20240102"],
            "open": [11.0, 10.0],
            "high": [11.5, 10.5],
            "low": [10.8, 9.9],
            "close": [11.2, 10.2],
            "change": [0.2, 0.2],
            "pct_chg": [1.82, 2.0],
            "vol": [12345, 10000],
            "amount": [123456.0, 100000.0],
        }
    )

    standardized = api._standardize_daily_price_data(raw, "000001.SZ")

    assert list(standardized.columns) == list(DataAPI.STANDARD_PRICE_COLUMNS)
    assert standardized["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert standardized["code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert standardized["turnover"].isnull().all()


def test_get_financial_data_tushare_daily_basic(tmpdir, monkeypatch):
    raw_dir, processed_dir = _build_tmp_dirs(tmpdir)

    class FakePro:
        def __init__(self):
            self.calls = []

        def daily_basic(self, **kwargs):
            self.calls.append(kwargs)
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_date": ["20240103", "20240102"],
                    "pb": [1.2, 1.1],
                }
            )

    fake_pro = FakePro()
    monkeypatch.setattr(DataAPI, "_require_tushare", staticmethod(lambda: None))
    monkeypatch.setattr(DataAPI, "_init_tushare", lambda self: setattr(self, "pro", fake_pro))

    api = DataAPI(source="tushare", cache_dir=raw_dir, processed_dir=processed_dir)
    result = api.get_financial_data(
        symbol="000001",
        start_date="2024-01-01",
        end_date="2024-01-31",
        fields=["ts_code", "trade_date", "pb"],
    )

    assert len(fake_pro.calls) == 1
    call = fake_pro.calls[0]
    assert call["ts_code"] == "000001.SZ"
    assert call["start_date"] == "20240101"
    assert call["end_date"] == "20240131"
    assert call["fields"] == "ts_code,trade_date,pb"
    assert result["trade_date"].tolist() == ["20240102", "20240103"]
