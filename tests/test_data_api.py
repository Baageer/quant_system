import pandas as pd
import pytest
from pathlib import Path

from data.data_api import DataAPI


class TestDataAPI:
    def test_default_adjust_mode_is_hfq(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        )

        assert api.adjust_mode == "hfq"
        assert api.adjust_mode_label == "hfq"

    def test_cache_is_isolated_by_adjust_mode(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        data = pd.DataFrame(
            {
                "日期": ["2024-01-01", "2024-01-02"],
                "开盘": [10.0, 10.5],
                "收盘": [10.2, 10.8],
            }
        )

        hfq_api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            adjust_mode="hfq",
        )
        qfq_api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            adjust_mode="qfq",
        )

        filename = "000001_20240101_20240102.csv"
        hfq_api.save_to_cache(data, filename, "price")
        qfq_api.save_to_cache(data.assign(收盘=[8.2, 8.8]), filename, "price")

        hfq_path = tmp_path / "raw" / "akshare" / "price" / "hfq" / filename
        qfq_path = tmp_path / "raw" / "akshare" / "price" / "qfq" / filename

        assert hfq_path.exists()
        assert qfq_path.exists()
        assert hfq_api.load_from_cache(filename, "price")["收盘"].tolist() == [10.2, 10.8]
        assert qfq_api.load_from_cache(filename, "price")["收盘"].tolist() == [8.2, 8.8]

    def test_detect_non_positive_prices_finds_invalid_values(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        )
        data = pd.DataFrame(
            {
                "open": [10.0, 0.0, 9.8],
                "close": [10.2, -1.0, 9.9],
                "high": [10.5, 10.0, 10.1],
                "low": [9.9, 9.7, 0.0],
            }
        )

        issues = api.detect_non_positive_prices(data)

        assert issues == {"open": 1, "close": 1, "low": 1}
        assert api.has_non_positive_prices(data) is True

    def test_raw_price_adjust_requires_raw_adjust_mode(self, tmpdir):
        tmp_path = Path(str(tmpdir))

        with pytest.raises(ValueError, match="raw_price_adjust can only be used"):
            DataAPI(
                source="akshare",
                cache_dir=str(tmp_path / "raw"),
                processed_dir=str(tmp_path / "processed"),
                adjust_mode="qfq",
                raw_price_adjust="hfq",
            )

    def test_raw_price_adjust_builds_distinct_cache_label(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            adjust_mode="raw",
            raw_price_adjust="qfq",
        )

        assert api.adjust_mode == ""
        assert api.raw_price_adjust == "qfq"
        assert api.adjust_mode_label == "raw_qfq_pct"

    def test_raw_price_adjust_rebuilds_qfq_prices_from_pct_change(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            adjust_mode="raw",
            raw_price_adjust="qfq",
        )
        raw = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "code": ["000001"] * 3,
                "open": [9.8, 10.8, 5.8],
                "close": [10.0, 11.0, 6.0],
                "high": [10.1, 11.2, 6.2],
                "low": [9.7, 10.7, 5.7],
                "volume": [1000, 1200, 1500],
                "amount": [10000, 13000, 9000],
                "amplitude": [0.0, 5.0, 8.0],
                "pct_change": [None, 10.0, 10.0],
                "change": [0.0, 1.0, -5.0],
                "turnover": [1.0, 1.1, 1.2],
            }
        )

        adjusted = api._standardize_daily_price_data(raw, "000001")

        assert adjusted["close"].tolist() == pytest.approx(
            [4.958677685950414, 5.454545454545455, 6.0]
        )
        assert adjusted["pct_change"].iloc[1:].tolist() == pytest.approx([10.0, 10.0])
        assert adjusted["open"].iloc[-1] == pytest.approx(5.8)

    def test_raw_price_adjust_rebuilds_hfq_prices_from_pct_change(self, tmpdir):
        tmp_path = Path(str(tmpdir))
        api = DataAPI(
            source="akshare",
            cache_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            adjust_mode="raw",
            raw_price_adjust="hfq",
        )
        raw = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "code": ["000001"] * 3,
                "open": [9.8, 10.8, 5.8],
                "close": [10.0, 11.0, 6.0],
                "high": [10.1, 11.2, 6.2],
                "low": [9.7, 10.7, 5.7],
                "volume": [1000, 1200, 1500],
                "amount": [10000, 13000, 9000],
                "amplitude": [0.0, 5.0, 8.0],
                "pct_change": [None, 10.0, 10.0],
                "change": [0.0, 1.0, -5.0],
                "turnover": [1.0, 1.1, 1.2],
            }
        )

        adjusted = api._standardize_daily_price_data(raw, "000001")

        assert adjusted["close"].tolist() == pytest.approx([10.0, 11.0, 12.1])
        assert adjusted["pct_change"].iloc[1:].tolist() == pytest.approx([10.0, 10.0])
        assert adjusted["open"].iloc[0] == pytest.approx(9.8)
