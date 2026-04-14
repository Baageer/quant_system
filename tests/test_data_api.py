import pandas as pd
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
