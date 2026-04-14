"""
Data access layer for price and fundamental data.
"""
import os
import time
from typing import Dict, List, Optional

import pandas as pd

try:
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs
    ak = None


class DataAPI:
    """Unified data access wrapper with cache isolation by adjust mode."""

    _RAW_PRICE_COLUMNS = ("open", "close", "high", "low")
    _CN_PRICE_COLUMNS = ("开盘", "收盘", "最高", "最低")

    def __init__(
        self,
        source: str,
        stock_file: str = ".test1.txt",
        cache_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        adjust_mode: Optional[str] = "hfq",
    ):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        self.stock_file = stock_file
        self.source = source
        self.adjust_mode = self._normalize_adjust_mode(adjust_mode)
        self.adjust_mode_label = self.adjust_mode or "raw"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    @staticmethod
    def _normalize_adjust_mode(adjust_mode: Optional[str]) -> str:
        if adjust_mode is None:
            return "hfq"

        normalized = str(adjust_mode).strip().lower()
        if normalized in {"", "none", "raw", "bfq", "unadjusted"}:
            return ""
        if normalized in {"qfq", "hfq"}:
            return normalized

        raise ValueError(
            f"Unsupported adjust_mode: {adjust_mode}. Expected one of hfq, qfq, none."
        )

    def _build_storage_dir(self, root_dir: str, data_type: str) -> str:
        path = os.path.join(root_dir, self.source, data_type, self.adjust_mode_label)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _require_akshare():
        if ak is None:
            raise ModuleNotFoundError(
                "akshare is required to fetch market data. Install it with `pip install akshare`."
            )

    def get_stock_list(self) -> List[str]:
        """Load the stock universe from the configured file."""
        stock_list = []
        with open(self.stock_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if self.source == "tushare":
                    if line[0] == "6":
                        line = line + ".SH"
                    else:
                        line = line + ".SZ"
                stock_list.append(line)
        return stock_list

    def get_price_history_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get historical daily price data and filter it to the requested window."""
        d_start_date = "20050101"
        d_end_date = "20241231"

        filename = f"{symbol}_{d_start_date}_{d_end_date}.csv"
        data = self.load_from_cache(filename, "price_history")
        if data is None:
            self._require_akshare()
            data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=d_start_date,
                end_date=d_end_date,
                adjust=self.adjust_mode,
            )
            self.save_to_cache(data, filename, "price_history")
            time.sleep(3)

        if "日期" in data.columns:
            data = data[(data["日期"] >= start_date) & (data["日期"] <= end_date)]

        return data

    def get_price_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get daily price data for a specific date window."""
        filename = f"{symbol}_{start_date}_{end_date}.csv"
        data = self.load_from_cache(filename, "price")
        if data is None:
            self._require_akshare()
            data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.adjust_mode,
            )
            self.save_to_cache(data, filename, "price")
            time.sleep(3)
        return data

    def get_financial_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Placeholder for future financial data integration."""
        pass

    @classmethod
    def detect_non_positive_prices(cls, data: pd.DataFrame) -> Dict[str, int]:
        """Count non-positive values in price columns."""
        issues: Dict[str, int] = {}
        candidate_columns = cls._RAW_PRICE_COLUMNS + cls._CN_PRICE_COLUMNS

        for column in candidate_columns:
            if column not in data.columns:
                continue

            numeric = pd.to_numeric(data[column], errors="coerce")
            invalid_mask = numeric.notnull() & (numeric <= 0)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                issues[column] = invalid_count

        return issues

    @classmethod
    def has_non_positive_prices(cls, data: pd.DataFrame) -> bool:
        """Return True when any known price column contains non-positive values."""
        return bool(cls.detect_non_positive_prices(data))

    def save_to_cache(self, data: pd.DataFrame, filename: str, data_type: str = "price"):
        """Save raw data to cache."""
        cache_dir = self._build_storage_dir(self.cache_dir, data_type)
        filepath = os.path.join(cache_dir, filename)
        data.to_csv(filepath, index=True, encoding="utf-8-sig")

    def load_from_cache(self, filename: str, data_type: str = "price") -> pd.DataFrame:
        """Load cached data if it exists."""
        cache_dir = self._build_storage_dir(self.cache_dir, data_type)
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, encoding="utf-8-sig")
        return None

    def save_processed(self, data: pd.DataFrame, filename: str, data_type: str = "price"):
        """Save processed data."""
        processed_dir = self._build_storage_dir(self.processed_dir, data_type)
        filepath = os.path.join(processed_dir, filename)
        data.to_csv(filepath, index=True, encoding="utf-8-sig")

    def load_processed(self, filename: str, data_type: str = "price") -> pd.DataFrame:
        """Load processed data if it exists."""
        processed_dir = self._build_storage_dir(self.processed_dir, data_type)
        filepath = os.path.join(processed_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, encoding="utf-8-sig")
        return None
