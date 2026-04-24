"""Data access layer for price and fundamental data."""

import os
import time
import warnings
from typing import Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - minimal runtime without optional deps
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()
TS_TOKEN = os.getenv("TS_TOKEN")

try:
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs
    ak = None

try:
    import tushare as ts
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs
    ts = None


class DataAPI:
    """Unified data access wrapper with cache isolation by source and adjust mode."""

    STANDARD_PRICE_COLUMNS = (
        "date",
        "code",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    )
    _RAW_PRICE_COLUMNS = ("open", "close", "high", "low")
    _CN_PRICE_COLUMNS = ("\u5f00\u76d8", "\u6536\u76d8", "\u6700\u9ad8", "\u6700\u4f4e")

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
        self.source = str(source).strip().lower()
        self.adjust_mode = self._normalize_adjust_mode(adjust_mode)
        self.adjust_mode_label = self.adjust_mode or "raw"
        self.pro = None

        if self.source not in {"akshare", "tushare"}:
            raise ValueError(f"Unsupported data source: {source}. Expected akshare or tushare.")

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        if self.source == "tushare":
            self._init_tushare()

    def _init_tushare(self):
        """Initialize tushare pro client."""
        self._require_tushare()
        if not TS_TOKEN:
            raise RuntimeError(
                "TS_TOKEN is not set. Please set TS_TOKEN in environment variables or .env."
            )
        ts.set_token(TS_TOKEN)
        self.pro = ts.pro_api()

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        try:
            return bool(value != value)
        except Exception:
            return False

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

    @staticmethod
    def _normalize_date(date_value: str) -> str:
        if date_value is None:
            raise ValueError("date_value cannot be None")

        date_text = str(date_value).strip()
        parsed = pd.to_datetime(date_text, errors="coerce")
        if DataAPI._is_missing(parsed):
            compact = date_text.replace("-", "").replace("/", "").replace(".", "")
            if len(compact) == 8 and compact.isdigit():
                return compact
            raise ValueError(f"Unsupported date format: {date_value}")
        return parsed.strftime("%Y%m%d")

    @staticmethod
    def _to_date_only(date_value: str) -> pd.Timestamp:
        return pd.to_datetime(date_value, errors="coerce").normalize()

    @staticmethod
    def _normalize_symbol_for_tushare(symbol: str) -> str:
        if symbol is None:
            raise ValueError("symbol cannot be None")

        normalized = str(symbol).strip().upper()
        if not normalized:
            raise ValueError("symbol cannot be empty")
        if "." in normalized:
            return normalized
        if normalized[0] in {"5", "6", "9"}:
            return f"{normalized}.SH"
        return f"{normalized}.SZ"

    @staticmethod
    def _normalize_symbol_for_akshare(symbol: str) -> str:
        if symbol is None:
            raise ValueError("symbol cannot be None")
        normalized = str(symbol).strip().upper()
        if "." in normalized:
            return normalized.split(".", 1)[0]
        return normalized

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

    @staticmethod
    def _require_tushare():
        if ts is None:
            raise ModuleNotFoundError(
                "tushare is required to fetch market data. Install it with `pip install tushare`."
            )

    @staticmethod
    def _read_csv_with_fallback(filepath: str) -> pd.DataFrame:
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(filepath, index_col=0, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return pd.read_csv(filepath, index_col=0)

    def _standardize_daily_price_data(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert different provider payloads to a common 12-column daily schema."""
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(columns=self.STANDARD_PRICE_COLUMNS)

        df = raw_df.copy()
        alias_map = {
            "trade_date": "date",
            "ts_code": "code",
            "vol": "volume",
            "pct_chg": "pct_change",
            "\u65e5\u671f": "date",
            "\u80a1\u7968\u4ee3\u7801": "code",
            "\u5f00\u76d8": "open",
            "\u6536\u76d8": "close",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6210\u4ea4\u91cf": "volume",
            "\u6210\u4ea4\u989d": "amount",
            "\u632f\u5e45": "amplitude",
            "\u6da8\u8dcc\u5e45": "pct_change",
            "\u6da8\u8dcc\u989d": "change",
            "\u6362\u624b\u7387": "turnover",
        }
        df = df.rename(columns=alias_map)

        # Some akshare versions return 11 columns without code.
        if "code" not in df.columns and len(df.columns) == 11 and "date" in df.columns:
            df.insert(1, "code", symbol)

        for column in self.STANDARD_PRICE_COLUMNS:
            if column not in df.columns:
                df[column] = float("nan")

        df["code"] = df["code"].fillna(symbol).astype(str)
        if self.source == "tushare":
            df["code"] = df["code"].str.upper()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notnull()].copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        numeric_columns = [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_change",
            "change",
            "turnover",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Backfill basic change metrics when provider payload does not include them.
        if df["change"].isnull().all():
            df["change"] = df["close"].diff()
        if df["pct_change"].isnull().all():
            previous_close = df["close"].shift(1)
            df["pct_change"] = (df["close"] / previous_close - 1.0) * 100.0
        if df["amplitude"].isnull().all():
            previous_close = df["close"].shift(1)
            previous_close = previous_close.where(previous_close != 0, float("nan"))
            df["amplitude"] = (df["high"] - df["low"]) / previous_close * 100.0

        df = df[list(self.STANDARD_PRICE_COLUMNS)].sort_values("date").reset_index(drop=True)
        return df

    def _filter_date_window(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if data is None or data.empty or "date" not in data.columns:
            return data

        start = self._to_date_only(start_date)
        end = self._to_date_only(end_date)
        if self._is_missing(start) or self._is_missing(end):
            return data

        date_index = pd.to_datetime(data["date"], errors="coerce")
        window_mask = date_index.between(start, end, inclusive="both")
        return data.loc[window_mask].reset_index(drop=True)

    def _select_columns(self, data: pd.DataFrame, fields: Optional[List[str]]) -> pd.DataFrame:
        if not fields:
            return data
        selected_columns = [column for column in fields if column in data.columns]
        if not selected_columns:
            return data
        return data[selected_columns].copy()

    def _fetch_daily_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.source == "akshare":
            self._require_akshare()
            ak_symbol = self._normalize_symbol_for_akshare(symbol)
            raw = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.adjust_mode,
            )
            return self._standardize_daily_price_data(raw, ak_symbol)

        self._require_tushare()
        if self.pro is None:
            raise RuntimeError("tushare client is not initialized.")

        ts_code = self._normalize_symbol_for_tushare(symbol)
        bar_kwargs = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "freq": "D",
        }
        if self.adjust_mode:
            bar_kwargs["adj"] = self.adjust_mode

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*Series\.fillna with 'method' is deprecated.*",
            )
            raw = ts.pro_bar(**bar_kwargs)
        if (raw is None or raw.empty) and not self.adjust_mode:
            # Fallback for raw mode where some environments expose only `pro.daily`.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r".*Series\.fillna with 'method' is deprecated.*",
                )
                raw = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return self._standardize_daily_price_data(raw, ts_code)

    def get_stock_list(self) -> List[str]:
        """Load the stock universe from the configured file."""
        stock_list: List[str] = []
        with open(self.stock_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                symbol = line.strip()
                if not symbol:
                    continue

                if self.source == "tushare":
                    symbol = self._normalize_symbol_for_tushare(symbol)
                else:
                    symbol = self._normalize_symbol_for_akshare(symbol)
                stock_list.append(symbol)
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
        d_end_date = self._normalize_date(end_date)
        cache_symbol = (
            self._normalize_symbol_for_tushare(symbol)
            if self.source == "tushare"
            else self._normalize_symbol_for_akshare(symbol)
        )

        filename = f"{cache_symbol}_{d_start_date}_{d_end_date}.csv"
        data = self.load_from_cache(filename, "price_history")
        if data is None or data.empty:
            data = self._fetch_daily_price_data(cache_symbol, d_start_date, d_end_date)
            self.save_to_cache(data, filename, "price_history")
            time.sleep(2)
        else:
            data = self._standardize_daily_price_data(data, cache_symbol)

        data = self._filter_date_window(data, start_date, end_date)
        return self._select_columns(data, fields)

    def get_price_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get daily price data for a specific date window."""
        d_start_date = self._normalize_date(start_date)
        d_end_date = self._normalize_date(end_date)
        cache_symbol = (
            self._normalize_symbol_for_tushare(symbol)
            if self.source == "tushare"
            else self._normalize_symbol_for_akshare(symbol)
        )

        filename = f"{cache_symbol}_{d_start_date}_{d_end_date}.csv"
        data = self.load_from_cache(filename, "price")
        if data is None or data.empty:
            data = self._fetch_daily_price_data(cache_symbol, d_start_date, d_end_date)
            self.save_to_cache(data, filename, "price")
            time.sleep(2)
        else:
            data = self._standardize_daily_price_data(data, cache_symbol)

        data = self._filter_date_window(data, start_date, end_date)
        return self._select_columns(data, fields)

    def get_financial_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get financial daily basic data. Currently available on tushare."""
        if self.source != "tushare":
            return pd.DataFrame()

        self._require_tushare()
        if self.pro is None:
            raise RuntimeError("tushare client is not initialized.")

        ts_code = self._normalize_symbol_for_tushare(symbol)
        d_start = self._normalize_date(start_date)
        d_end = self._normalize_date(end_date)
        query_fields = ",".join(fields) if fields else None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*Series\.fillna with 'method' is deprecated.*",
            )
            payload = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=d_start,
                end_date=d_end,
                fields=query_fields,
            )
        if payload is None or payload.empty:
            return pd.DataFrame()
        return payload.sort_values("trade_date").reset_index(drop=True)

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
            return self._read_csv_with_fallback(filepath)
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
            return self._read_csv_with_fallback(filepath)
        return None
