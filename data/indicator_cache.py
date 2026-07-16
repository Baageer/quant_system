"""Indicator result cache for reusable technical indicator values.

The cache layer keeps indicator formulas pure in ``signals.indicators`` while
providing a stable interface for applications and strategies to reuse computed
values. One symbol stores all indicator values in a single wide CSV table, with
per-indicator metadata keyed by indicator name and parameter hash.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from signals.indicators import atr, bollinger_bands, kdj, macd, rsi


class IndicatorCache:
    """Read-through cache for technical indicator values."""

    SUPPORTED_INDICATORS = {"bollinger_bands", "macd", "rsi", "atr", "kdj"}
    INDICATOR_ALIASES = {
        "bb": "bollinger_bands",
        "bollinger": "bollinger_bands",
        "bollinger_bands": "bollinger_bands",
        "macd": "macd",
        "rsi": "rsi",
        "atr": "atr",
        "kdj": "kdj",
    }
    OUTPUT_COLUMNS = {
        "bollinger_bands": ["upper", "middle", "lower", "bandwidth"],
        "macd": ["macd", "signal", "hist"],
        "rsi": ["rsi"],
        "atr": ["atr"],
        "kdj": ["k", "d", "j"],
    }
    INDICATOR_VERSION = "v1"
    DATA_FILENAME = "daily_indicators.csv"
    META_FILENAME = "daily_indicators.meta.json"

    def __init__(
        self,
        cache_dir: str = "./data/processed/indicators",
        source: str = "unknown",
        adjust_mode: str = "unknown",
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.source = self._safe_path_part(source)
        self.adjust_mode = self._safe_path_part(adjust_mode)
        self.enabled = enabled
        self.last_cache_status = "idle"
        self._table_cache = {}
        self._meta_cache = {}

    def get_indicator(
        self,
        name: str,
        data: pd.DataFrame,
        symbol: str,
        params: Optional[Mapping[str, Any]] = None,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return indicator values, loading from cache when metadata matches."""
        indicator_name = self._normalize_indicator_name(name)
        params = dict(params or {})

        if indicator_name == "bollinger_bands":
            return self.get_bollinger_bands(
                data=data,
                symbol=symbol,
                force_recompute=force_recompute,
                **params,
            )
        if indicator_name == "macd":
            return self.get_macd(
                data=data,
                symbol=symbol,
                force_recompute=force_recompute,
                **params,
            )
        if indicator_name == "rsi":
            return self.get_rsi(
                data=data,
                symbol=symbol,
                force_recompute=force_recompute,
                **params,
            )
        if indicator_name == "atr":
            return self.get_atr(
                data=data,
                symbol=symbol,
                force_recompute=force_recompute,
                **params,
            )
        if indicator_name == "kdj":
            return self.get_kdj(
                data=data,
                symbol=symbol,
                force_recompute=force_recompute,
                **params,
            )

        raise ValueError(f"Unsupported indicator: {name}")

    def get_bollinger_bands(
        self,
        data: pd.DataFrame,
        symbol: str,
        window: int = 20,
        num_std: float = 2.0,
        price_col: str = "close",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return Bollinger Bands columns: upper, middle, lower, bandwidth."""
        params = self._normalize_bollinger_params(window, num_std, price_col)
        return self._get_cached_indicator(
            indicator="bollinger_bands",
            data=data,
            symbol=symbol,
            params=params,
            input_columns=[params["price_col"]],
            compute_func=self._compute_bollinger_bands,
            force_recompute=force_recompute,
        )

    def get_macd(
        self,
        data: pd.DataFrame,
        symbol: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_col: str = "close",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return MACD columns: macd, signal, hist."""
        params = self._normalize_macd_params(
            fast_period, slow_period, signal_period, price_col
        )
        return self._get_cached_indicator(
            indicator="macd",
            data=data,
            symbol=symbol,
            params=params,
            input_columns=[params["price_col"]],
            compute_func=self._compute_macd,
            force_recompute=force_recompute,
        )

    def get_rsi(
        self,
        data: pd.DataFrame,
        symbol: str,
        window: int = 14,
        price_col: str = "close",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return RSI column: rsi."""
        params = self._normalize_rsi_params(window, price_col)
        return self._get_cached_indicator(
            indicator="rsi",
            data=data,
            symbol=symbol,
            params=params,
            input_columns=[params["price_col"]],
            compute_func=self._compute_rsi,
            force_recompute=force_recompute,
        )

    def get_atr(
        self,
        data: pd.DataFrame,
        symbol: str,
        window: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return ATR column: atr."""
        params = self._normalize_atr_params(window, high_col, low_col, close_col)
        return self._get_cached_indicator(
            indicator="atr",
            data=data,
            symbol=symbol,
            params=params,
            input_columns=[params["high_col"], params["low_col"], params["close_col"]],
            compute_func=self._compute_atr,
            force_recompute=force_recompute,
        )

    def get_kdj(
        self,
        data: pd.DataFrame,
        symbol: str,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Return KDJ columns: k, d, j."""
        params = self._normalize_kdj_params(n, m1, m2, high_col, low_col, close_col)
        return self._get_cached_indicator(
            indicator="kdj",
            data=data,
            symbol=symbol,
            params=params,
            input_columns=[params["high_col"], params["low_col"], params["close_col"]],
            compute_func=self._compute_kdj,
            force_recompute=force_recompute,
        )

    def _get_cached_indicator(
        self,
        indicator: str,
        data: pd.DataFrame,
        symbol: str,
        params: Mapping[str, Any],
        input_columns: List[str],
        compute_func: Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame],
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        self._validate_data_columns(data, input_columns)

        paths = self._cache_paths(symbol)
        params_hash = self._params_hash(params)
        indicator_key = self._indicator_key(indicator, params_hash)
        output_columns = self.OUTPUT_COLUMNS[indicator]
        storage_columns = self._storage_columns(indicator, params_hash, output_columns)
        input_fingerprint = self._fingerprint_frame(data, input_columns)

        if self.enabled and not force_recompute:
            cached = self._read_valid_cache(
                paths=paths,
                indicator_key=indicator_key,
                indicator=indicator,
                params=params,
                input_columns=input_columns,
                output_columns=output_columns,
                storage_columns=storage_columns,
                input_fingerprint=input_fingerprint,
            )
            if cached is not None:
                self.last_cache_status = "hit"
                return cached

        result = compute_func(data, params)[output_columns]
        self.last_cache_status = "disabled" if not self.enabled else "miss"

        if self.enabled:
            self._write_indicator_cache(
                paths=paths,
                result=result,
                indicator_key=indicator_key,
                indicator=indicator,
                symbol=symbol,
                params=params,
                input_columns=input_columns,
                output_columns=output_columns,
                storage_columns=storage_columns,
                data=data,
                input_fingerprint=input_fingerprint,
            )

        return result

    def _compute_bollinger_bands(
        self,
        data: pd.DataFrame,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        upper, middle, lower = bollinger_bands(
            data[params["price_col"]],
            window=params["window"],
            num_std=params["num_std"],
        )
        return pd.DataFrame(
            {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "bandwidth": (upper - lower) / middle,
            },
            index=data.index,
        )

    def _compute_macd(
        self,
        data: pd.DataFrame,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        macd_line, signal_line, histogram = macd(
            data[params["price_col"]],
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            signal_period=params["signal_period"],
        )
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "hist": histogram},
            index=data.index,
        )

    def _compute_rsi(
        self,
        data: pd.DataFrame,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"rsi": rsi(data[params["price_col"]], window=params["window"])},
            index=data.index,
        )

    def _compute_atr(
        self,
        data: pd.DataFrame,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "atr": atr(
                    data[params["high_col"]],
                    data[params["low_col"]],
                    data[params["close_col"]],
                    window=params["window"],
                )
            },
            index=data.index,
        )

    def _compute_kdj(
        self,
        data: pd.DataFrame,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        k, d, j = kdj(
            data[params["high_col"]],
            data[params["low_col"]],
            data[params["close_col"]],
            n=params["n"],
            m1=params["m1"],
            m2=params["m2"],
        )
        return pd.DataFrame({"k": k, "d": d, "j": j}, index=data.index)

    def _read_valid_cache(
        self,
        paths: Mapping[str, Path],
        indicator_key: str,
        indicator: str,
        params: Mapping[str, Any],
        input_columns: List[str],
        output_columns: List[str],
        storage_columns: Mapping[str, str],
        input_fingerprint: str,
    ) -> Optional[pd.DataFrame]:
        if not paths["data"].exists() or not paths["meta"].exists():
            return None

        meta = self._read_meta(paths["meta"])
        entry = meta.get("indicators", {}).get(indicator_key)
        if not self._is_meta_valid(
            meta=meta,
            entry=entry,
            indicator=indicator,
            params=params,
            input_columns=input_columns,
            output_columns=output_columns,
            storage_columns=storage_columns,
            input_fingerprint=input_fingerprint,
        ):
            return None

        table = self._read_cache_table(paths["data"])
        if table is None:
            return None

        required_storage_columns = [storage_columns[column] for column in output_columns]
        if any(column not in table.columns for column in required_storage_columns):
            return None

        result = pd.DataFrame(index=table.index)
        for output_column in output_columns:
            result[output_column] = table[storage_columns[output_column]]
        result.index.name = meta.get("index_name")
        return result[output_columns]

    def _write_indicator_cache(
        self,
        paths: Mapping[str, Path],
        result: pd.DataFrame,
        indicator_key: str,
        indicator: str,
        symbol: str,
        params: Mapping[str, Any],
        input_columns: List[str],
        output_columns: List[str],
        storage_columns: Mapping[str, str],
        data: pd.DataFrame,
        input_fingerprint: str,
    ) -> None:
        paths["data"].parent.mkdir(parents=True, exist_ok=True)

        table = self._read_cache_table(paths["data"])
        if table is None:
            table = pd.DataFrame(index=data.index)
        else:
            table = table.reindex(data.index)

        for output_column in output_columns:
            table[storage_columns[output_column]] = result[output_column].reindex(table.index)

        self._write_cache_table(table, paths["data"], data.index.name)

        meta = self._read_meta(paths["meta"])
        now = datetime.now().isoformat(timespec="seconds")
        meta.update(
            {
                "cache_format": "wide_csv",
                "symbol": symbol,
                "source": self.source,
                "adjust_mode": self.adjust_mode,
                "data_start": self._index_boundary(data, first=True),
                "data_end": self._index_boundary(data, first=False),
                "row_count": int(len(data)),
                "index_name": data.index.name,
                "updated_at": now,
            }
        )
        meta.setdefault("created_at", now)
        meta.setdefault("indicators", {})
        meta["indicators"][indicator_key] = {
            "indicator": indicator,
            "indicator_version": self.INDICATOR_VERSION,
            "params_hash": self._params_hash(params),
            "params": dict(params),
            "input_columns": list(input_columns),
            "input_fingerprint": input_fingerprint,
            "output_columns": list(output_columns),
            "storage_columns": dict(storage_columns),
            "updated_at": now,
        }
        meta["meta_version"] = 2

        self._write_meta(meta, paths["meta"])

    def _is_meta_valid(
        self,
        meta: Mapping[str, Any],
        entry: Optional[Mapping[str, Any]],
        indicator: str,
        params: Mapping[str, Any],
        input_columns: List[str],
        output_columns: List[str],
        storage_columns: Mapping[str, str],
        input_fingerprint: str,
    ) -> bool:
        if not entry:
            return False

        return (
            meta.get("cache_format") == "wide_csv"
            and meta.get("source") == self.source
            and meta.get("adjust_mode") == self.adjust_mode
            and entry.get("indicator") == indicator
            and entry.get("indicator_version") == self.INDICATOR_VERSION
            and entry.get("params_hash") == self._params_hash(params)
            and entry.get("params") == dict(params)
            and entry.get("input_columns") == list(input_columns)
            and entry.get("input_fingerprint") == input_fingerprint
            and entry.get("output_columns") == list(output_columns)
            and entry.get("storage_columns") == dict(storage_columns)
        )

    def _cache_paths(self, symbol: str) -> Dict[str, Path]:
        base_dir = (
            self.cache_dir
            / self.source
            / self.adjust_mode
            / self._safe_path_part(symbol)
        )
        return {
            "data": base_dir / self.DATA_FILENAME,
            "meta": base_dir / self.META_FILENAME,
        }

    def _read_cache_table(self, path: Path) -> Optional[pd.DataFrame]:
        cache_key = str(path)
        if cache_key in self._table_cache:
            return self._table_cache[cache_key].copy()

        if not path.exists():
            return None
        try:
            table = pd.read_csv(path, index_col=0, parse_dates=True)
            self._table_cache[cache_key] = table.copy()
            return table
        except (OSError, ValueError):
            return None

    def _write_cache_table(self, table: pd.DataFrame, path: Path, index_name: Any) -> None:
        table.index.name = index_name
        tmp_path = path.with_name(path.name + ".tmp")
        table.to_csv(tmp_path, index_label="date", encoding="utf-8-sig")
        tmp_path.replace(path)
        self._table_cache[str(path)] = table.copy()

    def _read_meta(self, path: Path) -> Dict[str, Any]:
        cache_key = str(path)
        if cache_key in self._meta_cache:
            return dict(self._meta_cache[cache_key])

        if not path.exists():
            return {}
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            self._meta_cache[cache_key] = dict(meta)
            return meta
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_meta(self, meta: Mapping[str, Any], path: Path) -> None:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        self._meta_cache[str(path)] = dict(meta)

    @classmethod
    def _normalize_indicator_name(cls, name: str) -> str:
        normalized = str(name).strip().lower()
        indicator_name = cls.INDICATOR_ALIASES.get(normalized, normalized)
        if indicator_name not in cls.SUPPORTED_INDICATORS:
            raise ValueError(
                f"Unsupported indicator: {name}. "
                f"Supported indicators: {sorted(cls.SUPPORTED_INDICATORS)}"
            )
        return indicator_name

    @staticmethod
    def _normalize_bollinger_params(
        window: int,
        num_std: float,
        price_col: str,
    ) -> Dict[str, Any]:
        normalized_window = int(window)
        normalized_num_std = float(num_std)
        normalized_price_col = str(price_col).strip()

        if normalized_window < 2:
            raise ValueError("window must be at least 2")
        if normalized_num_std <= 0:
            raise ValueError("num_std must be positive")
        if not normalized_price_col:
            raise ValueError("price_col cannot be empty")

        return {
            "window": normalized_window,
            "num_std": normalized_num_std,
            "price_col": normalized_price_col,
        }

    @staticmethod
    def _normalize_macd_params(
        fast_period: int,
        slow_period: int,
        signal_period: int,
        price_col: str,
    ) -> Dict[str, Any]:
        fast_period = int(fast_period)
        slow_period = int(slow_period)
        signal_period = int(signal_period)
        price_col = str(price_col).strip()

        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("MACD periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if not price_col:
            raise ValueError("price_col cannot be empty")

        return {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "price_col": price_col,
        }

    @staticmethod
    def _normalize_rsi_params(window: int, price_col: str) -> Dict[str, Any]:
        window = int(window)
        price_col = str(price_col).strip()

        if window <= 1:
            raise ValueError("window must be greater than 1")
        if not price_col:
            raise ValueError("price_col cannot be empty")

        return {"window": window, "price_col": price_col}

    @staticmethod
    def _normalize_atr_params(
        window: int,
        high_col: str,
        low_col: str,
        close_col: str,
    ) -> Dict[str, Any]:
        window = int(window)
        high_col = str(high_col).strip()
        low_col = str(low_col).strip()
        close_col = str(close_col).strip()

        if window <= 1:
            raise ValueError("window must be greater than 1")
        if not high_col or not low_col or not close_col:
            raise ValueError("high_col, low_col and close_col cannot be empty")

        return {
            "window": window,
            "high_col": high_col,
            "low_col": low_col,
            "close_col": close_col,
        }

    @staticmethod
    def _normalize_kdj_params(
        n: int,
        m1: int,
        m2: int,
        high_col: str,
        low_col: str,
        close_col: str,
    ) -> Dict[str, Any]:
        n = int(n)
        m1 = int(m1)
        m2 = int(m2)
        high_col = str(high_col).strip()
        low_col = str(low_col).strip()
        close_col = str(close_col).strip()

        if n <= 1 or m1 <= 0 or m2 <= 0:
            raise ValueError("KDJ periods must be positive and n must be greater than 1")
        if not high_col or not low_col or not close_col:
            raise ValueError("high_col, low_col and close_col cannot be empty")

        return {
            "n": n,
            "m1": m1,
            "m2": m2,
            "high_col": high_col,
            "low_col": low_col,
            "close_col": close_col,
        }

    @staticmethod
    def _validate_data_columns(data: pd.DataFrame, columns: List[str]) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data cannot be empty")

        missing_columns = [column for column in columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    @staticmethod
    def _params_hash(params: Mapping[str, Any]) -> str:
        payload = json.dumps(dict(params), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _indicator_key(indicator: str, params_hash: str) -> str:
        return f"{indicator}|{params_hash}"

    @classmethod
    def _storage_columns(
        cls,
        indicator: str,
        params_hash: str,
        output_columns: List[str],
    ) -> Dict[str, str]:
        return {
            output_column: (
                f"{cls._safe_path_part(indicator)}_"
                f"{params_hash}_"
                f"{cls._safe_path_part(output_column)}"
            )
            for output_column in output_columns
        }

    @staticmethod
    def _fingerprint_frame(data: pd.DataFrame, columns: List[str]) -> str:
        frame = data[columns].copy()
        hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        return hashlib.sha256(hashed).hexdigest()

    @staticmethod
    def _index_boundary(data: pd.DataFrame, first: bool) -> str:
        value = data.index[0] if first else data.index[-1]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _safe_path_part(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return "unknown"
        return re.sub(r"[^0-9A-Za-z._-]+", "_", text)
