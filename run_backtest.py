"""
Generic backtest entry script.

Supports:
- loading timing and stop strategies from YAML
- single-strategy and multi-strategy signal combinations
- exporting trade records after a backtest run
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_max_drawdown, calculate_sharpe_ratio
from backtest.performance import PerformanceAnalyzer
from data.data_api import DataAPI
from data.indicator_cache import IndicatorCache
from signals.signal_engine import SignalEngine
from signals.strategy_loader import StrategyLoader
from utils.logger import setup_logger


PRICE_COLUMNS = [
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
]


def create_strategy_function(trade_amount):
    """Create the broker-facing strategy callback from per-symbol signals.
    Optimized to reduce unnecessary iterations and handle filtered data.
    """

    def strategy_func(date, data, positions):
        signals = {}

        # Process only the filtered data passed in (already optimized by engine)
        for symbol, df in data.items():
            # Skip if date not in index (shouldn't happen with filtered data, but safe check)
            if date not in df.index:
                continue

            current_signal = df.loc[date, "signal"]
            if pd.isna(current_signal):
                continue

            current_pos = positions.get(symbol, 0)
            current_price = df.loc[date, "close"]
            if pd.isna(current_price) or current_price <= 0:
                continue
            shares = int(trade_amount / current_price)

            if current_signal == 1 and current_pos == 0:
                signals[symbol] = {"action": "buy", "shares": shares}
            elif current_signal == -1 and current_pos > 0:
                signals[symbol] = {"action": "sell", "shares": current_pos}

        return signals

    return strategy_func


def calculate_annual_portfolio_returns(results: pd.DataFrame) -> pd.Series:
    """Calculate annual portfolio returns from portfolio net value."""
    if results.empty or "portfolio_value" not in results.columns:
        return pd.Series(dtype=float)

    portfolio_value = results["portfolio_value"].copy()
    if not isinstance(portfolio_value.index, pd.DatetimeIndex):
        parsed_index = pd.to_datetime(portfolio_value.index, errors="coerce")
        valid_mask = ~parsed_index.isna()
        portfolio_value = portfolio_value.loc[valid_mask]
        portfolio_value.index = parsed_index[valid_mask]

    portfolio_value = portfolio_value.sort_index().dropna()
    if portfolio_value.empty:
        return pd.Series(dtype=float)

    grouped = portfolio_value.groupby(portfolio_value.index.year)
    year_start_values = grouped.first()
    year_end_values = grouped.last()

    annual_returns = (year_end_values / year_start_values - 1) * 100
    annual_returns = annual_returns.replace([np.inf, -np.inf], np.nan).dropna()
    return annual_returns.sort_index()


def normalize_code(value: object) -> str:
    """Normalize industry index code to 6-digit style when possible."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def normalize_text(value: object) -> str:
    """Normalize generic text values for robust CSV parsing."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first matching column by exact or case-insensitive name."""
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    return None


def read_csv_with_fallback(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read csv with common encodings used in this repo."""
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(filepath)


def attach_indicator_cache_to_strategies(
    strategies: List[object],
    indicator_cache: Optional[IndicatorCache],
    symbol: str,
) -> None:
    """Attach per-symbol indicator cache context to strategies that support it."""
    if indicator_cache is None:
        return

    for strategy in strategies:
        current = strategy
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            attrs = vars(current) if hasattr(current, "__dict__") else {}
            if "indicator_cache" in attrs:
                current.indicator_cache = indicator_cache
            if "cache_symbol" in attrs:
                current.cache_symbol = symbol
            current = attrs.get("_strategy")


def standardize_index_list(index_list_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize industry index universe file to a stable schema."""
    if index_list_df is None or index_list_df.empty:
        return pd.DataFrame(columns=["index_code", "industry_name", "link"])

    df = index_list_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    code_col = pick_column(df, ["index_code", "code", "industry_code", "板块代码", "行业代码"])
    name_col = pick_column(df, ["industry_name", "name", "board_name", "板块名称", "行业名称"])
    link_col = pick_column(df, ["link", "source_link", "url"])

    if code_col is None or name_col is None:
        raise ValueError("Industry index list must contain index_code and industry_name columns.")

    result = pd.DataFrame(
        {
            "index_code": df[code_col].map(normalize_code),
            "industry_name": df[name_col].map(normalize_text),
            "link": df[link_col].map(normalize_text) if link_col is not None else "",
        }
    )
    result = result[(result["index_code"] != "") & (result["industry_name"] != "")]
    return result.drop_duplicates(subset=["index_code"]).reset_index(drop=True)


def load_industry_universe(
    index_list_file: Union[str, Path],
    index_codes: Optional[List[str]] = None,
    max_indexes: int = -1,
) -> pd.DataFrame:
    """Load industry index universe with optional code filter and top-n cap."""
    list_df = read_csv_with_fallback(index_list_file)
    universe_df = standardize_index_list(list_df)

    if index_codes is not None:
        keep_codes = {normalize_code(code) for code in index_codes if normalize_code(code)}
        universe_df = universe_df[universe_df["index_code"].isin(keep_codes)].reset_index(drop=True)

    if max_indexes is not None and max_indexes > 0:
        universe_df = universe_df.head(max_indexes).reset_index(drop=True)

    return universe_df


def build_index_file_lookup(index_data_dir: Union[str, Path]) -> Dict[str, Path]:
    """Build code->latest csv path mapping from industry index directory."""
    data_dir = Path(index_data_dir)
    pattern = re.compile(r"^(?P<code>\d{6})_(?P<start>\d{8})_(?P<end>\d{8})\.csv$")
    best_file_map: Dict[str, tuple] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        code = match.group("code")
        start_date = match.group("start")
        end_date = match.group("end")
        if code not in best_file_map:
            best_file_map[code] = (start_date, end_date, csv_path)
            continue

        old_start, old_end, _ = best_file_map[code]
        if end_date > old_end or (end_date == old_end and start_date < old_start):
            best_file_map[code] = (start_date, end_date, csv_path)

    return {code: payload[2] for code, payload in best_file_map.items()}


def standardize_industry_price_frame(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert industry index data to the shared 12-column daily schema."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = raw_df.copy()
    rename_map = {}
    for column in df.columns:
        name = str(column).strip()
        name_lower = name.lower()

        if name in {"日期", "交易日期"} or name_lower in {"date", "datetime"}:
            rename_map[column] = "date"
        elif name in {"开盘价", "开盘"} or name_lower == "open":
            rename_map[column] = "open"
        elif name in {"最高价", "最高"} or name_lower == "high":
            rename_map[column] = "high"
        elif name in {"最低价", "最低"} or name_lower == "low":
            rename_map[column] = "low"
        elif name in {"收盘价", "收盘"} or name_lower == "close":
            rename_map[column] = "close"
        elif name in {"成交量"} or name_lower in {"volume", "vol"}:
            rename_map[column] = "volume"
        elif name in {"成交额"} or name_lower in {"amount", "turnover"}:
            rename_map[column] = "amount"
        elif name_lower in {"index_code", "code", "symbol"}:
            rename_map[column] = "code"

    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        raise ValueError("Industry index data is missing date column.")

    if "code" not in df.columns:
        df["code"] = symbol
    else:
        df["code"] = df["code"].map(normalize_code).replace("", symbol)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notnull()].copy()

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
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["change"].isnull().all():
        df["change"] = df["close"].diff()
    if df["pct_change"].isnull().all():
        previous_close = df["close"].shift(1)
        df["pct_change"] = (df["close"] / previous_close - 1.0) * 100.0
    if df["amplitude"].isnull().all():
        previous_close = df["close"].shift(1).where(df["close"].shift(1) != 0, np.nan)
        df["amplitude"] = (df["high"] - df["low"]) / previous_close * 100.0

    df = df[PRICE_COLUMNS].sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def apply_signals_to_dataframe(
    df: pd.DataFrame,
    strategies: List[object],
    signal_engine: SignalEngine,
    signal_weights: List[float],
    signal_combination: str,
    signal_threshold: float,
    symbol: Optional[str] = None,
    indicator_cache: Optional[IndicatorCache] = None,
) -> pd.DataFrame:
    """Generate and attach signal series using configured strategy logic."""
    if symbol is not None:
        attach_indicator_cache_to_strategies(strategies, indicator_cache, symbol)

    if len(strategies) == 1:
        df["signal"] = strategies[0].generate_signal(df)
        return df

    signals_list = [strategy.generate_signal(df) for strategy in strategies]
    combined_signal = signal_engine.combine_signals(signals_list, signal_weights)

    if signal_combination == "weighted":
        df["signal"] = combined_signal.apply(
            lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0)
        )
    elif signal_combination == "voting":
        vote_signal = pd.Series(0, index=df.index)
        for signal in signals_list:
            vote_signal += signal
        df["signal"] = vote_signal.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    elif signal_combination == "unanimous":
        df["signal"] = combined_signal.apply(
            lambda x: 1 if x >= (len(strategies) - 0.5) else (-1 if x <= -(len(strategies) - 0.5) else 0)
        )
    else:
        df["signal"] = combined_signal.apply(
            lambda x: 1 if x >= signal_threshold else (-1 if x <= -signal_threshold else 0)
        )

    return df


def load_single_symbol_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    source: str,
    stock_file: str,
    cache_dir: str,
    processed_dir: str,
    adjust_mode: str,
    raw_price_adjust: str,
    min_data_length: int,
    price_columns: List[str],
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Load and process a single stock symbol's data for parallel execution."""
    from data.data_api import DataAPI

    try:
        data_api = DataAPI(
            source=source,
            stock_file=stock_file,
            cache_dir=cache_dir,
            processed_dir=processed_dir,
            adjust_mode=adjust_mode,
            raw_price_adjust=raw_price_adjust,
        )
        df = data_api.get_price_history_data(symbol, start_date, end_date)
        df.columns = price_columns

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.sort_index()

        invalid_prices = DataAPI.detect_non_positive_prices(df)
        if invalid_prices:
            return symbol, None

        if len(df) < min_data_length:
            df["signal"] = np.nan
            return symbol, df

        return symbol, df
    except Exception:
        return symbol, None


def load_single_symbol_industry(
    symbol: str,
    csv_path: Path,
    start_date: str,
    end_date: str,
    min_data_length: int,
    price_columns: List[str],
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Load and process a single industry index's data for parallel execution."""
    try:
        raw_df = read_csv_with_fallback(csv_path)
        df = standardize_industry_price_frame(raw_df, symbol)
        df = df[
            pd.to_datetime(df["date"], errors="coerce").between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                inclusive="both",
            )
        ].reset_index(drop=True)

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.sort_index()

        invalid_prices = {}
        for col in ["open", "close", "high", "low"]:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                invalid_count = int((numeric.notnull() & (numeric <= 0)).sum())
                if invalid_count > 0:
                    invalid_prices[col] = invalid_count
        if invalid_prices:
            return symbol, None

        if len(df) < min_data_length:
            df["signal"] = np.nan
            return symbol, df

        return symbol, df
    except Exception:
        return symbol, None


def run_backtest(
    strategy_name: Union[str, List[str]],
    start_date: str = None,
    end_date: str = None,
    config_path: str = "./config/settings.yaml",
    strategy_config_path: str = "./config/strategies.yaml",
    stock_file: str = None,
    stock_max_number: int = -1,
    initial_capital: float = None,
    trade_amount: float = None,
    adjust_mode: Optional[str] = None,
    raw_price_adjust: Optional[str] = None,
    enable_stop_loss: bool = True,
    enable_stop_profit: bool = True,
    signal_combination: str = "weighted",
    signal_weights: Optional[List[float]] = None,
    signal_threshold: float = 0.5,
    universe_type: str = "stock",
    industry_index_list_file: Optional[str] = None,
    industry_index_data_dir: Optional[str] = None,
    industry_index_codes: Optional[List[str]] = None,
    max_workers: int = 4,
):
    """Run a configured backtest."""

    logger = setup_logger()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    strategy_loader = StrategyLoader(strategy_config_path)

    if isinstance(strategy_name, str):
        strategy_names = [strategy_name]
    else:
        strategy_names = strategy_name

    strategies, strategy_infos = strategy_loader.build_timing_strategies(strategy_names)
    signal_engine = SignalEngine()

    if signal_weights is None:
        signal_weights = [1.0 / len(strategies)] * len(strategies)
    elif len(signal_weights) != len(strategies):
        raise ValueError(
            f"Signal weight count ({len(signal_weights)}) does not match "
            f"strategy count ({len(strategies)})."
        )

    min_data_length = max(info.get("min_data_length", 20) for info in strategy_infos)
    backtest_config = strategy_loader.config.get("backtest", {})

    if start_date is None :
        start_date = config["backtest"].get("start_date", "2022-01-01")
    if end_date is None :
        end_date = config["backtest"].get("end_date", "2023-12-31")
    if initial_capital is None:
        initial_capital = backtest_config.get(
            "initial_capital", config["backtest"]["initial_capital"]
        )
    if trade_amount is None:
        trade_amount = config["backtest"].get("trade_amount", 100000)
    if stock_file is None:
        stock_file = config["data"].get("stock_file", "./data/test1.txt")
    if adjust_mode is None:
        adjust_mode = config["data"].get("adjust_mode", "hfq")
    if raw_price_adjust is None:
        raw_price_adjust = config["data"].get("raw_price_adjust", "")
    if industry_index_list_file is None:
        industry_index_list_file = config["data"].get(
            "industry_index_list_file",
            "./data/raw/akshare/industry_sector_index_list.csv",
        )
    if industry_index_data_dir is None:
        industry_index_data_dir = config["data"].get(
            "industry_index_data_dir",
            "./data/raw/akshare/industry_sector_index",
        )

    data_source = config["data"].get("source", "akshare")

    (
        stop_loss_strategy,
        stop_loss_info,
        stop_profit_strategy,
        stop_profit_info,
    ) = strategy_loader.build_stop_strategies(
        enable_stop_loss=enable_stop_loss,
        enable_stop_profit=enable_stop_profit,
    )

    logger.info("=" * 60)
    if len(strategies) == 1:
        logger.info(f"Strategy: {strategy_infos[0]['name']}")
        logger.info(f"Params: {strategy_infos[0]['params']}")
    else:
        logger.info(f"Strategy combination ({len(strategies)} strategies):")
        for i, (info, weight) in enumerate(zip(strategy_infos, signal_weights), 1):
            logger.info(f"  {i}. {info['name']} - weight: {weight:.2f}")
            logger.info(f"     Params: {info['params']}")
        logger.info(f"Signal combination mode: {signal_combination}")
        if signal_combination == "threshold":
            logger.info(f"Signal threshold: {signal_threshold}")

    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Universe type: {universe_type}")
    logger.info(f"Adjust mode: {adjust_mode or 'raw'}")
    logger.info(f"Initial capital: {initial_capital:,.2f}")
    logger.info(f"Trade amount: {trade_amount:,.2f}")
    if universe_type == "industry":
        logger.info(f"Industry index list: {industry_index_list_file}")
        logger.info(f"Industry data dir: {industry_index_data_dir}")
        if industry_index_codes:
            logger.info(f"Industry code filter: {','.join(industry_index_codes)}")
    else:
        logger.info(f"Stock file: {stock_file}")

    if stop_loss_strategy:
        logger.info(f"Stop loss enabled: {stop_loss_info['params']}")
    else:
        logger.info("Stop loss disabled")

    if stop_profit_strategy:
        logger.info(f"Stop profit enabled: {stop_profit_info['params']}")
    else:
        logger.info("Stop profit disabled")
    logger.info("=" * 60)

    data_api = None
    indicator_cache = None
    universe_type = (universe_type or "stock").strip().lower()
    if universe_type not in {"stock", "industry"}:
        raise ValueError(f"Unsupported universe_type: {universe_type}. Expected stock or industry.")

    symbols: List[str] = []
    industry_name_map: Dict[str, str] = {}
    industry_file_lookup: Dict[str, Path] = {}
    if universe_type == "industry":
        if not os.path.exists(industry_index_list_file):
            raise FileNotFoundError(f"Industry index list file not found: {industry_index_list_file}")
        if not os.path.exists(industry_index_data_dir):
            raise FileNotFoundError(f"Industry index data directory not found: {industry_index_data_dir}")

        universe_df = load_industry_universe(
            index_list_file=industry_index_list_file,
            index_codes=industry_index_codes,
            max_indexes=stock_max_number,
        )
        if universe_df.empty:
            raise ValueError("No industry indexes found in universe after filters.")

        symbols = universe_df["index_code"].tolist()
        industry_name_map = dict(zip(universe_df["index_code"], universe_df["industry_name"]))
        industry_file_lookup = build_index_file_lookup(industry_index_data_dir)
        logger.info(f"Industry index count: {len(symbols)}")
    else:
        data_api = DataAPI(
            source=data_source,
            stock_file=stock_file,
            cache_dir=config["data"]["cache_dir"],
            processed_dir=config["data"]["processed_dir"],
            adjust_mode=adjust_mode,
            raw_price_adjust=raw_price_adjust,
        )
        indicator_cache_config = config.get("indicator_cache", {})
        indicator_cache = IndicatorCache(
            cache_dir=indicator_cache_config.get(
                "dir",
                os.path.join(config["data"]["processed_dir"], "indicators"),
            ),
            source=data_source,
            adjust_mode=data_api.adjust_mode_label,
            enabled=indicator_cache_config.get("enabled", True),
        )
        logger.info(f"Indicator cache dir: {indicator_cache.cache_dir}")
        symbols = data_api.get_stock_list()
        # symbols = symbols[440:]
        if stock_max_number != -1 and len(symbols) > stock_max_number:
            symbols = symbols[:stock_max_number]
        logger.info(f"Stock count: {len(symbols)}")

    logger.info(f"Loading data with {max_workers} workers...")
    data = {}

    if max_workers > 1 and len(symbols) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            if universe_type == "industry":
                for symbol in symbols:
                    csv_path = industry_file_lookup.get(symbol)
                    if csv_path is None:
                        industry_name = industry_name_map.get(symbol, "")
                        logger.warning(f"Skipping {symbol} ({industry_name}): missing industry index csv file")
                        continue
                    future = executor.submit(
                        load_single_symbol_industry,
                        symbol=symbol,
                        csv_path=csv_path,
                        start_date=start_date,
                        end_date=end_date,
                        min_data_length=min_data_length,
                        price_columns=PRICE_COLUMNS,
                    )
                    futures[future] = symbol
            else:
                for symbol in symbols:
                    future = executor.submit(
                        load_single_symbol_stock,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        source=data_source,
                        stock_file=stock_file,
                        cache_dir=config["data"]["cache_dir"],
                        processed_dir=config["data"]["processed_dir"],
                        adjust_mode=adjust_mode,
                        raw_price_adjust=raw_price_adjust,
                        min_data_length=min_data_length,
                        price_columns=PRICE_COLUMNS,
                    )
                    futures[future] = symbol

            with tqdm(total=len(futures), desc="Data loading", disable=False) as pbar:
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        _, df = future.result()
                        if df is not None:
                            data[symbol] = df
                    except Exception as e:
                        logger.warning(f"Failed to load {symbol}: {e}")
                    pbar.update(1)
    else:
        data_iterator = tqdm(
            symbols,
            desc="Data loading",
            unit="index" if universe_type == "industry" else "symbol",
            disable=False,
        )
        for symbol in data_iterator:
            if universe_type == "industry":
                csv_path = industry_file_lookup.get(symbol)
                if csv_path is None:
                    industry_name = industry_name_map.get(symbol, "")
                    logger.warning(f"Skipping {symbol} ({industry_name}): missing industry index csv file")
                    continue
                raw_df = read_csv_with_fallback(csv_path)
                df = standardize_industry_price_frame(raw_df, symbol)
                df = df[
                    pd.to_datetime(df["date"], errors="coerce").between(
                        pd.Timestamp(start_date),
                        pd.Timestamp(end_date),
                        inclusive="both",
                    )
                ].reset_index(drop=True)
            else:
                if data_api is None:
                    raise RuntimeError("DataAPI is not initialized for stock universe.")
                df = data_api.get_price_history_data(symbol, start_date, end_date)
                df.columns = PRICE_COLUMNS

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()

            invalid_prices = DataAPI.detect_non_positive_prices(df)
            if invalid_prices:
                invalid_summary = ", ".join(
                    f"{column}={count}" for column, count in invalid_prices.items()
                )
                adjust_mode_label = data_api.adjust_mode_label if data_api is not None else "raw"
                logger.warning(
                    f"Skipping {symbol}: found non-positive prices under "
                    f"adjust_mode={adjust_mode_label} ({invalid_summary})"
                )
                continue

            if len(df) < min_data_length:
                df["signal"] = np.nan
                data[symbol] = df
                continue

            df = apply_signals_to_dataframe(
                df=df,
                strategies=strategies,
                signal_engine=signal_engine,
                signal_weights=signal_weights,
                signal_combination=signal_combination,
                signal_threshold=signal_threshold,
                symbol=symbol,
                indicator_cache=indicator_cache,
            )

            data[symbol] = df
            data_iterator.set_postfix({"loading": symbol})

    if data and "signal" not in next(iter(data.values())).columns:
        logger.info("Applying signals to loaded data...")
        for symbol in tqdm(data.keys(), desc="Applying signals", disable=False):
            df = data[symbol]
            if len(df) < min_data_length:
                df["signal"] = np.nan
            else:
                df = apply_signals_to_dataframe(
                    df=df,
                    strategies=strategies,
                    signal_engine=signal_engine,
                    signal_weights=signal_weights,
                    signal_combination=signal_combination,
                    signal_threshold=signal_threshold,
                    symbol=symbol,
                    indicator_cache=indicator_cache,
                )
            data[symbol] = df

    if not data:
        adjust_mode_label = data_api.adjust_mode_label if data_api is not None else "raw"
        raise ValueError(
            f"No valid price data loaded for backtest. "
            f"Please check universe and adjust_mode={adjust_mode_label}."
        )

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=config["backtest"]["commission_rate"],
        slippage=config["backtest"]["slippage"],
        config_path=config_path,
    )

    engine.set_stop_strategies(
        stop_loss_strategy=stop_loss_strategy,
        stop_profit_strategy=stop_profit_strategy,
    )

    logger.info("Backtest engine initialized")

    strategy_func = create_strategy_function(trade_amount)

    logger.info("Running backtest...")
    results = engine.run(data, strategy_func, start_date, end_date, show_progress=True)

    logger.info("Backtest completed, analyzing results...")

    results["returns"] = results["portfolio_value"].pct_change()
    results["cumulative_returns"] = (1 + results["returns"]).cumprod()

    total_cash = results["portfolio_value"].iloc[-1]
    total_return = (total_cash / initial_capital - 1) * 100
    sharpe = calculate_sharpe_ratio(results["returns"])
    max_drawdown = calculate_max_drawdown(results["portfolio_value"])
    position_counts = results["positions"].apply(len)
    max_position = position_counts.max()

    trades = engine.get_trades()
    if len(trades) == 0:
        logger.info("No trade records generated")
        return results

    filled_trades = trades
    if "status" in trades.columns:
        filled_trades = trades[trades["status"] == "filled"].copy()

    total_trades = len(filled_trades[filled_trades["action"] == "buy"])
    total_trades_done = len(filled_trades[filled_trades["action"] == "sell"])
    win_trades = len(filled_trades[filled_trades["profit"] > 0]) if "profit" in filled_trades.columns else 0
    win_rate = (win_trades / total_trades_done * 100) if total_trades_done > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("Backtest summary")
    logger.info("=" * 60)
    logger.info(f"Initial capital: {initial_capital:,.2f}")
    logger.info(f"Final portfolio value: {total_cash:,.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    annual_portfolio_returns = calculate_annual_portfolio_returns(results)
    if not annual_portfolio_returns.empty:
        logger.info("Annual portfolio return:")
        for year, annual_return in annual_portfolio_returns.items():
            logger.info(f"  {int(year)}: {annual_return:.2f}%")
    logger.info(f"Sharpe ratio: {sharpe:.4f}")
    logger.info(f"Max drawdown: {max_drawdown:.2f}%")
    logger.info(f"Trade count: {total_trades}")
    logger.info(f"Win rate: {win_rate:.2f}%")
    logger.info(f"Max concurrent positions: {max_position}")
    logger.info(f"Current positions: {len(results['positions'].iloc[-1].keys())}")
    if "status" in trades.columns:
        rejected_buy_count = len(
            trades[(trades["status"] == "rejected") & (trades["action"] == "buy")]
        )
        logger.info(f"Rejected buys (constraint): {rejected_buy_count}")

    if total_trades > 0 and "profit" in filled_trades.columns:
        avg_profit = filled_trades["profit"].mean()
        avg_profit_pct = filled_trades["profit_pct"].mean()
        logger.info(f"Average profit: {avg_profit:,.2f}")
        logger.info(f"Average profit pct: {avg_profit_pct:.2f}%")

        if "reason" in filled_trades.columns:
            stop_holding_count = len(filled_trades[filled_trades["reason"] == "stop_holding"])
            stop_loss_count = len(filled_trades[filled_trades["reason"] == "stop_loss"])
            stop_profit_count = len(filled_trades[filled_trades["reason"] == "stop_profit"])
            strategy_count = len(
                filled_trades[
                    (filled_trades["action"] == "sell")
                    & (filled_trades["reason"] == "strategy")
                ]
            )
            logger.info(f"Stop-holding exits: {stop_holding_count}")
            logger.info(f"Stop-loss exits: {stop_loss_count}")
            logger.info(f"Stop-profit exits: {stop_profit_count}")
            logger.info(f"Strategy exits: {strategy_count}")

    print_trades = False
    if len(trades) > 0 and print_trades:
        logger.info("\n" + "-" * 60)
        logger.info("Trade records")
        logger.info("-" * 60)
        for _, trade in trades.iterrows():
            reason = trade.get("reason", "strategy")
            reason_map = {
                "strategy": "strategy",
                "stop_holding": "Stop-holding",
                "stop_loss": "stop_loss",
                "stop_profit": "stop_profit",
            }
            reason_str = reason_map.get(reason, reason)

            if trade["action"] == "buy":
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"buy {trade['shares']} @ {trade['price']:.2f} | "
                    f"cost: {trade['cost']:,.2f}"
                )
            else:
                profit_str = (
                    f"profit: {trade['profit']:,.2f} ({trade['profit_pct']:.2f}%)"
                    if "profit" in trade
                    else ""
                )
                logger.info(
                    f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                    f"sell {trade['shares']} @ {trade['price']:.2f} | "
                    f"{profit_str} | {reason_str}"
                )

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name_str = "_".join(strategy_names) if len(strategy_names) > 1 else strategy_names[0]
    results_file = f"{output_dir}/backtest_{strategy_name_str}_{timestamp}.csv"
    trades_file = f"{output_dir}/trades_{strategy_name_str}_{timestamp}.csv"

    # results.to_csv(results_file)
    trades.to_csv(trades_file)

    logger.info("\nResults saved")
    logger.info(f"  Backtest results: {results_file}")
    logger.info(f"  Trade records: {trades_file}")

    return results, trades


def main():
    parser = argparse.ArgumentParser(
        description="Generic quant backtest runner with multi-signal support."
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="ma_cross",
        help="Strategy name. Use commas to pass multiple strategies, for example ma_cross,rsi",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config/settings.yaml",
        help="Path to the main config file",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="./config/strategies.yaml",
        help="Path to the strategy config file",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital override",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=None,
        help="Per-trade cash amount override",
    )
    parser.add_argument(
        "--stock-file",
        type=str,
        default=None,
        help="Path to the stock universe file",
    )
    parser.add_argument(
        "--no-stop-loss",
        action="store_true",
        help="Disable stop-loss strategy",
    )
    parser.add_argument(
        "--no-stop-profit",
        action="store_true",
        help="Disable stop-profit strategy",
    )
    parser.add_argument(
        "--adjust-mode",
        type=str,
        default=None,
        help="Price adjust mode override: hfq, qfq, or none",
    )
    parser.add_argument(
        "--raw-price-adjust",
        type=str,
        default=None,
        help="When adjust_mode is raw, rebuild prices from pct_change as qfq or hfq",
    )
    parser.add_argument(
        "--signal-combination",
        type=str,
        default="weighted",
        choices=["weighted", "voting", "unanimous"],
        help="How to combine multiple strategy signals",
    )
    parser.add_argument(
        "--signal-weights",
        type=str,
        default=None,
        help="Comma-separated weights for multiple signals, for example 0.6,0.4",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert combined signals into buy or sell actions",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available strategies",
    )
    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=-1,
        help="backtest max number of stock",
    )
    parser.add_argument(
        "--universe-type",
        type=str,
        default="stock",
        choices=["stock", "industry"],
        help="Backtest universe type: stock or industry index",
    )
    parser.add_argument(
        "--industry-index-list-file",
        type=str,
        default=None,
        help="Industry index universe list csv path",
    )
    parser.add_argument(
        "--industry-index-data-dir",
        type=str,
        default=None,
        help="Industry index price csv directory",
    )
    parser.add_argument(
        "--industry-index-codes",
        type=str,
        default=None,
        help="Comma-separated industry index codes, for example 881121,881273",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers for data loading. Use 1 for single process mode.",
    )

    args = parser.parse_args()

    if args.list:
        loader = StrategyLoader(args.strategy_config)
        loader.list_strategies()
        return

    strategy_names = [s.strip() for s in args.strategy.split(",")]

    signal_weights = None
    if args.signal_weights:
        signal_weights = [float(w.strip()) for w in args.signal_weights.split(",")]

    industry_index_codes = None
    if args.industry_index_codes:
        industry_index_codes = [code.strip() for code in args.industry_index_codes.split(",") if code.strip()]

    run_backtest(
        strategy_name=strategy_names if len(strategy_names) > 1 else strategy_names[0],
        start_date=args.start,
        end_date=args.end,
        config_path=args.config,
        strategy_config_path=args.strategy_config,
        stock_file=args.stock_file,
        stock_max_number=args.number,
        initial_capital=args.capital,
        trade_amount=args.trade_amount,
        adjust_mode=args.adjust_mode,
        raw_price_adjust=args.raw_price_adjust,
        enable_stop_loss=not args.no_stop_loss,
        enable_stop_profit=not args.no_stop_profit,
        signal_combination=args.signal_combination,
        signal_weights=signal_weights,
        signal_threshold=args.signal_threshold,
        universe_type=args.universe_type,
        industry_index_list_file=args.industry_index_list_file,
        industry_index_data_dir=args.industry_index_data_dir,
        industry_index_codes=industry_index_codes,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
