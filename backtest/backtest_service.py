"""Reusable backtest service functions for CLI and Streamlit workflows."""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_max_drawdown, calculate_sharpe_ratio
from data.data_api import DataAPI
from data.indicator_cache import IndicatorCache
from run_backtest import (
    PRICE_COLUMNS,
    apply_signals_to_dataframe,
    attach_indicator_cache_to_strategies,
    build_index_file_lookup,
    calculate_annual_portfolio_returns,
    load_industry_universe,
    load_single_symbol_industry,
    load_single_symbol_stock,
    read_csv_with_fallback,
    standardize_industry_price_frame,
)
from signals.signal_engine import SignalEngine
from signals.strategy_loader import StrategyLoader
from signals.timing.common_filters import (
    FilteredTimingStrategy,
    extract_common_timing_filter_params,
    has_common_timing_filters_enabled,
)


@dataclass
class BacktestRequest:
    """Parameters required for one interactive backtest run."""

    strategy_names: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    config_path: str = "./config/settings.yaml"
    strategy_config_path: str = "./config/strategies.yaml"
    stock_file: Optional[str] = None
    stock_max_number: int = -1
    initial_capital: Optional[float] = None
    trade_amount: Optional[float] = None
    commission_rate: Optional[float] = None
    slippage: Optional[float] = None
    adjust_mode: Optional[str] = None
    raw_price_adjust: Optional[str] = None
    enable_stop_loss: bool = True
    enable_stop_profit: bool = True
    stop_loss_params: Dict[str, Any] = field(default_factory=dict)
    stop_profit_params: Dict[str, Any] = field(default_factory=dict)
    signal_combination: str = "weighted"
    signal_weights: Optional[List[float]] = None
    signal_threshold: float = 0.5
    strategy_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    universe_type: str = "stock"
    industry_index_list_file: Optional[str] = None
    industry_index_data_dir: Optional[str] = None
    industry_index_codes: Optional[List[str]] = None
    max_workers: int = 1
    signal_workers: int = 1
    show_progress: bool = False


@dataclass
class LoadedMarketData:
    """Loaded price data and diagnostics."""

    data: Dict[str, pd.DataFrame]
    symbols: List[str]
    skipped_symbols: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source: str = ""
    adjust_mode_label: str = "raw"
    universe_type: str = "stock"


@dataclass
class BacktestRunResult:
    """Backtest output with summary metrics."""

    results: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    annual_returns: pd.Series
    request: BacktestRequest


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def list_timing_strategy_configs(strategy_config_path: str = "./config/strategies.yaml") -> Dict[str, Dict[str, Any]]:
    """Return timing strategy registry from YAML."""
    loader = StrategyLoader(strategy_config_path)
    return loader.config.get("timing_strategies", {})


def list_stop_strategy_configs(strategy_config_path: str = "./config/strategies.yaml") -> Dict[str, Dict[str, Any]]:
    """Return stop strategy registry from YAML."""
    loader = StrategyLoader(strategy_config_path)
    return loader.config.get("stop_strategies", {})


def resolve_request_defaults(request: BacktestRequest) -> Tuple[BacktestRequest, Dict[str, Any]]:
    """Fill request fields from project config without mutating the original request."""
    config = load_yaml_config(request.config_path)
    resolved = copy.deepcopy(request)
    backtest_config = config.get("backtest", {})
    data_config = config.get("data", {})

    if resolved.start_date is None:
        resolved.start_date = backtest_config.get("start_date", "2022-01-01")
    if resolved.end_date is None:
        resolved.end_date = backtest_config.get("end_date", "2023-12-31")
    if resolved.initial_capital is None:
        resolved.initial_capital = float(backtest_config.get("initial_capital", 1000000))
    if resolved.trade_amount is None:
        resolved.trade_amount = float(backtest_config.get("trade_amount", 100000))
    if resolved.commission_rate is None:
        resolved.commission_rate = float(backtest_config.get("commission_rate", 0.0003))
    if resolved.slippage is None:
        resolved.slippage = float(backtest_config.get("slippage", 0.0001))
    if resolved.stock_file is None:
        resolved.stock_file = data_config.get("stock_file", "./data/test1.txt")
    if resolved.adjust_mode is None:
        resolved.adjust_mode = data_config.get("adjust_mode", "hfq")
    if resolved.raw_price_adjust is None:
        resolved.raw_price_adjust = data_config.get("raw_price_adjust", "")
    if resolved.industry_index_list_file is None:
        resolved.industry_index_list_file = data_config.get(
            "industry_index_list_file",
            "./data/raw/akshare/industry_sector_index_list.csv",
        )
    if resolved.industry_index_data_dir is None:
        resolved.industry_index_data_dir = data_config.get(
            "industry_index_data_dir",
            "./data/raw/akshare/industry_sector_index",
        )

    resolved.universe_type = (resolved.universe_type or "stock").strip().lower()
    resolved.strategy_names = [name.strip() for name in resolved.strategy_names if name.strip()]
    if not resolved.strategy_names:
        raise ValueError("At least one timing strategy is required.")

    if resolved.signal_weights is None:
        resolved.signal_weights = [1.0 / len(resolved.strategy_names)] * len(resolved.strategy_names)
    if len(resolved.signal_weights) != len(resolved.strategy_names):
        raise ValueError("Signal weight count must match selected strategy count.")

    return resolved, config


def build_timing_strategy_bundle(
    request: BacktestRequest,
) -> Tuple[List[object], List[Dict[str, Any]], int]:
    """Instantiate timing strategies with optional UI parameter overrides."""
    loader = StrategyLoader(request.strategy_config_path)
    strategies: List[object] = []
    strategy_infos: List[Dict[str, Any]] = []

    for strategy_name in request.strategy_names:
        strategy_class, strategy_info = loader.get_strategy(strategy_name)
        params = dict(strategy_info.get("params", {}))
        params.update(request.strategy_params.get(strategy_name, {}))
        constructor_params = loader._filter_constructor_params(strategy_class, params)
        strategy = strategy_class(**constructor_params)

        common_filter_params = extract_common_timing_filter_params(params)
        if has_common_timing_filters_enabled(common_filter_params):
            strategy = FilteredTimingStrategy(strategy, common_filter_params)

        info = copy.deepcopy(strategy_info)
        info["key"] = strategy_name
        info["params"] = params
        strategies.append(strategy)
        strategy_infos.append(info)

    min_data_length = max(info.get("min_data_length", 20) for info in strategy_infos)
    return strategies, strategy_infos, min_data_length


def build_stop_strategy_bundle(
    request: BacktestRequest,
) -> Tuple[Optional[object], Optional[Dict[str, Any]], Optional[object], Optional[Dict[str, Any]]]:
    """Instantiate stop strategies with optional UI parameter overrides."""
    loader = StrategyLoader(request.strategy_config_path)
    stop_loss_strategy = None
    stop_loss_info = None
    stop_profit_strategy = None
    stop_profit_info = None

    if request.enable_stop_loss:
        strategy_class, raw_info = loader.get_stop_strategy("stop_loss")
        if strategy_class and raw_info:
            params = dict(raw_info.get("params", {}))
            params.update(request.stop_loss_params)
            stop_loss_strategy = strategy_class(**params)
            stop_loss_info = copy.deepcopy(raw_info)
            stop_loss_info["params"] = params

    if request.enable_stop_profit:
        strategy_class, raw_info = loader.get_stop_strategy("stop_profit")
        if strategy_class and raw_info:
            params = dict(raw_info.get("params", {}))
            params.update(request.stop_profit_params)
            stop_profit_strategy = strategy_class(**params)
            stop_profit_info = copy.deepcopy(raw_info)
            stop_profit_info["params"] = params

    return stop_loss_strategy, stop_loss_info, stop_profit_strategy, stop_profit_info


def create_trade_strategy_function(trade_amount: float):
    """Create broker-facing strategy callback using signal columns."""

    def is_missing(value: Any) -> bool:
        pd_isna = getattr(pd, "isna", None)
        if callable(pd_isna):
            return bool(pd_isna(value))
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except TypeError:
            return False

    def strategy_func(date, data, positions):
        signals = {}
        for symbol, df in data.items():
            if date not in df.index or "signal" not in df.columns:
                continue

            current_signal = df.loc[date, "signal"]
            if is_missing(current_signal):
                continue

            current_price = df.loc[date, "close"]
            if is_missing(current_price) or current_price <= 0:
                continue

            current_pos = positions.get(symbol, 0)
            shares = int(trade_amount / current_price)
            if current_signal == 1 and current_pos == 0:
                signals[symbol] = {"action": "buy", "shares": shares}
            elif current_signal == -1 and current_pos > 0:
                signals[symbol] = {"action": "sell", "shares": current_pos}

        return signals

    return strategy_func


def make_indicator_cache(config: Dict[str, Any], source: str, adjust_mode_label: str) -> IndicatorCache:
    """Create the shared indicator cache instance."""
    indicator_cache_config = config.get("indicator_cache", {})
    data_config = config.get("data", {})
    return IndicatorCache(
        cache_dir=indicator_cache_config.get(
            "dir",
            os.path.join(data_config.get("processed_dir", "./data/processed"), "indicators"),
        ),
        source=source,
        adjust_mode=adjust_mode_label,
        enabled=indicator_cache_config.get("enabled", True),
    )


_SIGNAL_WORKER_CONTEXT: Dict[str, Any] = {}


def _initialize_signal_worker(
    request: BacktestRequest,
    config: Dict[str, Any],
    source: str,
    adjust_mode_label: str,
    universe_type: str,
) -> None:
    """Initialize isolated strategy state for one signal worker process."""
    strategies, _, min_data_length = build_timing_strategy_bundle(request)
    indicator_cache = None
    if universe_type == "stock":
        indicator_cache = make_indicator_cache(config, source, adjust_mode_label)

    global _SIGNAL_WORKER_CONTEXT
    _SIGNAL_WORKER_CONTEXT = {
        "strategies": strategies,
        "signal_engine": SignalEngine(),
        "min_data_length": min_data_length,
        "signal_weights": list(request.signal_weights or []),
        "signal_combination": request.signal_combination,
        "signal_threshold": request.signal_threshold,
        "indicator_cache": indicator_cache,
    }


def _generate_symbol_signal(
    index: int,
    symbol: str,
    source_df: pd.DataFrame,
    strategies: List[object],
    signal_engine: SignalEngine,
    min_data_length: int,
    signal_weights: List[float],
    signal_combination: str,
    signal_threshold: float,
    indicator_cache: Optional[IndicatorCache],
) -> Tuple[int, str, pd.DataFrame]:
    """Generate one symbol's signal data with isolated input data."""
    df = source_df.copy()
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
        attach_indicator_cache_to_strategies(strategies, indicator_cache, symbol)
    return index, symbol, df


def _generate_symbol_signal_in_worker(
    index: int,
    symbol: str,
    source_df: pd.DataFrame,
) -> Tuple[int, str, pd.DataFrame]:
    """Run one symbol task using process-local signal worker state."""
    if not _SIGNAL_WORKER_CONTEXT:
        raise RuntimeError("Signal worker is not initialized.")

    return _generate_symbol_signal(
        index=index,
        symbol=symbol,
        source_df=source_df,
        strategies=_SIGNAL_WORKER_CONTEXT["strategies"],
        signal_engine=_SIGNAL_WORKER_CONTEXT["signal_engine"],
        min_data_length=_SIGNAL_WORKER_CONTEXT["min_data_length"],
        signal_weights=_SIGNAL_WORKER_CONTEXT["signal_weights"],
        signal_combination=_SIGNAL_WORKER_CONTEXT["signal_combination"],
        signal_threshold=_SIGNAL_WORKER_CONTEXT["signal_threshold"],
        indicator_cache=_SIGNAL_WORKER_CONTEXT["indicator_cache"],
    )


def load_market_data(
    request: BacktestRequest,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> LoadedMarketData:
    """Load price data without generating strategy signals."""
    resolved, config = resolve_request_defaults(request)
    _, _, min_data_length = build_timing_strategy_bundle(resolved)
    data_source = config.get("data", {}).get("source", "akshare")
    data: Dict[str, pd.DataFrame] = {}
    skipped_symbols: List[str] = []
    warnings: List[str] = []
    symbols: List[str] = []
    adjust_mode_label = "raw"

    if resolved.universe_type not in {"stock", "industry"}:
        raise ValueError(f"Unsupported universe_type: {resolved.universe_type}")

    industry_name_map: Dict[str, str] = {}
    industry_file_lookup: Dict[str, Path] = {}
    data_api = None

    if resolved.universe_type == "industry":
        if not os.path.exists(str(resolved.industry_index_list_file)):
            raise FileNotFoundError(f"Industry index list file not found: {resolved.industry_index_list_file}")
        if not os.path.exists(str(resolved.industry_index_data_dir)):
            raise FileNotFoundError(f"Industry index data directory not found: {resolved.industry_index_data_dir}")

        universe_df = load_industry_universe(
            index_list_file=str(resolved.industry_index_list_file),
            index_codes=resolved.industry_index_codes,
            max_indexes=resolved.stock_max_number,
        )
        symbols = universe_df["index_code"].tolist()
        industry_name_map = dict(zip(universe_df["index_code"], universe_df["industry_name"]))
        industry_file_lookup = build_index_file_lookup(str(resolved.industry_index_data_dir))
    else:
        data_api = DataAPI(
            source=data_source,
            stock_file=str(resolved.stock_file),
            cache_dir=config["data"]["cache_dir"],
            processed_dir=config["data"]["processed_dir"],
            adjust_mode=resolved.adjust_mode,
            raw_price_adjust=resolved.raw_price_adjust,
        )
        adjust_mode_label = data_api.adjust_mode_label
        symbols = data_api.get_stock_list()
        if resolved.stock_max_number != -1 and len(symbols) > resolved.stock_max_number:
            symbols = symbols[: resolved.stock_max_number]

    if resolved.max_workers > 1 and len(symbols) > 1:
        if progress_callback is not None:
            progress_callback(0, len(symbols), "Submitting data loading tasks...")
        with ProcessPoolExecutor(max_workers=resolved.max_workers) as executor:
            futures = {}
            if resolved.universe_type == "industry":
                for symbol in symbols:
                    csv_path = industry_file_lookup.get(symbol)
                    if csv_path is None:
                        skipped_symbols.append(symbol)
                        warnings.append(f"Skipping {symbol} ({industry_name_map.get(symbol, '')}): missing csv file")
                        continue
                    futures[
                        executor.submit(
                            load_single_symbol_industry,
                            symbol=symbol,
                            csv_path=csv_path,
                            start_date=str(resolved.start_date),
                            end_date=str(resolved.end_date),
                            min_data_length=min_data_length,
                            price_columns=PRICE_COLUMNS,
                        )
                    ] = symbol
            else:
                for symbol in symbols:
                    futures[
                        executor.submit(
                            load_single_symbol_stock,
                            symbol=symbol,
                            start_date=str(resolved.start_date),
                            end_date=str(resolved.end_date),
                            source=data_source,
                            stock_file=str(resolved.stock_file),
                            cache_dir=config["data"]["cache_dir"],
                            processed_dir=config["data"]["processed_dir"],
                            adjust_mode=resolved.adjust_mode,
                            raw_price_adjust=resolved.raw_price_adjust,
                            min_data_length=min_data_length,
                            price_columns=PRICE_COLUMNS,
                        )
                    ] = symbol

            completed = len(symbols) - len(futures)
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, df = future.result()
                    if df is None:
                        skipped_symbols.append(symbol)
                    else:
                        data[symbol] = df
                except Exception as exc:
                    skipped_symbols.append(symbol)
                    warnings.append(f"Failed to load {symbol}: {exc}")
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(symbols), f"Loaded {symbol}")
    else:
        if progress_callback is not None:
            progress_callback(0, len(symbols), "Starting data loading...")
        for index, symbol in enumerate(symbols, 1):
            try:
                if resolved.universe_type == "industry":
                    csv_path = industry_file_lookup.get(symbol)
                    if csv_path is None:
                        skipped_symbols.append(symbol)
                        warnings.append(f"Skipping {symbol} ({industry_name_map.get(symbol, '')}): missing csv file")
                        continue
                    raw_df = read_csv_with_fallback(csv_path)
                    df = standardize_industry_price_frame(raw_df, symbol)
                    df = df[
                        pd.to_datetime(df["date"], errors="coerce").between(
                            pd.Timestamp(resolved.start_date),
                            pd.Timestamp(resolved.end_date),
                            inclusive="both",
                        )
                    ].reset_index(drop=True)
                else:
                    if data_api is None:
                        raise RuntimeError("DataAPI is not initialized for stock universe.")
                    df = data_api.get_price_history_data(symbol, str(resolved.start_date), str(resolved.end_date))
                    df.columns = PRICE_COLUMNS

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()

                invalid_prices = DataAPI.detect_non_positive_prices(df)
                if invalid_prices:
                    skipped_symbols.append(symbol)
                    invalid_summary = ", ".join(f"{column}={count}" for column, count in invalid_prices.items())
                    warnings.append(f"Skipping {symbol}: non-positive prices ({invalid_summary})")
                    continue

                if len(df) < min_data_length:
                    df["signal"] = np.nan
                data[symbol] = df
            except Exception as exc:
                skipped_symbols.append(symbol)
                warnings.append(f"Failed to load {symbol}: {exc}")
            if progress_callback is not None:
                progress_callback(index, len(symbols), f"Loaded {symbol}")

    return LoadedMarketData(
        data=data,
        symbols=symbols,
        skipped_symbols=skipped_symbols,
        warnings=warnings,
        source=data_source,
        adjust_mode_label=adjust_mode_label,
        universe_type=resolved.universe_type,
    )


def generate_signals(
    loaded_data: LoadedMarketData,
    request: BacktestRequest,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate combined timing signals for loaded market data."""
    resolved, config = resolve_request_defaults(request)
    strategies, _, min_data_length = build_timing_strategy_bundle(resolved)
    signal_engine = SignalEngine()
    indicator_cache = None
    if loaded_data.universe_type == "stock":
        indicator_cache = make_indicator_cache(config, loaded_data.source, loaded_data.adjust_mode_label)

    signaled_data: Dict[str, pd.DataFrame] = {}
    total = len(loaded_data.data)
    if progress_callback is not None:
        progress_callback(0, total, "Starting signal generation...")

    signal_workers = max(1, int(resolved.signal_workers))
    if signal_workers > 1 and total > 1:
        completed_results: Dict[int, Tuple[str, pd.DataFrame]] = {}
        with ProcessPoolExecutor(
            max_workers=signal_workers,
            initializer=_initialize_signal_worker,
            initargs=(
                resolved,
                config,
                loaded_data.source,
                loaded_data.adjust_mode_label,
                loaded_data.universe_type,
            ),
        ) as executor:
            futures = {
                executor.submit(_generate_symbol_signal_in_worker, index, symbol, source_df): (index, symbol)
                for index, (symbol, source_df) in enumerate(loaded_data.data.items(), 1)
            }
            for completed, future in enumerate(as_completed(futures), 1):
                _, expected_symbol = futures[future]
                try:
                    result_index, symbol, df = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to generate signal for {expected_symbol}") from exc
                completed_results[result_index] = (symbol, df)
                if progress_callback is not None:
                    progress_callback(completed, total, f"Generated signal for {symbol}")

        for index in range(1, total + 1):
            symbol, df = completed_results[index]
            signaled_data[symbol] = df
    else:
        for index, (symbol, source_df) in enumerate(loaded_data.data.items(), 1):
            _, _, df = _generate_symbol_signal(
                index=index,
                symbol=symbol,
                source_df=source_df,
                strategies=strategies,
                signal_engine=signal_engine,
                min_data_length=min_data_length,
                signal_weights=list(resolved.signal_weights or []),
                signal_combination=resolved.signal_combination,
                signal_threshold=resolved.signal_threshold,
                indicator_cache=indicator_cache,
            )
            signaled_data[symbol] = df
            if progress_callback is not None:
                progress_callback(index, total, f"Generated signal for {symbol}")

    return signaled_data


def run_engine_with_signals(
    signaled_data: Dict[str, pd.DataFrame],
    request: BacktestRequest,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BacktestRunResult:
    """Run the backtest engine using already-generated signal data."""
    resolved, config = resolve_request_defaults(request)
    if not signaled_data:
        raise ValueError("No valid price data loaded for backtest.")

    stop_loss_strategy, _, stop_profit_strategy, _ = build_stop_strategy_bundle(resolved)
    engine = BacktestEngine(
        initial_capital=float(resolved.initial_capital),
        commission_rate=float(resolved.commission_rate),
        slippage=float(resolved.slippage),
        config_path=resolved.config_path,
    )
    engine.set_stop_strategies(
        stop_loss_strategy=stop_loss_strategy,
        stop_profit_strategy=stop_profit_strategy,
    )

    strategy_func = create_trade_strategy_function(float(resolved.trade_amount))
    results = engine.run(
        signaled_data,
        strategy_func,
        str(resolved.start_date),
        str(resolved.end_date),
        show_progress=resolved.show_progress,
        progress_callback=progress_callback,
    )
    trades = engine.get_trades()

    if not results.empty:
        results["returns"] = results["portfolio_value"].pct_change()
        results["cumulative_returns"] = (1 + results["returns"]).cumprod()

    annual_returns = calculate_annual_portfolio_returns(results)
    metrics = summarize_backtest(results, trades, float(resolved.initial_capital))
    return BacktestRunResult(
        results=results,
        trades=trades,
        metrics=metrics,
        annual_returns=annual_returns,
        request=resolved,
    )


def run_backtest_request(
    request: BacktestRequest,
    loaded_data: Optional[LoadedMarketData] = None,
) -> BacktestRunResult:
    """Convenience function for a full load -> signal -> engine workflow."""
    market_data = loaded_data or load_market_data(request)
    signaled_data = generate_signals(market_data, request)
    return run_engine_with_signals(signaled_data, request)


def summarize_backtest(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, Any]:
    """Build UI-friendly performance metrics."""
    if results.empty or "portfolio_value" not in results.columns:
        return {}

    final_value = float(results["portfolio_value"].iloc[-1])
    total_return = (final_value / initial_capital - 1) * 100
    returns = results["returns"] if "returns" in results.columns else results["portfolio_value"].pct_change()
    sharpe = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(results["portfolio_value"])
    position_counts = results["positions"].apply(len) if "positions" in results.columns else pd.Series(dtype=int)
    max_position = int(position_counts.max()) if not position_counts.empty else 0
    current_positions = int(len(results["positions"].iloc[-1])) if "positions" in results.columns and len(results) else 0

    filled_trades = trades
    if not trades.empty and "status" in trades.columns:
        filled_trades = trades[trades["status"] == "filled"].copy()

    total_buys = int(len(filled_trades[filled_trades["action"] == "buy"])) if "action" in filled_trades.columns else 0
    total_sells = int(len(filled_trades[filled_trades["action"] == "sell"])) if "action" in filled_trades.columns else 0
    win_trades = int(len(filled_trades[filled_trades["profit"] > 0])) if "profit" in filled_trades.columns else 0
    win_rate = (win_trades / total_sells * 100) if total_sells > 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown,
        "buy_count": total_buys,
        "sell_count": total_sells,
        "win_rate_pct": win_rate,
        "max_position": max_position,
        "current_positions": current_positions,
    }


def export_backtest_result(
    result: BacktestRunResult,
    output_dir: Union[str, Path] = "./output",
) -> Tuple[Path, Path]:
    """Save backtest result and trade records to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = "_".join(result.request.strategy_names)
    results_file = output_path / f"backtest_{strategy_name}_{timestamp}.csv"
    trades_file = output_path / f"trades_{strategy_name}_{timestamp}.csv"

    result.results.to_csv(results_file, encoding="utf-8-sig")
    result.trades.to_csv(trades_file, encoding="utf-8-sig")
    return results_file, trades_file
