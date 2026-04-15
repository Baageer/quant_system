"""
Bollinger parameter study example.

This script shows a practical workflow for tuning indicator parameters
without relying on a single in-sample backtest:

1. build a parameter grid for one Bollinger mode
2. evaluate every candidate on train and validation windows
3. rank candidates by validation quality with a stability penalty
4. re-run only the top candidates on the test window
5. export csv files for later inspection

Example:
    python research/bollinger_parameter_study.py --mode breakout --stock-max-number 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import itertools
import json
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


STANDARD_PRICE_COLUMNS = [
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

_WORKER_CONTEXT: Dict[str, object] = {}


def find_project_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "run_backtest.py").exists() and (candidate / "config" / "settings.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine  # noqa: E402
from backtest.performance import PerformanceAnalyzer  # noqa: E402
from data.data_api import DataAPI  # noqa: E402
from signals.strategy_loader import StrategyLoader  # noqa: E402
from signals.timing.bollinger_bands import BollingerBandsStrategy  # noqa: E402


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def normalize_price_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if len(df.columns) != len(STANDARD_PRICE_COLUMNS):
        raise ValueError(
            "Unexpected price history column count. "
            f"Expected {len(STANDARD_PRICE_COLUMNS)}, got {len(df.columns)}."
        )

    df.columns = STANDARD_PRICE_COLUMNS
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def load_universe_data(
    settings: Mapping[str, object],
    stock_file: Optional[str],
    stock_max_number: int,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    data_api = DataAPI(
        source=str(settings["data"].get("source", "akshare")),
        stock_file=stock_file or str(settings["data"].get("stock_file", "./data/HS300.txt")),
        cache_dir=str(settings["data"]["cache_dir"]),
        processed_dir=str(settings["data"]["processed_dir"]),
        adjust_mode=str(settings["data"].get("adjust_mode", "qfq")),
    )

    stock_list = data_api.get_stock_list()
    if stock_max_number > 0:
        stock_list = stock_list[:stock_max_number]

    loaded: Dict[str, pd.DataFrame] = {}
    for symbol in tqdm(stock_list, desc="Loading universe", unit="symbol"):
        raw_df = data_api.get_price_history_data(symbol, start_date, end_date)
        if raw_df is None or raw_df.empty:
            continue

        df = normalize_price_frame(raw_df)
        invalid_prices = data_api.detect_non_positive_prices(df)
        if invalid_prices:
            continue

        loaded[symbol] = df

    if not loaded:
        raise ValueError("No valid price data was loaded for the requested universe and dates.")

    return loaded


def create_strategy_callback(trade_amount: float):
    def strategy_func(date, data, positions):
        signals = {}

        for symbol, df in data.items():
            if date not in df.index:
                continue

            current_signal = df.loc[date, "signal"]
            if _is_missing(current_signal):
                continue

            current_price = df.loc[date, "close"]
            if _is_missing(current_price) or float(current_price) <= 0:
                continue

            current_pos = positions.get(symbol, 0)
            shares = int(trade_amount / float(current_price))
            if shares <= 0:
                continue

            if current_signal == 1 and current_pos == 0:
                signals[symbol] = {"action": "buy", "shares": shares}
            elif current_signal == -1 and current_pos > 0:
                signals[symbol] = {"action": "sell", "shares": current_pos}

        return signals

    return strategy_func


def compute_min_data_length(params: Mapping[str, object]) -> int:
    lengths = [int(params["window"]) + 2, int(params.get("signal_delay", 0)) + 1]

    if bool(params.get("use_trend_filter", False)):
        trend_window = int(params.get("trend_window", params["window"]))
        trend_slope_window = int(params.get("trend_slope_window", 3))
        lengths.append(trend_window + trend_slope_window)

    if bool(params.get("use_volume_filter", False)):
        lengths.append(int(params.get("volume_window", 20)) + 1)

    mode = str(params.get("mode", "breakout"))
    if mode == "squeeze":
        lengths.append(int(params.get("squeeze_lookback", 60)) + int(params["window"]))
    if mode == "double":
        lengths.append(int(params.get("confirm_bars", 2)) + int(params["window"]))

    return max(lengths)


def prepare_signal_data(
    source_data: Mapping[str, pd.DataFrame],
    params: Mapping[str, object],
) -> Dict[str, pd.DataFrame]:
    strategy = BollingerBandsStrategy(**params)
    min_data_length = compute_min_data_length(params)

    prepared: Dict[str, pd.DataFrame] = {}
    for symbol, df in source_data.items():
        working = df.copy()
        if len(working) < min_data_length:
            working["signal"] = 0
        else:
            working["signal"] = strategy.generate_signal(working)
        prepared[symbol] = working
    return prepared


def _initialize_worker_context(
    source_data: Mapping[str, pd.DataFrame],
    commission_rate: float,
    slippage: float,
    config_path: str,
    strategy_config_path: str,
    periods: Mapping[str, Mapping[str, str]],
    initial_capital: float,
    trade_amount: float,
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    signal_horizons: Sequence[int],
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "source_data": dict(source_data),
        "commission_rate": float(commission_rate),
        "slippage": float(slippage),
        "config_path": str(config_path),
        "strategy_config_path": str(strategy_config_path),
        "periods": dict(periods),
        "initial_capital": float(initial_capital),
        "trade_amount": float(trade_amount),
        "enable_stop_loss": bool(enable_stop_loss),
        "enable_stop_profit": bool(enable_stop_profit),
        "signal_horizons": list(signal_horizons),
    }


def _evaluate_grid_candidate_task(params: Mapping[str, object]) -> MutableMapping[str, object]:
    context = _WORKER_CONTEXT
    if not context:
        raise RuntimeError("Worker context is not initialized.")

    strategy_loader = StrategyLoader(str(context["strategy_config_path"]))
    data_with_signals = prepare_signal_data(context["source_data"], params)

    train_metrics = run_period_backtest(
        data_with_signals=data_with_signals,
        commission_rate=float(context["commission_rate"]),
        slippage=float(context["slippage"]),
        config_path=Path(str(context["config_path"])),
        start_date=str(context["periods"]["train"]["start"]),
        end_date=str(context["periods"]["train"]["end"]),
        initial_capital=float(context["initial_capital"]),
        trade_amount=float(context["trade_amount"]),
        enable_stop_loss=bool(context["enable_stop_loss"]),
        enable_stop_profit=bool(context["enable_stop_profit"]),
        strategy_loader=strategy_loader,
        signal_horizons=context["signal_horizons"],
    )
    valid_metrics = run_period_backtest(
        data_with_signals=data_with_signals,
        commission_rate=float(context["commission_rate"]),
        slippage=float(context["slippage"]),
        config_path=Path(str(context["config_path"])),
        start_date=str(context["periods"]["valid"]["start"]),
        end_date=str(context["periods"]["valid"]["end"]),
        initial_capital=float(context["initial_capital"]),
        trade_amount=float(context["trade_amount"]),
        enable_stop_loss=bool(context["enable_stop_loss"]),
        enable_stop_profit=bool(context["enable_stop_profit"]),
        strategy_loader=strategy_loader,
        signal_horizons=context["signal_horizons"],
    )

    row: MutableMapping[str, object] = dict(params)
    row.update(flatten_metrics("train", train_metrics))
    row.update(flatten_metrics("valid", valid_metrics))
    return row


def _evaluate_test_candidate_task(candidate_row: Mapping[str, object]) -> MutableMapping[str, object]:
    context = _WORKER_CONTEXT
    if not context:
        raise RuntimeError("Worker context is not initialized.")

    allowed_parameter_keys = {
        "mode",
        "window",
        "num_std",
        "squeeze_threshold",
        "confirm_bars",
        "squeeze_quantile",
        "squeeze_lookback",
        "use_supertrend_filter",
        "use_trend_filter",
        "trend_window",
        "trend_slope_window",
        "use_volume_filter",
        "volume_window",
        "volume_multiplier",
        "signal_delay",
        "price_col",
        "high_col",
        "low_col",
        "volume_col",
    }

    params = {
        key: candidate_row[key]
        for key in candidate_row
        if key in allowed_parameter_keys and not _is_missing(candidate_row[key])
    }

    strategy_loader = StrategyLoader(str(context["strategy_config_path"]))
    data_with_signals = prepare_signal_data(context["source_data"], params)
    test_metrics = run_period_backtest(
        data_with_signals=data_with_signals,
        commission_rate=float(context["commission_rate"]),
        slippage=float(context["slippage"]),
        config_path=Path(str(context["config_path"])),
        start_date=str(context["periods"]["test"]["start"]),
        end_date=str(context["periods"]["test"]["end"]),
        initial_capital=float(context["initial_capital"]),
        trade_amount=float(context["trade_amount"]),
        enable_stop_loss=bool(context["enable_stop_loss"]),
        enable_stop_profit=bool(context["enable_stop_profit"]),
        strategy_loader=strategy_loader,
        signal_horizons=context["signal_horizons"],
    )

    combined = dict(candidate_row)
    combined.update(flatten_metrics("test", test_metrics))
    return combined


def _evaluate_test_candidate_task_with_local_context(
    candidate_row: Mapping[str, object],
    source_data: Mapping[str, pd.DataFrame],
    commission_rate: float,
    slippage: float,
    config_path: Path,
    strategy_config_path: Path,
    periods: Mapping[str, Mapping[str, str]],
    initial_capital: float,
    trade_amount: float,
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    signal_horizons: Sequence[int],
) -> MutableMapping[str, object]:
    _initialize_worker_context(
        source_data=source_data,
        commission_rate=commission_rate,
        slippage=slippage,
        config_path=str(config_path),
        strategy_config_path=str(strategy_config_path),
        periods=periods,
        initial_capital=initial_capital,
        trade_amount=trade_amount,
        enable_stop_loss=enable_stop_loss,
        enable_stop_profit=enable_stop_profit,
        signal_horizons=signal_horizons,
    )
    return _evaluate_test_candidate_task(candidate_row)


def _resolve_index_position(index: pd.Index, label: pd.Timestamp) -> Optional[int]:
    loc = index.get_loc(label)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        if len(loc) == 0:
            return None
        return int(loc[0])
    return int(loc)


def summarize_signal_events(
    source_data: Mapping[str, pd.DataFrame],
    trades: pd.DataFrame,
    horizons: Sequence[int],
) -> Dict[str, float]:
    horizons = sorted({int(horizon) for horizon in horizons if int(horizon) > 0})
    if not horizons:
        raise ValueError("signal horizons must contain at least one positive integer")

    max_horizon = max(horizons)
    empty_metrics: Dict[str, float] = {
        "signal_event_count": 0,
        "signal_buy_event_count": 0,
        "signal_sell_event_count": 0,
    }
    for horizon in horizons:
        empty_metrics[f"signal_sample_count_{horizon}"] = 0
        empty_metrics[f"signal_mean_edge_{horizon}"] = 0.0
        empty_metrics[f"signal_median_edge_{horizon}"] = 0.0
        empty_metrics[f"signal_hit_rate_{horizon}"] = 0.0
    empty_metrics[f"signal_mean_mfe_{max_horizon}"] = 0.0
    empty_metrics[f"signal_mean_mae_{max_horizon}"] = 0.0

    if trades.empty:
        return empty_metrics

    event_rows: List[Dict[str, float]] = []
    for _, trade in trades.iterrows():
        action = trade.get("action")
        if action not in {"buy", "sell"}:
            continue

        symbol = trade.get("symbol")
        trade_date = pd.Timestamp(trade.get("date"))
        df = source_data.get(symbol)
        if df is None or trade_date not in df.index:
            continue

        position = _resolve_index_position(df.index, trade_date)
        if position is None:
            continue

        event_close = df.loc[trade_date, "close"]
        if _is_missing(event_close) or float(event_close) <= 0:
            continue

        sign = 1.0 if action == "buy" else -1.0
        row: Dict[str, float] = {"action_sign": sign}

        for horizon in horizons:
            future_position = position + horizon
            if future_position >= len(df):
                row[f"edge_{horizon}"] = np.nan
                row[f"hit_{horizon}"] = np.nan
                continue

            future_close = df["close"].iloc[future_position]
            if _is_missing(future_close) or float(future_close) <= 0:
                row[f"edge_{horizon}"] = np.nan
                row[f"hit_{horizon}"] = np.nan
                continue

            directed_return = sign * (float(future_close) / float(event_close) - 1.0)
            row[f"edge_{horizon}"] = directed_return
            row[f"hit_{horizon}"] = 1.0 if directed_return > 0 else 0.0

        path_end = min(len(df), position + max_horizon + 1)
        future_path = df["close"].iloc[position + 1 : path_end].astype(float)
        if future_path.empty:
            row[f"mfe_{max_horizon}"] = 0.0
            row[f"mae_{max_horizon}"] = 0.0
        else:
            directed_path = sign * (future_path / float(event_close) - 1.0)
            row[f"mfe_{max_horizon}"] = float(directed_path.max())
            row[f"mae_{max_horizon}"] = float(directed_path.min())

        event_rows.append(row)

    if not event_rows:
        return empty_metrics

    events = pd.DataFrame(event_rows)
    metrics = {
        "signal_event_count": int(len(events)),
        "signal_buy_event_count": int((events["action_sign"] > 0).sum()),
        "signal_sell_event_count": int((events["action_sign"] < 0).sum()),
    }
    for horizon in horizons:
        edge_col = f"edge_{horizon}"
        hit_col = f"hit_{horizon}"
        valid_mask = events[edge_col].notna()
        metrics[f"signal_sample_count_{horizon}"] = int(valid_mask.sum())
        metrics[f"signal_mean_edge_{horizon}"] = (
            float(events.loc[valid_mask, edge_col].mean()) if valid_mask.any() else 0.0
        )
        metrics[f"signal_median_edge_{horizon}"] = (
            float(events.loc[valid_mask, edge_col].median()) if valid_mask.any() else 0.0
        )
        metrics[f"signal_hit_rate_{horizon}"] = (
            float(events.loc[valid_mask, hit_col].mean()) if valid_mask.any() else 0.0
        )

    metrics[f"signal_mean_mfe_{max_horizon}"] = float(events[f"mfe_{max_horizon}"].mean())
    metrics[f"signal_mean_mae_{max_horizon}"] = float(events[f"mae_{max_horizon}"].mean())
    return metrics


def summarize_period(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, float]:
    analyzer = PerformanceAnalyzer()

    if results.empty:
        return {
            "final_portfolio_value": initial_capital,
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "buy_trade_count": 0,
            "sell_trade_count": 0,
            "avg_profit_pct": 0.0,
            "final_position_count": 0,
        }

    analysis = analyzer.analyze(results["portfolio_value"])
    sell_trades = trades[trades["action"] == "sell"] if not trades.empty else pd.DataFrame()
    buy_trades = trades[trades["action"] == "buy"] if not trades.empty else pd.DataFrame()

    win_rate = 0.0
    avg_profit_pct = 0.0
    if not sell_trades.empty and "profit" in sell_trades.columns:
        win_rate = float((sell_trades["profit"] > 0).mean())
    if not sell_trades.empty and "profit_pct" in sell_trades.columns:
        avg_profit_pct = float(sell_trades["profit_pct"].mean())

    final_positions = results["positions"].iloc[-1] if "positions" in results.columns else {}
    return {
        "final_portfolio_value": float(results["portfolio_value"].iloc[-1]),
        "total_return": float(analysis["total_return"]),
        "annual_return": float(analysis["annual_return"]),
        "annual_volatility": float(analysis["annual_volatility"]),
        "sharpe_ratio": float(analysis["sharpe_ratio"]),
        "max_drawdown": float(analysis["max_drawdown"]),
        "calmar_ratio": float(analysis["calmar_ratio"]),
        "win_rate": win_rate,
        "profit_loss_ratio": float(analysis["profit_loss_ratio"]),
        "buy_trade_count": int(len(buy_trades)),
        "sell_trade_count": int(len(sell_trades)),
        "avg_profit_pct": avg_profit_pct,
        "final_position_count": int(len(final_positions)),
    }


def run_period_backtest(
    data_with_signals: Mapping[str, pd.DataFrame],
    commission_rate: float,
    slippage: float,
    config_path: Path,
    start_date: str,
    end_date: str,
    initial_capital: float,
    trade_amount: float,
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    strategy_loader: StrategyLoader,
    signal_horizons: Sequence[int],
) -> Dict[str, float]:
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage=slippage,
        config_path=str(config_path),
    )

    stop_loss_strategy, _, stop_profit_strategy, _ = strategy_loader.build_stop_strategies(
        enable_stop_loss=enable_stop_loss,
        enable_stop_profit=enable_stop_profit,
    )
    engine.set_stop_strategies(
        stop_loss_strategy=stop_loss_strategy,
        stop_profit_strategy=stop_profit_strategy,
    )

    callback = create_strategy_callback(trade_amount)
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        results = engine.run(
            dict(data_with_signals),
            callback,
            start_date=start_date,
            end_date=end_date,
            show_progress=False,
        )
    trades = engine.get_trades()
    metrics = summarize_period(results, trades, initial_capital)
    metrics.update(summarize_signal_events(data_with_signals, trades, signal_horizons))
    return metrics


def build_parameter_grid(mode: str) -> List[Dict[str, object]]:
    common = {
        "mode": mode,
        "price_col": "close",
        "high_col": "high",
        "low_col": "low",
        "volume_col": "volume",
    }

    candidates: List[Dict[str, object]] = []
    if mode == "breakout":
        for window, num_std, signal_delay in itertools.product(
            [14, 20, 26],
            [1.8, 2.0, 2.2],
            [0, 1],
        ):
            for use_trend_filter in [False, True]:
                trend_windows = [60] if not use_trend_filter else [40, 60]
                for trend_window in trend_windows:
                    for use_volume_filter in [False, True]:
                        volume_multipliers = [1.2] if not use_volume_filter else [1.2, 1.5]
                        for volume_multiplier in volume_multipliers:
                            candidates.append(
                                {
                                    **common,
                                    "window": window,
                                    "num_std": num_std,
                                    "signal_delay": signal_delay,
                                    "use_trend_filter": use_trend_filter,
                                    "trend_window": trend_window,
                                    "trend_slope_window": 3,
                                    "use_volume_filter": use_volume_filter,
                                    "volume_window": 20,
                                    "volume_multiplier": volume_multiplier,
                                }
                            )
    elif mode == "mean_reversion":
        for window, num_std, signal_delay in itertools.product(
            [14, 20, 26, 34],
            [1.6, 1.8, 2.0, 2.2],
            [0, 1],
        ):
            candidates.append(
                {
                    **common,
                    "window": window,
                    "num_std": num_std,
                    "signal_delay": signal_delay,
                }
            )
    elif mode == "squeeze":
        for window, num_std, squeeze_quantile, signal_delay in itertools.product(
            [18, 20, 26],
            [1.8, 2.0, 2.2],
            [0.1, 0.2],
            [0, 1],
        ):
            for use_trend_filter, use_supertrend_filter in itertools.product([False, True], [False, True]):
                candidates.append(
                    {
                        **common,
                        "window": window,
                        "num_std": num_std,
                        "squeeze_threshold": None,
                        "squeeze_quantile": squeeze_quantile,
                        "squeeze_lookback": 60,
                        "signal_delay": signal_delay,
                        "use_supertrend_filter": use_supertrend_filter,
                        "use_trend_filter": use_trend_filter,
                        "trend_window": 60,
                        "trend_slope_window": 3,
                        "use_volume_filter": False,
                        "volume_window": 20,
                        "volume_multiplier": 1.2,
                    }
                )
    elif mode == "double":
        for window, num_std, confirm_bars, signal_delay in itertools.product(
            [14, 20, 26],
            [1.8, 2.0, 2.2],
            [2, 3],
            [0, 1],
        ):
            candidates.append(
                {
                    **common,
                    "window": window,
                    "num_std": num_std,
                    "confirm_bars": confirm_bars,
                    "signal_delay": signal_delay,
                    "use_trend_filter": True,
                    "trend_window": 60,
                    "trend_slope_window": 3,
                    "use_volume_filter": False,
                    "volume_window": 20,
                    "volume_multiplier": 1.2,
                }
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    deduped = {}
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        deduped[key] = candidate
    return list(deduped.values())


def score_candidates(
    frame: pd.DataFrame,
    evaluation_mode: str,
    signal_horizons: Sequence[int],
    signal_primary_horizon: int,
    min_validation_trades: int,
    min_signal_events: int,
    max_drawdown_limit: float,
    portfolio_weight: float,
    signal_weight: float,
) -> pd.DataFrame:
    scored = frame.copy()
    horizons = sorted({int(horizon) for horizon in signal_horizons if int(horizon) > 0})
    primary_horizon = int(signal_primary_horizon)
    if primary_horizon not in horizons:
        raise ValueError("signal_primary_horizon must be included in signal_horizons")
    max_horizon = max(horizons)

    scored["train_valid_sharpe_gap"] = (
        scored["train_sharpe_ratio"] - scored["valid_sharpe_ratio"]
    ).abs()
    scored["train_valid_return_gap"] = (
        scored["train_annual_return"] - scored["valid_annual_return"]
    ).abs()
    scored["stability_penalty"] = (
        scored["train_valid_sharpe_gap"] + 0.5 * scored["train_valid_return_gap"]
    )
    scored["portfolio_validation_pass"] = (
        (scored["valid_sell_trade_count"] >= min_validation_trades)
        & (scored["valid_max_drawdown"] >= -abs(max_drawdown_limit))
    )
    scored["portfolio_selection_score"] = (
        scored["valid_sharpe_ratio"]
        + 0.35 * scored["valid_calmar_ratio"]
        + 0.10 * scored["valid_annual_return"]
        - 0.50 * scored["stability_penalty"]
    )
    scored["signal_validation_pass"] = scored["valid_signal_event_count"] >= min_signal_events

    valid_edge_cols = [f"valid_signal_mean_edge_{horizon}" for horizon in horizons]
    train_edge_cols = [f"train_signal_mean_edge_{horizon}" for horizon in horizons]
    valid_hit_cols = [f"valid_signal_hit_rate_{horizon}" for horizon in horizons]

    scored["valid_signal_edge_mean"] = scored[valid_edge_cols].mean(axis=1)
    scored["train_signal_edge_mean"] = scored[train_edge_cols].mean(axis=1)
    scored["signal_stability_penalty"] = (
        scored["train_signal_edge_mean"] - scored["valid_signal_edge_mean"]
    ).abs()
    scored["signal_selection_score"] = (
        scored[f"valid_signal_mean_edge_{primary_horizon}"] * 100.0
        + scored[valid_hit_cols].mean(axis=1)
        + 0.50 * scored[f"valid_signal_mean_mfe_{max_horizon}"] * 100.0
        + 0.25 * scored[f"valid_signal_mean_mae_{max_horizon}"] * 100.0
        - 0.50 * scored["signal_stability_penalty"] * 100.0
    )

    if evaluation_mode == "portfolio":
        scored["validation_pass"] = scored["portfolio_validation_pass"]
        scored["selection_score"] = scored["portfolio_selection_score"]
    elif evaluation_mode == "signal":
        scored["validation_pass"] = scored["signal_validation_pass"]
        scored["selection_score"] = scored["signal_selection_score"]
    elif evaluation_mode == "combined":
        scored["validation_pass"] = (
            scored["portfolio_validation_pass"] & scored["signal_validation_pass"]
        )
        scored["selection_score"] = (
            portfolio_weight * scored["portfolio_selection_score"]
            + signal_weight * scored["signal_selection_score"]
        )
    else:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")

    if evaluation_mode == "signal":
        scored = scored.sort_values(
            by=["validation_pass", "selection_score"],
            ascending=[False, False],
        ).reset_index(drop=True)
    else:
        scored = scored.sort_values(
            by=["validation_pass", "selection_score", "valid_sharpe_ratio", "valid_annual_return"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def flatten_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parameter_columns(mode: str) -> List[str]:
    ordered = [
        "mode",
        "window",
        "num_std",
        "signal_delay",
        "use_trend_filter",
        "trend_window",
        "trend_slope_window",
        "use_volume_filter",
        "volume_window",
        "volume_multiplier",
        "use_supertrend_filter",
        "squeeze_threshold",
        "squeeze_quantile",
        "squeeze_lookback",
        "confirm_bars",
    ]
    if mode == "mean_reversion":
        return ["mode", "window", "num_std", "signal_delay"]
    return ordered


def evaluate_grid(
    source_data: Mapping[str, pd.DataFrame],
    parameter_grid: Sequence[Mapping[str, object]],
    commission_rate: float,
    slippage: float,
    config_path: Path,
    periods: Mapping[str, Mapping[str, str]],
    initial_capital: float,
    trade_amount: float,
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    strategy_loader: StrategyLoader,
    signal_horizons: Sequence[int],
    strategy_config_path: Path,
    jobs: int,
    chunksize: int,
) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []

    if jobs <= 1:
        iterator = tqdm(parameter_grid, desc="Evaluating parameters", unit="combo")
        for params in iterator:
            data_with_signals = prepare_signal_data(source_data, params)
            train_metrics = run_period_backtest(
                data_with_signals=data_with_signals,
                commission_rate=commission_rate,
                slippage=slippage,
                config_path=config_path,
                start_date=periods["train"]["start"],
                end_date=periods["train"]["end"],
                initial_capital=initial_capital,
                trade_amount=trade_amount,
                enable_stop_loss=enable_stop_loss,
                enable_stop_profit=enable_stop_profit,
                strategy_loader=strategy_loader,
                signal_horizons=signal_horizons,
            )
            valid_metrics = run_period_backtest(
                data_with_signals=data_with_signals,
                commission_rate=commission_rate,
                slippage=slippage,
                config_path=config_path,
                start_date=periods["valid"]["start"],
                end_date=periods["valid"]["end"],
                initial_capital=initial_capital,
                trade_amount=trade_amount,
                enable_stop_loss=enable_stop_loss,
                enable_stop_profit=enable_stop_profit,
                strategy_loader=strategy_loader,
                signal_horizons=signal_horizons,
            )

            row: MutableMapping[str, object] = dict(params)
            row.update(flatten_metrics("train", train_metrics))
            row.update(flatten_metrics("valid", valid_metrics))
            rows.append(row)

            iterator.set_postfix(
                {
                    "valid_sharpe": f"{valid_metrics['sharpe_ratio']:.2f}",
                    "valid_return": f"{valid_metrics['annual_return']:.2%}",
                }
            )
        return pd.DataFrame(rows)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_initialize_worker_context,
        initargs=(
            source_data,
            commission_rate,
            slippage,
            str(config_path),
            str(strategy_config_path),
            periods,
            initial_capital,
            trade_amount,
            enable_stop_loss,
            enable_stop_profit,
            signal_horizons,
        ),
    ) as executor:
        mapped_rows = executor.map(
            _evaluate_grid_candidate_task,
            parameter_grid,
            chunksize=max(int(chunksize), 1),
        )
        for row in tqdm(mapped_rows, total=len(parameter_grid), desc="Evaluating parameters", unit="combo"):
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_top_candidates_on_test(
    source_data: Mapping[str, pd.DataFrame],
    ranked_candidates: pd.DataFrame,
    commission_rate: float,
    slippage: float,
    config_path: Path,
    periods: Mapping[str, Mapping[str, str]],
    initial_capital: float,
    trade_amount: float,
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    strategy_loader: StrategyLoader,
    top_k: int,
    signal_horizons: Sequence[int],
    strategy_config_path: Path,
    jobs: int,
    chunksize: int,
) -> pd.DataFrame:
    top_rows: List[MutableMapping[str, object]] = []

    top_candidates = ranked_candidates.head(top_k).to_dict(orient="records")
    if jobs <= 1:
        for candidate in top_candidates:
            top_rows.append(_evaluate_test_candidate_task_with_local_context(
                candidate_row=candidate,
                source_data=source_data,
                commission_rate=commission_rate,
                slippage=slippage,
                config_path=config_path,
                strategy_config_path=strategy_config_path,
                periods=periods,
                initial_capital=initial_capital,
                trade_amount=trade_amount,
                enable_stop_loss=enable_stop_loss,
                enable_stop_profit=enable_stop_profit,
                signal_horizons=signal_horizons,
            ))
        return pd.DataFrame(top_rows)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_initialize_worker_context,
        initargs=(
            source_data,
            commission_rate,
            slippage,
            str(config_path),
            str(strategy_config_path),
            periods,
            initial_capital,
            trade_amount,
            enable_stop_loss,
            enable_stop_profit,
            signal_horizons,
        ),
    ) as executor:
        mapped_rows = executor.map(
            _evaluate_test_candidate_task,
            top_candidates,
            chunksize=max(int(chunksize), 1),
        )
        for row in tqdm(mapped_rows, total=len(top_candidates), desc="Evaluating test candidates", unit="combo"):
            top_rows.append(row)

    return pd.DataFrame(top_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bollinger parameter study example.")
    parser.add_argument(
        "--evaluation-mode",
        choices=["portfolio", "signal", "combined"],
        default="combined",
        help="How to rank parameter candidates.",
    )
    parser.add_argument(
        "--mode",
        choices=["breakout", "mean_reversion", "squeeze", "double"],
        default="breakout",
        help="Bollinger mode to study.",
    )
    parser.add_argument(
        "--stock-file",
        default=None,
        help="Override the stock universe file. Defaults to config/settings.yaml.",
    )
    parser.add_argument(
        "--stock-max-number",
        type=int,
        default=20,
        help="Maximum number of symbols to load. Use -1 for the full universe.",
    )
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked candidates to re-run on the test window.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 to keep single-process execution.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Task chunksize passed to the process pool map call.",
    )
    parser.add_argument(
        "--min-validation-trades",
        type=int,
        default=5,
        help="Validation sell-trade threshold used by the ranking filter.",
    )
    parser.add_argument(
        "--min-signal-events",
        type=int,
        default=5,
        help="Minimum trade events required by signal-level validation.",
    )
    parser.add_argument(
        "--max-drawdown-limit",
        type=float,
        default=0.30,
        help="Absolute maximum drawdown tolerated by the ranking filter.",
    )
    parser.add_argument(
        "--signal-horizons",
        default="1,3,5,10,20",
        help="Comma-separated future horizons used by signal-level evaluation.",
    )
    parser.add_argument(
        "--signal-primary-horizon",
        type=int,
        default=10,
        help="Primary horizon used by signal ranking. Must be included in --signal-horizons.",
    )
    parser.add_argument(
        "--portfolio-weight",
        type=float,
        default=0.5,
        help="Weight of portfolio score when --evaluation-mode combined is used.",
    )
    parser.add_argument(
        "--signal-weight",
        type=float,
        default=0.5,
        help="Weight of signal score when --evaluation-mode combined is used.",
    )
    parser.add_argument(
        "--enable-stop-loss",
        action="store_true",
        help="Enable the repo default stop-loss strategy.",
    )
    parser.add_argument(
        "--enable-stop-profit",
        action="store_true",
        help="Enable the repo default stop-profit strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    strategy_config_path = PROJECT_ROOT / "config" / "strategies.yaml"
    settings = load_yaml(settings_path)
    strategy_loader = StrategyLoader(str(strategy_config_path))

    periods = {
        "train": {"start": args.train_start, "end": args.train_end},
        "valid": {"start": args.valid_start, "end": args.valid_end},
        "test": {"start": args.test_start, "end": args.test_end},
    }
    full_start = min(period["start"] for period in periods.values())
    full_end = max(period["end"] for period in periods.values())

    initial_capital = float(settings["backtest"]["initial_capital"])
    trade_amount = float(settings["backtest"].get("trade_amount", 10000))
    commission_rate = float(settings["backtest"]["commission_rate"])
    slippage = float(settings["backtest"]["slippage"])
    signal_horizons = [int(token.strip()) for token in args.signal_horizons.split(",") if token.strip()]
    signal_horizons = sorted({horizon for horizon in signal_horizons if horizon > 0})
    if not signal_horizons:
        raise ValueError("--signal-horizons must contain at least one positive integer")
    if args.signal_primary_horizon not in signal_horizons:
        raise ValueError("--signal-primary-horizon must be one of --signal-horizons")
    if args.portfolio_weight < 0 or args.signal_weight < 0:
        raise ValueError("--portfolio-weight and --signal-weight must be non-negative")
    if args.evaluation_mode == "combined" and (args.portfolio_weight + args.signal_weight) == 0:
        raise ValueError("Combined evaluation requires a positive portfolio or signal weight")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")

    parameter_grid = build_parameter_grid(args.mode)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"Mode = {args.mode}")
    print(f"Evaluation mode = {args.evaluation_mode}")
    print(f"Jobs = {args.jobs}")
    print(f"Chunksize = {args.chunksize}")
    print(f"Parameter combinations = {len(parameter_grid)}")
    print(f"Universe file = {args.stock_file or settings['data']['stock_file']}")
    print(f"Signal horizons = {signal_horizons}")
    print(f"Signal primary horizon = {args.signal_primary_horizon}")
    print(
        "Periods = "
        f"train[{args.train_start} -> {args.train_end}], "
        f"valid[{args.valid_start} -> {args.valid_end}], "
        f"test[{args.test_start} -> {args.test_end}]"
    )

    source_data = load_universe_data(
        settings=settings,
        stock_file=args.stock_file,
        stock_max_number=args.stock_max_number,
        start_date=full_start,
        end_date=full_end,
    )
    print(f"Loaded symbols = {len(source_data)}")

    evaluated = evaluate_grid(
        source_data=source_data,
        parameter_grid=parameter_grid,
        commission_rate=commission_rate,
        slippage=slippage,
        config_path=settings_path,
        periods=periods,
        initial_capital=initial_capital,
        trade_amount=trade_amount,
        enable_stop_loss=args.enable_stop_loss,
        enable_stop_profit=args.enable_stop_profit,
        strategy_loader=strategy_loader,
        signal_horizons=signal_horizons,
        strategy_config_path=strategy_config_path,
        jobs=args.jobs,
        chunksize=args.chunksize,
    )
    ranked = score_candidates(
        evaluated,
        evaluation_mode=args.evaluation_mode,
        signal_horizons=signal_horizons,
        signal_primary_horizon=args.signal_primary_horizon,
        min_validation_trades=args.min_validation_trades,
        min_signal_events=args.min_signal_events,
        max_drawdown_limit=args.max_drawdown_limit,
        portfolio_weight=args.portfolio_weight,
        signal_weight=args.signal_weight,
    )
    tested = evaluate_top_candidates_on_test(
        source_data=source_data,
        ranked_candidates=ranked,
        commission_rate=commission_rate,
        slippage=slippage,
        config_path=settings_path,
        periods=periods,
        initial_capital=initial_capital,
        trade_amount=trade_amount,
        enable_stop_loss=args.enable_stop_loss,
        enable_stop_profit=args.enable_stop_profit,
        strategy_loader=strategy_loader,
        top_k=args.top_k,
        signal_horizons=signal_horizons,
        strategy_config_path=strategy_config_path,
        jobs=args.jobs,
        chunksize=args.chunksize,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"bollinger_study_{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(output_dir / "grid_train_valid.csv", index=False, encoding="utf-8-sig")
    tested.to_csv(output_dir / "top_candidates_with_test.csv", index=False, encoding="utf-8-sig")

    heatmap = (
        ranked.groupby(["window", "num_std"], dropna=False)["valid_sharpe_ratio"]
        .mean()
        .unstack()
        .sort_index()
    )
    heatmap.to_csv(output_dir / "validation_sharpe_heatmap.csv", encoding="utf-8-sig")

    best_row = tested.head(1)
    if not best_row.empty:
        best_payload = best_row.iloc[0].to_dict()
        with open(output_dir / "best_candidate.json", "w", encoding="utf-8") as file_obj:
            json.dump(best_payload, file_obj, indent=2, ensure_ascii=False, default=str)

    show_columns = [
        column
        for column in (
            parameter_columns(args.mode)
            + [
                "portfolio_validation_pass",
                "signal_validation_pass",
                "validation_pass",
                "portfolio_selection_score",
                "signal_selection_score",
                "selection_score",
                "valid_sharpe_ratio",
                "valid_annual_return",
                "valid_max_drawdown",
                "valid_sell_trade_count",
                f"valid_signal_mean_edge_{args.signal_primary_horizon}",
                f"valid_signal_hit_rate_{args.signal_primary_horizon}",
                "valid_signal_event_count",
                "test_sharpe_ratio",
                "test_annual_return",
                "test_max_drawdown",
                "test_sell_trade_count",
                f"test_signal_mean_edge_{args.signal_primary_horizon}",
                f"test_signal_hit_rate_{args.signal_primary_horizon}",
                "test_signal_event_count",
            ]
        )
        if column in tested.columns
    ]
    print("\nTop candidates:")
    if tested.empty:
        print("No candidates were evaluated on the test window.")
    else:
        print(tested[show_columns].head(min(args.top_k, 10)).to_string(index=False))

    print("\nSaved files:")
    print(f"  {output_dir / 'grid_train_valid.csv'}")
    print(f"  {output_dir / 'top_candidates_with_test.csv'}")
    print(f"  {output_dir / 'validation_sharpe_heatmap.csv'}")
    if (output_dir / "best_candidate.json").exists():
        print(f"  {output_dir / 'best_candidate.json'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
