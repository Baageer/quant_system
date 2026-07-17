"""Interactive Streamlit backtest runner."""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from backtest.backtest_service import (
    BacktestRequest,
    LoadedMarketData,
    export_backtest_result,
    generate_signals,
    list_stop_strategy_configs,
    list_timing_strategy_configs,
    load_market_data,
    load_yaml_config,
    run_engine_with_signals,
)


CONFIG_PATH = "./config/settings.yaml"
STRATEGY_CONFIG_PATH = "./config/strategies.yaml"
ADJUST_MODE_OPTIONS = {
    "": "原始价格 / 不直接复权",
    "qfq": "前复权 qfq",
    "hfq": "后复权 hfq",
}
RAW_PRICE_ADJUST_OPTIONS = {
    "": "不按涨跌幅重建",
    "qfq": "按涨跌幅重建前复权",
    "hfq": "按涨跌幅重建后复权",
}
STOP_TYPE_OPTIONS = {
    "absolute": "固定比例",
    "trailing": "移动跟踪",
    "atr": "ATR",
    "holding_day": "持仓天数",
}


def get_file_mtime(path: str) -> float:
    """Return file mtime for cache invalidation."""
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def option_index(options: Dict[str, str], value: Any) -> int:
    """Return a safe selectbox index for an option value."""
    keys = list(options.keys())
    normalized = "" if value is None else str(value).strip().lower()
    if normalized in {"none", "raw", "bfq", "unadjusted"}:
        normalized = ""
    return keys.index(normalized) if normalized in options else 0


def save_yaml_config(config_path: str, config: Dict[str, Any]) -> None:
    """Atomically save a YAML configuration file."""
    absolute_config_path = os.path.abspath(config_path)
    directory = os.path.dirname(absolute_config_path)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            delete=False,
        ) as file:
            yaml.safe_dump(
                config,
                file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
            temp_path = file.name
        os.replace(temp_path, absolute_config_path)
        temp_path = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def save_page_configuration(
    settings: Dict[str, Any],
    universe_type: str,
    start_date: Any,
    end_date: Any,
    stock_file: str,
    industry_index_list_file: Optional[str],
    industry_index_data_dir: Optional[str],
    industry_index_codes: Optional[List[str]],
    stock_max_number: int,
    max_workers: int,
    signal_workers: int,
    initial_capital: float,
    trade_amount: float,
    commission_rate: float,
    slippage: float,
    adjust_mode: str,
    raw_price_adjust: str,
    selected_strategies: List[str],
    strategy_params: Dict[str, Dict[str, Any]],
    signal_combination: str,
    signal_threshold: float,
    signal_weights: List[float],
    enable_stop_loss: bool,
    enable_stop_profit: bool,
    stop_loss_params: Dict[str, Any],
    stop_profit_params: Dict[str, Any],
) -> None:
    """Persist editable page values to the project YAML configuration files."""
    data_config = settings.setdefault("data", {})
    data_config.update(
        {
            "stock_file": stock_file,
            "adjust_mode": adjust_mode,
            "raw_price_adjust": raw_price_adjust,
        }
    )
    if industry_index_list_file is not None:
        data_config["industry_index_list_file"] = industry_index_list_file
    if industry_index_data_dir is not None:
        data_config["industry_index_data_dir"] = industry_index_data_dir

    settings.setdefault("backtest", {}).update(
        {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": float(initial_capital),
            "trade_amount": float(trade_amount),
            "commission_rate": float(commission_rate),
            "slippage": float(slippage),
        }
    )
    settings["live_backtest"] = {
        "universe_type": universe_type,
        "industry_index_codes": industry_index_codes or [],
        "stock_max_number": int(stock_max_number),
        "max_workers": int(max_workers),
        "signal_workers": int(signal_workers),
        "selected_strategies": selected_strategies,
        "signal_combination": signal_combination,
        "signal_threshold": float(signal_threshold),
        "signal_weights": dict(zip(selected_strategies, signal_weights)),
        "enable_stop_loss": bool(enable_stop_loss),
        "enable_stop_profit": bool(enable_stop_profit),
    }

    strategy_document = load_yaml_config(STRATEGY_CONFIG_PATH)
    timing_configs = strategy_document.setdefault("timing_strategies", {})
    for strategy_name, params in strategy_params.items():
        if strategy_name in timing_configs:
            timing_configs[strategy_name]["params"] = params

    stop_configs = strategy_document.setdefault("stop_strategies", {})
    if "stop_loss" in stop_configs:
        stop_configs["stop_loss"]["params"] = stop_loss_params
    if "stop_profit" in stop_configs:
        stop_configs["stop_profit"]["params"] = stop_profit_params

    save_yaml_config(CONFIG_PATH, settings)
    save_yaml_config(STRATEGY_CONFIG_PATH, strategy_document)


def load_market_data_for_cache(
    request_payload: Dict[str, Any],
    config_mtime: float,
    strategy_config_mtime: float,
    _progress_callback=None,
) -> LoadedMarketData:
    """Load market data for the Streamlit cache wrapper."""
    _ = config_mtime, strategy_config_mtime
    return load_market_data(
        BacktestRequest(**request_payload),
        progress_callback=_progress_callback,
    )


def parse_code_list(value: str) -> List[str]:
    """Parse comma-separated code input."""
    return [item.strip() for item in value.split(",") if item.strip()]


def coerce_text_value(text: str, default_value: Any) -> Any:
    """Convert text input back to a YAML-like scalar."""
    stripped = text.strip()
    if stripped == "":
        return None
    if default_value is None:
        return stripped
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(stripped)
    if isinstance(default_value, float):
        return float(stripped)
    return stripped


def render_param_input(scope: str, name: str, value: Any) -> Any:
    """Render an input widget based on default value type."""
    key = f"{scope}_{name}"
    if name in {"loss_type", "profit_type"}:
        normalized = str(value or "absolute").strip().lower()
        if normalized == "holding_days":
            normalized = "holding_day"
        options = list(STOP_TYPE_OPTIONS.keys())
        index = options.index(normalized) if normalized in STOP_TYPE_OPTIONS else 0
        return st.selectbox(
            name,
            options=options,
            index=index,
            format_func=lambda option: f"{STOP_TYPE_OPTIONS[option]} ({option})",
            key=key,
        )
    if isinstance(value, bool):
        return st.checkbox(name, value=value, key=key)
    if isinstance(value, int) and not isinstance(value, bool):
        return st.number_input(name, value=int(value), step=1, key=key)
    if isinstance(value, float):
        return st.number_input(name, value=float(value), step=0.01, format="%.6f", key=key)
    if value is None:
        text_value = st.text_input(name, value="", key=key, placeholder="空值")
        return coerce_text_value(text_value, value)
    text_value = st.text_input(name, value=str(value), key=key)
    return coerce_text_value(text_value, value)


def make_request_payload(request: BacktestRequest) -> Dict[str, Any]:
    """Convert request to a cache-friendly payload."""
    return asdict(request)


def make_market_payload(request: BacktestRequest) -> Dict[str, Any]:
    """Keep only fields that affect price data loading."""
    payload = make_request_payload(request)
    payload["strategy_params"] = {}
    payload["stop_loss_params"] = {}
    payload["stop_profit_params"] = {}
    payload["signal_combination"] = "weighted"
    payload["signal_weights"] = None
    payload["signal_threshold"] = 0.5
    payload["initial_capital"] = None
    payload["trade_amount"] = None
    payload["commission_rate"] = None
    payload["slippage"] = None
    payload["enable_stop_loss"] = False
    payload["enable_stop_profit"] = False
    payload["signal_workers"] = 1
    return payload


def make_signal_payload(request: BacktestRequest, market_key: str) -> Dict[str, Any]:
    """Keep fields that affect generated signal columns."""
    payload = make_request_payload(request)
    keep_keys = {
        "strategy_names",
        "strategy_params",
        "signal_combination",
        "signal_weights",
        "signal_threshold",
    }
    return {"market_key": market_key, **{key: payload[key] for key in keep_keys}}


def stable_key(payload: Dict[str, Any]) -> str:
    """Build a stable JSON key for session invalidation."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def set_progress(progress_bar, status_text, value: int, text: str) -> None:
    """Update a Streamlit progress bar with status text."""
    progress_bar.progress(value)
    status_text.caption(text)


def make_progress_callback(progress_bar, status_text, prefix: str, start: int = 0, end: int = 100):
    """Create a progress callback that maps completed/total to a range."""

    def update(completed: int, total: int, message: str) -> None:
        if total <= 0:
            percent = end
            detail = message
        else:
            ratio = max(0.0, min(1.0, completed / total))
            percent = int(start + (end - start) * ratio)
            detail = f"{completed}/{total} | {message}"
        set_progress(progress_bar, status_text, percent, f"{prefix}：{detail}")

    return update


def draw_equity_chart(results: pd.DataFrame) -> None:
    """Draw portfolio value and drawdown chart."""
    if results.empty:
        st.info("暂无回测结果。")
        return

    chart_df = results.copy()
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    portfolio_value = chart_df["portfolio_value"]
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("组合净值", "回撤"),
    )
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=portfolio_value, name="组合净值"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=drawdown * 100, name="回撤(%)", fill="tozeroy"),
        row=2,
        col=1,
    )
    fig.update_layout(height=620, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def draw_symbol_chart(symbol: str, df: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Draw one symbol's price, volume, signals, and trades."""
    if df.empty:
        st.info("该标的暂无数据。")
        return

    plot_df = df.copy()
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.to_datetime(plot_df.index)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.2],
        subplot_titles=(f"{symbol} K线与交易点", "成交量", "信号"),
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["volume"], name="成交量"),
        row=2,
        col=1,
    )
    if "signal" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["signal"], name="信号", mode="lines"),
            row=3,
            col=1,
        )

    if trades is not None and not trades.empty and "symbol" in trades.columns:
        symbol_trades = trades[trades["symbol"] == symbol].copy()
        if not symbol_trades.empty and "date" in symbol_trades.columns:
            symbol_trades["date"] = pd.to_datetime(symbol_trades["date"])
            buy_trades = symbol_trades[symbol_trades["action"] == "buy"]
            sell_trades = symbol_trades[symbol_trades["action"] == "sell"]
            fig.add_trace(
                go.Scatter(
                    x=buy_trades["date"],
                    y=buy_trades["price"],
                    mode="markers",
                    name="买入",
                    marker=dict(color="red", size=10, symbol="triangle-up"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_trades["date"],
                    y=sell_trades["price"],
                    mode="markers",
                    name="卖出",
                    marker=dict(color="green", size=10, symbol="triangle-down"),
                ),
                row=1,
                col=1,
            )

    fig.update_layout(height=760, hovermode="x unified", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Render the Streamlit application in its script-runner process."""
    st.set_page_config(page_title="BACKTEST LIVE", layout="wide")
    st.title("BACKTEST LIVE")

    @st.cache_data(show_spinner=False)
    def cached_load_market_data(
        request_payload: Dict[str, Any],
        config_mtime: float,
        strategy_config_mtime: float,
        _progress_callback=None,
    ) -> LoadedMarketData:
        """Load market data with Streamlit cache."""
        return load_market_data_for_cache(
            request_payload,
            config_mtime,
            strategy_config_mtime,
            _progress_callback,
        )


    settings = load_yaml_config(CONFIG_PATH)
    strategy_configs = list_timing_strategy_configs(STRATEGY_CONFIG_PATH)
    stop_strategy_configs = list_stop_strategy_configs(STRATEGY_CONFIG_PATH)
    backtest_defaults = settings.get("backtest", {})
    data_defaults = settings.get("data", {})
    live_defaults = settings.get("live_backtest") or {}
    if not isinstance(live_defaults, dict):
        live_defaults = {}

    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = None
    if "signaled_data" not in st.session_state:
        st.session_state.signaled_data = None
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "loaded_data_key" not in st.session_state:
        st.session_state.loaded_data_key = None
    if "signaled_data_key" not in st.session_state:
        st.session_state.signaled_data_key = None

    with st.sidebar:
        st.header("运行参数")

        universe_options = ["stock", "industry"]
        saved_universe_type = str(live_defaults.get("universe_type", "stock"))
        universe_type = st.selectbox(
            "标的类型",
            universe_options,
            index=universe_options.index(saved_universe_type) if saved_universe_type in universe_options else 0,
            format_func=lambda x: "股票" if x == "stock" else "行业指数",
        )
        start_date = st.date_input("开始日期", pd.to_datetime(backtest_defaults.get("start_date", "2022-01-01")))
        end_date = st.date_input("结束日期", pd.to_datetime(backtest_defaults.get("end_date", "2023-12-31")))

        source = data_defaults.get("source", "akshare")
        adjust_mode = st.selectbox(
            "复权模式 adjust_mode",
            options=list(ADJUST_MODE_OPTIONS.keys()),
            index=option_index(ADJUST_MODE_OPTIONS, data_defaults.get("adjust_mode", "")),
            format_func=lambda value: ADJUST_MODE_OPTIONS[value],
            help="直接向数据源请求复权价格。选择原始价格时，才可以使用 raw_price_adjust。",
        )
        raw_price_adjust_default = data_defaults.get("raw_price_adjust", "") if adjust_mode == "" else ""
        raw_price_adjust = st.selectbox(
            "raw_price_adjust",
            options=list(RAW_PRICE_ADJUST_OPTIONS.keys()),
            index=option_index(RAW_PRICE_ADJUST_OPTIONS, raw_price_adjust_default),
            format_func=lambda value: RAW_PRICE_ADJUST_OPTIONS[value],
            disabled=adjust_mode != "",
            help="仅在 adjust_mode 为原始价格时，用涨跌幅重建前/后复权价格。",
        )
        if adjust_mode != "":
            raw_price_adjust = ""
            st.caption("已选择直接复权模式，raw_price_adjust 自动置空。")

        if universe_type == "stock":
            stock_file = st.text_input("股票池文件", value=str(data_defaults.get("stock_file", "./data/test1.txt")))
            industry_index_list_file = None
            industry_index_data_dir = None
            industry_index_codes = None
        else:
            stock_file = str(data_defaults.get("stock_file", "./data/test1.txt"))
            industry_index_list_file = st.text_input(
                "行业指数列表",
                value=str(data_defaults.get("industry_index_list_file", "./data/raw/akshare/industry_sector_index_list.csv")),
            )
            industry_index_data_dir = st.text_input(
                "行业指数数据目录",
                value=str(data_defaults.get("industry_index_data_dir", "./data/raw/akshare/industry_sector_index")),
            )
            saved_industry_codes = live_defaults.get("industry_index_codes", [])
            if not isinstance(saved_industry_codes, list):
                saved_industry_codes = []
            industry_codes_text = st.text_input(
                "行业指数代码过滤",
                value=",".join(str(code) for code in saved_industry_codes),
                placeholder="例如 881121,881273",
            )
            industry_index_codes = parse_code_list(industry_codes_text) or None

        stock_max_number = st.number_input(
            "最大标的数（-1 为全部）",
            value=int(live_defaults.get("stock_max_number", -1)),
            step=1,
        )
        max_workers = st.number_input(
            "加载进程数",
            min_value=1,
            max_value=16,
            value=max(1, min(16, int(live_defaults.get("max_workers", 1)))),
            step=1,
        )
        signal_workers = st.number_input(
            "信号生成进程数",
            min_value=1,
            max_value=16,
            value=max(1, min(16, int(live_defaults.get("signal_workers", 1)))),
            step=1,
            help="大于 1 时按标的使用多进程生成信号。",
        )

        st.subheader("资金与交易")
        initial_capital = st.number_input(
            "初始资金",
            value=float(backtest_defaults.get("initial_capital", 1000000)),
            step=10000.0,
        )
        trade_amount = st.number_input(
            "单笔交易金额",
            value=float(backtest_defaults.get("trade_amount", 100000)),
            step=1000.0,
        )
        commission_rate = st.number_input(
            "佣金率",
            value=float(backtest_defaults.get("commission_rate", 0.0003)),
            step=0.0001,
            format="%.6f",
        )
        slippage = st.number_input(
            "滑点",
            value=float(backtest_defaults.get("slippage", 0.0001)),
            step=0.0001,
            format="%.6f",
        )

        st.caption(f"当前数据源：{source}")

    tab_strategy, tab_run, tab_summary, tab_trades, tab_symbol = st.tabs(
        ["策略配置", "运行控制", "绩效总览", "交易明细", "单标的分析"]
    )

    with tab_strategy:
        st.subheader("择时策略")
        strategy_options = list(strategy_configs.keys())
        saved_selected_strategies = live_defaults.get("selected_strategies", [])
        if not isinstance(saved_selected_strategies, list):
            saved_selected_strategies = []
        selected_strategy_defaults = [
            strategy_name
            for strategy_name in saved_selected_strategies
            if strategy_name in strategy_options
        ]
        if not selected_strategy_defaults and strategy_options:
            selected_strategy_defaults = [strategy_options[0]]
        selected_strategies = st.multiselect(
            "选择策略",
            options=strategy_options,
            default=selected_strategy_defaults,
            format_func=lambda key: f"{key} - {strategy_configs[key].get('name', key)}",
        )

        strategy_params: Dict[str, Dict[str, Any]] = {}
        if selected_strategies:
            for strategy_key in selected_strategies:
                info = strategy_configs[strategy_key]
                with st.expander(f"{strategy_key} 参数", expanded=len(selected_strategies) == 1):
                    st.caption(info.get("description", ""))
                    params = {}
                    for param_name, default_value in info.get("params", {}).items():
                        params[param_name] = render_param_input(f"strategy_{strategy_key}", param_name, default_value)
                    strategy_params[strategy_key] = params

        st.subheader("组合信号")
        signal_combination_options = ["weighted", "voting", "unanimous"]
        saved_signal_combination = str(live_defaults.get("signal_combination", "weighted"))
        signal_combination = st.selectbox(
            "组合方式",
            signal_combination_options,
            index=(
                signal_combination_options.index(saved_signal_combination)
                if saved_signal_combination in signal_combination_options
                else 0
            ),
        )
        signal_threshold = st.number_input(
            "信号阈值",
            value=float(live_defaults.get("signal_threshold", 0.5)),
            min_value=0.0,
            max_value=5.0,
            step=0.05,
        )
        signal_weights = []
        if selected_strategies:
            default_weight = 1.0 / len(selected_strategies)
            saved_signal_weights = live_defaults.get("signal_weights", {})
            cols = st.columns(len(selected_strategies))
            for index, strategy_key in enumerate(selected_strategies):
                with cols[index]:
                    saved_weight = (
                        saved_signal_weights.get(strategy_key, default_weight)
                        if isinstance(saved_signal_weights, dict)
                        else default_weight
                    )
                    signal_weights.append(
                        st.number_input(
                            f"{strategy_key} 权重",
                            value=float(saved_weight),
                            step=0.05,
                            format="%.4f",
                            key=f"weight_{strategy_key}",
                        )
                    )

        st.subheader("止损止盈")
        enable_stop_loss = st.checkbox("启用止损", value=bool(live_defaults.get("enable_stop_loss", True)))
        enable_stop_profit = st.checkbox("启用止盈", value=bool(live_defaults.get("enable_stop_profit", True)))
        stop_loss_params: Dict[str, Any] = {}
        stop_profit_params: Dict[str, Any] = {}

        stop_loss_config = stop_strategy_configs.get("stop_loss", {})
        stop_profit_config = stop_strategy_configs.get("stop_profit", {})
        col_loss, col_profit = st.columns(2)
        with col_loss:
            with st.expander("止损参数", expanded=False):
                for param_name, default_value in stop_loss_config.get("params", {}).items():
                    stop_loss_params[param_name] = render_param_input("stop_loss", param_name, default_value)
        with col_profit:
            with st.expander("止盈参数", expanded=False):
                for param_name, default_value in stop_profit_config.get("params", {}).items():
                    stop_profit_params[param_name] = render_param_input("stop_profit", param_name, default_value)

    save_clicked = st.sidebar.button("保存页面配置", use_container_width=True)
    if save_clicked:
        try:
            save_page_configuration(
                settings=settings,
                universe_type=universe_type,
                start_date=start_date,
                end_date=end_date,
                stock_file=stock_file,
                industry_index_list_file=industry_index_list_file,
                industry_index_data_dir=industry_index_data_dir,
                industry_index_codes=industry_index_codes,
                stock_max_number=int(stock_max_number),
                max_workers=int(max_workers),
                signal_workers=int(signal_workers),
                initial_capital=float(initial_capital),
                trade_amount=float(trade_amount),
                commission_rate=float(commission_rate),
                slippage=float(slippage),
                adjust_mode=adjust_mode,
                raw_price_adjust=raw_price_adjust,
                selected_strategies=selected_strategies,
                strategy_params=strategy_params,
                signal_combination=signal_combination,
                signal_threshold=float(signal_threshold),
                signal_weights=[float(weight) for weight in signal_weights],
                enable_stop_loss=enable_stop_loss,
                enable_stop_profit=enable_stop_profit,
                stop_loss_params=stop_loss_params,
                stop_profit_params=stop_profit_params,
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            st.sidebar.error(f"配置保存失败：{error}")
        else:
            st.sidebar.success("配置已保存到 settings.yaml 和 strategies.yaml。")

    request = BacktestRequest(
        strategy_names=selected_strategies,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        config_path=CONFIG_PATH,
        strategy_config_path=STRATEGY_CONFIG_PATH,
        stock_file=stock_file,
        stock_max_number=int(stock_max_number),
        initial_capital=float(initial_capital),
        trade_amount=float(trade_amount),
        commission_rate=float(commission_rate),
        slippage=float(slippage),
        adjust_mode=adjust_mode,
        raw_price_adjust=raw_price_adjust,
        enable_stop_loss=enable_stop_loss,
        enable_stop_profit=enable_stop_profit,
        stop_loss_params=stop_loss_params,
        stop_profit_params=stop_profit_params,
        signal_combination=signal_combination,
        signal_weights=signal_weights if signal_weights else None,
        signal_threshold=float(signal_threshold),
        strategy_params=strategy_params,
        universe_type=universe_type,
        industry_index_list_file=industry_index_list_file,
        industry_index_data_dir=industry_index_data_dir,
        industry_index_codes=industry_index_codes,
        max_workers=int(max_workers),
        signal_workers=int(signal_workers),
        show_progress=False,
    )
    market_payload = make_market_payload(request)
    market_key = stable_key(market_payload)
    signal_key = stable_key(make_signal_payload(request, market_key))

    with tab_run:
        st.subheader("分步执行")
        if not selected_strategies:
            st.warning("请至少选择一个策略。")

        col_load, col_signal, col_run, col_clear = st.columns(4)
        with col_load:
            load_clicked = st.button("加载数据", type="primary", disabled=not selected_strategies)
        with col_signal:
            signal_clicked = st.button("生成信号", disabled=not selected_strategies)
        with col_run:
            run_clicked = st.button("运行回测", disabled=not selected_strategies)
        with col_clear:
            clear_clicked = st.button("清空页面缓存")

        if clear_clicked:
            st.session_state.loaded_data = None
            st.session_state.signaled_data = None
            st.session_state.backtest_result = None
            st.session_state.loaded_data_key = None
            st.session_state.signaled_data_key = None
            cached_load_market_data.clear()
            st.success("页面缓存已清空。")

        if load_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()
            load_progress = make_progress_callback(progress_bar, status_text, "加载行情")
            set_progress(progress_bar, status_text, 0, "加载行情：准备开始...")
            st.session_state.loaded_data = cached_load_market_data(
                market_payload,
                get_file_mtime(CONFIG_PATH),
                get_file_mtime(STRATEGY_CONFIG_PATH),
                _progress_callback=load_progress,
            )
            st.session_state.loaded_data_key = market_key
            st.session_state.signaled_data = None
            st.session_state.signaled_data_key = None
            st.session_state.backtest_result = None
            set_progress(progress_bar, status_text, 100, "加载行情：完成。")
            st.success(f"加载完成：{len(st.session_state.loaded_data.data)} / {len(st.session_state.loaded_data.symbols)} 个标的有效。")

        if signal_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()
            needs_load = st.session_state.loaded_data is None or st.session_state.loaded_data_key != market_key
            if needs_load:
                load_progress = make_progress_callback(progress_bar, status_text, "加载行情", 0, 40)
                st.session_state.loaded_data = cached_load_market_data(
                    market_payload,
                    get_file_mtime(CONFIG_PATH),
                    get_file_mtime(STRATEGY_CONFIG_PATH),
                    _progress_callback=load_progress,
                )
                st.session_state.loaded_data_key = market_key
            signal_progress = make_progress_callback(
                progress_bar,
                status_text,
                "生成信号",
                40 if needs_load else 0,
                100,
            )
            st.session_state.signaled_data = generate_signals(
                st.session_state.loaded_data,
                request,
                progress_callback=signal_progress,
            )
            st.session_state.signaled_data_key = signal_key
            st.session_state.backtest_result = None
            set_progress(progress_bar, status_text, 100, "生成信号：完成。")
            st.success("信号生成完成。")

        if run_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()
            needs_load = st.session_state.loaded_data is None or st.session_state.loaded_data_key != market_key
            needs_signal = st.session_state.signaled_data is None or st.session_state.signaled_data_key != signal_key
            if needs_load:
                load_progress = make_progress_callback(progress_bar, status_text, "加载行情", 0, 30)
                st.session_state.loaded_data = cached_load_market_data(
                    market_payload,
                    get_file_mtime(CONFIG_PATH),
                    get_file_mtime(STRATEGY_CONFIG_PATH),
                    _progress_callback=load_progress,
                )
                st.session_state.loaded_data_key = market_key
                st.session_state.signaled_data = None
                st.session_state.signaled_data_key = None
                needs_signal = True
            if needs_signal:
                signal_start = 30 if needs_load else 0
                signal_end = 60 if needs_load else 40
                signal_progress = make_progress_callback(
                    progress_bar,
                    status_text,
                    "生成信号",
                    signal_start,
                    signal_end,
                )
                st.session_state.signaled_data = generate_signals(
                    st.session_state.loaded_data,
                    request,
                    progress_callback=signal_progress,
                )
                st.session_state.signaled_data_key = signal_key
            engine_start = 60 if needs_load else (40 if needs_signal else 0)
            engine_progress = make_progress_callback(
                progress_bar,
                status_text,
                "运行回测",
                engine_start,
                100,
            )
            st.session_state.backtest_result = run_engine_with_signals(
                st.session_state.signaled_data,
                request,
                progress_callback=engine_progress,
            )
            set_progress(progress_bar, status_text, 100, "回测完成。")
            st.success("回测完成。")

        loaded_data = st.session_state.loaded_data
        if loaded_data is not None:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("股票/指数总数", len(loaded_data.symbols))
            col_b.metric("有效数据", len(loaded_data.data))
            col_c.metric("跳过数量", len(loaded_data.skipped_symbols))
            if loaded_data.warnings:
                with st.expander("加载警告", expanded=False):
                    st.write(pd.DataFrame({"warning": loaded_data.warnings}))

        result = st.session_state.backtest_result
        if result is not None:
            if st.button("导出当前结果"):
                results_file, trades_file = export_backtest_result(result)
                st.success(f"已导出：{results_file}，{trades_file}")

    with tab_summary:
        result = st.session_state.backtest_result
        if result is None:
            st.info("请先在“运行控制”中执行回测。")
        else:
            metrics = result.metrics
            cols = st.columns(6)
            cols[0].metric("最终资产", f"{metrics.get('final_value', 0):,.2f}")
            cols[1].metric("总收益率", f"{metrics.get('total_return_pct', 0):.2f}%")
            cols[2].metric("最大回撤", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
            cols[3].metric("夏普", f"{metrics.get('sharpe', 0):.2f}")
            cols[4].metric("胜率", f"{metrics.get('win_rate_pct', 0):.2f}%")
            cols[5].metric("买入次数", metrics.get("buy_count", 0))

            draw_equity_chart(result.results)

            col_year, col_pos = st.columns(2)
            with col_year:
                st.subheader("年度收益")
                if result.annual_returns.empty:
                    st.info("年度收益暂无数据。")
                else:
                    st.dataframe(result.annual_returns.rename("annual_return_pct").to_frame(), width="stretch")
            with col_pos:
                st.subheader("持仓统计")
                st.write(
                    {
                        "最大同时持仓": metrics.get("max_position", 0),
                        "当前持仓": metrics.get("current_positions", 0),
                        "卖出次数": metrics.get("sell_count", 0),
                    }
                )

    with tab_trades:
        result = st.session_state.backtest_result
        if result is None:
            st.info("请先运行回测。")
        elif result.trades.empty:
            st.info("本次回测没有交易记录。")
        else:
            trades_df = result.trades.copy()
            if "date" in trades_df.columns:
                trades_df["date"] = pd.to_datetime(trades_df["date"])
            symbols = ["全部"] + sorted(trades_df["symbol"].dropna().unique().tolist()) if "symbol" in trades_df.columns else ["全部"]
            actions = ["全部"] + sorted(trades_df["action"].dropna().unique().tolist()) if "action" in trades_df.columns else ["全部"]
            col_symbol_filter, col_action_filter = st.columns(2)
            selected_symbol_filter = col_symbol_filter.selectbox("标的过滤", symbols)
            selected_action_filter = col_action_filter.selectbox("方向过滤", actions)
            if selected_symbol_filter != "全部":
                trades_df = trades_df[trades_df["symbol"] == selected_symbol_filter]
            if selected_action_filter != "全部":
                trades_df = trades_df[trades_df["action"] == selected_action_filter]
            st.dataframe(trades_df, width="stretch", height=560)

    with tab_symbol:
        signaled_data = st.session_state.signaled_data
        result = st.session_state.backtest_result
        if not signaled_data:
            st.info("请先加载数据并生成信号。")
        else:
            symbol = st.selectbox("选择标的", sorted(signaled_data.keys()))
            trades = result.trades if result is not None else pd.DataFrame()
            draw_symbol_chart(symbol, signaled_data[symbol], trades)
            with st.expander("信号数据预览", expanded=False):
                preview_cols = [
                    col
                    for col in ["open", "high", "low", "close", "volume", "signal"]
                    if col in signaled_data[symbol].columns
                ]
                st.dataframe(signaled_data[symbol][preview_cols].tail(200), width="stretch")


if __name__ == "__main__":
    main()
