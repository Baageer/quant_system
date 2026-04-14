from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from event_study_utils import (
    DEFAULT_EVENT_STUDY_DIR,
    find_event_window_files,
    get_event_window,
    get_symbol_event_catalog,
    load_event_study_data,
)


st.set_page_config(page_title="事件窗口查看器", page_icon="🕯️", layout="wide")


def format_event_label(event_row: pd.Series, event_index: int, total_events: int) -> str:
    window_text = f"{int(event_row['min_relative_day'])} ~ {int(event_row['max_relative_day'])}"
    return f"片段 {event_index + 1}/{total_events} | {event_row['event_date']:%Y-%m-%d} | 窗口 {window_text}"


def build_volume_colors(window_df: pd.DataFrame) -> np.ndarray:
    return np.where(window_df["close"] >= window_df["open"], "#EF5350", "#26A69A")


def build_event_figure(window_df: pd.DataFrame, event_row: pd.Series) -> go.Figure:
    volume_colors = build_volume_colors(window_df)
    close_to_event = (
        window_df["close_to_event"] * 100
        if "close_to_event" in window_df.columns
        else pd.Series(index=window_df.index, dtype=float)
    )
    custom_data = np.column_stack(
        [
            window_df["relative_day"],
            close_to_event.fillna(np.nan),
        ]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"{event_row['symbol']} | 事件日 {event_row['event_date']:%Y-%m-%d}",
            "成交量",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=window_df["trade_date"],
            open=window_df["open"],
            high=window_df["high"],
            low=window_df["low"],
            close=window_df["close"],
            name="K线",
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            customdata=custom_data,
            hovertemplate=(
                "日期: %{x|%Y-%m-%d}<br>"
                "开: %{open:.2f}<br>"
                "高: %{high:.2f}<br>"
                "低: %{low:.2f}<br>"
                "收: %{close:.2f}<br>"
                "相对日: %{customdata[0]:.0f}<br>"
                "相对事件涨跌: %{customdata[1]:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=window_df["trade_date"],
            y=window_df["volume"],
            name="成交量",
            marker_color=volume_colors,
            customdata=window_df["relative_day"],
            hovertemplate=(
                "日期: %{x|%Y-%m-%d}<br>"
                "成交量: %{y:,.0f}<br>"
                "相对日: %{customdata:.0f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    event_day_row = window_df[window_df["relative_day"] == 0]
    if not event_day_row.empty:
        event_trade_date = event_day_row.iloc[0]["trade_date"]
        event_close = event_day_row.iloc[0]["close"]

        fig.add_vline(
            x=event_trade_date,
            line_dash="dash",
            line_color="#FF9800",
            line_width=1.5,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=event_trade_date,
            line_dash="dash",
            line_color="#FF9800",
            line_width=1.5,
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[event_trade_date],
                y=[event_close],
                mode="markers",
                name="事件日",
                marker=dict(size=12, color="#FF9800", symbol="diamond"),
                hovertemplate=(
                    "事件日: %{x|%Y-%m-%d}<br>"
                    "收盘价: %{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=760,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    return fig


def build_event_overview_table(symbol_events: pd.DataFrame, current_index: int) -> pd.DataFrame:
    display_df = symbol_events.copy()
    display_df.insert(
        0,
        "当前",
        ["◀ 当前" if index == current_index else "" for index in range(len(display_df))],
    )
    display_df["event_date"] = display_df["event_date"].dt.strftime("%Y-%m-%d")
    display_df["window_start"] = display_df["window_start"].dt.strftime("%Y-%m-%d")
    display_df["window_end"] = display_df["window_end"].dt.strftime("%Y-%m-%d")

    ordered_columns = [
        "当前",
        "event_date",
        "window_start",
        "window_end",
        "bar_count",
        "event_close",
    ]
    optional_columns = [
        "bandwidth",
        "condition_threshold",
        "event_direction",
        "breakout_valid",
        "volume_confirmation",
    ]
    ordered_columns.extend([column for column in optional_columns if column in display_df.columns])

    display_df = display_df[ordered_columns].rename(
        columns={
            "event_date": "事件日",
            "window_start": "窗口开始",
            "window_end": "窗口结束",
            "bar_count": "K线数",
            "event_close": "事件日收盘价",
            "bandwidth": "带宽",
            "condition_threshold": "阈值",
            "event_direction": "方向",
            "breakout_valid": "突破确认",
            "volume_confirmation": "放量确认",
        }
    )

    numeric_columns = ["事件日收盘价", "带宽", "阈值"]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else "-"
            )

    return display_df


def shift_event_index(selector_key: str, step: int, total_events: int) -> None:
    if total_events <= 0:
        st.session_state[selector_key] = 0
        return

    current_index = int(st.session_state.get(selector_key, 0))
    next_index = min(max(current_index + step, 0), total_events - 1)
    st.session_state[selector_key] = next_index


def shift_symbol_selection(symbol_key: str, symbols: list[str], step: int) -> None:
    if not symbols:
        st.session_state[symbol_key] = ""
        return

    current_symbol = st.session_state.get(symbol_key, symbols[0])
    try:
        current_index = symbols.index(current_symbol)
    except ValueError:
        current_index = 0

    next_index = min(max(current_index + step, 0), len(symbols) - 1)
    st.session_state[symbol_key] = symbols[next_index]


@st.cache_data(show_spinner=False)
def load_cached_event_study(event_windows_file: str):
    return load_event_study_data(event_windows_file)


def main():
    st.title("🕯️ 事件窗口查看器")
    st.caption("浏览 `event_windows_*.csv` 中的单个事件片段，查看 K 线、成交量与事件日位置。")

    event_window_files = find_event_window_files(DEFAULT_EVENT_STUDY_DIR)
    if not event_window_files:
        st.warning("当前没有可用的事件窗口文件。")
        st.stop()

    st.sidebar.header("⚙️ 参数设置")
    selected_file = st.sidebar.selectbox(
        "选择事件窗口文件",
        event_window_files,
        format_func=lambda value: Path(value).name,
    )

    try:
        event_windows, _, event_catalog = load_cached_event_study(selected_file)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"加载事件窗口文件失败: {exc}")
        st.stop()

    if event_catalog.empty:
        st.warning("当前文件没有可显示的事件片段。")
        st.stop()

    symbols = event_catalog["symbol"].drop_duplicates().tolist()
    symbol_key = f"symbol_selector::{selected_file}"
    if symbol_key not in st.session_state or st.session_state[symbol_key] not in symbols:
        st.session_state[symbol_key] = symbols[0]

    st.sidebar.selectbox(
        "选择股票",
        symbols,
        key=symbol_key,
    )
    selected_symbol = st.session_state[symbol_key]

    stock_nav_cols = st.sidebar.columns(2)
    previous_symbol_disabled = symbols.index(selected_symbol) <= 0
    next_symbol_disabled = symbols.index(selected_symbol) >= len(symbols) - 1
    with stock_nav_cols[0]:
        st.button(
            "⬅ 上一个",
            disabled=previous_symbol_disabled,
            on_click=shift_symbol_selection,
            args=(symbol_key, symbols, -1),
            use_container_width=True,
            key=f"prev_symbol::{selected_file}",
        )
    with stock_nav_cols[1]:
        st.button(
            "下一个 ➡",
            disabled=next_symbol_disabled,
            on_click=shift_symbol_selection,
            args=(symbol_key, symbols, 1),
            use_container_width=True,
            key=f"next_symbol::{selected_file}",
        )

    symbol_events = get_symbol_event_catalog(event_catalog, selected_symbol)

    selector_key = f"event_selector::{selected_file}::{selected_symbol}"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = 0
    current_index = int(st.session_state[selector_key])
    if current_index >= len(symbol_events):
        current_index = 0
        st.session_state[selector_key] = current_index

    st.sidebar.selectbox(
        "选择片段",
        options=list(range(len(symbol_events))),
        key=selector_key,
        format_func=lambda idx: format_event_label(symbol_events.iloc[idx], idx, len(symbol_events)),
    )
    current_index = st.session_state[selector_key]

    sidebar_nav_cols = st.sidebar.columns(2)
    previous_disabled = current_index <= 0
    next_disabled = current_index >= len(symbol_events) - 1
    with sidebar_nav_cols[0]:
        st.button(
            "⬅ 上一个",
            disabled=previous_disabled,
            on_click=shift_event_index,
            args=(selector_key, -1, len(symbol_events)),
            use_container_width=True,
        )
    with sidebar_nav_cols[1]:
        st.button(
            "下一个 ➡",
            disabled=next_disabled,
            on_click=shift_event_index,
            args=(selector_key, 1, len(symbol_events)),
            use_container_width=True,
        )
    st.sidebar.caption(f"当前第 {current_index + 1} / {len(symbol_events)} 个片段")

    current_event = symbol_events.iloc[current_index]
    max_visible_days = int(
        max(abs(current_event["min_relative_day"]), abs(current_event["max_relative_day"]))
    )
    visible_days = st.sidebar.slider(
        "显示事件日前后天数",
        min_value=1,
        max_value=max(1, max_visible_days),
        value=max(1, max_visible_days),
    )

    current_window = get_event_window(
        event_windows,
        symbol=current_event["symbol"],
        event_date=current_event["event_date"],
        visible_days=visible_days,
    )
    if current_window.empty:
        st.warning("当前片段没有可显示的数据。")
        st.stop()

    summary_cols = st.columns(6)
    summary_cols[0].metric("股票", current_event["symbol"])
    summary_cols[1].metric("事件日", current_event["event_date"].strftime("%Y-%m-%d"))
    summary_cols[2].metric("当前片段", f"{current_index + 1}/{len(symbol_events)}")
    summary_cols[3].metric(
        "窗口范围",
        f"{int(current_event['min_relative_day'])} ~ {int(current_event['max_relative_day'])}",
    )
    summary_cols[4].metric("事件日收盘价", f"{current_event['event_close']:.2f}")
    if "bandwidth" in current_event.index and pd.notna(current_event["bandwidth"]):
        summary_cols[5].metric("带宽", f"{current_event['bandwidth']:.4f}")
    elif "condition_threshold" in current_event.index and pd.notna(current_event["condition_threshold"]):
        summary_cols[5].metric("阈值", f"{current_event['condition_threshold']:.4f}")
    else:
        summary_cols[5].metric("显示窗口", f"±{visible_days} 天")

    st.plotly_chart(build_event_figure(current_window, current_event), width="stretch")

    st.caption(f"当前浏览 `{Path(selected_file).name}` 中 `{selected_symbol}` 的第 {current_index + 1} 个片段。")

    with st.expander("📋 当前股票的事件片段列表", expanded=False):
        st.dataframe(
            build_event_overview_table(symbol_events, current_index),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
