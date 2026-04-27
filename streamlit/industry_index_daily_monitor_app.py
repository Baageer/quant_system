from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.param_search.industry_index_daily_monitor import (  # noqa: E402
    INDEX_DATA_DIR,
    OUTPUT_DIR,
    REPORT_COLUMNS,
    build_index_file_lookup,
    normalize_code,
    normalize_text,
    pick_column,
    safe_float,
    standardize_index_price_frame,
)


STREAMLIT_KLINE_BARS = 160


def read_csv_with_fallback(csv_path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(csv_path)


def find_opportunity_report_files(output_dir):
    if not output_dir.exists():
        return []

    report_files = sorted(
        output_dir.glob("industry_opportunities_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    latest_path = output_dir / "latest_industry_opportunities.csv"
    if latest_path.exists():
        report_files = [latest_path, *[path for path in report_files if path != latest_path]]
    return report_files


def parse_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "nan", "none", "null", ""}:
        return False
    return bool(value)


def standardize_monitor_report(report_df):
    if report_df is None or report_df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    df = report_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    column_aliases = {
        "symbol": ["symbol", "index_code", "code"],
        "industry_name": ["industry_name", "industry", "name", "板块名称", "行业名称"],
        "trade_date": ["trade_date", "date", "日期"],
        "snapshot_time": ["snapshot_time", "snapshot", "run_time"],
        "run_label": ["run_label"],
        "is_today": ["is_today"],
        "is_new_signal": ["is_new_signal", "new_signal"],
        "event_direction": ["event_direction", "direction"],
    }
    rename_map = {}
    for target, candidates in column_aliases.items():
        source_column = pick_column(df, candidates)
        if source_column is not None and source_column != target:
            rename_map[source_column] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    if "symbol" not in df.columns:
        raise ValueError("机会报告缺少 symbol/index_code 列，无法构建可视化。")

    df["symbol"] = df["symbol"].map(normalize_code)
    if "industry_name" not in df.columns:
        df["industry_name"] = ""
    else:
        df["industry_name"] = df["industry_name"].map(normalize_text)

    date_columns = ["trade_date", "snapshot_time"]
    for column in date_columns:
        if column not in df.columns:
            df[column] = pd.NaT
        df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns = [
        "close",
        "daily_return",
        "volume_ratio",
        "bandwidth",
        "condition_threshold",
        "rsrs_score",
        "rsrs_zscore",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = None
        df[column] = pd.to_numeric(df[column], errors="coerce")

    boolean_columns = [
        "is_today",
        "is_new_signal",
        "condition",
        "breakout_valid",
        "volume_confirmation",
        "trend_long_confirmation",
    ]
    for column in boolean_columns:
        if column not in df.columns:
            df[column] = False
        df[column] = df[column].map(parse_bool)

    for column in REPORT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[(df["symbol"] != "") & df["trade_date"].notna()].copy()
    if df.empty:
        return df[REPORT_COLUMNS]

    return (
        df.sort_values(
            by=["is_today", "is_new_signal", "condition", "trade_date", "daily_return", "volume_ratio"],
            ascending=[False, False, False, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)[REPORT_COLUMNS]
    )


def format_opportunity_label(row, index, total):
    trade_date = "-"
    if pd.notna(row.get("trade_date")):
        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")

    snapshot_text = "-"
    if pd.notna(row.get("snapshot_time")):
        snapshot_text = pd.Timestamp(row["snapshot_time"]).strftime("%m-%d %H:%M")

    signal_state = "观察"
    if parse_bool(row.get("is_new_signal")):
        signal_state = "新信号"
    elif parse_bool(row.get("condition")):
        signal_state = "条件满足"

    return f"机会 {index + 1}/{total} | {trade_date} | {signal_state} | {snapshot_text}"


def format_float_text(value, digits=2, suffix=""):
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{suffix}"


def load_symbol_price_frame(symbol):
    csv_path = build_index_file_lookup(INDEX_DATA_DIR).get(normalize_code(symbol))
    if csv_path is None:
        return pd.DataFrame()
    return standardize_index_price_frame(read_csv_with_fallback(csv_path))


def slice_price_window(price_df, trade_date, kline_bars):
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    sorted_df = price_df.sort_values("date").reset_index(drop=True)
    if trade_date is None or pd.isna(trade_date):
        return sorted_df.tail(max(kline_bars, 20)).copy()

    trade_date = pd.Timestamp(trade_date).normalize()
    candidate = sorted_df[sorted_df["date"] <= trade_date]
    if candidate.empty:
        return sorted_df.tail(max(kline_bars, 20)).copy()
    return candidate.tail(max(kline_bars, 20)).copy()


def build_opportunity_figure(window_df, selected_row):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    volume_colors = [
        "#EF5350" if close_price >= open_price else "#26A69A"
        for close_price, open_price in zip(window_df["close"], window_df["open"])
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"{selected_row['symbol']} | {selected_row.get('industry_name', '')} | 交易日 {selected_row['trade_date']:%Y-%m-%d}",
            "成交量",
        ),
    )
    fig.add_trace(
        go.Candlestick(
            x=window_df["date"],
            open=window_df["open"],
            high=window_df["high"],
            low=window_df["low"],
            close=window_df["close"],
            name="K线",
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            hovertemplate=(
                "日期: %{x|%Y-%m-%d}<br>"
                "开: %{open:.2f}<br>"
                "高: %{high:.2f}<br>"
                "低: %{low:.2f}<br>"
                "收: %{close:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=window_df["date"],
            y=window_df["volume"],
            name="成交量",
            marker_color=volume_colors,
            hovertemplate="日期: %{x|%Y-%m-%d}<br>成交量: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    trade_date = pd.Timestamp(selected_row["trade_date"]).normalize()
    event_rows = window_df[window_df["date"] == trade_date]
    if not event_rows.empty:
        event_close = event_rows.iloc[-1]["close"]
        fig.add_vline(
            x=trade_date,
            line_dash="dash",
            line_color="#FF9800",
            line_width=1.5,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=trade_date,
            line_dash="dash",
            line_color="#FF9800",
            line_width=1.5,
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[trade_date],
                y=[event_close],
                mode="markers",
                marker=dict(size=11, color="#FF9800", symbol="diamond"),
                name="机会触发日",
                hovertemplate="触发日: %{x|%Y-%m-%d}<br>收盘: %{y:.2f}<extra></extra>",
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


def build_opportunity_overview_table(symbol_df, current_index):
    display_df = symbol_df.copy().reset_index(drop=True)
    display_df.insert(
        0,
        "当前",
        ["● 当前" if idx == current_index else "" for idx in range(len(display_df))],
    )
    if "trade_date" in display_df.columns:
        display_df["trade_date"] = pd.to_datetime(display_df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "snapshot_time" in display_df.columns:
        display_df["snapshot_time"] = pd.to_datetime(display_df["snapshot_time"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    percent_columns = ["daily_return"]
    decimal_columns = ["close", "volume_ratio", "bandwidth", "condition_threshold", "rsrs_score", "rsrs_zscore"]
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: format_float_text(value, digits=2, suffix="%"))
    for column in decimal_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: format_float_text(value, digits=4))

    ordered_columns = [
        "当前",
        "trade_date",
        "snapshot_time",
        "industry_name",
        "event_direction",
        "close",
        "daily_return",
        "volume_ratio",
        "condition",
        "is_new_signal",
        "breakout_valid",
        "volume_confirmation",
        "trend_long_confirmation",
        "bandwidth",
        "condition_threshold",
        "rsrs_score",
        "rsrs_zscore",
    ]
    ordered_columns = [column for column in ordered_columns if column in display_df.columns]
    display_df = display_df[ordered_columns].rename(
        columns={
            "trade_date": "交易日",
            "snapshot_time": "快照时间",
            "industry_name": "行业",
            "event_direction": "方向",
            "close": "收盘价",
            "daily_return": "日涨跌幅",
            "volume_ratio": "量比",
            "condition": "条件触发",
            "is_new_signal": "新信号",
            "breakout_valid": "突破确认",
            "volume_confirmation": "放量确认",
            "trend_long_confirmation": "趋势确认",
            "bandwidth": "带宽",
            "condition_threshold": "阈值",
            "rsrs_score": "RSRS分数",
            "rsrs_zscore": "RSRS Z分数",
        }
    )
    return display_df


def classify_strength_level(strength_pct, strong_threshold):
    if pd.isna(strength_pct):
        return "缺失"
    if strength_pct >= strong_threshold:
        return "强势"
    if strength_pct >= 0:
        return "偏强"
    if strength_pct > -strong_threshold:
        return "偏弱"
    return "弱势"


@st.cache_data(ttl=600, show_spinner=False)
def build_ma20_heatmap_frames(symbol_items, lookback_days=30, ma_window=20):
    file_lookup = build_index_file_lookup(INDEX_DATA_DIR)
    rows = []
    for symbol, industry_name in symbol_items:
        csv_path = file_lookup.get(normalize_code(symbol))
        if csv_path is None:
            continue
        price_df = standardize_index_price_frame(read_csv_with_fallback(csv_path))
        if price_df.empty or "close" not in price_df.columns:
            continue

        price_df = price_df.sort_values("date").copy()
        price_df["ma"] = price_df["close"].rolling(ma_window, min_periods=ma_window).mean()
        price_df["strength_pct"] = (price_df["close"] / price_df["ma"] - 1.0) * 100.0
        strength_df = price_df.dropna(subset=["strength_pct"])[["date", "strength_pct"]].tail(lookback_days)
        if strength_df.empty:
            continue

        row_label = f"{industry_name} ({symbol})" if industry_name else symbol
        strength_df = strength_df.assign(symbol=symbol, industry_name=industry_name, row_label=row_label)
        rows.append(strength_df)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce").dt.normalize()
    long_df = long_df.dropna(subset=["date"]).copy()
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    recent_trade_dates = (
        pd.Series(long_df["date"].drop_duplicates().sort_values().tolist()).tail(max(int(lookback_days), 1)).tolist()
    )

    latest_available_df = (
        long_df.sort_values("date")
        .groupby("row_label", as_index=False)
        .tail(1)[["row_label", "symbol", "industry_name", "date"]]
        .rename(columns={"date": "latest_available_date"})
        .reset_index(drop=True)
    )

    pivot_df = long_df.pivot_table(index="row_label", columns="date", values="strength_pct", aggfunc="last")
    pivot_df = pivot_df.reindex(columns=recent_trade_dates)
    pivot_df = pivot_df.dropna(axis=0, how="all")
    if pivot_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    reference_date = pivot_df.columns[-1]
    latest_df = latest_available_df[latest_available_df["row_label"].isin(pivot_df.index)].copy()
    latest_df = latest_df.set_index("row_label").reindex(pivot_df.index).reset_index()
    latest_df["date"] = pd.Timestamp(reference_date)
    latest_df["strength_pct"] = pivot_df[reference_date].reindex(latest_df["row_label"]).values
    latest_df = latest_df.sort_values(
        by=["strength_pct", "latest_available_date"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    pivot_df = pivot_df.reindex(index=latest_df["row_label"].tolist())
    return pivot_df, latest_df


def build_ma20_heatmap_figure(pivot_df):
    import plotly.graph_objects as go

    if pivot_df is None or pivot_df.empty:
        return None

    rows_count = len(pivot_df.index)
    max_abs = float(pivot_df.abs().quantile(0.95).max()) if not pivot_df.empty else 5.0
    max_abs = max(3.0, min(max_abs, 15.0))

    x_values = pd.to_datetime(pivot_df.columns, errors="coerce")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=x_values,
            y=pivot_df.index.tolist(),
            colorscale="RdYlGn_r",
            zmid=0.0,
            zmin=-max_abs,
            zmax=max_abs,
            colorbar=dict(title="收盘-MA20偏离(%)"),
            hovertemplate="行业: %{y}<br>日期: %{x|%Y-%m-%d}<br>偏离: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=min(max(420, 22 * rows_count + 140), 1400),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="交易日",
        yaxis_title="行业",
    )
    fig.update_xaxes(type="date", tickformat="%m-%d", tickangle=-45)
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=pivot_df.index.tolist(),
    )
    return fig


def shift_selection_index(selector_key, step, total_count):
    if total_count <= 0:
        st.session_state[selector_key] = 0
        return
    current_index = int(st.session_state.get(selector_key, 0))
    next_index = min(max(current_index + step, 0), total_count - 1)
    st.session_state[selector_key] = next_index


def shift_symbol_in_list(symbol_key, symbols, step):
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


def run_dashboard():
    st.set_page_config(page_title="行业指数机会监控", page_icon="📱", layout="wide")
    st.title("📱 行业指数 Top Opportunities 可视化")
    st.caption("查看机会列表，并联动展示行业指数 K 线与成交量。")

    report_files = find_opportunity_report_files(OUTPUT_DIR)
    if not report_files:
        st.warning(f"未找到机会报告文件，请先运行监控脚本生成 CSV：{OUTPUT_DIR}")
        st.stop()

    st.sidebar.header("⚙️ 参数设置")
    selected_report = st.sidebar.selectbox(
        "选择机会报告文件",
        options=report_files,
        format_func=lambda path: Path(path).name,
    )

    try:
        monitor_df = standardize_monitor_report(read_csv_with_fallback(selected_report))
    except Exception as exc:  # noqa: BLE001
        st.error(f"加载机会报告失败: {exc}")
        st.stop()

    if monitor_df.empty:
        st.warning("当前报告没有可展示的机会数据。")
        st.stop()

    only_today = st.sidebar.checkbox("仅显示当日数据", value=True)
    only_new_signal = st.sidebar.checkbox("仅显示新信号", value=False)
    only_condition_true = st.sidebar.checkbox("仅显示条件触发", value=True)
    industry_keyword = st.sidebar.text_input("行业关键词过滤", value="").strip().lower()

    filtered_df = monitor_df.copy()
    if only_today and "is_today" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_today"]]
    if only_new_signal and "is_new_signal" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_new_signal"]]
    if only_condition_true and "condition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["condition"]]
    if industry_keyword:
        filtered_df = filtered_df[
            filtered_df["industry_name"].fillna("").astype(str).str.lower().str.contains(industry_keyword)
        ]

    if filtered_df.empty:
        st.warning("筛选后没有可展示的机会，请放宽筛选条件。")
        st.stop()

    symbols = filtered_df["symbol"].dropna().astype(str).drop_duplicates().tolist()
    symbol_key = f"industry_symbol::{Path(selected_report).name}"
    if symbol_key not in st.session_state or st.session_state[symbol_key] not in symbols:
        st.session_state[symbol_key] = symbols[0]

    st.sidebar.selectbox("选择行业指数", symbols, key=symbol_key)
    selected_symbol = st.session_state[symbol_key]

    symbol_nav_cols = st.sidebar.columns(2)
    previous_symbol_disabled = symbols.index(selected_symbol) <= 0
    next_symbol_disabled = symbols.index(selected_symbol) >= len(symbols) - 1
    with symbol_nav_cols[0]:
        st.button(
            "⬅️ 上一个",
            disabled=previous_symbol_disabled,
            on_click=shift_symbol_in_list,
            args=(symbol_key, symbols, -1),
            width='stretch',
            key=f"prev_symbol::{Path(selected_report).name}",
        )
    with symbol_nav_cols[1]:
        st.button(
            "下一个 ➡️",
            disabled=next_symbol_disabled,
            on_click=shift_symbol_in_list,
            args=(symbol_key, symbols, 1),
            width='stretch',
            key=f"next_symbol::{Path(selected_report).name}",
        )

    selected_symbol = st.session_state[symbol_key]
    symbol_rows = (
        filtered_df[filtered_df["symbol"] == selected_symbol]
        .sort_values(by=["trade_date", "snapshot_time"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    if symbol_rows.empty:
        st.warning("当前行业指数没有可展示的机会记录。")
        st.stop()

    row_selector_key = f"industry_row::{Path(selected_report).name}::{selected_symbol}"
    if row_selector_key not in st.session_state:
        st.session_state[row_selector_key] = 0
    current_index = int(st.session_state[row_selector_key])
    if current_index >= len(symbol_rows):
        current_index = 0
        st.session_state[row_selector_key] = 0

    st.sidebar.selectbox(
        "选择机会记录",
        options=list(range(len(symbol_rows))),
        key=row_selector_key,
        format_func=lambda idx: format_opportunity_label(symbol_rows.iloc[idx], idx, len(symbol_rows)),
    )
    current_index = int(st.session_state[row_selector_key])

    row_nav_cols = st.sidebar.columns(2)
    previous_disabled = current_index <= 0
    next_disabled = current_index >= len(symbol_rows) - 1
    with row_nav_cols[0]:
        st.button(
            "⬅️ 上一条",
            disabled=previous_disabled,
            on_click=shift_selection_index,
            args=(row_selector_key, -1, len(symbol_rows)),
            width='stretch',
            key=f"prev_row::{Path(selected_report).name}::{selected_symbol}",
        )
    with row_nav_cols[1]:
        st.button(
            "下一条 ➡️",
            disabled=next_disabled,
            on_click=shift_selection_index,
            args=(row_selector_key, 1, len(symbol_rows)),
            width='stretch',
            key=f"next_row::{Path(selected_report).name}::{selected_symbol}",
        )
    st.sidebar.caption(f"当前第 {current_index + 1} / {len(symbol_rows)} 条")

    selected_row = symbol_rows.iloc[current_index]
    price_df = load_symbol_price_frame(selected_symbol)
    if price_df.empty:
        st.warning(f"未找到 {selected_symbol} 的历史行情文件，无法绘制 K 线。")
        st.stop()

    max_bars = min(max(len(price_df), 30), 600)
    default_bars = min(STREAMLIT_KLINE_BARS, max_bars)
    kline_bars = st.sidebar.slider("显示最近K线数量", min_value=30, max_value=max_bars, value=default_bars, step=10)
    window_df = slice_price_window(price_df, selected_row.get("trade_date"), kline_bars)
    if window_df.empty:
        st.warning("当前机会缺少可展示的行情窗口。")
        st.stop()

    tab_detail, tab_heatmap = st.tabs(["机会详情", "30天MA20热力图"])

    with tab_detail:
        metric_cols = st.columns(6)
        metric_cols[0].metric("行业指数", selected_row["symbol"])
        metric_cols[1].metric("行业名称", selected_row.get("industry_name", "-") or "-")
        metric_cols[2].metric("交易日", selected_row["trade_date"].strftime("%Y-%m-%d"))
        snapshot_value = "-"
        if pd.notna(selected_row.get("snapshot_time")):
            snapshot_value = pd.Timestamp(selected_row["snapshot_time"]).strftime("%Y-%m-%d %H:%M:%S")
        metric_cols[3].metric("快照时间", snapshot_value)
        metric_cols[4].metric("收盘价", format_float_text(selected_row.get("close"), digits=2))
        metric_cols[5].metric("日涨跌幅", format_float_text(selected_row.get("daily_return"), digits=2, suffix="%"))

        flag_cols = st.columns(4)
        flag_cols[0].metric("条件触发", "是" if parse_bool(selected_row.get("condition")) else "否")
        flag_cols[1].metric("新信号", "是" if parse_bool(selected_row.get("is_new_signal")) else "否")
        flag_cols[2].metric("突破确认", "是" if parse_bool(selected_row.get("breakout_valid")) else "否")
        flag_cols[3].metric("放量确认", "是" if parse_bool(selected_row.get("volume_confirmation")) else "否")

        st.plotly_chart(build_opportunity_figure(window_df, selected_row), width='stretch')
        st.caption(
            f"当前浏览 `{Path(selected_report).name}` 中 `{selected_symbol}` 的第 {current_index + 1} 条机会记录。"
        )

        with st.expander("📋 当前行业指数机会列表", expanded=False):
            st.dataframe(
                build_opportunity_overview_table(symbol_rows, current_index),
                width='stretch',
                hide_index=True,
            )

        with st.expander("📊 筛选后的 Top opportunities（全部）", expanded=False):
            st.dataframe(
                build_opportunity_overview_table(filtered_df.reset_index(drop=True), current_index=-1),
                width='stretch',
                hide_index=True,
            )

    with tab_heatmap:
        control_cols = st.columns(4)
        source_mode = control_cols[0].selectbox(
            "行业范围",
            options=["当前筛选行业", "报告全部行业"],
            index=0,
            key=f"heatmap_source::{Path(selected_report).name}",
        )
        lookback_days = control_cols[1].slider(
            "观察天数",
            min_value=15,
            max_value=60,
            value=30,
            step=1,
            key=f"heatmap_lookback::{Path(selected_report).name}",
        )
        ma_window = control_cols[2].slider(
            "均线窗口",
            min_value=10,
            max_value=60,
            value=20,
            step=1,
            key=f"heatmap_ma::{Path(selected_report).name}",
        )
        strong_threshold = control_cols[3].slider(
            "强弱阈值(%)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key=f"heatmap_threshold::{Path(selected_report).name}",
        )

        source_df = filtered_df if source_mode == "当前筛选行业" else monitor_df
        symbol_map_df = (
            source_df[["symbol", "industry_name"]]
            .dropna(subset=["symbol"])
            .drop_duplicates(subset=["symbol"], keep="first")
            .reset_index(drop=True)
        )
        symbol_items = tuple(
            (normalize_code(row.symbol), normalize_text(row.industry_name))
            for row in symbol_map_df.itertuples(index=False)
        )

        pivot_df, latest_df = build_ma20_heatmap_frames(
            symbol_items=symbol_items,
            lookback_days=lookback_days,
            ma_window=ma_window,
        )
        if pivot_df.empty or latest_df.empty:
            st.warning("当前范围内行业数据不足，无法构建 MA20 热力图。")
        else:
            max_rows_upper = max(1, min(len(latest_df), 200))
            default_rows = min(60, max_rows_upper)
            max_rows = st.slider(
                "展示行业数量",
                min_value=1,
                max_value=max_rows_upper,
                value=default_rows,
                step=1,
                key=f"heatmap_rows::{Path(selected_report).name}",
            )

            latest_view = latest_df.head(max_rows).copy()
            latest_view["强弱级别"] = latest_view["strength_pct"].apply(
                lambda value: classify_strength_level(value, strong_threshold=strong_threshold)
            )
            row_labels = latest_view["row_label"].tolist()
            pivot_view = pivot_df.reindex(row_labels)

            summary_cols = st.columns(4)
            strong_count = int((latest_view["强弱级别"] == "强势").sum())
            weak_count = int((latest_view["强弱级别"] == "弱势").sum())
            valid_count = int(latest_view["strength_pct"].notna().sum())
            strong_ratio = (strong_count / valid_count * 100.0) if valid_count > 0 else 0.0
            summary_cols[0].metric("样本行业", f"{len(latest_view)}")
            summary_cols[1].metric("强势行业", f"{strong_count}")
            summary_cols[2].metric("弱势行业", f"{weak_count}")
            summary_cols[3].metric("强势占比", f"{strong_ratio:.1f}%")

            heatmap_fig = build_ma20_heatmap_figure(pivot_view)
            st.plotly_chart(heatmap_fig, width='stretch')
            reference_trade_date = pd.to_datetime(pivot_view.columns[-1]).strftime("%Y-%m-%d")
            st.caption(f"基准日 {reference_trade_date}：颜色越红越强，越绿越弱；排序按该基准日偏离度。")

            table_df = latest_view.copy()
            table_df["基准交易日"] = pd.to_datetime(table_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            table_df["行业最新数据日"] = pd.to_datetime(table_df["latest_available_date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            table_df["收盘相对MA20(%)"] = table_df["strength_pct"].map(lambda value: format_float_text(value, digits=2))
            table_df = table_df.rename(columns={"industry_name": "行业", "symbol": "代码", "row_label": "行业标识"})
            show_columns = ["行业", "代码", "基准交易日", "行业最新数据日", "收盘相对MA20(%)", "强弱级别"]
            show_columns = [column for column in show_columns if column in table_df.columns]
            with st.expander("📌 最新强弱分级明细", expanded=False):
                st.dataframe(table_df[show_columns], width='stretch', hide_index=True)


if __name__ == "__main__":
    run_dashboard()
