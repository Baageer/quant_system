"""HS300 个股离线行情阶段标签研究页面。

运行方式：streamlit run research/market_regime/app_streamlit.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.market_regime.data_loader import (  # noqa: E402
    DEFAULT_PRICE_HISTORY_DIR,
    load_hs300_symbols,
    load_local_price_history,
)
from research.market_regime.labeler import MarketRegimeLabeler, RegimeLabelerConfig  # noqa: E402


REGIME_COLORS = {
    "up": "#2ca02c",
    "sideways": "#7f7f7f",
    "down": "#d62728",
}
REGIME_NAMES = {
    "up": "上涨",
    "sideways": "盘整",
    "down": "下跌",
}


if hasattr(st, "cache_data"):
    cache_data = st.cache_data(show_spinner=False)
else:
    cache_data = st.cache


@cache_data
def load_cached_price_history(symbol, adjustment):
    return load_local_price_history(symbol, DEFAULT_PRICE_HISTORY_DIR, adjustment)


def build_regime_figure(data, segments, pivots):
    """创建收盘价、阶段背景和结构变点叠加图。"""

    figure = go.Figure()
    for segment in segments.itertuples(index=False):
        color = REGIME_COLORS[segment.regime]
        suffix = "（未闭合）" if segment.open_segment else ""
        figure.add_vrect(
            x0=segment.start_date,
            x1=segment.end_date,
            fillcolor=color,
            opacity=0.14,
            line_width=0,
            annotation_text="{}{}".format(REGIME_NAMES[segment.regime], suffix),
            annotation_position="top left",
            annotation_font_color=color,
        )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            mode="lines",
            name="收盘价",
            line={"color": "#1f77b4", "width": 1.5},
        )
    )

    boundaries = segments.iloc[:-1]
    if not boundaries.empty:
        figure.add_trace(
            go.Scatter(
                x=boundaries["end_date"],
                y=[data.loc[date, "close"] for date in boundaries["end_date"]],
                mode="markers",
                name="分段线性变点",
                marker={"color": "#ff7f0e", "size": 9, "symbol": "diamond"},
                hovertemplate="变点<br>%{x|%Y-%m-%d}<br>收盘价 %{y:.2f}<extra></extra>",
            )
        )

    if not pivots.empty:
        pivot_colors = pivots["pivot_type"].map({"top": "#d62728", "bottom": "#2ca02c"}).fillna("#9467bd")
        figure.add_trace(
            go.Scatter(
                x=pivots["pivot_date"],
                y=[data.loc[date, "close"] for date in pivots["pivot_date"]],
                mode="markers",
                name="状态切换拐点",
                marker={"color": pivot_colors.tolist(), "size": 11, "symbol": "x"},
                text=["{} → {}".format(REGIME_NAMES[row.previous_regime], REGIME_NAMES[row.next_regime]) for row in pivots.itertuples()],
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>收盘价 %{y:.2f}<extra></extra>",
            )
        )

    figure.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        xaxis_title="日期",
        yaxis_title="收盘价（本地复权口径）",
    )
    return figure


def main():
    st.set_page_config(page_title="HS300 行情阶段标注", layout="wide")
    st.title("第一版研究原型")
    st.caption("离线结构标签会使用完整历史路径，仅用于复盘与研究，不能直接作为实盘信号。")

    try:
        symbols = load_hs300_symbols()
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("数据与参数")
    symbol = st.sidebar.selectbox("HS300 标的", symbols)
    adjustment = st.sidebar.selectbox("本地价格口径", ["raw_hfq_pct", "qfq", "hfq"], index=0)
    st.sidebar.caption("仅读取 data/raw/tushare/price_history 下的本地 CSV 缓存。")
    st.sidebar.subheader("阶段划分")
    min_segment_length = st.sidebar.slider("最短阶段交易日", 10, 60, 20, 1)
    max_segment_length = st.sidebar.slider("最大候选阶段交易日", min_segment_length, 252, max(120, min_segment_length), 1)
    smoothing_span = st.sidebar.slider("对数价格平滑周期", 1, 20, 7, 1)
    min_return = st.sidebar.slider("固定最小幅度", 0.01, 0.20, 0.03, 0.01)
    volatility_multiplier = st.sidebar.slider("波动率倍数", 0.0, 3.0, 0.75, 0.1)
    volatility_horizon = st.sidebar.slider("波动率累计期限上限", 10, 120, 60, 5)
    dynamic_threshold_cap = st.sidebar.slider("动态幅度上限", 0.05, 0.50, 0.20, 0.01)
    min_efficiency_ratio = st.sidebar.slider("最小趋势效率", 0.1, 0.9, 0.15, 0.05)
    min_r_squared = st.sidebar.slider("最小 R²", 0.0, 0.9, 0.10, 0.05)
    segmentation_penalty = st.sidebar.slider("分段惩罚系数", 0.0, 10.0, 3.0, 0.5)
    boundary_window = st.sidebar.slider("边界校准窗口", 0, 15, 5, 1)
    st.sidebar.subheader("拐点确认")
    pivot_min_reversal = st.sidebar.slider("最小反转幅度", 0.01, 0.20, 0.05, 0.01)
    pivot_reversal_multiplier = st.sidebar.slider("反转波动率倍数", 0.0, 5.0, 2.0, 0.25)

    try:
        data = load_cached_price_history(symbol, adjustment)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(str(exc))
        st.stop()

    start_date = st.sidebar.date_input("开始日期", value=data.index.min().date(), min_value=data.index.min().date(), max_value=data.index.max().date())
    end_date = st.sidebar.date_input("结束日期", value=data.index.max().date(), min_value=data.index.min().date(), max_value=data.index.max().date())
    selected = data.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
    if len(selected) < min_segment_length:
        st.warning("所选区间只有 {} 个交易日，少于最短阶段长度 {}。".format(len(selected), min_segment_length))
        st.stop()

    config = RegimeLabelerConfig(
        min_segment_length=min_segment_length,
        min_return=min_return,
        volatility_multiplier=volatility_multiplier,
        max_segment_length=max_segment_length,
        smoothing_span=smoothing_span,
        volatility_horizon=volatility_horizon,
        dynamic_threshold_cap=dynamic_threshold_cap,
        min_efficiency_ratio=min_efficiency_ratio,
        min_r_squared=min_r_squared,
        segmentation_penalty=segmentation_penalty,
        boundary_window=boundary_window,
        pivot_min_reversal=pivot_min_reversal,
        pivot_reversal_multiplier=pivot_reversal_multiplier,
    )
    labeler = MarketRegimeLabeler(config)
    try:
        segments, daily, pivots = labeler.label(selected, symbol)
    except ValueError as exc:
        st.error("无法标注：{}".format(exc))
        st.stop()

    audit = labeler.audit(selected)
    metric_columns = st.columns(4)
    metric_columns[0].metric("交易日", len(selected))
    metric_columns[1].metric("阶段数", len(segments))
    metric_columns[2].metric("结构变点", max(len(segments) - 1, 0))
    metric_columns[3].metric("状态切换", len(pivots))

    st.plotly_chart(build_regime_figure(selected, segments, pivots), width="stretch")
    st.caption("橙色菱形表示最终阶段边界；叉号表示标签发生状态切换的拐点。")

    overview, segment_tab, daily_tab, pivot_tab, audit_tab = st.tabs(
        ["阶段概览", "阶段表", "逐日状态", "拐点事件", "数据审计"]
    )
    with overview:
        summary = segments.copy()
        summary["regime_name"] = summary["regime"].map(REGIME_NAMES)
        regime_days = summary.groupby("regime_name")["duration"].sum().reindex(["上涨", "盘整", "下跌"], fill_value=0)
        distribution_columns = st.columns(3)
        for column, (regime_name, duration) in zip(distribution_columns, regime_days.items()):
            column.metric("{}占比".format(regime_name), "{:.1%}".format(duration / len(selected)))
        longest_sideways = summary.loc[summary["regime"] == "sideways", "duration"]
        if not longest_sideways.empty and longest_sideways.max() > max_segment_length:
            st.warning("最长盘整段超过最大候选阶段长度，请检查相邻盘整合并与参数设置。")
        st.dataframe(
            summary[["segment_id", "regime_name", "start_date", "end_date", "duration", "segment_return", "confidence", "open_segment"]],
            width="stretch",
            hide_index=True,
        )
    with segment_tab:
        st.dataframe(segments, width="stretch", hide_index=True)
        st.download_button("下载阶段表 CSV", segments.to_csv(index=False).encode("utf-8-sig"), "market_regime_segments_{}.csv".format(symbol), "text/csv")
    with daily_tab:
        st.dataframe(daily, width="stretch", hide_index=True)
        st.download_button("下载逐日状态 CSV", daily.to_csv(index=False).encode("utf-8-sig"), "market_regime_daily_{}.csv".format(symbol), "text/csv")
    with pivot_tab:
        st.dataframe(pivots, width="stretch", hide_index=True)
        st.download_button("下载拐点表 CSV", pivots.to_csv(index=False).encode("utf-8-sig"), "market_regime_pivots_{}.csv".format(symbol), "text/csv")
    with audit_tab:
        audit_display = {key: value for key, value in audit.items() if key != "parameters"}
        st.dataframe(pd.DataFrame([audit_display]), width="stretch", hide_index=True)
        st.json(audit["parameters"])


if __name__ == "__main__":
    main()
