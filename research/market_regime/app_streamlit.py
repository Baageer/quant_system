"""HS300 个股离线行情阶段标签研究页面。

运行方式：streamlit run research/market_regime/app_streamlit.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.market_regime.data_loader import (  # noqa: E402
    DEFAULT_PRICE_HISTORY_DIR,
    load_hs300_symbols,
    load_local_price_history,
)
from research.market_regime.labeler import MarketRegimeLabeler, RegimeLabelerConfig  # noqa: E402
# The datasets live on the D: volume (the checkout is on FAT32); this app used
# to hard-code the in-repo ``output/datasets``, which no longer exists -- so the
# prediction tab was silently reading an empty directory.
from research.market_regime.output_paths import DATASET_DIR  # noqa: E402
from research.market_regime.stock_valuation_diagnostic import (  # noqa: E402
    load_pool_membership,
)


MODEL_OUTPUT_DIR = Path(DATASET_DIR)

# Valuation-state runs, written by index_valuation_diagnostic and
# stock_valuation_diagnostic.  Prefix -> the manifest stem that identifies a run.
VALUATION_RUNS = {
    "index": "index_valuation",
    "stock": "stock_valuation",
}
# Descriptive-only banner, shown with every valuation figure.  The predictive use
# of valuation is closed (see docs/research/03_factor_and_valuation/value_factor_regime_study_summary.md)
# and this page must not imply otherwise.
VALUATION_CAPTION = (
    "估值状态为离线**描述型**指标：它说明当前价格处在自身历史或池内的什么位置，"
    "不预测未来收益，也不是交易信号。分位越低越便宜；非正倍数（亏损、负现金流）"
    "不参与排名，单独计数。"
)


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


def normalize_symbol(symbol):
    value = str(symbol).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(6) if value.isdigit() and len(value) <= 6 else value


def list_logistic_prediction_runs():
    prefix = "trend_logistic_test_predictions_"
    runs = []
    for path in MODEL_OUTPUT_DIR.glob("{}*.csv".format(prefix)):
        run_id = path.stem[len(prefix) :]
        manifest = MODEL_OUTPUT_DIR / "trend_logistic_manifest_{}.json".format(run_id)
        if manifest.is_file():
            runs.append(run_id)
    return sorted(runs, reverse=True)


@cache_data
def load_logistic_test_predictions(run_id):
    path = MODEL_OUTPUT_DIR / "trend_logistic_test_predictions_{}.csv".format(run_id)
    if not path.is_file():
        return pd.DataFrame()
    predictions = pd.read_csv(path)
    required = {"date", "symbol", "p_down", "p_none", "p_up", "predicted_target"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["symbol"] = predictions["symbol"].map(normalize_symbol)
    return predictions.dropna(subset=["date"]).sort_values(["symbol", "date"])


def build_logistic_prediction_figure(data, predictions, up_threshold, down_threshold):
    prediction_index = predictions.set_index("date")
    aligned = data.loc[:, ["open", "high", "low", "close"]].join(prediction_index, how="inner")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
    )
    figure.add_trace(
        go.Candlestick(
            x=aligned.index,
            open=aligned["open"],
            high=aligned["high"],
            low=aligned["low"],
            close=aligned["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    up_signals = aligned[aligned["p_up"] >= up_threshold]
    down_signals = aligned[aligned["p_down"] >= down_threshold]
    if not up_signals.empty:
        figure.add_trace(
            go.Scatter(
                x=up_signals.index,
                y=up_signals["low"] * 0.985,
                mode="markers",
                name="上涨预警",
                marker={"color": "#2ca02c", "size": 10, "symbol": "triangle-up"},
                customdata=up_signals[["p_up", "p_down", "p_none"]],
                hovertemplate="上涨预警<br>%{x|%Y-%m-%d}<br>p_up=%{customdata[0]:.2%}<br>p_down=%{customdata[1]:.2%}<br>p_none=%{customdata[2]:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if not down_signals.empty:
        figure.add_trace(
            go.Scatter(
                x=down_signals.index,
                y=down_signals["high"] * 1.015,
                mode="markers",
                name="下跌预警",
                marker={"color": "#d62728", "size": 10, "symbol": "triangle-down"},
                customdata=down_signals[["p_up", "p_down", "p_none"]],
                hovertemplate="下跌预警<br>%{x|%Y-%m-%d}<br>p_down=%{customdata[1]:.2%}<br>p_up=%{customdata[0]:.2%}<br>p_none=%{customdata[2]:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(x=aligned.index, y=aligned["p_up"], mode="lines", name="p_up", line={"color": "#2ca02c"}),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=aligned.index, y=aligned["p_down"], mode="lines", name="p_down", line={"color": "#d62728"}),
        row=2,
        col=1,
    )
    figure.add_hline(y=up_threshold, line_dash="dot", line_color="#2ca02c", row=2, col=1)
    figure.add_hline(y=down_threshold, line_dash="dot", line_color="#d62728", row=2, col=1)
    probability_ceiling = max(0.05, float(aligned[["p_up", "p_down"]].max().max()) * 1.15)
    figure.update_yaxes(title_text="价格", row=1, col=1)
    figure.update_yaxes(title_text="趋势概率", range=[0, probability_ceiling], tickformat=".0%", row=2, col=1)
    figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    figure.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06},
    )
    return figure, aligned, up_signals, down_signals


def stock_coverage_note(symbol, own_state, manifest):
    """Why a requested symbol has no rows in this run.

    "Not in the pool" and "in the pool, but no data" are different facts and
    they send the reader to different files.  A pool member whose fetch failed
    -- a 2025 delisting or merger -- is *in* the snapshot and absent from the
    run, so reporting it as outside the pool is simply wrong.
    """

    pool_files = (manifest or {}).get("pool_files") or {}
    resolved = {
        name: str(path if Path(path).is_absolute() else PROJECT_ROOT / path)
        for name, path in pool_files.items()
    }
    membership = load_pool_membership(resolved) if resolved else {}
    label = " + ".join(sorted(membership)) or "股票池"
    pool = set().union(*membership.values()) if membership else set()
    covered = set(own_state["symbol"].astype(str).str.zfill(6))
    symbol = str(symbol).zfill(6)

    if symbol in pool:
        missing = sorted(pool - covered)
        return (
            "{} 在池内（{} 快照），但本次运行的估值数据里没有它——通常是抓取失败"
            "（退市 / 吸收合并 / 换代码）。该运行覆盖 {} 只，池内还缺 {} 只{}。".format(
                symbol,
                label,
                len(covered),
                len(missing),
                "：" + ", ".join(missing) if missing else "",
            )
        )
    return "{} 不在本次运行覆盖的池内（{} 快照）。".format(symbol, label)


@cache_data
def list_valuation_runs(kind):
    """Run ids with a manifest for the given valuation layer, newest first."""

    prefix = VALUATION_RUNS[kind]
    runs = []
    for path in MODEL_OUTPUT_DIR.glob("{}_manifest_*.json".format(prefix)):
        run_id = path.stem[len(prefix) + len("_manifest_"):]
        runs.append(run_id)
    return sorted(runs)


@cache_data
def load_valuation_table(kind, run_id, table):
    """One CSV from a valuation run, or an empty frame when it is missing."""

    path = MODEL_OUTPUT_DIR / "{}_{}_{}.csv".format(VALUATION_RUNS[kind], table, run_id)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for column in ("date", "as_of", "period_end", "observed_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


@cache_data
def load_valuation_manifest(kind, run_id):
    """Run metadata, or an empty dict when the manifest is missing or unreadable."""

    path = MODEL_OUTPUT_DIR / "{}_manifest_{}.json".format(VALUATION_RUNS[kind], run_id)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


BAND_LABELS = {
    "very_cheap": "极便宜",
    "cheap": "便宜",
    "fair": "合理",
    "expensive": "贵",
    "very_expensive": "极贵",
}
BAND_COLORS = {
    "very_cheap": "#1a9850",
    "cheap": "#91cf60",
    "fair": "#bdbdbd",
    "expensive": "#fc8d59",
    "very_expensive": "#d73027",
}


def build_anchor_cheapness_figure(anchors, window):
    """Horizontal bars of cheapness per anchor, floored at 0 and coloured by band."""

    subset = anchors.loc[anchors["window"].eq(window) & anchors["cheapness"].notna()].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("cheapness")
    labels = ["{}  [{}]".format(row.anchor_label, row.scope) for row in subset.itertuples()]
    figure = go.Figure(go.Bar(
        x=subset["cheapness"],
        y=labels,
        orientation="h",
        marker_color=[BAND_COLORS.get(band, "#bdbdbd") for band in subset["band"]],
        text=["{}  {}".format(round(value, 3), BAND_LABELS.get(band, band))
              for value, band in zip(subset["cheapness"], subset["band"])],
        textposition="outside",
        hovertext=["{} (as-of {})".format(level, as_of)
                   for level, as_of in zip(subset["level"].round(4), subset["as_of"].dt.date)],
    ))
    figure.update_layout(
        title="各锚点便宜度（1 = 自身历史最便宜，window={}）".format(window),
        xaxis_title="便宜度",
        xaxis_range=[0, 1.15],
        height=max(320, 26 * len(subset) + 120),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return figure


def build_scope_summary_figure(summary):
    """Median cheapness per scope with the anchor spread drawn as an error bar."""

    subset = summary.loc[summary["median_cheapness"].notna()].sort_values("median_cheapness")
    if subset.empty:
        return None
    figure = go.Figure(go.Bar(
        x=subset["median_cheapness"],
        y=["{} (n={})".format(scope, n) for scope, n in zip(subset["scope"], subset["n_anchors"])],
        orientation="h",
        marker_color="#4c78a8",
        error_x=dict(
            type="data",
            symmetric=False,
            array=subset["max_cheapness"] - subset["median_cheapness"],
            arrayminus=subset["median_cheapness"] - subset["min_cheapness"],
            color="#333333",
        ),
        text=["分歧 {:.2f}".format(spread) for spread in subset["spread"]],
        textposition="outside",
    ))
    windows = sorted(set(subset["window"])) if "window" in subset.columns else []
    figure.update_layout(
        title="按口径的中位便宜度（横线为锚点分歧带，window={}）".format(
            "/".join(str(value) for value in windows) or "?"),
        xaxis_title="便宜度",
        xaxis_range=[0, 1.25],
        height=max(300, 30 * len(subset) + 120),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return figure


def build_symbol_own_state_figure(own_state, symbol, window):
    """One symbol's own-history percentile per metric over time."""

    subset = own_state.loc[own_state["window"].eq(window)].copy()
    if subset.empty:
        return None
    figure = go.Figure()
    for metric, group in subset.groupby("metric"):
        group = group.sort_values("date")
        figure.add_trace(go.Scatter(
            x=group["date"],
            y=group["cheapness"],
            mode="lines",
            name=metric,
            hovertemplate="%{x|%Y-%m-%d}<br>便宜度 %{y:.3f}<extra>" + metric + "</extra>",
        ))
    for cut, color in ((0.10, "#1a9850"), (0.30, "#91cf60"), (0.70, "#fc8d59"), (0.90, "#d73027")):
        figure.add_hline(y=cut, line_dash="dot", line_color=color, opacity=0.5)
    figure.update_layout(
        title="{} 自身历史便宜度（window={}，虚线为分档阈值）".format(symbol, window),
        yaxis_title="便宜度（1 = 自身历史最便宜）",
        yaxis_range=[0, 1],
        height=420,
        margin=dict(l=10, r=10, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


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
    logistic_runs = list_logistic_prediction_runs()
    logistic_run = None
    up_threshold = 0.02
    down_threshold = 0.02
    if logistic_runs:
        st.sidebar.subheader("Logistic 趋势预测")
        default_run_index = logistic_runs.index("hs300_logistic_v2") if "hs300_logistic_v2" in logistic_runs else 0
        logistic_run = st.sidebar.selectbox("预测模型版本", logistic_runs, index=default_run_index)
        up_threshold = st.sidebar.slider("上涨预警阈值 p_up", 0.0, 0.10, 0.02, 0.005)
        down_threshold = st.sidebar.slider("下跌预警阈值 p_down", 0.0, 0.10, 0.02, 0.005)
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
    logistic_predictions = load_logistic_test_predictions(logistic_run) if logistic_run else pd.DataFrame()
    symbol_predictions = logistic_predictions[
        (logistic_predictions["symbol"] == normalize_symbol(symbol))
        & (logistic_predictions["date"] >= selected.index.min())
        & (logistic_predictions["date"] <= selected.index.max())
    ].copy() if not logistic_predictions.empty else pd.DataFrame()
    metric_columns = st.columns(4)
    metric_columns[0].metric("交易日", len(selected))
    metric_columns[1].metric("阶段数", len(segments))
    metric_columns[2].metric("结构变点", max(len(segments) - 1, 0))
    metric_columns[3].metric("状态切换", len(pivots))

    st.plotly_chart(build_regime_figure(selected, segments, pivots), width="stretch")
    st.caption("橙色菱形表示最终阶段边界；叉号表示标签发生状态切换的拐点。")

    overview, segment_tab, daily_tab, pivot_tab, prediction_tab, valuation_tab, audit_tab = st.tabs(
        ["阶段概览", "阶段表", "逐日状态", "拐点事件", "模型预测", "估值状态", "数据审计"]
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
    with prediction_tab:
        if not logistic_runs:
            st.info("未找到 output/datasets/trend_logistic_test_predictions_*.csv，无法显示模型预测。")
        elif symbol_predictions.empty:
            st.info("所选标的或日期区间没有 {} 的测试期预测。模型预测仅覆盖其导出测试集日期。".format(logistic_run))
        else:
            figure, aligned_predictions, up_signals, down_signals = build_logistic_prediction_figure(
                selected, symbol_predictions, up_threshold, down_threshold
            )
            prediction_metrics = st.columns(4)
            prediction_metrics[0].metric("预测交易日", len(aligned_predictions))
            prediction_metrics[1].metric("上涨预警", len(up_signals))
            prediction_metrics[2].metric("下跌预警", len(down_signals))
            prediction_metrics[3].metric("最大趋势概率", "{:.2%}".format(aligned_predictions[["p_up", "p_down"]].max().max()))
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "预测来自 {} 的历史测试集输出；绿色/红色三角分别表示 p_up / p_down 达到当前阈值。"
                " target 为离线研究标签，仅用于复盘，不可作为当日可得信息。".format(logistic_run)
            )
            columns = ["p_up", "p_down", "p_none", "predicted_target"]
            if "target" in aligned_predictions.columns:
                columns.append("target")
            signal_display = aligned_predictions.loc[:, columns].reset_index().rename(columns={"index": "date"})
            st.dataframe(signal_display.sort_values("date", ascending=False), width="stretch", hide_index=True)
            st.download_button(
                "下载模型预测 CSV",
                signal_display.to_csv(index=False).encode("utf-8-sig"),
                "logistic_predictions_{}_{}.csv".format(logistic_run, symbol),
                "text/csv",
            )
    with valuation_tab:
        st.caption(VALUATION_CAPTION)

        index_runs = list_valuation_runs("index")
        st.subheader("指数 / 全市场")
        if not index_runs:
            st.info("未找到 index_valuation_manifest_*.json。先运行 "
                    "`python -m research.market_regime.index_valuation_diagnostic`。")
        else:
            index_run = st.selectbox("指数估值运行", index_runs, index=0)
            anchors = load_valuation_table("index", index_run, "anchors")
            summary = load_valuation_table("index", index_run, "summary")
            if anchors.empty:
                st.info("该运行没有锚点表。")
            else:
                available_windows = [w for w in ["36m", "60m", "120m", "all"] if anchors["window"].eq(w).any()]
                window = st.radio("窗口", available_windows or ["60m"], horizontal=True,
                                  help="分位随窗口变化，标签也会变。窗口未填满时不给分位。")
                anchor_figure = build_anchor_cheapness_figure(anchors, window)
                if anchor_figure is not None:
                    st.plotly_chart(anchor_figure, width="stretch")
                # The scope summary is exported at one window only (the run's
                # ``--summary-window``); the figure must say which, or a reader
                # comparing it against a different radio selection would be
                # comparing two windows without knowing it.
                manifest = load_valuation_manifest("index", index_run)
                summary_window = str(manifest.get("summary_window") or "60m")
                summary_figure = build_scope_summary_figure(
                    summary.loc[summary["window"].eq(summary_window)]
                    if not summary.empty else summary
                )
                if summary_figure is not None:
                    st.plotly_chart(summary_figure, width="stretch")
                    if summary_window != window:
                        st.caption("上方口径摘要固定在 window={}，与当前选择的 {} 不同——"
                                   "两者不可直接比较。".format(summary_window, window))
                st.caption(
                    "注意不同 `scope` 是不同总体（hs300 / market / board_*），不可跨口径拼成一句结论。"
                    "as-of 为该锚点真实最新观测日，不是月末标签。"
                )
                st.dataframe(anchors.loc[:, ["anchor_label", "scope", "window", "level", "as_of",
                                             "percentile", "n_obs", "band", "period_end"]],
                             width="stretch", hide_index=True)
                st.download_button("下载锚点状态 CSV",
                                   anchors.to_csv(index=False).encode("utf-8-sig"),
                                   "index_valuation_anchors_{}.csv".format(index_run), "text/csv")

        stock_runs = list_valuation_runs("stock")
        st.divider()
        st.subheader("个股：自身历史 vs 池内")
        if not stock_runs:
            st.info("未找到 stock_valuation_manifest_*.json。先运行 "
                    "`python -m research.market_regime.stock_valuation_diagnostic`。")
        else:
            stock_run = st.selectbox("个股估值运行", stock_runs, index=0)
            own_state = load_valuation_table("stock", stock_run, "own_state")
            composite = load_valuation_table("stock", stock_run, "composite_state")
            symbol_own = own_state.loc[
                own_state["symbol"].astype(str).str.zfill(6).eq(symbol)
            ] if not own_state.empty else own_state

            if own_state.empty:
                st.info("该运行没有个股状态表。")
            elif symbol_own.empty:
                st.info(stock_coverage_note(
                    symbol, own_state, load_valuation_manifest("stock", stock_run)))
            else:
                windows = [w for w in ["36m", "60m", "all"] if symbol_own["window"].eq(w).any()]
                chosen = st.radio("窗口", windows or ["36m"], horizontal=True,
                                  help="个股数据自 2018 起，120m 窗口填不满，因此不提供。")
                figure = build_symbol_own_state_figure(symbol_own, symbol, chosen)
                if figure is not None:
                    st.plotly_chart(figure, width="stretch")
                latest = symbol_own.loc[symbol_own["window"].eq(chosen)].sort_values("date").groupby("metric").tail(1)
                # ``date`` is the month-end bucket label, which for an in-progress
                # month is a date that has not happened yet; ``observed_date`` is
                # when the data actually exists.  Both go on screen.
                wanted = ["metric", "observed_date", "date", "level", "percentile_own",
                          "n_obs", "band_own", "undefined"]
                display = latest.loc[:, [column for column in wanted if column in latest.columns]].copy()
                display = display.rename(columns={"observed_date": "as_of", "date": "period_end"})
                display["band_own"] = display["band_own"].map(lambda value: BAND_LABELS.get(value, value))
                st.dataframe(display, width="stretch", hide_index=True)
                undefined_latest = latest.loc[latest["undefined"].astype(bool)]
                if not undefined_latest.empty:
                    st.warning("倍数非正，未参与排名：{}".format(
                        ", ".join(
                            "{} ({})".format(
                                row.metric,
                                (row.observed_date if pd.notna(row.observed_date) else row.date).date(),
                            )
                            for row in undefined_latest.itertuples()
                        )))
                st.caption(
                    "`n_obs` 是该分位实际使用的样本月数；窗口未填满时 `percentile_own` 为空。"
                    "亏损股的 PE 被排除而非当作'极便宜'，其占比见下方非正倍数统计。"
                )
                if not composite.empty:
                    symbol_composite = composite.loc[
                        composite["symbol"].astype(str).str.zfill(6).eq(symbol) & composite["window"].eq(chosen)
                    ].sort_values("date")
                    if not symbol_composite.empty:
                        st.caption("多指标合成（各指标便宜度的中位数）：最新 {}，投票指标数 {}".format(
                            BAND_LABELS.get(symbol_composite["band_composite"].iloc[-1],
                                            symbol_composite["band_composite"].iloc[-1]),
                            int(symbol_composite["n_metrics"].iloc[-1])))
                st.download_button("下载该标的状态 CSV",
                                   symbol_own.to_csv(index=False).encode("utf-8-sig"),
                                   "stock_valuation_own_{}_{}.csv".format(stock_run, symbol), "text/csv")

            undefined = load_valuation_table("stock", stock_run, "undefined_share")
            if not undefined.empty:
                recent = undefined.loc[undefined["date"].eq(undefined["date"].max())]
                as_of = recent["observed_date"].max() if "observed_date" in recent.columns else pd.NaT
                st.caption("最近一期（数据截至 {}）非正倍数占比".format(
                    (as_of if pd.notna(as_of) else recent["date"].max()).date()))
                st.dataframe(recent.loc[:, ["metric", "n_symbols", "n_undefined", "share_undefined"]],
                             width="stretch", hide_index=True)
    with audit_tab:
        audit_display = {key: value for key, value in audit.items() if key != "parameters"}
        st.dataframe(pd.DataFrame([audit_display]), width="stretch", hide_index=True)
        st.json(audit["parameters"])


if __name__ == "__main__":
    main()
