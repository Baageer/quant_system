"""Shared interactive candlestick charts for Streamlit backtest pages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from signals import indicators


OVERLAY_INDICATORS = {
    "ma": "移动平均线 (MA)",
    "ema": "指数平均线 (EMA)",
    "bollinger": "布林带 (BOLL)",
    "vwap": "成交量加权均价 (VWAP)",
    "supertrend": "SuperTrend",
}

PANEL_INDICATORS = {
    "signal": "策略信号",
    "rsi": "RSI",
    "macd": "MACD",
    "kdj": "KDJ",
    "atr": "ATR",
    "obv": "OBV",
}

DEFAULT_INDICATOR_PARAMS: Dict[str, Any] = {
    "ma_short": 10,
    "ma_long": 50,
    "ema_short": 12,
    "ema_long": 26,
    "bollinger_window": 20,
    "bollinger_std": 2.0,
    "supertrend_period": 10,
    "supertrend_multiplier": 3.0,
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "kdj_window": 9,
    "kdj_k_smooth": 3,
    "kdj_d_smooth": 3,
    "atr_window": 14,
}

PLOTLY_CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}


@dataclass
class CandlestickChartSettings:
    """User-selectable chart presentation and indicator settings."""

    overlay_indicators: Sequence[str] = field(default_factory=tuple)
    panel_indicators: Sequence[str] = field(default_factory=tuple)
    show_volume: bool = True
    show_trades: bool = True
    auto_scale_visible_y: bool = True
    show_range_slider: bool = False
    chart_height: int = 800
    indicator_params: Mapping[str, Any] = field(default_factory=dict)


def render_candlestick_chart_controls(
    key_prefix: str,
    default_panel_indicators: Sequence[str] = (),
) -> CandlestickChartSettings:
    """Render a consistent set of chart controls on a Streamlit page."""
    st.markdown("##### 图表设置")
    overlay_col, panel_col = st.columns(2)
    with overlay_col:
        overlay_indicators = st.multiselect(
            "主图叠加指标",
            options=list(OVERLAY_INDICATORS),
            default=[],
            format_func=OVERLAY_INDICATORS.get,
            key=f"{key_prefix}_overlay_indicators",
        )
    with panel_col:
        panel_indicators = st.multiselect(
            "副图指标",
            options=list(PANEL_INDICATORS),
            default=[
                indicator
                for indicator in default_panel_indicators
                if indicator in PANEL_INDICATORS
            ],
            format_func=PANEL_INDICATORS.get,
            key=f"{key_prefix}_panel_indicators",
        )

    volume_col, trades_col, auto_y_col, slider_col, height_col = st.columns(5)
    show_volume = volume_col.checkbox(
        "显示成交量",
        value=True,
        key=f"{key_prefix}_show_volume",
    )
    show_trades = trades_col.checkbox(
        "显示交易点",
        value=True,
        key=f"{key_prefix}_show_trades",
    )
    auto_scale_visible_y = auto_y_col.checkbox(
        "可视区Y轴自适应",
        value=True,
        help="横向缩放或拖动后，按当前可见数据自动调整各图层纵轴范围。",
        key=f"{key_prefix}_auto_scale_visible_y",
    )
    show_range_slider = slider_col.checkbox(
        "显示区间滑块",
        value=False,
        key=f"{key_prefix}_show_range_slider",
    )
    chart_height = height_col.slider(
        "图表高度",
        min_value=600,
        max_value=1600,
        value=800,
        step=50,
        key=f"{key_prefix}_chart_height",
    )

    indicator_params = dict(DEFAULT_INDICATOR_PARAMS)
    selected_indicators = set(overlay_indicators) | set(panel_indicators)
    configurable_indicators = selected_indicators - {"signal", "vwap", "obv"}
    if configurable_indicators:
        with st.expander("指标参数", expanded=False):
            _render_indicator_param_inputs(
                key_prefix,
                selected_indicators,
                indicator_params,
            )

    return CandlestickChartSettings(
        overlay_indicators=tuple(overlay_indicators),
        panel_indicators=tuple(panel_indicators),
        show_volume=show_volume,
        show_trades=show_trades,
        auto_scale_visible_y=auto_scale_visible_y,
        show_range_slider=show_range_slider,
        chart_height=chart_height,
        indicator_params=indicator_params,
    )


def _render_indicator_param_inputs(
    key_prefix: str,
    selected_indicators: set[str],
    params: Dict[str, Any],
) -> None:
    """Render parameter inputs only for indicators currently in use."""
    if "ma" in selected_indicators:
        short_col, long_col = st.columns(2)
        params["ma_short"] = int(
            short_col.number_input(
                "MA 短周期",
                min_value=1,
                value=int(params["ma_short"]),
                step=1,
                key=f"{key_prefix}_ma_short",
            )
        )
        params["ma_long"] = int(
            long_col.number_input(
                "MA 长周期",
                min_value=1,
                value=int(params["ma_long"]),
                step=1,
                key=f"{key_prefix}_ma_long",
            )
        )

    if "ema" in selected_indicators:
        short_col, long_col = st.columns(2)
        params["ema_short"] = int(
            short_col.number_input(
                "EMA 短周期",
                min_value=1,
                value=int(params["ema_short"]),
                step=1,
                key=f"{key_prefix}_ema_short",
            )
        )
        params["ema_long"] = int(
            long_col.number_input(
                "EMA 长周期",
                min_value=1,
                value=int(params["ema_long"]),
                step=1,
                key=f"{key_prefix}_ema_long",
            )
        )

    if "bollinger" in selected_indicators:
        window_col, std_col = st.columns(2)
        params["bollinger_window"] = int(
            window_col.number_input(
                "BOLL 周期",
                min_value=2,
                value=int(params["bollinger_window"]),
                step=1,
                key=f"{key_prefix}_bollinger_window",
            )
        )
        params["bollinger_std"] = float(
            std_col.number_input(
                "BOLL 标准差倍数",
                min_value=0.1,
                value=float(params["bollinger_std"]),
                step=0.1,
                key=f"{key_prefix}_bollinger_std",
            )
        )

    if "supertrend" in selected_indicators:
        period_col, multiplier_col = st.columns(2)
        params["supertrend_period"] = int(
            period_col.number_input(
                "SuperTrend ATR 周期",
                min_value=1,
                value=int(params["supertrend_period"]),
                step=1,
                key=f"{key_prefix}_supertrend_period",
            )
        )
        params["supertrend_multiplier"] = float(
            multiplier_col.number_input(
                "SuperTrend 倍数",
                min_value=0.1,
                value=float(params["supertrend_multiplier"]),
                step=0.1,
                key=f"{key_prefix}_supertrend_multiplier",
            )
        )

    if "rsi" in selected_indicators:
        params["rsi_window"] = int(
            st.number_input(
                "RSI 周期",
                min_value=2,
                value=int(params["rsi_window"]),
                step=1,
                key=f"{key_prefix}_rsi_window",
            )
        )

    if "macd" in selected_indicators:
        fast_col, slow_col, signal_col = st.columns(3)
        params["macd_fast"] = int(
            fast_col.number_input(
                "MACD 快线",
                min_value=1,
                value=int(params["macd_fast"]),
                step=1,
                key=f"{key_prefix}_macd_fast",
            )
        )
        params["macd_slow"] = int(
            slow_col.number_input(
                "MACD 慢线",
                min_value=2,
                value=int(params["macd_slow"]),
                step=1,
                key=f"{key_prefix}_macd_slow",
            )
        )
        params["macd_signal"] = int(
            signal_col.number_input(
                "MACD 信号线",
                min_value=1,
                value=int(params["macd_signal"]),
                step=1,
                key=f"{key_prefix}_macd_signal",
            )
        )

    if "kdj" in selected_indicators:
        window_col, k_col, d_col = st.columns(3)
        params["kdj_window"] = int(
            window_col.number_input(
                "KDJ 周期",
                min_value=2,
                value=int(params["kdj_window"]),
                step=1,
                key=f"{key_prefix}_kdj_window",
            )
        )
        params["kdj_k_smooth"] = int(
            k_col.number_input(
                "K 平滑周期",
                min_value=1,
                value=int(params["kdj_k_smooth"]),
                step=1,
                key=f"{key_prefix}_kdj_k_smooth",
            )
        )
        params["kdj_d_smooth"] = int(
            d_col.number_input(
                "D 平滑周期",
                min_value=1,
                value=int(params["kdj_d_smooth"]),
                step=1,
                key=f"{key_prefix}_kdj_d_smooth",
            )
        )

    if "atr" in selected_indicators:
        params["atr_window"] = int(
            st.number_input(
                "ATR 周期",
                min_value=1,
                value=int(params["atr_window"]),
                step=1,
                key=f"{key_prefix}_atr_window",
            )
        )


def build_candlestick_chart(
    price_data: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    symbol: Any,
    settings: Optional[CandlestickChartSettings] = None,
    ui_revision: Optional[str] = None,
) -> go.Figure:
    """Build an interactive candlestick chart with dynamic indicator rows."""
    chart_settings = settings or CandlestickChartSettings()
    plot_df = _normalize_price_data(price_data)
    params = {**DEFAULT_INDICATOR_PARAMS, **dict(chart_settings.indicator_params)}

    active_panels = [
        indicator
        for indicator in chart_settings.panel_indicators
        if indicator in PANEL_INDICATORS
        and not (indicator == "signal" and "signal" not in plot_df.columns)
    ]
    show_volume = chart_settings.show_volume and "volume" in plot_df.columns
    _validate_indicator_columns(plot_df, chart_settings, active_panels)

    row_titles = [f"{symbol} K线与交易点"]
    row_heights = [0.62]
    if show_volume:
        row_titles.append("成交量")
        row_heights.append(0.16)
    for indicator in active_panels:
        row_titles.append(PANEL_INDICATORS[indicator])
        row_heights.append(0.2)

    fig = make_subplots(
        rows=len(row_titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=tuple(row_titles),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="K线",
            increasing={"line": {"color": "red"}, "fillcolor": "red"},
            decreasing={"line": {"color": "green"}, "fillcolor": "green"},
        ),
        row=1,
        col=1,
    )
    _add_overlay_indicators(fig, plot_df, chart_settings.overlay_indicators, params)

    if chart_settings.show_trades:
        _add_trade_markers(fig, trades, symbol)

    next_row = 2
    if show_volume:
        candle_colors = [
            "red" if close >= open_price else "green"
            for open_price, close in zip(plot_df["open"], plot_df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df["volume"],
                name="成交量",
                marker_color=candle_colors,
                opacity=0.65,
            ),
            row=next_row,
            col=1,
        )
        fig.update_yaxes(title_text="成交量", row=next_row, col=1)
        next_row += 1

    for panel_indicator in active_panels:
        _add_panel_indicator(fig, plot_df, panel_indicator, next_row, params)
        next_row += 1

    fig.update_layout(
        height=chart_settings.chart_height,
        showlegend=True,
        hovermode="x unified",
        dragmode="pan",
        template="plotly_white",
        uirevision=ui_revision or f"candlestick-{symbol}",
        margin=dict(t=95, r=35, b=45, l=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=3, label="3月", step="month", stepmode="backward"),
                dict(count=6, label="半年", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(label="全部", step="all"),
            ]
        ),
        rangeslider_visible=chart_settings.show_range_slider,
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="日期", row=len(row_titles), col=1)
    fig.update_yaxes(title_text="价格", fixedrange=False, row=1, col=1)
    return fig


def render_candlestick_chart(
    fig: go.Figure,
    settings: CandlestickChartSettings,
    state_key: str,
) -> None:
    """Render a chart, optionally adapting Y axes to the visible X range."""
    if not settings.auto_scale_visible_y:
        st.plotly_chart(
            fig,
            width="stretch",
            config=PLOTLY_CHART_CONFIG,
            key=state_key,
        )
        return

    chart_html = build_visible_y_autoscale_html(fig, state_key)
    components.html(
        chart_html,
        height=settings.chart_height + 20,
        scrolling=False,
        tab_index=0,
    )


def build_visible_y_autoscale_html(
    fig: go.Figure,
    state_key: str,
    include_plotlyjs: bool = True,
) -> str:
    """Build standalone Plotly HTML that rescales Y axes after X-axis zoom."""
    fixed_y_axes = [
        axis_name
        for axis_name in fig.layout
        if str(axis_name).startswith("yaxis")
        and (
            fig.layout[axis_name].range is not None
            or fig.layout[axis_name].fixedrange is True
        )
    ]
    domain_signature = _chart_domain_signature(fig)
    storage_key = (
        f"quant-system-visible-range:{state_key}:{domain_signature}"
    )
    div_hash = hashlib.sha1(state_key.encode("utf-8")).hexdigest()[:16]
    post_script = _VISIBLE_Y_AUTOSCALE_SCRIPT.replace(
        "__STORAGE_KEY__",
        json.dumps(storage_key, ensure_ascii=False),
    ).replace(
        "__FIXED_Y_AXES__",
        json.dumps(fixed_y_axes),
    )
    chart_html = pio.to_html(
        fig,
        config=PLOTLY_CHART_CONFIG,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        post_script=post_script,
        div_id=f"candlestick-{div_hash}",
        default_width="100%",
        default_height=f"{fig.layout.height or 800}px",
    )
    return (
        "<style>html,body{margin:0;padding:0;overflow:hidden;}</style>"
        f"{chart_html}"
    )


def _chart_domain_signature(fig: go.Figure) -> str:
    """Return a stable signature that changes with the chart's date domain."""
    for trace in fig.data:
        x_values = getattr(trace, "x", None)
        if x_values is not None and len(x_values) > 0:
            domain = f"{x_values[0]}|{x_values[-1]}|{len(x_values)}"
            return hashlib.sha1(domain.encode("utf-8")).hexdigest()[:12]
    return "no-x-domain"


_VISIBLE_Y_AUTOSCALE_SCRIPT = r"""
(function () {
    const chart = document.getElementById('{plot_id}');
    const storageKey = __STORAGE_KEY__;
    const fixedYAxes = new Set(__FIXED_Y_AXES__);
    let adaptiveRelayoutPending = false;
    let debounceTimer = null;

    function arrayValues(values) {
        if (Array.isArray(values) || ArrayBuffer.isView(values)) {
            return values;
        }
        if (!values || !values.bdata || !values.dtype) {
            return [];
        }
        try {
            const binary = atob(values.bdata);
            const bytes = new Uint8Array(binary.length);
            for (let index = 0; index < binary.length; index += 1) {
                bytes[index] = binary.charCodeAt(index);
            }
            const dataType = values.dtype.replace(/[<>|]/g, '');
            const constructors = {
                f8: Float64Array,
                f4: Float32Array,
                i1: Int8Array,
                i2: Int16Array,
                i4: Int32Array,
                i8: typeof BigInt64Array === 'undefined' ? null : BigInt64Array,
                u1: Uint8Array,
                u2: Uint16Array,
                u4: Uint32Array,
                u8: typeof BigUint64Array === 'undefined' ? null : BigUint64Array,
            };
            const TypedArray = constructors[dataType];
            return TypedArray ? new TypedArray(bytes.buffer) : [];
        } catch (error) {
            return [];
        }
    }

    function dateValue(value) {
        if (value instanceof Date) {
            return value.getTime();
        }
        if (typeof value === 'number') {
            return value;
        }
        const parsed = Date.parse(value);
        return Number.isNaN(parsed) ? null : parsed;
    }

    function numericValue(value) {
        if (value === null || value === undefined || value === '') {
            return null;
        }
        const converted = Number(value);
        return Number.isFinite(converted) ? converted : null;
    }

    function visibleRangeFromEvent(eventData) {
        const axisRanges = new Map();
        let reset = false;
        Object.entries(eventData).forEach(([key, value]) => {
            const autorangeMatch = key.match(/^(xaxis\d*)\.autorange$/);
            if (autorangeMatch && value === true) {
                reset = true;
            }
            const pairMatch = key.match(/^(xaxis\d*)\.range$/);
            if (pairMatch && Array.isArray(value) && value.length === 2) {
                axisRanges.set(pairMatch[1], [value[0], value[1]]);
            }
            const itemMatch = key.match(/^(xaxis\d*)\.range\[(0|1)\]$/);
            if (itemMatch) {
                const current = axisRanges.get(itemMatch[1]) || [null, null];
                current[Number(itemMatch[2])] = value;
                axisRanges.set(itemMatch[1], current);
            }
        });
        if (reset) {
            return {reset: true, range: null};
        }
        for (const range of axisRanges.values()) {
            if (range[0] !== null && range[1] !== null) {
                return {reset: false, range: range};
            }
        }
        return null;
    }

    function yAxisName(trace) {
        const reference = trace.yaxis || 'y';
        return reference === 'y' ? 'yaxis' : `yaxis${reference.slice(1)}`;
    }

    function adaptiveYAxisUpdates(range) {
        const firstDate = dateValue(range[0]);
        const secondDate = dateValue(range[1]);
        if (firstDate === null || secondDate === null) {
            return {};
        }
        const start = Math.min(firstDate, secondDate);
        const end = Math.max(firstDate, secondDate);

        const extents = new Map();
        chart.data.forEach((trace) => {
            if (trace.visible === false || trace.visible === 'legendonly') {
                return;
            }
            const axisName = yAxisName(trace);
            if (fixedYAxes.has(axisName)) {
                return;
            }
            const xValues = arrayValues(trace.x);
            const valueArrays = trace.type === 'candlestick'
                ? [arrayValues(trace.low), arrayValues(trace.high)]
                : [arrayValues(trace.y)];
            if (!xValues.length || !valueArrays.some((values) => values.length)) {
                return;
            }

            const extent = extents.get(axisName) || {
                min: Infinity,
                max: -Infinity,
                includeZero: false,
            };
            if (trace.type === 'bar') {
                extent.includeZero = true;
            }
            for (let index = 0; index < xValues.length; index += 1) {
                const xValue = dateValue(xValues[index]);
                if (xValue === null || xValue < start || xValue > end) {
                    continue;
                }
                valueArrays.forEach((values) => {
                    const value = numericValue(values[index]);
                    if (value !== null) {
                        extent.min = Math.min(extent.min, value);
                        extent.max = Math.max(extent.max, value);
                    }
                });
            }
            extents.set(axisName, extent);
        });

        const updates = {};
        extents.forEach((extent, axisName) => {
            if (!Number.isFinite(extent.min) || !Number.isFinite(extent.max)) {
                return;
            }
            let lower = extent.includeZero ? Math.min(0, extent.min) : extent.min;
            let upper = extent.includeZero ? Math.max(0, extent.max) : extent.max;
            const span = upper - lower;
            const padding = span > 0
                ? span * 0.07
                : Math.max(Math.abs(upper) * 0.05, 1);
            lower = extent.includeZero && lower === 0 ? 0 : lower - padding;
            upper += padding;
            updates[`${axisName}.range`] = [lower, upper];
            updates[`${axisName}.autorange`] = false;
        });
        return updates;
    }

    function resetAdaptiveYAxes() {
        const updates = {};
        Object.keys(chart._fullLayout).forEach((axisName) => {
            if (/^yaxis\d*$/.test(axisName) && !fixedYAxes.has(axisName)) {
                updates[`${axisName}.autorange`] = true;
            }
        });
        return updates;
    }

    function saveRange(range) {
        try {
            localStorage.setItem(storageKey, JSON.stringify(range));
        } catch (error) {
        }
    }

    function clearSavedRange() {
        try {
            localStorage.removeItem(storageKey);
        } catch (error) {
        }
    }

    function loadSavedRange() {
        try {
            const stored = JSON.parse(localStorage.getItem(storageKey));
            return Array.isArray(stored) && stored.length === 2 ? stored : null;
        } catch (error) {
            return null;
        }
    }

    function masterXAxisName() {
        const axisNames = Object.keys(chart._fullLayout).filter(
            (axisName) => /^xaxis\d*$/.test(axisName),
        );
        return axisNames.find(
            (axisName) => !chart._fullLayout[axisName].matches,
        ) || 'xaxis';
    }

    function applyVisibleRange(range, restoreXAxis) {
        const updates = adaptiveYAxisUpdates(range);
        if (restoreXAxis) {
            const axisName = masterXAxisName();
            updates[`${axisName}.range`] = range;
            updates[`${axisName}.autorange`] = false;
        }
        if (!Object.keys(updates).length) {
            return;
        }
        adaptiveRelayoutPending = true;
        Plotly.relayout(chart, updates).finally(() => {
            adaptiveRelayoutPending = false;
        });
    }

    chart.on('plotly_relayout', (eventData) => {
        if (adaptiveRelayoutPending) {
            return;
        }
        const visibleChange = visibleRangeFromEvent(eventData);
        if (!visibleChange) {
            return;
        }
        window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
            if (visibleChange.reset) {
                clearSavedRange();
                adaptiveRelayoutPending = true;
                Plotly.relayout(chart, resetAdaptiveYAxes()).finally(() => {
                    adaptiveRelayoutPending = false;
                });
                return;
            }
            saveRange(visibleChange.range);
            applyVisibleRange(visibleChange.range, false);
        }, 40);
    });

    const savedRange = loadSavedRange();
    if (savedRange) {
        applyVisibleRange(savedRange, true);
    }
})();
"""


def _normalize_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Return chronologically sorted OHLCV data with a DatetimeIndex."""
    if price_data is None or price_data.empty:
        raise ValueError("K线数据为空。")

    plot_df = price_data.copy()
    if "date" in plot_df.columns:
        plot_df.index = pd.to_datetime(plot_df["date"])
    elif not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.to_datetime(plot_df.index)

    required_columns = {"open", "high", "low", "close"}
    missing_columns = sorted(required_columns - set(plot_df.columns))
    if missing_columns:
        raise ValueError(f"K线数据缺少字段：{', '.join(missing_columns)}")

    return plot_df.sort_index()


def _validate_indicator_columns(
    plot_df: pd.DataFrame,
    settings: CandlestickChartSettings,
    active_panels: Sequence[str],
) -> None:
    """Validate optional columns required by selected volume indicators."""
    volume_required = (
        settings.show_volume
        or "vwap" in settings.overlay_indicators
        or "obv" in active_panels
    )
    if volume_required and "volume" not in plot_df.columns:
        raise ValueError("成交量或所选指标需要 volume 字段。")


def _add_overlay_indicators(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    selected_indicators: Sequence[str],
    params: Mapping[str, Any],
) -> None:
    """Add selected price-scale indicators to the candlestick row."""
    close = plot_df["close"]
    if "ma" in selected_indicators:
        short_window = int(params["ma_short"])
        long_window = int(params["ma_long"])
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=indicators.sma(close, short_window),
                name=f"MA{short_window}",
                line=dict(color="#F39C12", width=1.3),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=indicators.sma(close, long_window),
                name=f"MA{long_window}",
                line=dict(color="#2980B9", width=1.3),
            ),
            row=1,
            col=1,
        )

    if "ema" in selected_indicators:
        short_span = int(params["ema_short"])
        long_span = int(params["ema_long"])
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=indicators.ema(close, short_span),
                name=f"EMA{short_span}",
                line=dict(color="#8E44AD", width=1.3),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=indicators.ema(close, long_span),
                name=f"EMA{long_span}",
                line=dict(color="#16A085", width=1.3),
            ),
            row=1,
            col=1,
        )

    if "bollinger" in selected_indicators:
        upper, middle, lower = indicators.bollinger_bands(
            close,
            window=int(params["bollinger_window"]),
            num_std=float(params["bollinger_std"]),
        )
        for values, name, dash in (
            (upper, "BOLL 上轨", "dash"),
            (middle, "BOLL 中轨", "solid"),
            (lower, "BOLL 下轨", "dash"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=values,
                    name=name,
                    line=dict(color="#7F8C8D", width=1, dash=dash),
                ),
                row=1,
                col=1,
            )

    if "vwap" in selected_indicators:
        vwap_values = indicators.vwap(
            plot_df["high"],
            plot_df["low"],
            close,
            plot_df["volume"],
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=vwap_values,
                name="VWAP",
                line=dict(color="#D35400", width=1.4),
            ),
            row=1,
            col=1,
        )

    if "supertrend" in selected_indicators:
        supertrend_values, direction = indicators.supertrend(
            plot_df["high"],
            plot_df["low"],
            close,
            atr_period=int(params["supertrend_period"]),
            multiplier=float(params["supertrend_multiplier"]),
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=supertrend_values.where(direction > 0),
                name="SuperTrend 多头",
                line=dict(color="red", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=supertrend_values.where(direction < 0),
                name="SuperTrend 空头",
                line=dict(color="green", width=1.5),
            ),
            row=1,
            col=1,
        )


def _add_trade_markers(
    fig: go.Figure,
    trades: Optional[pd.DataFrame],
    symbol: Any,
) -> None:
    """Add buy and sell markers for one symbol when trade data is present."""
    if trades is None or trades.empty:
        return
    required_columns = {"date", "action", "price"}
    if not required_columns.issubset(trades.columns):
        return

    symbol_trades = trades.copy()
    if "symbol" in symbol_trades.columns:
        symbol_trades = symbol_trades[
            symbol_trades["symbol"].astype(str) == str(symbol)
        ]
    if symbol_trades.empty:
        return

    symbol_trades["date"] = pd.to_datetime(symbol_trades["date"])
    actions = symbol_trades["action"].astype(str).str.lower()
    for action, name, marker_symbol, color in (
        ("buy", "买入", "triangle-up", "red"),
        ("sell", "卖出", "triangle-down", "green"),
    ):
        action_trades = symbol_trades[actions == action]
        if action_trades.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=action_trades["date"],
                y=action_trades["price"],
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=11,
                    symbol=marker_symbol,
                    line=dict(color="DarkSlateGrey", width=1),
                ),
                text=[f"{name}: {price:.2f}" for price in action_trades["price"]],
                hoverinfo="text+x",
            ),
            row=1,
            col=1,
        )


def _add_panel_indicator(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    indicator: str,
    row: int,
    params: Mapping[str, Any],
) -> None:
    """Add one selected indicator to its own subplot row."""
    if indicator == "signal":
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["signal"],
                name="策略信号",
                mode="lines",
                line_shape="hv",
                line=dict(color="#2C3E50", width=1.4),
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=1)
        fig.update_yaxes(
            title_text="信号",
            tickvals=[-1, 0, 1],
            range=[-1.2, 1.2],
            row=row,
            col=1,
        )
        return

    if indicator == "rsi":
        rsi_values = indicators.rsi(
            plot_df["close"],
            window=int(params["rsi_window"]),
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=rsi_values,
                name="RSI",
                line=dict(color="#8E44AD", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)
        return

    if indicator == "macd":
        macd_line, signal_line, histogram = indicators.macd(
            plot_df["close"],
            fast_period=int(params["macd_fast"]),
            slow_period=int(params["macd_slow"]),
            signal_period=int(params["macd_signal"]),
        )
        histogram_colors = ["red" if value >= 0 else "green" for value in histogram]
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=macd_line,
                name="DIF",
                line=dict(color="#2980B9", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=signal_line,
                name="DEA",
                line=dict(color="#F39C12", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=histogram,
                name="MACD 柱",
                marker_color=histogram_colors,
                opacity=0.65,
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)
        return

    if indicator == "kdj":
        k_values, d_values, j_values = indicators.kdj(
            plot_df["high"],
            plot_df["low"],
            plot_df["close"],
            n=int(params["kdj_window"]),
            m1=int(params["kdj_k_smooth"]),
            m2=int(params["kdj_d_smooth"]),
        )
        for values, name, color in (
            (k_values, "K", "#2980B9"),
            (d_values, "D", "#F39C12"),
            (j_values, "J", "#8E44AD"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=values,
                    name=name,
                    line=dict(color=color, width=1.2),
                ),
                row=row,
                col=1,
            )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1)
        fig.update_yaxes(title_text="KDJ", row=row, col=1)
        return

    if indicator == "atr":
        atr_values = indicators.atr(
            plot_df["high"],
            plot_df["low"],
            plot_df["close"],
            window=int(params["atr_window"]),
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=atr_values,
                name="ATR",
                line=dict(color="#16A085", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="ATR", row=row, col=1)
        return

    if indicator == "obv":
        obv_values = indicators.obv(plot_df["close"], plot_df["volume"])
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=obv_values,
                name="OBV",
                line=dict(color="#D35400", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="OBV", row=row, col=1)
