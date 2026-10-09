"""选股结果查看页：2026-08-01 → 2026-09-15，冻结模型，两个池。

运行方式：streamlit run research/market_regime/app_stock_selection.py

页面只读 ``build_stock_selection_2026`` 的产物，不重新打分、不重跑回测、不重选
``top_k``。因此这里看到的每一行都是已发布链路的结果，而不是界面自己算出来的。

两个口径分开展示，因为它们的收益数字**不可互换**：

- **信号口径** —— 每个调仓日横截面 top_k 等权，收益是模型的目标标签（相对基准的
  已实现 10 日超额），不含仓位上限、持有连续性和成本。
- **组合口径** —— 引擎实际持仓，扣成本后的净值。它的窗口被刻意延长到 2026-09-16
  （前向发布版本因标签缺失停在 2026-09-01），所以必须带 ``extension_check``；
  该检查不通过时本页面拒绝展示组合数字。

**这不是实盘界面。** 页面不连行情、不下单、不含任何交易通道。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.build_forward_2026_samples import load_benchmark  # noqa: E402
from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR  # noqa: E402
from research.market_regime.stock_selection import (  # noqa: E402
    FORWARD_ROOT,
    POOLS,
    book_as_of,
    book_with_values,
    extension_trustworthy,
    load_selection_artifacts,
    load_selection_manifest,
    load_symbol_names,
    manifest_pool,
    price_window,
)


FORWARD_PRICE_DIR = Path(FORWARD_ROOT) / "price_history"
PRICE_DIRS = [DEFAULT_PRICE_HISTORY_DIR, FORWARD_PRICE_DIR]

#: 选股区间只有 32 个交易日，K 线要往前多给一段才看得出位置。
CHART_SESSIONS = 120


if hasattr(st, "cache_data"):
    cache_data = st.cache_data(show_spinner=False)
else:
    cache_data = st.cache


@cache_data
def cached_names():
    return load_symbol_names()


@cache_data
def cached_artifacts(pool):
    return load_selection_artifacts(pool, None, cached_names())


@cache_data
def cached_manifest():
    return load_selection_manifest()


@cache_data
def cached_benchmark(pool):
    return load_benchmark(pool)


@cache_data
def cached_prices(symbol, end):
    return price_window(symbol, tuple(PRICE_DIRS), end, CHART_SESSIONS)


def as_percent(frame, columns):
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = (pd.to_numeric(out[column], errors="coerce") * 100).round(2)
    return out


def signal_display(picks, names=None):
    """选股明细的可读版本：中文名靠前，收益换成百分数。"""

    if picks.empty:
        return picks
    display = as_percent(picks, ["entry_score", "realized_excess", "realized_mae",
                                 "predicted_excess", "predicted_mae"])
    rename = {
        "date": "调仓日", "rank": "排序", "symbol": "代码", "name": "名称",
        "entry_score": "打分", "predicted_excess": "预测超额%",
        "predicted_mae": "预测回撤%", "realized_excess": "已实现超额%",
        "realized_mae": "已实现回撤%", "label_available": "已实现",
    }
    wanted = ["date", "rank", "symbol", "name", "entry_score", "predicted_excess",
              "predicted_mae", "realized_excess", "realized_mae", "label_available"]
    display = display.loc[:, [c for c in wanted if c in display.columns]]
    return display.rename(columns=rename)


def summary_display(summary):
    if summary.empty:
        return summary
    display = as_percent(summary, ["basket_excess", "mean_predicted", "pool_excess", "edge"])
    rename = {
        "date": "调仓日", "basket_excess": "篮子超额%", "graded_symbols": "已实现只数",
        "basket_symbols": "篮内只数", "mean_predicted": "平均预测%",
        "pool_excess": "池内均值%", "pool_symbols": "池内只数", "edge": "选股增量%",
        "label_available": "标签已出",
    }
    return display.rename(columns=rename)


def build_equity_figure(equity, benchmark, start, end, pool_label):
    """组合净值 vs 基准，双双归一到区间前最后一个交易日。"""

    value = equity.loc[:, "portfolio_value"].astype(float)
    bench = benchmark.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.loc[bench.index <= pd.Timestamp(end)]

    prior = value.loc[value.index <= pd.Timestamp(start) - pd.Timedelta(days=1)]
    base = float(prior.iloc[-1]) if len(prior) else float(value.iloc[0])
    bench_prior = bench.loc[bench.index <= pd.Timestamp(start) - pd.Timedelta(days=1)]
    bench_base = float(bench_prior.iloc[-1]) if len(bench_prior) else float(bench.iloc[0])

    window = value.loc[(value.index >= pd.Timestamp(start)) & (value.index <= pd.Timestamp(end))]
    bench_window = bench.loc[(bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=window.index, y=(window / base - 1.0) * 100, name="模型组合",
        mode="lines", line=dict(color="#d62728", width=2)))
    figure.add_trace(go.Scatter(
        x=bench_window.index, y=(bench_window / bench_base - 1.0) * 100,
        name="{}".format(pool_label), mode="lines",
        line=dict(color="#7f7f7f", width=1.6, dash="dash")))
    figure.add_hline(y=0.0, line_color="#cccccc", line_width=1)
    figure.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="区间收益（%）", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0))
    return figure


def build_edge_figure(summary):
    """每个调仓日的选股增量：篮子超额 − 池内均值。"""

    graded = summary.dropna(subset=["edge"]) if "edge" in summary.columns else summary
    figure = go.Figure()
    if not graded.empty:
        colors = ["#d62728" if v >= 0 else "#2ca02c" for v in graded["edge"]]
        figure.add_trace(go.Bar(
            x=graded["date"], y=graded["edge"] * 100,
            marker_color=colors, name="选股增量"))
    figure.add_hline(y=0.0, line_color="#888888", line_width=1)
    figure.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="选股增量（%）", xaxis_title=None, showlegend=False)
    return figure


def build_weight_figure(book):
    if book.empty:
        return go.Figure()
    frame = book.dropna(subset=["weight"])
    labels = frame["name"].where(frame["name"].notna(), frame["symbol"])
    figure = go.Figure(go.Bar(
        x=frame["weight"] * 100, y=labels, orientation="h",
        marker_color="#1f77b4"))
    figure.update_layout(
        height=max(260, 24 * len(frame)), margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="占组合比重（%）", yaxis_title=None, showlegend=False)
    return figure


def build_price_figure(prices, symbol, name, picks_on_symbol):
    """个股 K 线，并在模型选中它的调仓日打标记。"""

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                           row_heights=[0.72, 0.28],
                           subplot_titles=("{} {}".format(symbol, name or ""), "成交量"))
    figure.add_trace(go.Candlestick(
        x=prices["date"], open=prices["open"], high=prices["high"],
        low=prices["low"], close=prices["close"], name="K 线",
        increasing_line_color="#d62728", decreasing_line_color="#2ca02c",
        increasing_fillcolor="#d62728", decreasing_fillcolor="#2ca02c",
        line=dict(width=1)), row=1, col=1)
    if "volume" in prices.columns:
        figure.add_trace(go.Bar(
            x=prices["date"], y=prices["volume"], name="成交量",
            marker_color="#bbbbbb"), row=2, col=1)
    if not picks_on_symbol.empty:
        merged = prices.merge(
            picks_on_symbol.loc[:, ["date", "realized_excess"]], on="date", how="inner")
        figure.add_trace(go.Scatter(
            x=merged["date"], y=merged["low"] * 0.985, mode="markers",
            name="模型选中", marker=dict(symbol="triangle-up", size=11,
                                        color="#ff7f0e")), row=1, col=1)
    figure.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, showlegend=False)
    return figure


def render_controls(manifest):
    pool_label = st.radio(
        "股票池", options=list(POOLS), key="pool",
        format_func=lambda key: "{}（{}）".format(POOLS[key]["index_name"], key),
        horizontal=True)
    window_start = pd.Timestamp(manifest.get("window_start", "2026-08-01"))
    window_end = pd.Timestamp(manifest.get("window_end", "2026-09-15"))
    return pool_label, window_start, window_end


def render_header(manifest, entry, pool, window_start, window_end):
    pools = POOLS[pool]
    st.title("选股结果 · {}".format(pools["index_name"]))
    st.caption(
        "区间 {} → {} ｜ 冻结模型 `{}` ｜ top_k {:.0%} ｜ 周频调仓 ｜ **纸面研究，不涉及实盘**".format(
            window_start.date(), window_end.date(), pools["model_id"], float(entry.get("top_k", 0))))

    if not entry:
        st.warning("未找到该池的选股产物。先运行 `python -m research.experiments.build_stock_selection_2026`。")
        return False

    if not extension_trustworthy(entry):
        st.error(
            "组合口径的**扩展一致性检查未通过**（extension_check = {}）：延长后的净值不是已发布的那本账，"
            "下面的组合收益不可引用。信号口径不受影响。".format(
                (entry.get("extension_check") or {}).get("status", "缺失")))
    else:
        check = entry["extension_check"]
        st.caption(
            "组合口径的扩展一致性检查 **PASS**：与已发布前向账本在重叠的 {} 个交易日上最大权益差 "
            "{:.1e}（相对 {:.1e}），即延长的只是报告窗口，不是策略本身。".format(
                check["overlap_sessions"], check["max_abs_equity_delta"],
                check["max_rel_equity_delta"]))
    return True


def render_metrics(entry, summary, picks):
    portfolio = entry.get("portfolio", {}) or {}
    signal = entry.get("signal", {}) or {}
    columns = st.columns(4)
    columns[0].metric(
        "组合区间收益", "{:+.2%}".format(portfolio.get("window_return", float("nan"))),
        help="引擎实际持仓，扣成本，基点 = 区间前最后一个交易日（{}）".format(
            portfolio.get("base_date", "—")))
    columns[1].metric(
        "基准同期", "{:+.2%}".format(portfolio.get("benchmark_return", float("nan"))))
    columns[2].metric(
        "组合超额", "{:+.2%}".format(portfolio.get("window_excess_vs_benchmark", float("nan"))))
    columns[3].metric(
        "组合最大回撤", "{:.2%}".format(portfolio.get("window_max_drawdown", float("nan"))))

    second = st.columns(4)
    graded = summary.dropna(subset=["edge"]) if "edge" in summary.columns else summary
    second[0].metric(
        "选股增量均值", "{:+.2%}".format(graded["edge"].mean() if len(graded) else float("nan")),
        help="每期（篮子等权超额 − 池内等权超额）的均值，仅计标签已出的调仓日")
    second[1].metric(
        "信号已实现只数", "{} / {}".format(signal.get("graded", 0), signal.get("picks", 0)),
        help="未被计入的是仍然持仓、10 日标签尚未发布的选股")
    second[2].metric("调仓次数", str(len(signal.get("rebalances", []))))
    second[3].metric(
        "持仓上限", "{} 只".format(entry.get("execution", {}).get("max_positions", "—")))


def render_overview(summary):
    st.subheader("信号口径：每个调仓日，top_k 篮子 vs 整个池")
    st.caption(
        "收益为模型的目标标签——相对基准的已实现 10 日超额，等权、不含成本与仓位限制。"
        "**label_available 为 False 的调仓日仍持有中**，10 日窗口尚未走完，因此没有已实现数字；"
        "它们被保留在表里而不是删掉。")
    if summary.empty:
        st.info("没有信号汇总表。")
        return
    st.dataframe(summary_display(summary), width="stretch", hide_index=True)
    st.plotly_chart(build_edge_figure(summary), width="stretch")
    st.download_button(
        "下载信号汇总 CSV",
        summary.to_csv(index=False).encode("utf-8-sig"),
        "stock_selection_signal_summary.csv", "text/csv")


def render_picks(picks):
    st.subheader("选股明细：每期篮子的逐个标的")
    if picks.empty:
        st.info("没有选股明细表。")
        return
    dates = sorted(picks["date"].dropna().unique())
    labels = [pd.Timestamp(d) for d in dates]
    chosen = st.selectbox(
        "调仓日", labels, index=len(labels) - 1, key="rebalance_date",
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    block = picks.loc[picks["date"] == pd.Timestamp(chosen)].copy()
    st.dataframe(signal_display(block), width="stretch", hide_index=True)

    open_count = int((~block["label_available"]).sum())
    if open_count:
        st.warning(
            "该期 {} 只标的的 10 日窗口尚未走完（调仓日 {} 加 10 个交易日已超过最新行情），"
            "已实现超额为空是**持有中**，不是缺失。".format(open_count, chosen.strftime("%Y-%m-%d")))
    st.download_button(
        "下载全部选股 CSV",
        picks.to_csv(index=False).encode("utf-8-sig"),
        "stock_selection_signals.csv", "text/csv")


def render_equity(entry, equity, pool, window_start, window_end):
    st.subheader("组合口径：扣成本净值 vs 基准")
    if equity.empty:
        st.info("没有组合净值表。")
        return
    if not extension_trustworthy(entry):
        st.error("扩展一致性检查未通过，不展示组合净值。")
        return
    benchmark = cached_benchmark(pool)
    st.plotly_chart(
        build_equity_figure(equity, benchmark, window_start, window_end,
                            POOLS[pool]["index_name"]),
        width="stretch")
    st.caption(
        "两条线都归一到区间前最后一个交易日（{}）。基点为 50 万初始资金，"
        "引擎按周频调仓、单票不超过 2.5 万、最少持有 10 个交易日，已计佣金/滑点/印花税。".format(
            (entry.get("portfolio") or {}).get("base_date", "—")))
    detail = equity.reset_index().loc[:, ["date", "portfolio_value", "cash"]].copy()
    for column in ("portfolio_value", "cash"):
        detail[column] = pd.to_numeric(detail[column], errors="coerce").round(2)
    st.dataframe(
        detail.rename(columns={"date": "日期", "portfolio_value": "组合权益", "cash": "现金"}),
        width="stretch", hide_index=True)
    st.download_button(
        "下载组合净值 CSV",
        equity.to_csv().encode("utf-8-sig"),
        "stock_selection_portfolio_equity.csv", "text/csv")


def render_book(entry, equity, pool, window_end):
    st.subheader("截至 {} 的持仓".format(window_end.date()))
    if equity.empty:
        st.info("没有组合净值表。")
        return
    if not extension_trustworthy(entry):
        st.error("扩展一致性检查未通过，不展示持仓。")
        return

    book = book_as_of(equity, window_end)
    if book.empty:
        st.info("区间结束时组合为空仓。")
        return

    closes = {}
    for symbol in book["symbol"]:
        try:
            frame = price_window(symbol, tuple(PRICE_DIRS), window_end, 1)
        except (FileNotFoundError, ValueError):
            continue
        if len(frame):
            closes[symbol] = float(frame["close"].iloc[-1])

    last = equity.loc[equity.index <= pd.Timestamp(window_end)].iloc[-1]
    portfolio_value = float(last["portfolio_value"])
    cash = float(last["cash"])
    valued = book_with_values(book, closes, portfolio_value, cached_names())
    if portfolio_value:
        st.caption(
            "持仓 {:.1%}，现金 {:.1%}。市价用 {} 收盘；行情缓存里没有的标的保留行、市值留空。".format(
                1.0 - cash / portfolio_value, cash / portfolio_value, window_end.date()))

    display = valued.copy()
    display["weight"] = (display["weight"] * 100).round(2)
    display = display.loc[:, ["symbol", "name", "shares", "close", "market_value", "weight"]]
    st.dataframe(
        display.rename(columns={"symbol": "代码", "name": "名称", "shares": "股数",
                                "close": "收盘", "market_value": "市值", "weight": "占比%"}),
        width="stretch", hide_index=True)
    st.plotly_chart(build_weight_figure(valued), width="stretch")
    st.download_button(
        "下载持仓 CSV",
        valued.to_csv(index=False).encode("utf-8-sig"),
        "stock_selection_portfolio_holdings.csv", "text/csv")


def render_symbol_chart(pool, picks, window_end):
    st.subheader("个股行情与选中时点")
    if picks.empty:
        st.info("没有选股明细表。")
        return
    names = cached_names()
    symbols = sorted(picks["symbol"].unique())
    # Keyed by pool: a symbol picked in one pool is not necessarily in the other,
    # and a stale cross-pool value would otherwise be carried into a new option
    # list.
    chosen = st.selectbox(
        "标的", symbols, key="symbol_{}".format(pool),
        format_func=lambda s: "{} {}".format(s, names.get(s, "")).strip())
    on_symbol = picks.loc[picks["symbol"] == chosen]
    try:
        prices = price_window(chosen, tuple(PRICE_DIRS), window_end, CHART_SESSIONS)
    except (FileNotFoundError, ValueError) as exc:
        st.warning("读不到 {} 的本地行情：{}".format(chosen, exc))
        return
    st.plotly_chart(
        build_price_figure(prices, chosen, names.get(chosen, ""), on_symbol),
        width="stretch")
    st.caption(
        "橙色三角为模型在该调仓日把这只股票选进 top_k 的位置（按信号日收盘显示，"
        "实际成交在下一个交易日）。K 线为本地缓存口径，未含分红再投。")
    table = on_symbol.loc[:, [c for c in ("date", "rank", "entry_score", "predicted_excess",
                                          "realized_excess", "label_available")
                              if c in on_symbol.columns]]
    st.dataframe(signal_display(table), width="stretch", hide_index=True)


def render_notes():
    st.subheader("这个页面没说什么")
    st.markdown(
        "- **不是实盘界面。** 没有行情连接、没有下单通道、没有账户。\n"
        "- **组合收益来自扩展窗口。** 已发布的前向回测因 10 日标签缺失停在 2026-09-01；"
        "本页把引擎喂了不带标签的打分帧，让它跑到 2026-09-16。一致性检查证明重叠部分逐位相同，"
        "但 2026-09-01 之后的那段**没有**独立对照。\n"
        "- **信号收益与组合收益不可互换。** 前者是标签本身（无成本、无仓位限制、无限额），"
        "后者是引擎扣成本后的净值。两者差多少，就是这个执行链的代价。\n"
        "- **区间太短。** 32 个交易日、7 次调仓，单期结果无法与「一次好抽样」区分。"
        "这里只报告，不据此下结论。\n"
        "- **未做**：随机零假设、常仓位对照、前向窗口内的任何参数重选。前两项见 "
        "`docs/research/04_validation_and_diagnostics/forward_paper_tracking_2026.md`。")
    with st.expander("重新生成这些产物"):
        st.code("python -m research.experiments.fetch_tushare_stock_basic\n"
                "python -m research.experiments.build_stock_selection_2026 --pools hs300,zz1000",
                language="bash")
        st.caption("产物落在 `{}` 与 `data/raw/tushare/forward_2026/stock_selection_manifest.json`。"
                   "生成器在扩展一致性检查不通过时返回 2 并拒绝写 manifest。".format(
                       "D:\\temp\\quant_output\\datasets"))


def main():
    st.set_page_config(page_title="选股结果 2026", layout="wide")
    manifest = cached_manifest()

    if not manifest:
        st.title("选股结果")
        st.warning("未找到 `stock_selection_manifest.json`。先运行 "
                   "`python -m research.experiments.build_stock_selection_2026`。")
        return

    pool, window_start, window_end = render_controls(manifest)
    entry = manifest_pool(manifest, pool)
    if not render_header(manifest, entry, pool, window_start, window_end):
        return

    artifacts = cached_artifacts(pool)
    picks, summary, equity = artifacts["signals"], artifacts["summary"], artifacts["equity"]

    render_metrics(entry, summary, picks)

    overview_tab, picks_tab, equity_tab, book_tab, symbol_tab, notes_tab = st.tabs(
        ["信号汇总", "选股明细", "组合净值", "当前持仓", "个股行情", "口径与边界"])
    with overview_tab:
        render_overview(summary)
    with picks_tab:
        render_picks(picks)
    with equity_tab:
        render_equity(entry, equity, pool, window_start, window_end)
    with book_tab:
        render_book(entry, equity, pool, window_end)
    with symbol_tab:
        render_symbol_chart(pool, picks, window_end)
    with notes_tab:
        render_notes()


if __name__ == "__main__":
    main()
