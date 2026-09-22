"""当日选股页：最新已发布交易日的 top_k 名单。

运行方式：streamlit run research/market_regime/app_stock_selection_live.py

与 ``app_stock_selection.py``（回看已完成窗口的净值）分开，因为数据来源不同：本页
只读 ``daily_selection_2026`` 生成的当期产物，不重跑任何链路、不抓行情、不打分。

**本页的核心事实：这批选股没有已实现收益。** 标签是入场后未来 10 个交易日的超额，
最新可打分的交易日按定义就缺这 10 天的未来数据，所以它**不可能**有已实现数字。任何
出现在这里的收益都会是个 bug。信号日 T 出名单，生产链路在 T+1 成交——所以每行都带
一个 ``entry_date``。

**这不是实盘界面。** 无行情连接、无下单通道、无账户。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.market_regime.stock_selection import (  # noqa: E402
    POOLS,
    daily_pool_entry,
    load_daily_selection,
)


if hasattr(st, "cache_data"):
    cache_data = st.cache_data(show_spinner=False)
else:
    cache_data = st.cache


@cache_data
def cached_daily():
    return load_daily_selection()


def as_percent(frame, columns):
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = (pd.to_numeric(out[column], errors="coerce") * 100).round(2)
    return out


def picks_display(picks):
    """名单的可读版本：中文名靠前，收益换百分数。"""

    display = as_percent(picks, ["entry_score", "predicted_excess", "predicted_mae"])
    wanted = ["rank", "symbol", "name", "signal_close", "entry_score",
              "predicted_excess", "predicted_mae", "entry_date"]
    rename = {
        "rank": "排序", "symbol": "代码", "name": "名称", "signal_close": "信号日收盘",
        "entry_score": "打分", "predicted_excess": "预测超额%",
        "predicted_mae": "预测回撤%", "entry_date": "拟成交日",
    }
    display = display.loc[:, [c for c in wanted if c in display.columns]]
    return display.rename(columns=rename)


def build_score_figure(picks, pool_label):
    """按打分排序的横向条形，正负分色。"""

    frame = picks.sort_values("entry_score")
    labels = frame["name"].where(frame["name"].notna(), frame["symbol"])
    colors = ["#d62728" if value >= 0 else "#2ca02c" for value in frame["entry_score"]]
    figure = go.Figure(go.Bar(
        x=frame["entry_score"], y=labels, orientation="h", marker_color=colors))
    figure.add_vline(x=0.0, line_color="#888888", line_width=1)
    figure.update_layout(
        height=max(300, 22 * len(frame) + 120), margin=dict(l=10, r=10, t=40, b=10),
        title="{} 当日打分（entry_score）".format(pool_label),
        xaxis_title="entry_score = 预测超额 − risk_penalty × max(−预测回撤, 0)",
        yaxis_title=None, showlegend=False)
    return figure


def render_freshness(manifest):
    session = manifest.get("session", "—")
    entry = manifest.get("entry_date") or "未解析（trade_cal 不可达）"
    price_session = manifest.get("latest_price_session", "—")
    columns = st.columns(4)
    columns[0].metric("信号日（T）", session)
    columns[1].metric("拟成交日（T+1）", entry)
    columns[2].metric("行情数据截至", price_session)
    columns[3].metric("生成时间", str(manifest.get("generated_at", "—")).replace("T", " "))

    if manifest.get("stale"):
        st.warning(
            "**这份名单不是最新的。** 行情已到 {}，但打分表止于 {}——中间的重链路"
            "（拼价 → 特征 → 打分）还没跑。要最新名单，请重新运行：\n\n"
            "```bash\npython -m research.experiments.daily_selection_2026 --refresh\n```\n\n"
            "实测耗时约 1 小时（2026-09-18 那次 58 分钟），几乎全在样本重建："
            "两池共 1300 只标的按 150 一批、每批 6~9 分钟。这不是可省的开销——"
            "`vwap_distance` / `obv` 是自上市以来累积的量，单日增量打分与已发布链路会有漂移。"
            .format(price_session, session))
    else:
        st.caption(
            "打分表与行情树同步（均止于 {}）。信号日为 T，生产链路在 T+1 成交，"
            "因此上面的「拟成交日」才是可执行日。".format(session))


def render_picks(pool, picks, entry, manifest):
    pool_label = POOLS[pool]["index_name"]
    block = picks.loc[picks["pool"] == pool] if "pool" in picks.columns else picks
    if block.empty:
        st.info("该池没有当日选股。")
        return

    st.subheader("{} · 当日名单".format(pool_label))
    st.caption(
        "横截面 top_k 等权。**这是信号口径**：不含成本、仓位上限与持有连续性，"
        "与引擎实际持仓不是一回事（后者见回测页）。当日全池参与打分的标的数 {}。".format(
            entry.get("cross_section", "—")))
    st.dataframe(picks_display(block), width="stretch", hide_index=True)

    if "entry_score" in block.columns:
        st.plotly_chart(build_score_figure(block, pool_label), width="stretch")

    missed = block["signal_close"].isna().sum() if "signal_close" in block.columns else 0
    if missed:
        st.caption("{} 只标的在本地行情树里没有信号日收盘价，收盘留空。".format(int(missed)))

    st.download_button(
        "下载该池名单 CSV",
        block.to_csv(index=False).encode("utf-8-sig"),
        "daily_selection_{}.csv".format(pool), "text/csv")


def render_notes(manifest):
    st.subheader("这批数字是什么、不是什么")
    st.markdown(
        "- **没有已实现收益，而且不可能有。** 标签是入场后未来 {} 个交易日的超额，"
        "最新交易日按定义缺这段未来数据。本页只给预测。\n"
        "- **`entry_score` 的含义**：`预测超额 − risk_penalty × max(−预测回撤, 0)`，"
        "即对下行风险的惩罚从预测超额里扣掉。它不是预期收益的校准估计。\n"
        "- **名单是信号日的，不是今天的。** 信号日 {}，拟成交日 {}。"
        "停牌、涨跌停、资金不足都会让 T+1 实际买不进去——本页不做可执行性判断。\n"
        "- **模型是冻结的**：不重训、不重选 `top_k`（来自已发布 ranking manifest）。"
        "名单随行情更新，模型不随行情更新。\n"
        "- **这不是实盘界面**：无行情连接、无下单通道、无账户。".format(
            manifest.get("horizon", 10), manifest.get("session", "—"),
            manifest.get("entry_date") or "—"))
    st.caption(manifest.get("label_note", ""))
    with st.expander("重新生成"):
        st.code("python -m research.experiments.daily_selection_2026 --refresh",
                language="bash")
        st.caption("不加 `--refresh` 只用现有打分表出名单；stale 时页面会显式告警。")


def main():
    st.set_page_config(page_title="当日选股", layout="wide")
    payload = cached_daily()
    picks, manifest = payload["picks"], payload["manifest"]

    st.title("当日选股")
    if picks.empty or not manifest:
        st.warning(
            "还没有当日选股产物。先运行：\n\n"
            "```bash\npython -m research.experiments.daily_selection_2026 --refresh\n```")
        return

    st.caption("最新已发布交易日 · 冻结模型 · **纸面研究，不涉及实盘**")
    render_freshness(manifest)

    available = [pool for pool in POOLS if pool in set(picks.get("pool", []))]
    if not available:
        st.info("产物里没有已知股票池。")
        return
    pool = st.radio(
        "股票池", options=available, key="live_pool",
        format_func=lambda key: "{}（{}）".format(POOLS[key]["index_name"], key),
        horizontal=True)

    render_picks(pool, picks, daily_pool_entry(manifest, pool), manifest)
    st.divider()
    render_notes(manifest)


if __name__ == "__main__":
    main()
