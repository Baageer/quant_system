import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ======================
# 页面配置
# ======================
st.set_page_config(layout="wide")
st.title("📊 可视化 Demo")

# ======================
# 侧边栏参数
# ======================
st.sidebar.header("⚙️ 参数设置")

symbol = st.sidebar.text_input("股票代码", "AAPL")
start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("2024-01-01"))

short_window = st.sidebar.slider("短期均线", 5, 50, 10)
long_window = st.sidebar.slider("长期均线", 20, 200, 50)

initial_cash = st.sidebar.number_input("初始资金", 1000, 1000000, 100000)

# ======================
# 数据获取
# ======================
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df.reset_index()
    return df

df = load_data(symbol, start_date, end_date)

if df.empty:
    st.warning("数据为空，请检查股票代码")
    st.stop()

# ======================
# 技术指标
# ======================
df['MA_short'] = df['Close'].rolling(short_window).mean()
df['MA_long'] = df['Close'].rolling(long_window).mean()

# ======================
# 交易信号（均线策略）
# ======================
df['signal'] = 0
df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1

df['position'] = df['signal'].diff()

# 买卖点
buy_signals = df[df['position'] == 2]
sell_signals = df[df['position'] == -2]

# ======================
# K线图
# ======================
st.subheader("📉 K线图 + 买卖点")

fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="K线"
))

# 均线
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_short'], name='MA短期'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_long'], name='MA长期'))

# 买点
fig.add_trace(go.Scatter(
    x=buy_signals['Date'],
    y=buy_signals['Close'],
    mode='markers',
    name='买点',
    marker=dict(symbol='triangle-up', size=10)
))

# 卖点
fig.add_trace(go.Scatter(
    x=sell_signals['Date'],
    y=sell_signals['Close'],
    mode='markers',
    name='卖点',
    marker=dict(symbol='triangle-down', size=10)
))

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# ======================
# 回测逻辑
# ======================
df['returns'] = df['Close'].pct_change()

df['strategy_returns'] = df['returns'] * df['signal'].shift(1)

df['nav'] = (1 + df['strategy_returns']).cumprod()
df['benchmark'] = (1 + df['returns']).cumprod()

# ======================
# 净值曲线
# ======================
st.subheader("📈 回测净值曲线")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df['Date'],
    y=df['nav'],
    name='策略净值'
))

fig2.add_trace(go.Scatter(
    x=df['Date'],
    y=df['benchmark'],
    name='基准（买入持有）'
))

fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# ======================
# 绩效指标
# ======================
st.subheader("📊 策略绩效")

total_return = df['nav'].iloc[-1] - 1
benchmark_return = df['benchmark'].iloc[-1] - 1

sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252)

max_drawdown = (df['nav'] / df['nav'].cummax() - 1).min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("策略收益", f"{total_return:.2%}")
col2.metric("基准收益", f"{benchmark_return:.2%}")
col3.metric("夏普比率", f"{sharpe:.2f}")
col4.metric("最大回撤", f"{max_drawdown:.2%}")

# ======================
# 数据表
# ======================
with st.expander("📋 查看数据"):
    st.dataframe(df.tail(100))