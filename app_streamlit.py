import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_api import DataAPI
from signals import indicators

st.set_page_config(layout="wide")
st.title("📊 量化分析系统")

st.sidebar.header("⚙️ 参数设置")

st.sidebar.subheader("数据源设置")
source = st.sidebar.selectbox("数据源", ["akshare", "tushare"], index=0)
uploaded_file = st.sidebar.file_uploader("上传股票列表文件", type=["txt", "csv"])

if uploaded_file is not None:
    stock_file_content = uploaded_file.read().decode('utf-8')
    stock_list = [line.strip() for line in stock_file_content.split('\n') if line.strip()]
    if source == 'tushare':
        stock_list = [s + ".SH" if s[0] == '6' else s + ".SZ" for s in stock_list]
else:
    stock_list = ["000001"]

data_api = DataAPI(source=source, stock_file=".test1.txt",cache_dir="./data/raw",processed_dir="./data/processed")


symbol = st.sidebar.selectbox("股票代码", stock_list if stock_list else ["000001"])
start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("2024-12-31"))

st.sidebar.subheader("均线参数")
short_window = st.sidebar.slider("短期均线", 5, 50, 10)
long_window = st.sidebar.slider("长期均线", 20, 200, 50)

st.sidebar.subheader("技术指标参数")
show_ma = st.sidebar.checkbox("显示均线", value=True)
show_bb = st.sidebar.checkbox("显示布林带", value=False)
show_rsi = st.sidebar.checkbox("显示RSI", value=True)
show_macd = st.sidebar.checkbox("显示MACD", value=True)
show_kdj = st.sidebar.checkbox("显示KDJ", value=True)

bb_window = st.sidebar.number_input("布林带周期", 5, 50, 20)
bb_std = st.sidebar.number_input("布林带标准差倍数", 1.0, 3.0, 2.0, 0.1)
rsi_window = st.sidebar.number_input("RSI周期", 5, 30, 14)
macd_fast = st.sidebar.number_input("MACD快线", 5, 20, 12)
macd_slow = st.sidebar.number_input("MACD慢线", 15, 40, 26)
macd_signal = st.sidebar.number_input("MACD信号线", 5, 15, 9)
kdj_n = st.sidebar.number_input("KDJ周期", 5, 20, 9)

initial_cash = st.sidebar.number_input("初始资金", 1000, 1000000, 100000)

@st.cache_data
def load_data(symbol, start, end):
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    df = data_api.get_price_history_data(symbol, start_str, end_str)
    return df

df = load_data(symbol, start_date, end_date)
if len(df.columns) == 12:
    df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 
              'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
elif len(df.columns) == 13:
    df.columns = ['index', 'date', 'code', 'open', 'close', 'high', 'low', 
              'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']

if df is None or df.empty:
    st.warning("数据为空，请检查股票代码或日期范围")
    st.stop()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
elif df.index.name != 'date':
    df.index = pd.to_datetime(df.index)

df = df.sort_index()


if show_ma:
    df['MA_short'] = indicators.sma(df['close'], short_window)
    df['MA_long'] = indicators.sma(df['close'], long_window)

if show_bb:
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = indicators.bollinger_bands(
        df['close'], window=bb_window, num_std=bb_std
    )

if show_rsi:
    df['RSI'] = indicators.rsi(df['close'], window=rsi_window)

if show_macd:
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = indicators.macd(
        df['close'], fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
    )

if show_kdj:
    df['K'], df['D'], df['J'] = indicators.kdj(
        df['high'], df['low'], df['close'], n=kdj_n
    )

num_rows = 2
if show_rsi:
    num_rows += 1
if show_macd:
    num_rows += 1
if show_kdj:
    num_rows += 1

row_heights = [0.5, 0.15]
if show_rsi:
    row_heights.append(0.15)
if show_macd:
    row_heights.append(0.15)
if show_kdj:
    row_heights.append(0.15)

subplot_titles = ['K线图', '成交量']
if show_rsi:
    subplot_titles.append('RSI')
if show_macd:
    subplot_titles.append('MACD')
if show_kdj:
    subplot_titles.append('KDJ')

fig = make_subplots(
    rows=num_rows, 
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=subplot_titles
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="K线",
        increasing_line_color='red',
        decreasing_line_color='green'
    ),
    row=1, col=1
)

if show_ma:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_short'], name=f'MA{short_window}', 
                   line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_long'], name=f'MA{long_window}',
                   line=dict(color='blue', width=1)),
        row=1, col=1
    )

if show_bb:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_upper'], name='BB上轨',
                   line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_middle'], name='BB中轨',
                   line=dict(color='gray', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_lower'], name='BB下轨',
                   line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )

colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green' 
          for i in range(len(df))]
fig.add_trace(
    go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color=colors, opacity=0.7),
    row=2, col=1
)

current_row = 3

if show_rsi:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)),
        row=current_row, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)
    current_row += 1

if show_macd:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)),
        row=current_row, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='orange', width=1)),
        row=current_row, col=1
    )
    colors_macd = ['red' if v >= 0 else 'green' for v in df['MACD_hist'].fillna(0)]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram', marker_color=colors_macd, opacity=0.7),
        row=current_row, col=1
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=current_row, col=1)
    current_row += 1

if show_kdj:
    fig.add_trace(
        go.Scatter(x=df.index, y=df['K'], name='K', line=dict(color='blue', width=1)),
        row=current_row, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['D'], name='D', line=dict(color='orange', width=1)),
        row=current_row, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['J'], name='J', line=dict(color='purple', width=1)),
        row=current_row, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)

fig.update_layout(
    height=200 + num_rows * 150,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    dragmode='zoom'
)

fig.update_xaxes(title_text="日期", row=num_rows, col=1, rangeslider_visible=False)

fig.update_yaxes(title_text="价格", row=1, col=1, autorange=True, fixedrange=False)
fig.update_yaxes(title_text="成交量", row=2, col=1, autorange=True, fixedrange=False)

current_row = 3
if show_rsi:
    fig.update_yaxes(title_text="RSI", row=current_row, col=1, autorange=True, fixedrange=False)
    current_row += 1
if show_macd:
    fig.update_yaxes(title_text="MACD", row=current_row, col=1, autorange=True, fixedrange=False)
    current_row += 1
if show_kdj:
    fig.update_yaxes(title_text="KDJ", row=current_row, col=1, autorange=True, fixedrange=False)

st.subheader("📉 K线图与技术指标")

chart_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <div id="chart" style="width:100%;height:{200 + num_rows * 150}px;"></div>
    <script>
        var graphData = {fig.to_json()};
        
        Plotly.newPlot('chart', graphData.data, graphData.layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape']
        }});
        
        var chartDiv = document.getElementById('chart');
        
        chartDiv.on('plotly_relayout', function(eventdata) {{
            if (eventdata['xaxis.range[0]'] !== undefined) {{
                var xStart = eventdata['xaxis.range[0]'];
                var xEnd = eventdata['xaxis.range[1]'];
                
                var update = {{}};
                var traces = graphData.data;
                
                for (var i = 0; i < traces.length; i++) {{
                    var trace = traces[i];
                    var visibleY = [];
                    
                    if (trace.x && trace.y) {{
                        for (var j = 0; j < trace.x.length; j++) {{
                            var xVal = trace.x[j];
                            if (xVal >= xStart && xVal <= xEnd) {{
                                if (trace.y[j] !== null && !isNaN(trace.y[j])) {{
                                    visibleY.push(trace.y[j]);
                                }}
                            }}
                        }}
                        
                        if (visibleY.length > 0) {{
                            var minY = Math.min.apply(null, visibleY);
                            var maxY = Math.max.apply(null, visibleY);
                            var padding = (maxY - minY) * 0.05;
                            
                            var yaxis = 'yaxis';
                            if (i > 0) {{
                                var yAxisNum = trace.yaxis ? trace.yaxis.replace('y', '') : (i + 1);
                                if (yAxisNum === '') yAxisNum = '1';
                                yaxis = 'yaxis' + yAxisNum;
                            }}
                            
                            update[yaxis + '.range'] = [minY - padding, maxY + padding];
                        }}
                    }}
                }}
                
                if (Object.keys(update).length > 0) {{
                    Plotly.relayout('chart', update);
                }}
            }}
        }});
    </script>
</body>
</html>
"""

components.html(chart_html, height=220 + num_rows * 150)

st.info("""
**指标说明：**
- **与K线叠加显示**：均线(MA)、布林带(BB) - 这些指标与价格范围相近，适合叠加在K线上
- **单独显示**：RSI、MACD、KDJ - 这些震荡类指标数值范围不同，需要单独显示
- **同步交互**：所有图表共享X轴，缩放和拖动时自动同步
""")

# if show_ma:
#     df['signal'] = 0
#     df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
#     df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1
#     df['position'] = df['signal'].diff()
    
#     df['returns'] = df['close'].pct_change()
#     df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
#     df['nav'] = (1 + df['strategy_returns']).cumprod()
#     df['benchmark'] = (1 + df['returns']).cumprod()
    
#     st.subheader("📈 回测净值曲线")
    
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=df.index, y=df['nav'], name='策略净值', line=dict(color='blue')))
#     fig2.add_trace(go.Scatter(x=df.index, y=df['benchmark'], name='基准（买入持有）', line=dict(color='gray')))
#     fig2.update_layout(height=400, hovermode='x unified')
#     st.plotly_chart(fig2, use_container_width=True)
    
#     st.subheader("📊 策略绩效")
    
#     total_return = df['nav'].iloc[-1] - 1
#     benchmark_return = df['benchmark'].iloc[-1] - 1
#     sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252)
#     max_drawdown = (df['nav'] / df['nav'].cummax() - 1).min()
    
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("策略收益", f"{total_return:.2%}")
#     col2.metric("基准收益", f"{benchmark_return:.2%}")
#     col3.metric("夏普比率", f"{sharpe:.2f}")
#     col4.metric("最大回撤", f"{max_drawdown:.2%}")

# with st.expander("📋 查看数据"):
#     display_df = df.copy()
#     display_df = display_df.reset_index()
#     st.dataframe(display_df.tail(100))
