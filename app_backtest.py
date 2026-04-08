import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from datetime import datetime
from typing import Tuple, List, Dict, Optional


def load_backtest_files() -> Tuple[List[str], List[str]]:
    trades_files = sorted(glob.glob('./output/trades_*.csv'))
    backtest_files = sorted(glob.glob('./output/backtest_*.csv'))
    return trades_files, backtest_files


def load_trades_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_backtest_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_stock_price_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    price_files = glob.glob(f'./data/raw/tmp_akshare/price_history/{symbol}_*.csv')
    
    if not price_files:
        st.warning(f"未找到股票 {symbol} 的历史价格数据")
        return None
    
    price_file = price_files[0]
    
    try:
        df = pd.read_csv(price_file, index_col=0)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        })
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        st.error(f"加载股票价格数据失败: {str(e)}")
        return None


def calculate_performance_metrics(backtest_df: pd.DataFrame) -> Dict:
    metrics = {}
    
    initial_value = backtest_df['portfolio_value'].iloc[0]
    final_value = backtest_df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    start_date = backtest_df['date'].iloc[0]
    end_date = backtest_df['date'].iloc[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0
    
    returns = backtest_df['returns'].dropna()
    if len(returns) > 0:
        cumulative_returns = backtest_df['cumulative_returns'].dropna()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if returns.std() != 0:
            risk_free_rate = 0.03
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        max_drawdown = 0
        sharpe_ratio = 0
        win_rate = 0
        profit_loss_ratio = 0
    
    metrics['初始资金'] = f"{initial_value:,.2f}"
    metrics['最终资金'] = f"{final_value:,.2f}"
    metrics['总收益率'] = f"{total_return * 100:.2f}%"
    metrics['年化收益率'] = f"{annualized_return * 100:.2f}%"
    metrics['最大回撤'] = f"{max_drawdown * 100:.2f}%"
    metrics['夏普比率'] = f"{sharpe_ratio:.2f}"
    metrics['胜率'] = f"{win_rate * 100:.2f}%"
    metrics['盈亏比'] = f"{profit_loss_ratio:.2f}"
    metrics['回测天数'] = f"{days} 天"
    metrics['回测年数'] = f"{years:.2f} 年"
    
    return metrics


def plot_portfolio_value(backtest_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('组合价值', '累计收益率')
    )
    
    fig.add_trace(
        go.Scatter(
            x=backtest_df['date'],
            y=backtest_df['portfolio_value'],
            mode='lines',
            name='组合价值',
            line=dict(color='#2E86AB', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=backtest_df['date'],
            y=backtest_df['cumulative_returns'],
            mode='lines',
            name='累计收益率',
            line=dict(color='#A23B72', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价值 (元)", row=1, col=1)
    fig.update_yaxes(title_text="收益率", row=2, col=1)
    
    return fig


def plot_candlestick_with_signals(
    price_df: pd.DataFrame, 
    trades_df: pd.DataFrame,
    symbol: str
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} K线图', '成交量')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=price_df['date'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='K线',
            increasing_line_color='#EF476F',
            decreasing_line_color='#06D6A0'
        ),
        row=1, col=1
    )
    
    symbol_trades = trades_df[trades_df['symbol'] == symbol]
    
    buy_trades = symbol_trades[symbol_trades['action'] == 'buy']
    sell_trades = symbol_trades[symbol_trades['action'] == 'sell']
    
    if not buy_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_trades['date'],
                y=buy_trades['price'],
                mode='markers',
                name='买入',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#FF6B6B',
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=[f'买入: {p:.2f}' for p in buy_trades['price']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )
    
    if not sell_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_trades['date'],
                y=sell_trades['price'],
                mode='markers',
                name='卖出',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#4ECDC4',
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=[f'卖出: {p:.2f}' for p in sell_trades['price']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Bar(
            x=price_df['date'],
            y=price_df['volume'],
            name='成交量',
            marker_color='#95E1D3'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig


def display_trades_table(trades_df: pd.DataFrame):
    display_df = trades_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['price'] = display_df['price'].apply(lambda x: f"{x:.2f}")
    display_df['cost'] = display_df['cost'].apply(lambda x: f"{x:,.2f}")
    
    if 'revenue' in display_df.columns:
        display_df['revenue'] = display_df['revenue'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '-')
    if 'profit' in display_df.columns:
        display_df['profit'] = display_df['profit'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '-')
    if 'profit_pct' in display_df.columns:
        display_df['profit_pct'] = display_df['profit_pct'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else '-')
    
    display_df = display_df.rename(columns={
        'date': '日期',
        'symbol': '股票代码',
        'action': '操作',
        'shares': '股数',
        'price': '价格',
        'cost': '成本',
        'revenue': '收入',
        'profit': '利润',
        'profit_pct': '利润率'
    })
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )


def main():
    st.set_page_config(
        page_title="回测分析系统",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 量化回测分析系统")
    st.markdown("---")
    
    trades_files, backtest_files = load_backtest_files()
    
    if not trades_files or not backtest_files:
        st.error("未找到回测数据文件，请确保 ./output/ 目录下存在 trades_*.csv 和 backtest_*.csv 文件")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_backtest = st.selectbox(
            "选择回测文件",
            backtest_files,
            format_func=lambda x: os.path.basename(x)
        )
    
    with col2:
        trades_file = selected_backtest.replace('backtest_', 'trades_')
        if trades_file in trades_files:
            selected_trades = trades_file
        else:
            selected_trades = st.selectbox(
                "选择交易记录文件",
                trades_files,
                format_func=lambda x: os.path.basename(x)
            )
    
    backtest_df = load_backtest_data(selected_backtest)
    trades_df = load_trades_data(selected_trades)
    
    st.markdown("---")
    
    st.header("📈 回测效能指标")
    
    metrics = calculate_performance_metrics(backtest_df)
    
    cols = st.columns(4)
    metric_items = list(metrics.items())
    
    for idx, (key, value) in enumerate(metric_items):
        with cols[idx % 4]:
            st.metric(label=key, value=value)
    
    st.markdown("---")
    
    st.header("💰 组合价值曲线")
    
    portfolio_fig = plot_portfolio_value(backtest_df)
    st.plotly_chart(portfolio_fig, width='stretch')
    
    st.markdown("---")
    
    st.header("📋 交易记录")
    
    display_trades_table(trades_df)
    
    st.markdown("---")
    
    st.header("🕯️ K线图与买卖点")
    
    unique_symbols = trades_df['symbol'].unique()
    
    if len(unique_symbols) > 0:
        selected_symbol = st.selectbox(
            "选择股票",
            unique_symbols,
            format_func=lambda x: f"股票代码: {x}"
        )
        
        start_date = backtest_df['date'].min().strftime('%Y-%m-%d')
        end_date = backtest_df['date'].max().strftime('%Y-%m-%d')
        
        price_df = load_stock_price_data(selected_symbol, start_date, end_date)
        
        if price_df is not None and not price_df.empty:
            candlestick_fig = plot_candlestick_with_signals(price_df, trades_df, selected_symbol)
            st.plotly_chart(candlestick_fig, width='stretch')
        else:
            st.warning(f"无法加载股票 {selected_symbol} 的价格数据")
    else:
        st.info("交易记录中没有股票数据")
    
    st.markdown("---")
    
    with st.expander("📊 数据统计"):
        st.subheader("回测数据概览")
        st.write(f"回测时间范围: {backtest_df['date'].min().strftime('%Y-%m-%d')} 至 {backtest_df['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"交易次数: {len(trades_df)}")
        st.write(f"买入次数: {len(trades_df[trades_df['action'] == 'buy'])}")
        st.write(f"卖出次数: {len(trades_df[trades_df['action'] == 'sell'])}")
        st.write(f"涉及股票: {', '.join(trades_df['symbol'].astype(str).unique())}")
        
        st.subheader("交易数据样本")
        st.dataframe(backtest_df.head(10), width='stretch')


if __name__ == "__main__":
    main()
