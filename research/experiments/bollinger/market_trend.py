import pandas as pd
import numpy as np

# =========================
# Data Loading
# =========================
def load_market_data(index_codes):
    """
    加载市场指数数据
    
    Args:
        index_codes: 指数代码列表，如 ['000001.SH', '399001.SZ']
    
    Returns:
        dict: 包含各指数数据的字典
    """
    data_dict = {}
    
    for code in index_codes:
        # 这里假设数据存储在data目录下的CSV文件中
        # 实际项目中可能需要使用数据接口获取数据
        file_path = f"../../data/{code}.csv"
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            data_dict[code] = df
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            # 生成模拟数据用于测试
            date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            close = 3000 + np.cumsum(np.random.randn(len(date_range)) * 10)
            volume = np.random.randint(10000000, 100000000, len(date_range))
            high = close * (1 + np.random.rand(len(date_range)) * 0.02)
            low = close * (1 - np.random.rand(len(date_range)) * 0.02)
            open_ = close * (1 + np.random.rand(len(date_range)) * 0.01)
            
            df = pd.DataFrame({
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }, index=date_range)
            data_dict[code] = df
    
    return data_dict

# =========================
# Market Trend Analysis
# =========================
def calculate_technical_indicators(df):
    """
    计算技术指标
    
    Args:
        df: 包含价格数据的DataFrame
    
    Returns:
        df: 添加了技术指标的DataFrame
    """
    # 移动平均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    
    # 成交量移动平均
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    
    # 价格变化率
    df['pct_change'] = df['close'].pct_change() * 100
    
    # 趋势判断
    df['uptrend'] = (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])
    df['downtrend'] = (df['ma5'] < df['ma20']) & (df['ma20'] < df['ma60'])
    
    # 市场宽度（基于移动平均线）
    df['price_above_ma20'] = df['close'] > df['ma20']
    
    return df

def determine_market_regime(df):
    """
    判断市场状态
    
    Args:
        df: 包含技术指标的DataFrame
    
    Returns:
        dict: 市场状态分析结果
    """
    # 最近一段时间的数据
    recent_df = df.tail(60)  # 最近60天
    
    # 计算市场景气度指标
    # 1. 价格趋势
    price_trend = recent_df['close'].iloc[-1] / recent_df['close'].iloc[0] - 1
    
    # 2. 成交量趋势
    volume_trend = recent_df['volume'].mean() / recent_df['volume'].iloc[0] - 1
    
    # 3. 移动平均线状态
    ma_status = 0
    if recent_df['ma5'].iloc[-1] > recent_df['ma20'].iloc[-1] > recent_df['ma60'].iloc[-1] > recent_df['ma120'].iloc[-1]:
        ma_status = 3  # 强上升趋势
    elif recent_df['ma5'].iloc[-1] > recent_df['ma20'].iloc[-1] > recent_df['ma60'].iloc[-1]:
        ma_status = 2  # 中等上升趋势
    elif recent_df['ma5'].iloc[-1] > recent_df['ma20'].iloc[-1]:
        ma_status = 1  # 弱上升趋势
    elif recent_df['ma5'].iloc[-1] < recent_df['ma20'].iloc[-1] < recent_df['ma60'].iloc[-1] < recent_df['ma120'].iloc[-1]:
        ma_status = -3  # 强下降趋势
    elif recent_df['ma5'].iloc[-1] < recent_df['ma20'].iloc[-1] < recent_df['ma60'].iloc[-1]:
        ma_status = -2  # 中等下降趋势
    elif recent_df['ma5'].iloc[-1] < recent_df['ma20'].iloc[-1]:
        ma_status = -1  # 弱下降趋势
    else:
        ma_status = 0  # 震荡
    
    # 4. 价格在移动平均线上方的比例
    price_above_ma_ratio = recent_df['price_above_ma20'].mean()
    
    # 综合判断市场景气度
    market_sentiment = 0
    if price_trend > 0.05 and volume_trend > 0.1 and ma_status > 1 and price_above_ma_ratio > 0.7:
        market_sentiment = 3  # 高景气
    elif price_trend > 0 and volume_trend > 0 and ma_status > 0 and price_above_ma_ratio > 0.5:
        market_sentiment = 2  # 中等景气
    elif price_trend > -0.05 and volume_trend > -0.1 and ma_status >= -1 and price_above_ma_ratio > 0.3:
        market_sentiment = 1  # 低景气
    elif price_trend < -0.05 and volume_trend < -0.1 and ma_status < -1 and price_above_ma_ratio < 0.3:
        market_sentiment = -3  # 低景气（熊市）
    elif price_trend < 0 and volume_trend < 0 and ma_status < 0 and price_above_ma_ratio < 0.5:
        market_sentiment = -2  # 中等不景气
    else:
        market_sentiment = -1  # 低不景气
    
    # 判断牛熊市
    bull_bear = ""
    if ma_status >= 2 and price_trend > 0.1:
        bull_bear = "牛市"
    elif ma_status <= -2 and price_trend < -0.1:
        bull_bear = "熊市"
    else:
        bull_bear = "震荡市"
    
    # 判断短期趋势
    short_term_trend = ""
    if recent_df['close'].iloc[-1] > recent_df['close'].iloc[-5]:
        short_term_trend = "上升"
    elif recent_df['close'].iloc[-1] < recent_df['close'].iloc[-5]:
        short_term_trend = "下降"
    else:
        short_term_trend = "震荡"
    
    return {
        'market_sentiment': market_sentiment,
        'bull_bear': bull_bear,
        'short_term_trend': short_term_trend,
        'price_trend': price_trend,
        'volume_trend': volume_trend,
        'ma_status': ma_status,
        'price_above_ma_ratio': price_above_ma_ratio
    }

# =========================
# Industry Analysis
# =========================
def analyze_industry_sectors(industry_data):
    """
    分析行业板块景气度
    
    Args:
        industry_data: 行业指数数据字典
    
    Returns:
        dict: 行业景气度分析结果
    """
    industry_analysis = {}
    
    for industry_name, df in industry_data.items():
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 分析行业景气度
        analysis = determine_market_regime(df)
        
        # 添加行业特定指标
        # 行业相对大盘表现（假设大盘数据在industry_data中）
        if '000001.SH' in industry_data:
            market_df = industry_data['000001.SH']
            recent_days = min(len(df), len(market_df), 60)
            industry_return = df['close'].iloc[-1] / df['close'].iloc[-recent_days] - 1
            market_return = market_df['close'].iloc[-1] / market_df['close'].iloc[-recent_days] - 1
            analysis['relative_performance'] = industry_return - market_return
        else:
            analysis['relative_performance'] = 0
        
        industry_analysis[industry_name] = analysis
    
    return industry_analysis

# =========================
# Main Function
# =========================
def main():
    """
    主函数，演示市场趋势分析和行业景气度判断
    """
    # 加载大盘指数数据
    market_indices = ['000001.SH', '399001.SZ']  # 上证指数和深证成指
    market_data = load_market_data(market_indices)
    
    # 分析大盘趋势
    print("=== 大盘趋势分析 ===")
    for index_code, df in market_data.items():
        df = calculate_technical_indicators(df)
        regime = determine_market_regime(df)
        
        print(f"\n指数: {index_code}")
        print(f"市场景气度: {regime['market_sentiment']}")
        print(f"牛熊市判断: {regime['bull_bear']}")
        print(f"短期趋势: {regime['short_term_trend']}")
        print(f"价格趋势: {regime['price_trend']:.2%}")
        print(f"成交量趋势: {regime['volume_trend']:.2%}")
        print(f"移动平均线状态: {regime['ma_status']}")
        print(f"价格在MA20上方比例: {regime['price_above_ma_ratio']:.2%}")
    
    # 加载行业指数数据
    industry_indices = ['000016.SH', '000905.SH', '000913.SH']  # 上证50、中证500、沪深300
    industry_data = load_market_data(industry_indices)
    
    # 分析行业景气度
    print("\n=== 行业景气度分析 ===")
    industry_analysis = analyze_industry_sectors(industry_data)
    
    for industry_name, analysis in industry_analysis.items():
        print(f"\n行业: {industry_name}")
        print(f"景气度: {analysis['market_sentiment']}")
        print(f"牛熊市判断: {analysis['bull_bear']}")
        print(f"短期趋势: {analysis['short_term_trend']}")
        print(f"相对大盘表现: {analysis['relative_performance']:.2%}")

if __name__ == "__main__":
    main()
