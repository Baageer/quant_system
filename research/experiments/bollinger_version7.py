import pandas as pd
import numpy as np
# 布林带突破 + 趋势判断
# RSI 
# 根据趋势判断分仓比例
# =========================
# Indicators
# =========================
def bollinger_bands(df, period=20, std=2):
    df['ma'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['upper'] = df['ma'] + std * df['std']
    df['lower'] = df['ma'] - std * df['std']
    df['bandwidth'] = (df['upper'] - df['lower']) / df['ma']
    return df


def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(period).mean()
    return df


def moving_average(df, fast=50, slow=200):
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    return df


def rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# =========================
# Market Regime Detection
# =========================
def detect_market_regime(df):
    df['trend_strength'] = abs(df['ma_fast'] - df['ma_slow']) / df['close']

    # Threshold can be optimized
    df['is_trending'] = df['trend_strength'] > 0.02

    return df

# =========================
# Signals
# =========================
def generate_signals(df, squeeze_quantile=0.2, volume_filter=False):
    df = df.copy()

    threshold = df['bandwidth'].quantile(squeeze_quantile)
    df['squeeze'] = df['bandwidth'] < threshold

    df['uptrend'] = df['ma_fast'] > df['ma_slow']
    df['downtrend'] = df['ma_fast'] < df['ma_slow']

    # Volume filter
    if volume_filter and 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(20).mean()
        vol_condition = df['volume'] > df['vol_ma']
    else:
        vol_condition = True

    df['bb_long'] = df['squeeze'].shift(1) & (df['close'] > df['upper']) & df['uptrend'] & vol_condition
    df['bb_short'] = df['squeeze'].shift(1) & (df['close'] < df['lower']) & df['downtrend'] & vol_condition

    df['rsi_long'] = (df['rsi'] < 30) & (df['close'] < df['lower']) 
    df['rsi_short'] = (df['rsi'] > 70) & (df['close'] > df['upper'])

    return df

# =========================
# Dynamic Weight Function
# =========================
def get_weights(row):
    if row['is_trending']:
        return 0.8, 0.2  # favor BB
    else:
        return 0.3, 0.7  # favor RSI

# =========================
# Backtest (Dynamic Allocation)
# =========================
def backtest(df, capital=100000, risk_per_stock=0.1):
    bb_capital = capital * 0.5
    rsi_capital = capital * 0.5

    bb_pos = rsi_pos = 0
    bb_entry = rsi_entry = 0
    bb_stop = rsi_stop = 0
    bb_size = rsi_size = 0
    bb_take = rsi_take = 0

    equity_curve = []
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]

        bb_w, rsi_w = get_weights(row)
        total_equity = bb_capital + rsi_capital

        bb_capital = total_equity * bb_w
        rsi_capital = total_equity * rsi_w

        # BB strategy
        if bb_pos == 0:
            risk = bb_capital * risk_per_stock
            if row['bb_long']:
                if row["pct_change"] > 9:
                    print(row["pct_change"])
                    continue
                bb_entry = row['close']
                bb_stop = bb_entry - 1.5 * row['atr']
                bb_take = bb_entry + (bb_entry - bb_stop) * 2
                bb_size = risk / (bb_entry - bb_stop)
                bb_pos = 1
            # elif row['bb_short']:
            #     bb_entry = row['close']
            #     bb_stop = bb_entry + 1.5 * row['atr']
            #     bb_size = risk / (bb_stop - bb_entry)
            #     bb_pos = -1
        elif bb_pos == 1 :
            if row['low'] <= bb_stop:
                if row["pct_change"] < -9:
                    print(row["pct_change"])
                    continue
                pnl = (bb_stop - bb_entry) * bb_size
                bb_capital += pnl
                trades.append(pnl)
                bb_pos = 0
            elif row['high'] >= bb_take:
                pnl = (bb_take - bb_entry) * bb_size
                bb_capital += pnl
                trades.append(pnl)
                bb_pos = 0
        # elif bb_pos == -1 and row['high'] >= bb_stop:
        #     pnl = (bb_entry - bb_stop) * bb_size
        #     bb_capital += pnl
        #     trades.append(pnl)
        #     bb_pos = 0

        # RSI strategy
        if rsi_pos == 0:
            risk = rsi_capital * risk_per_stock
            if row['rsi_long']:
                rsi_entry = row['close']
                rsi_stop = rsi_entry - 1.5 * row['atr']
                rsi_take = rsi_entry + (rsi_entry - rsi_stop) * 2
                rsi_size = risk / (rsi_entry - rsi_stop)
                rsi_pos = 1
            # elif row['rsi_short']:
            #     rsi_entry = row['close']
            #     rsi_stop = rsi_entry + 1.5 * row['atr']
            #     rsi_size = risk / (rsi_stop - rsi_entry)
            #     rsi_pos = -1
        elif rsi_pos == 1:
            if row['low'] <= rsi_stop:
                pnl = (rsi_stop - rsi_entry) * rsi_size
                rsi_capital += pnl
                trades.append(pnl)
                rsi_pos = 0
            elif row['high'] >= rsi_take:
                pnl = (rsi_take - rsi_entry) * rsi_size
                rsi_capital += pnl
                trades.append(pnl)
                rsi_pos = 0
        # elif rsi_pos == -1 and row['high'] >= rsi_stop:
        #     pnl = (rsi_entry - rsi_stop) * rsi_size
        #     rsi_capital += pnl
        #     trades.append(pnl)
        #     rsi_pos = 0

        equity_curve.append(bb_capital + rsi_capital)

    return trades, equity_curve

# =========================
# Performance
# =========================
def performance(trades, equity_curve):
    trades = np.array(trades)

    if len(trades) == 0:
        return None

    wins = trades[trades > 0]
    losses = trades[trades <= 0]

    win_rate = len(wins) / len(trades)
    expectancy = trades.mean()

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    return {
        'win_rate': win_rate,
        'expectancy': expectancy,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'total_trades': len(trades)
    }

# =========================
# Pipeline
# =========================
def run_strategy(df):
    df = bollinger_bands(df)
    df = atr(df)
    df = moving_average(df)
    df = rsi(df)
    df = detect_market_regime(df)
    df = generate_signals(df)

    trades, equity_curve = backtest(df)
    stats = performance(trades, equity_curve)

    return stats
