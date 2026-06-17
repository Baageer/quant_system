import pandas as pd
import numpy as np

df = pd.read_csv('output/trades_vwap_distance_rsrs_beta_20260617_153234.csv')

print('=' * 70)
print('交易记录分析报告')
print('=' * 70)

print(f'\n时间范围: {df["date"].min()} ~ {df["date"].max()}')
print(f'总交易笔数: {len(df)}')

filled = df[df['status'] == 'filled']
rejected = df[df['status'] == 'rejected']
print(f'\n成交笔数: {len(filled)} ({len(filled)/len(df)*100:.1f}%)')
print(f'拒绝笔数: {len(rejected)} ({len(rejected)/len(df)*100:.1f}%)')

if len(rejected) > 0:
    print(f'拒绝原因分布:')
    print(rejected['rejection_reason'].value_counts())

buys = df[df['action'] == 'buy']
sells = df[df['action'] == 'sell']
print(f'\n买入请求: {len(buys)}')
print(f'买入成交: {len(buys[buys["status"] == "filled"])}')
print(f'买入拒绝: {len(buys[buys["status"] == "rejected"])}')
print(f'卖出请求: {len(sells)}')
print(f'卖出成交: {len(sells[sells["status"] == "filled"])}')
print(f'卖出拒绝: {len(sells[sells["status"] == "rejected"])}')

sell_filled = sells[sells['status'] == 'filled'].copy()
sell_filled = sell_filled.dropna(subset=['profit'])
print(f'\n有效交易(含盈亏): {len(sell_filled)}')
print(f'盈利交易: {len(sell_filled[sell_filled["profit"] > 0])}')
print(f'亏损交易: {len(sell_filled[sell_filled["profit"] <= 0])}')

if len(sell_filled) > 0:
    win_rate = len(sell_filled[sell_filled['profit'] > 0]) / len(sell_filled)
    avg_profit = sell_filled[sell_filled['profit'] > 0]['profit'].mean()
    avg_loss = sell_filled[sell_filled['profit'] <= 0]['profit'].mean()
    profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
    
    print(f'\n胜率: {win_rate*100:.1f}%')
    print(f'平均盈利: {avg_profit:.2f} 元')
    print(f'平均亏损: {avg_loss:.2f} 元')
    print(f'盈亏比: {profit_loss_ratio:.2f}')
    print(f'总盈亏: {sell_filled["profit"].sum():.2f} 元')
    print(f'平均盈亏: {sell_filled["profit"].mean():.2f} 元')
    print(f'最大盈利: {sell_filled["profit"].max():.2f} 元')
    print(f'最大亏损: {sell_filled["profit"].min():.2f} 元')
    
    print(f'\n盈利百分比统计:')
    print(sell_filled['profit_pct'].describe())

print(f'\n交易原因分布:')
print(df['reason'].value_counts())

print(f'\n按交易原因统计盈亏:')
for reason in df['reason'].unique():
    if pd.isna(reason):
        continue
    reason_sells = sell_filled[sell_filled['reason'] == reason]
    if len(reason_sells) > 0:
        reason_win_rate = len(reason_sells[reason_sells['profit'] > 0]) / len(reason_sells)
        reason_avg_profit = reason_sells[reason_sells['profit'] > 0]['profit'].mean()
        reason_avg_loss = reason_sells[reason_sells['profit'] <= 0]['profit'].mean()
        reason_pl_ratio = abs(reason_avg_profit / reason_avg_loss) if avg_loss != 0 else float('inf')
        print(f'  {reason}: 交易数={len(reason_sells)}, 胜率={reason_win_rate*100:.1f}%, 盈亏比={reason_pl_ratio:.2f}, 总盈亏={reason_sells["profit"].sum():.2f}')

print(f'\n按年份统计:')
df['year'] = pd.to_datetime(df['date']).dt.year
for year in sorted(df['year'].unique()):
    year_sells = sell_filled[sell_filled['date'].str.startswith(str(year))]
    if len(year_sells) > 0:
        year_win_rate = len(year_sells[year_sells['profit'] > 0]) / len(year_sells)
        year_avg_profit = year_sells[year_sells['profit'] > 0]['profit'].mean()
        year_avg_loss = year_sells[year_sells['profit'] <= 0]['profit'].mean()
        year_pl_ratio = abs(year_avg_profit / year_avg_loss) if year_avg_loss != 0 else float('inf')
        print(f'  {year}: 交易数={len(year_sells)}, 胜率={year_win_rate*100:.1f}%, 盈亏比={year_pl_ratio:.2f}, 总盈亏={year_sells["profit"].sum():.2f}')

print(f'\n最大亏损交易详情:')
print(sell_filled.nsmallest(5, 'profit')[['date', 'symbol', 'action', 'reason', 'profit', 'profit_pct']])

print(f'\n最大盈利交易详情:')
print(sell_filled.nlargest(5, 'profit')[['date', 'symbol', 'action', 'reason', 'profit', 'profit_pct']])
