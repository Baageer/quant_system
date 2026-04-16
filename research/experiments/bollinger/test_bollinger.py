from bollinger_version7 import *
import os
from tqdm import tqdm

CSV_DIR= r"J:\workspace\2_quant\quant_system\data\raw\akshare\price_history"

trade_sum = 0
win_sum = 0
expectancy_sum = 0

count = 0
for filename in os.listdir(CSV_DIR):
    count += 1
    if count >10: break
    print("\n ==============================")
    print(filename)
    df = pd.read_csv(os.path.join(CSV_DIR , filename))
    df.columns = ['index', 'date', 'code', 'open', 'close', 'high', 'low', 
                        'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']

    result = run_strategy(df)
    # print(result)
    trade_sum += result["total_trades"]
    win_sum   += result["total_trades"] * result["win_rate"]
    expectancy_sum += result["total_trades"] * result["expectancy"]

    
    # for k, item in result.items():
    #     print(k,'\t', item)

print("\n\n ==============================")
print(" trade_sum ", trade_sum)
print(" win_sum ", win_sum)
print(" expectancy_sum ", expectancy_sum)
print(" win_rate ", win_sum/trade_sum)
print(" expectancy ", expectancy_sum/trade_sum)