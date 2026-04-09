from bollinger_version7 import *

df = pd.read_csv('000001_20050101_20241231.csv')
df.columns = ['index', 'date', 'code', 'open', 'close', 'high', 'low', 
                    'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
result = run_strategy(df)
# print(result)
for k, item in result.items():
    print(k,'\t', item)