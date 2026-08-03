import numpy as np
import pandas as pd

d = np.load('paper_trade/mt5_backtest/fundednext_audusd_m1.npy', allow_pickle=True)
df = pd.DataFrame(d)
print('Columns:', list(df.columns))
print('First 5 rows:')
for i in range(5):
    print(f'  time={df["time"].iloc[i]}  open={df["open"].iloc[i]:.5f}  high={df["high"].iloc[i]:.5f}  low={df["low"].iloc[i]:.5f}  close={df["close"].iloc[i]:.5f}  spread_raw={df["spread"].iloc[i]}')

print(f'\nSpread stats (raw):')
print(f'  min={df["spread"].min()}, max={df["spread"].max()}, median={df["spread"].median()}')
print(f'  p25={df["spread"].quantile(0.25)}, p75={df["spread"].quantile(0.75)}, p90={df["spread"].quantile(0.90)}, p99={df["spread"].quantile(0.99)}')

# Check digits from a typical price
print(f'\nPrice examples: open={df["open"].iloc[0]:.5f}, close={df["close"].iloc[0]:.5f}')

# For EURAUD and GBPAUD
for pair in ['euraud', 'gbpaud']:
    d2 = np.load(f'paper_trade/mt5_backtest/fundednext_{pair}_m1.npy', allow_pickle=True)
    df2 = pd.DataFrame(d2)
    s = df2['spread']
    print(f'\n{pair.upper()} spread raw:')
    print(f'  min={s.min()}, max={s.max()}, median={s.median()}')
    print(f'  p25={s.quantile(0.25)}, p75={s.quantile(0.75)}')
    print(f'  p90={s.quantile(0.90)}, p99={s.quantile(0.99)}')
    print(f'  Price example: open={df2["open"].iloc[0]:.5f}')
