"""Check available data"""
import pandas as pd, os

for pair in ['EURJPY', 'NZDUSD', 'USDCHF', 'EURUSD']:
    df = pd.read_parquet(f'data/market/{pair}.parquet').sort_values('timestamp')
    tcol = 'timestamp' if 'timestamp' in df.columns else 'time'
    t = df[tcol]
    avg = df['close'].mean()
    print(f'{pair}: {df.shape[0]} rows, {t.min()} -> {t.max()}, avg_close={avg:.4f}')

print()
df = pd.read_parquet('data/temp/mt5_m1_9day.parquet')
print(f'temp data: {df.shape}')
print(f'time range: {df["time"].min()} -> {df["time"].max()}')
print(f'pairs: {df["pair"].unique()}')
