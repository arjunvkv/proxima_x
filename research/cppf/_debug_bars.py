"""Check actual bar data around live trade times."""
import pandas as pd
import datetime

pair = "gbpnzd"
df = pd.read_parquet(f"research/cppf/_mt5_data/{pair}.parquet")
df.index = pd.to_datetime(df.index, utc=True)

print(f"Total bars: {len(df)}")
print(f"Range: {df.index[0]} to {df.index[-1]}")

t = pd.Timestamp(2026, 7, 23, 18, 9, tz="UTC")
mask = (df.index >= t - pd.Timedelta(minutes=10)) & (df.index <= t + pd.Timedelta(minutes=10))
window = df[mask]
print(f"\nBars around 18:09 UTC:")
if len(window) == 0:
    print("  NO BARS IN THIS RANGE")
    # Find nearest bar
    mask2 = df.index <= t
    if mask2.any():
        idx = mask2.sum() - 1
        print(f"  Nearest before: {df.index[idx]} close={df.iloc[idx]['close']:.5f}")
        if idx + 1 < len(df):
            print(f"  Next after: {df.index[idx+1]} close={df.iloc[idx+1]['close']:.5f}")
else:
    for idx, row in window.iterrows():
        print(f"  {idx} O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")
