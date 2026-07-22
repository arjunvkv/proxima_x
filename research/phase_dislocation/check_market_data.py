"""Check market data structure."""
import os, pandas as pd
md = sorted(os.listdir('data/market'))
print(f"Files: {len(md)}")

pairs = {}
for f in md:
    df = pd.read_parquet(f'data/market/{f}')
    avg = df['close'].mean()
    if 'pair' in df.columns:
        pair = df['pair'].iloc[0]
    else:
        if avg > 150:
            pair = 'JPY_PAIR'
        elif avg > 0.9:
            pair = 'EUR_GBP_AUD'
        else:
            pair = 'OTHER'
    pairs[f] = {'shape': df.shape, 'avg_close': avg, 'pair': pair}
    print(f"  {f}: {df.shape}, avg_close={avg:.4f}, cols={list(df.columns)}")

# Check if 30 files are 30 days of same pair or different pairs
# by looking at min/max time range
times = []
for f in md[:3]:
    df = pd.read_parquet(f'data/market/{f}')
    t = df['timestamp'] if 'timestamp' in df.columns else df['time']
    times.append((f, t.min(), t.max()))
print("\nTime ranges:")
for f, tmin, tmax in times:
    print(f"  {f}: {tmin} -> {tmax}")
