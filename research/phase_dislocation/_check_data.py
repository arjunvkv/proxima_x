"""Check available parquet data for Phase Dislocation backtest."""
import pandas as pd
import os

print("=" * 60)
print("DATA SOURCE CHECK")
print("=" * 60)

# 1. data/market/ parquet files
mkt_dir = "data/market"
if os.path.exists(mkt_dir):
    files = [f for f in os.listdir(mkt_dir) if f.endswith(".parquet")]
    print(f"\n--- data/market/ ({len(files)} files) ---")
    for f in sorted(files):
        path = os.path.join(mkt_dir, f)
        df = pd.read_parquet(path)
        pair = f.replace(".parquet", "")
        print(f"  {pair:>7s}: {len(df):>5d} rows, {list(df.columns)}")
else:
    print("\n--- data/market/ NOT FOUND ---")

# 2. data/temp/ multi-pair parquet
temp_file = "data/temp/mt5_m1_9day.parquet"
if os.path.exists(temp_file):
    df = pd.read_parquet(temp_file)
    print(f"\n--- data/temp/mt5_m1_9day.parquet ---")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    pairs = sorted(df["pair"].unique())
    print(f"  Pairs: {pairs}")
    print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    for p in pairs:
        n = len(df[df["pair"] == p])
        print(f"    {p:>7s}: {n} rows")
else:
    print(f"\n--- {temp_file} NOT FOUND ---")
