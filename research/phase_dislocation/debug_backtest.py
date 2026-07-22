"""Debug the backtest — check data alignment and Z-scores."""
import numpy as np
import pandas as pd
import os, glob

datadir = 'research/phase_dislocation/dukascopy_data'
all_data = {}
for fpath in sorted(glob.glob(os.path.join(datadir, '*.parquet'))):
    pair = os.path.basename(fpath).replace('.parquet', '')
    df = pd.read_parquet(fpath).sort_values('timestamp')
    all_data[pair] = df

pairs_list = sorted(all_data.keys())
print(f"Pairs: {len(pairs_list)}")
print(f"First pair: {pairs_list[0]}, {len(all_data[pairs_list[0]])} rows")
print(f"Last pair: {pairs_list[-1]}, {len(all_data[pairs_list[-1]])} rows")

# Check timestamp ranges
for pair in ['nzdusd', 'eurusd', 'eurjpy', 'usdchf']:
    df = all_data[pair]
    print(f"{pair}: {df['timestamp'].min()} -> {df['timestamp'].max()} ({len(df)} rows)")

# Check alignment
ref = all_data[pairs_list[0]]
timestamps = ref['timestamp'].values
n = len(timestamps)

# Count matching timestamps for each pair
for pair in pairs_list[:5]:
    df = all_data[pair]
    common = len(set(timestamps) & set(df['timestamp'].values))
    print(f"{pair}: {common}/{n} timestamps match reference")

# Build price matrix and check for NaN
price_matrix = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    times = df['timestamp'].values
    prices = df['close'].values
    for i, t in enumerate(timestamps):
        match = np.where(times == t)[0]
        if len(match) > 0:
            price_matrix[i, j] = prices[match[0]]

nan_cols = np.sum(np.isnan(price_matrix), axis=0)
nan_rows = np.sum(np.isnan(price_matrix), axis=1)
print(f"\nNaN per column (first 5): {nan_cols[:5]}")
print(f"NaN per row: min={nan_rows.min()}, max={nan_rows.max()}, {np.sum(nan_rows > 0)} rows with NaN")

# Remove leading NaN rows
first_valid = np.where(~np.isnan(price_matrix).any(axis=1))[0]
print(f"First valid row: {first_valid[0] if len(first_valid) > 0 else 'NONE'}")

if len(first_valid) > 0:
    price_matrix = price_matrix[first_valid[0]:]
    print(f"After trimming: {price_matrix.shape}")
    print(f"NaN remaining: {np.sum(np.isnan(price_matrix))}")
