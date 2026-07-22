"""Compare live CP Z-scores against backtest rolling Z-score distribution.

Live fills:
  NZDCAD short, z_score=2.15, trigger_currency=CAD @ 1784556720
  AUDUSD long,  z_score=-2.14, trigger_currency=USD @ 1784556840

Backtest methodology: rolling z_window=2000, fixed vol from first 199 returns.
"""
import numpy as np
import pandas as pd
import os, glob, sys

datadir = 'research/phase_dislocation/dukascopy_data'
parquet_files = sorted(glob.glob(os.path.join(datadir, '*.parquet')))

ALL_CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']

ALL_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD",
    "AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF",
    "CADJPY","CADCHF",
    "CHFJPY",
]

BEST_PAIR = {
    "USD": "AUDUSD", "EUR": "EURUSD", "JPY": "NZDJPY",
    "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "NZDCAD", "CHF": "USDCHF",
}

def base_quote(pair):
    currs = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    for c in currs:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None

# Load all pairs
all_dfs = {}
for fpath in parquet_files:
    pair = os.path.basename(fpath).replace('.parquet', '')
    pair_upper = pair.upper()
    if pair_upper not in ALL_PAIRS:
        continue
    df = pd.read_parquet(fpath)
    df = df[['timestamp', 'close']].rename(columns={'close': pair_upper})
    all_dfs[pair_upper] = df

# Merge on timestamp
merged = None
for pair, df in all_dfs.items():
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on='timestamp', how='outer')
merged = merged.sort_values('timestamp').dropna().reset_index(drop=True)

pairs_list = [c for c in merged.columns if c != 'timestamp']
timestamps = merged['timestamp'].values.astype('int64') // 10**9
price_matrix = merged[pairs_list].values.astype(np.float64)
n = len(merged)
print(f"Bars: {n}, Pairs: {len(pairs_list)}")
print(f"Time range: {pd.Timestamp(timestamps[0], unit='s')} -> {pd.Timestamp(timestamps[-1], unit='s')}")
print()

# Currency pair mapping
curr_pairs = {c: [] for c in ALL_CURRENCIES}
for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0, pairs_list[j]))
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0, pairs_list[j]))

# Compute log returns
lr = np.diff(np.log(price_matrix), axis=0)
timestamps = timestamps[1:]
n_lr = len(lr)

# Rolling Z-score computation (matching live strategy)
Z_WINDOW = 2000
VOL_WINDOW = 200
VOL_COUNT = VOL_WINDOW - 1  # 199

# Fixed vol from first 199 returns
pair_vol_fixed = {}
for j, pair in enumerate(pairs_list):
    first_rets = lr[1:VOL_COUNT+1, j]
    pair_vol_fixed[pair] = np.std(first_rets) + 1e-10

# Compute rolling currency returns and Z-scores
all_curr_rets = np.full((n_lr, len(ALL_CURRENCIES)), np.nan)
all_curr_z = np.full((n_lr, len(ALL_CURRENCIES)), np.nan)

# We need at least VOL_COUNT+1 returns before we can compute Z (need vol + 1 return)
# Z needs z_window returns in history before computing

print("Computing rolling Z-scores (window=2000, matching live strategy)...")

for i in range(1, n_lr):
    if i < VOL_COUNT + 1:
        continue

    # Compute currency returns for this bar
    pair_returns = {}
    for j, pair in enumerate(pairs_list):
        pair_returns[pair] = lr[i, j]

    curr_rets = {}
    for c in ALL_CURRENCIES:
        pairs = curr_pairs[c]
        rets = []
        vols = []
        for j, sign, pair_name in pairs:
            ret = pair_returns.get(pair_name)
            if ret is None:
                continue
            vol = pair_vol_fixed.get(pair_name, 1e-10)
            rets.append(ret * sign)
            vols.append(vol)
        if len(rets) >= 2:
            w = np.array([1.0 / v for v in vols])
            w = w / np.sum(w)
            curr_rets[c] = np.dot(rets, w)

    ci = {c: idx for idx, c in enumerate(ALL_CURRENCIES)}
    for c, ret in curr_rets.items():
        all_curr_rets[i, ci[c]] = ret

    # Once we have enough history (z_window), compute Z-scores
    if i >= Z_WINDOW + VOL_COUNT:
        for c in ALL_CURRENCIES:
            hist = all_curr_rets[i-Z_WINDOW+1:i+1, ci[c]]
            hist = hist[~np.isnan(hist)]
            if len(hist) < 5:
                continue
            mean = np.mean(hist)
            std = np.std(hist)
            if std < 1e-12:
                continue
            all_curr_z[i, ci[c]] = (curr_rets.get(c, 0) - mean) / std

print("Done.")
print()

# Find signals matching the live fills
live_fills = [
    {"pair": "NZDCAD", "direction": -1, "z_score": 2.15, "currency": "CAD"},
    {"pair": "AUDUSD", "direction": 1, "z_score": -2.14, "currency": "USD"},
]

for fill in live_fills:
    pair = fill["pair"]
    direction = fill["direction"]
    live_z = fill["z_score"]
    currency = fill["currency"]
    
    ci = ALL_CURRENCIES.index(currency)
    j = pairs_list.index(pair)
    sign = next((s for _, s, pn in curr_pairs[currency] if pn == pair), None)
    
    print(f"{'='*60}")
    print(f"LIVE FILL: {pair} {'SHORT' if direction==-1 else 'LONG'}")
    print(f"  Z-score: {live_z} | trigger: {currency}")
    print(f"{'='*60}")
    
    # Find all matching signals in backtest
    matching = []  # (timestamp, z_score)
    for i in range(Z_WINDOW + VOL_COUNT, n_lr):
        z = all_curr_z[i, ci]
        if np.isnan(z):
            continue
        if abs(z) < 2.0:
            continue
        expected_dir = int(np.sign(z) * sign)
        if expected_dir != direction:
            continue
        
        matching.append({
            'bar_index': i,
            'timestamp': timestamps[i],
            'z_score': z,
            'currency_return': all_curr_rets[i, ci],
        })
    
    dir_label = "SHORT" if direction == -1 else "LONG"
    if matching:
        mdf = pd.DataFrame(matching)
        print(f"\n  Backtest matches (Z>2.0, {pair} {dir_label}): {len(mdf)} events")
        print(f"  Z-score stats:")
        print(f"    min: {mdf['z_score'].min():.2f}")
        print(f"    max: {mdf['z_score'].max():.2f}")
        print(f"    mean: {mdf['z_score'].mean():.2f}")
        print(f"    std: {mdf['z_score'].std():.2f}")
        print(f"    median: {mdf['z_score'].median():.2f}")
        
        # Percentile of live Z-score
        pct = (mdf['z_score'] < live_z).mean() * 100
        print(f"\n  Live Z-score {live_z:.2f} is at the {pct:.0f}th percentile of backtest distribution")
        
        # Show closest matches
        mdf['z_diff'] = abs(mdf['z_score'] - live_z)
        closest = mdf.nsmallest(5, 'z_diff')
        print(f"\n  Top 5 closest backtest matches:")
        for _, row in closest.iterrows():
            ts = pd.Timestamp(row['timestamp'], unit='s')
            print(f"    bar={row['bar_index']} | {ts} | Z={row['z_score']:.2f} | ret={row['currency_return']:.6f}")
    else:
        print(f"\n  No matching backtest events found (Z>2.0, {pair} {dir_label})")
    
    print()

# Also show the global Z-score distribution for CAD and USD
print(f"{'='*60}")
print("GLOBAL Z-SCORE DISTRIBUTION (backtest, all bars)")
print(f"{'='*60}")
for currency in ['CAD', 'USD']:
    ci = ALL_CURRENCIES.index(currency)
    z_vals = all_curr_z[Z_WINDOW + VOL_COUNT:, ci]
    z_vals = z_vals[~np.isnan(z_vals)]
    print(f"\n  {currency} Z-scores: n={len(z_vals)}")
    print(f"    mean={np.mean(z_vals):.2f} std={np.std(z_vals):.2f}")
    print(f"    min={np.min(z_vals):.2f} max={np.max(z_vals):.2f}")
    for thresh in [1.0, 1.5, 2.0, 2.5]:
        pct_above = (np.abs(z_vals) > thresh).mean() * 100
        print(f"    |Z|>{thresh:.1f}: {pct_above:.1f}%")
