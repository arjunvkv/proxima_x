"""Currency Inventory Pressure — build synthetic currency indices from 28 pairs.
Detect extreme accumulation/liquidation of specific currencies.
"""
import numpy as np
import pandas as pd
import os

# Load all 28 FX pairs
mdir = 'data/market'
fx_pairs = sorted([f.replace('.parquet', '') for f in os.listdir(mdir) 
                   if f.endswith('.parquet') and f.replace('.parquet', '') not in ('NAS100', 'XAUUSD')])

print(f"FX pairs loaded: {len(fx_pairs)}")

def base_quote(pair):
    """Split pair into base and quote currencies."""
    currs = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    for c in currs:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None

# Load all pairs into aligned DataFrame
all_data = {}
for pair in fx_pairs:
    df = pd.read_parquet(f'{mdir}/{pair}.parquet').sort_values('timestamp')
    all_data[pair] = df

# Use EURJPY as reference for time alignment
ref = all_data['EURJPY']
timestamps = ref['timestamp' if 'timestamp' in ref.columns else 'time'].values
n = len(timestamps)

# Build price matrix: rows=time, cols=pairs
pairs_list = sorted(all_data.keys())
price_matrix = np.full((n, len(pairs_list)), np.nan)
for j, pair in enumerate(pairs_list):
    df = all_data[pair]
    tcol = 'timestamp' if 'timestamp' in df.columns else 'time'
    times = df[tcol].values
    # Align to reference timestamps
    p_col = df['close'].values
    # Simple alignment: find matching timestamps
    for i, t in enumerate(timestamps):
        match = np.where(times == t)[0]
        if len(match) > 0:
            price_matrix[i, j] = p_col[match[0]]

# Remove rows with any NaN
valid_rows = ~np.isnan(price_matrix).any(axis=1)
price_matrix = price_matrix[valid_rows]
timestamps = timestamps[valid_rows]
n = len(price_matrix)
print(f"Aligned bars: {n}")

# Compute log returns
lr = np.diff(np.log(price_matrix), axis=0)
lr_timestamps = timestamps[1:]
n_lr = len(lr)

# Map pairs to currencies
# For each currency, collect which pair-columns contribute and how
currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']
curr_pairs = {c: [] for c in currencies}

for j, pair in enumerate(pairs_list):
    base, quote = base_quote(pair)
    if base and quote:
        if base in curr_pairs:
            curr_pairs[base].append((j, 1.0))  # base currency: positive
        if quote in curr_pairs:
            curr_pairs[quote].append((j, -1.0))  # quote currency: negative

print(f"\nCurrency pair counts:")
for c, pairs in curr_pairs.items():
    print(f"  {c}: {len(pairs)} pairs")

# Compute currency returns for each bar
# Weight by inverse volatility (adaptive)
curr_rets = np.zeros((n_lr, len(currencies)))
for ci, c in enumerate(currencies):
    pairs = curr_pairs[c]
    if len(pairs) == 0:
        continue
    cols = [p[0] for p in pairs]
    signs = [p[1] for p in pairs]
    rets = lr[:, cols] * np.array(signs)
    # Equal-weight for now
    curr_rets[:, ci] = np.nanmean(rets, axis=1)

# Z-score normalize
curr_z = np.zeros_like(curr_rets)
for ci in range(len(currencies)):
    m = np.mean(curr_rets[:, ci])
    s = np.std(curr_rets[:, ci])
    curr_z[:, ci] = (curr_rets[:, ci] - m) / s

print(f"\nCurrency return stats:")
for ci, c in enumerate(currencies):
    print(f"  {c}: mean={np.mean(curr_rets[:,ci]):.6f} std={np.std(curr_rets[:,ci]):.6f}")

# ================================================================
# TEST: After extreme currency pressure, do exposed pairs react?
# ================================================================
print("\n" + "=" * 60)
print("CURRENCY PRESSURE → PAIR REACTION")
print("=" * 60)

# For each extreme currency pressure event, find the most exposed pair
# and check its subsequent return
for z_thresh in [1.5, 2.0, 2.5, 3.0]:
    results = []
    for i in range(30, n_lr - 30):
        for ci, c in enumerate(currencies):
            if abs(curr_z[i, ci]) < z_thresh:
                continue
            
            direction = np.sign(curr_z[i, ci])
            
            # The most exposed pair = the one where this currency contributed most
            # For now: find pairs where this currency is a component
            exposed_pairs = curr_pairs[c]
            if len(exposed_pairs) == 0:
                continue
            
            for j, sign in exposed_pairs:
                pair = pairs_list[j]
                for fwd in [5, 10, 15, 30]:
                    if i + fwd >= n_lr:
                        continue
                    fwd_ret = lr[i:i+fwd, j].sum()
                    # Expected: if currency is being accumulated (z>0), pairs where it's base should go up
                    expected_dir = direction * sign
                    correct = np.sign(fwd_ret) == expected_dir
                    
                    results.append({
                        'currency': c,
                        'z_score': curr_z[i, ci],
                        'direction': direction,
                        'pair': pair,
                        'sign': sign,
                        'expected_dir': expected_dir,
                        'fwd': fwd,
                        'fwd_ret': fwd_ret,
                        'correct': correct,
                    })
    
    rdf = pd.DataFrame(results)
    if len(rdf) < 10:
        continue
    
    for fwd in [5, 10, 15, 30]:
        sub = rdf[rdf['fwd'] == fwd]
        if len(sub) < 5:
            continue
        wr = np.mean(sub['correct']) * 100
        avg = np.mean(sub['fwd_ret']) * 10000
        print(f"  |Z|>{z_thresh:.1f} fwd={fwd:>2d}: n={len(sub):>4d} WR={wr:.0f}% avg={avg:+.1f}bps")

# Per-currency breakdown for best parameters
print("\n" + "=" * 60)
print("PER-CURRENCY BREAKDOWN (|Z|>2.0, fwd=15)")
print("=" * 60)

sub = pd.DataFrame()
for i in range(30, n_lr - 30):
    for ci, c in enumerate(currencies):
        if abs(curr_z[i, ci]) < 2.0:
            continue
        direction = np.sign(curr_z[i, ci])
        exposed_pairs = curr_pairs[c]
        for j, sign in exposed_pairs:
            pair = pairs_list[j]
            for fwd in [5, 10, 15]:
                if i + fwd >= n_lr:
                    continue
                fwd_ret = lr[i:i+fwd, j].sum()
                expected_dir = direction * sign
                correct = np.sign(fwd_ret) == expected_dir
                sub = pd.concat([sub, pd.DataFrame([{
                    'currency': c, 'z': curr_z[i, ci], 'pair': pair,
                    'fwd': fwd, 'fwd_ret': fwd_ret, 'correct': correct
                }])])

if len(sub) > 0:
    for fwd in [5, 10, 15]:
        print(f"\n  Fwd={fwd}m:")
        fsub = sub[sub['fwd'] == fwd]
        for c in currencies:
            csub = fsub[fsub['currency'] == c]
            if len(csub) < 5:
                continue
            wr = np.mean(csub['correct']) * 100
            avg = np.mean(csub['fwd_ret']) * 10000
            print(f"    {c:>4s}: n={len(csub):>4d} WR={wr:.0f}% avg={avg:+.1f}bps")
