"""
Tick-Level Triangle Arbitrage Detection.
EURJPY ≠ EURUSD × USDJPY at tick level = forced convergence.
Market cannot hide this — it's arithmetic.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path
from datetime import timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]

t0 = time.time()

# Load ALL 3 pairs
all_t = {}
for pair in ['EURUSD', 'USDJPY', 'EURJPY']:
    dfs = []
    for y, m in MONTHS:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts': str, 'B': np.float64, 'A': np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['Mid'] = (df['B'] + df['A']) / 2
    df['Sprd_p'] = (df['A'] - df['B']) * 10000 if pair == 'EURUSD' else (df['A'] - df['B']) * 100
    all_t[pair] = df[['Mid', 'Sprd_p']]
    print(f"  {pair}: {len(df):,d} ticks  ({time.time()-t0:.1f}s)")

# Check overlap
print("\nAligning ticks...")
# Use EURUSD as reference, find nearest EURJPY and USDJPY ticks
eus = all_t['EURUSD']
usj = all_t['USDJPY']
eur = all_t['EURJPY']

# Merge all three on timestamp: inner join, nearest 100ms
# Since timestamps aren't exact across pairs, create aligned 100ms windows
aligned = pd.DataFrame(index=eus.index)
aligned['eurusd'] = eus['Mid'].values
aligned['eurusd_s'] = eus['Sprd_p'].values
aligned['ts_eus'] = eus.index

# For each EURUSD tick, find nearest USDJPY and EURJPY tick within 500ms
# Vectorized: reindex with tolerance
print("Merging USDJPY...")
usj_idx = usj.index
aligned['usdjpy'] = np.nan
aligned['ts_usj'] = pd.NaT

# Use merge_asof for nearest-tick matching
temp_eus = eus[['Mid']].rename(columns={'Mid': 'eurusd'}).sort_index()
temp_usj = usj[['Mid']].rename(columns={'Mid': 'usdjpy'}).sort_index()
temp_eur = eur[['Mid']].rename(columns={'Mid': 'eurjpy'}).sort_index()

merged = pd.merge_asof(temp_eus, temp_usj, left_index=True, right_index=True, tolerance=pd.Timedelta('100ms'), direction='nearest')
merged = pd.merge_asof(merged, temp_eur, left_index=True, right_index=True, tolerance=pd.Timedelta('100ms'), direction='nearest')
merged = merged.dropna()
print(f"  Aligned ticks with 100ms tolerance: {len(merged):,d}")

# Compute synthetic EURJPY = EURUSD * USDJPY
merged['synthetic'] = merged['eurusd'] * merged['usdjpy']
merged['triangle_error'] = (merged['eurjpy'] - merged['synthetic'])
# In pip terms for EURJPY (100 scale)
merged['error_pips'] = merged['triangle_error'] * 100
merged['error_abs'] = merged['error_pips'].abs()
merged['error_sign'] = np.sign(merged['error_pips'])

# Rolling stats on error
merged['roll_err_mean'] = merged['error_pips'].rolling(500).mean()
merged['roll_err_std'] = merged['error_pips'].rolling(500).std()
merged['error_z'] = (merged['error_pips'] - merged['roll_err_mean']) / merged['roll_err_std'].clip(lower=1e-8)

print(f"\n  Mean error: {merged['error_pips'].mean():+.4f}p")
print(f"  Std error: {merged['error_pips'].std():.4f}p")
print(f"  Max error: {merged['error_pips'].max():+.4f}p")
print(f"  Min error: {merged['error_pips'].min():+.4f}p")
print(f"  |error| > 0.5p: {(merged['error_abs'] > 0.5).mean():.2%}")
print(f"  |error| > 1.0p: {(merged['error_abs'] > 1.0).mean():.2%}")

# Build 100ms bars to test predictive power
merged['time_bin'] = merged.index.floor('100ms')
bars_100ms = merged.groupby('time_bin').agg({
    'error_pips': 'last', 'error_z': 'last',
    'eurjpy': 'last', 'synthetic': 'last',
    'eurusd': 'last', 'usdjpy': 'last',
    'error_abs': 'max'
})
print(f"\n  100ms bars: {len(bars_100ms):,d}")

# Forward EURJPY return over next 500ms
bars_100ms['eurjpy_fwd_5'] = bars_100ms['eurjpy'].shift(-5) - bars_100ms['eurjpy']
bars_100ms['eurjpy_fwd_10'] = bars_100ms['eurjpy'].shift(-10) - bars_100ms['eurjpy']
bars_100ms['eurjpy_fwd_20'] = bars_100ms['eurjpy'].shift(-20) - bars_100ms['eurjpy']

# Direction: if synthetic > actual (error < 0), EURJPY should go UP, so go LONG
# direction-adjusted return = -sign(error) * fwd_ret
bars_100ms['dir_ret_5'] = -np.sign(bars_100ms['error_pips'].fillna(0)) * bars_100ms['eurjpy_fwd_5'].fillna(0) * 100
bars_100ms['dir_ret_10'] = -np.sign(bars_100ms['error_pips'].fillna(0)) * bars_100ms['eurjpy_fwd_10'].fillna(0) * 100
bars_100ms['dir_ret_20'] = -np.sign(bars_100ms['error_pips'].fillna(0)) * bars_100ms['eurjpy_fwd_20'].fillna(0) * 100

# Test: when |error_z| > threshold, does EURJPY converge?
print(f"\n--- Predictive Power: Error Z-score thresholds ---")
print(f"  {'|error_z|>':<10s} {'n':>8s} {'n/hr':>6s} {'WR_5':>7s} {'WR_10':>7s} {'WR_20':>7s} {'avg_5':>9s} {'avg_10':>9s} {'avg_20':>9s}")
print(f"  {'-'*75}")

for thr in [0.5, 1.0, 1.5, 2.0, 3.0]:
    mask = bars_100ms['error_z'].abs() > thr
    n = mask.sum()
    if n < 20: continue
    sub = bars_100ms.loc[mask]
    w5 = (sub['dir_ret_5'] > 0).mean()
    w10 = (sub['dir_ret_10'] > 0).mean()
    w20 = (sub['dir_ret_20'] > 0).mean()
    a5 = sub['dir_ret_5'].mean()
    a10 = sub['dir_ret_10'].mean()
    a20 = sub['dir_ret_20'].mean()
    n_hr = n / (len(bars_100ms) / 36000)  # 36000 = 100ms bars per hour
    print(f"  |z|>{thr:<4.1f}  {n:>8,d} {n_hr:>5.0f}  {w5:>6.1%} {w10:>6.1%} {w20:>6.1%} {a5:>+9.4f}p {a10:>+9.4f}p {a20:>+9.4f}p")

# More important: the derivative of error (error velocity) 
bars_100ms['err_dot'] = bars_100ms['error_pips'].diff()
bars_100ms['err_dot_z'] = bars_100ms['err_dot'] / bars_100ms['err_dot'].rolling(100).std().clip(lower=1e-8)

print(f"\n--- Error Velocity (ACCELERATING error) ---")
print(f"  {'err_dot_z>':<12s} {'n':>8s} {'n/hr':>6s} {'WR_5':>7s} {'WR_10':>7s} {'WR_20':>7s} {'avg_5':>9s} {'avg_10':>9s} {'avg_20':>9s}")
print(f"  {'-'*75}")

for thr in [1.0, 2.0, 3.0, 5.0]:
    mask = bars_100ms['err_dot_z'].abs() > thr
    n = mask.sum()
    if n < 20: continue
    sub = bars_100ms.loc[mask]
    w5 = (sub['dir_ret_5'] > 0).mean()
    w10 = (sub['dir_ret_10'] > 0).mean()
    w20 = (sub['dir_ret_20'] > 0).mean()
    a5 = sub['dir_ret_5'].mean()
    a10 = sub['dir_ret_10'].mean()
    a20 = sub['dir_ret_20'].mean()
    n_hr = n / (len(bars_100ms) / 36000)
    print(f"  |ed|>{thr:<4.1f}  {n:>8,d} {n_hr:>5.0f}  {w5:>6.1%} {w10:>6.1%} {w20:>6.1%} {a5:>+9.4f}p {a10:>+9.4f}p {a20:>+9.4f}p")

# Key test: when error spikes AND spread on EURJPY is wide (dealer not correcting)
print(f"\n--- Error Spike + Spread Confirmation ---")
merged['sprd_z'] = merged['Sprd_p'] / merged['Sprd_p'].rolling(500).mean().clip(lower=1e-8)
bars_100ms['sprd_mean'] = merged['Sprd_p'].resample('100ms').mean()
bars_100ms['sprd_z'] = bars_100ms['sprd_mean'] / bars_100ms['sprd_mean'].rolling(100).mean().clip(lower=1e-8)

for thr in [1.5, 2.0, 3.0]:
    err_mask = bars_100ms['error_z'].abs() > thr
    sprd_mask = bars_100ms['sprd_z'] > 1.5
    mask = err_mask & sprd_mask
    n = mask.sum()
    if n < 10: continue
    sub = bars_100ms.loc[mask]
    w5 = (sub['dir_ret_5'] > 0).mean()
    w10 = (sub['dir_ret_10'] > 0).mean()
    a5 = sub['dir_ret_5'].mean()
    a10 = sub['dir_ret_10'].mean()
    n_hr = n / (len(bars_100ms) / 36000)
    print(f"  err|z|>{thr}+sprd  {n:>5,d} {n_hr:>5.0f}  {w5:>6.1%} {w10:>6.1%} {a5:>+9.4f}p {a10:>+9.4f}p")

print(f"\nTotal: {time.time()-t0:.1f}s")
