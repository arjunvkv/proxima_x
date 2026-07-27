"""Test dual-pair tick exhaustion confirmation.
When EURUSD AND GBPUSD both show tick exhaustion simultaneously, does WR improve?"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

def compute_streaks(df):
    dirs = np.sign(df.mid.diff().fillna(0)).astype(np.int8)
    streaks = np.zeros(len(df), dtype=np.int32)
    cur = 0
    for i in range(1, len(df)):
        d = int(dirs[i])
        if d == 1: cur = cur + 1 if cur >= 0 else 1
        elif d == -1: cur = cur - 1 if cur <= 0 else -1
        streaks[i] = cur
    df['streak'] = streaks
    df['abs_streak'] = abs(streaks)
    df['dir'] = dirs
    return df

print("=== Dual-Pair Tick Exhaustion ===")

# Load EU + GU and align tick-by-tick
eu = pd.read_csv(DATA / "EURUSD_mt5.csv", parse_dates=['time'])
gu = pd.read_csv(DATA / "GBPUSD_mt5.csv", parse_dates=['time'])
eu['mid'] = (eu.bid + eu.ask) / 2
gu['mid'] = (gu.bid + gu.ask) / 2

# Merge on nearest timestamp (asof merge for tick alignment)
eu = eu.sort_values('time')
gu = gu.sort_values('time')
merged = pd.merge_asof(eu, gu, on='time', suffixes=('_eu', '_gu'), tolerance=pd.Timedelta('50ms'))
merged = merged.dropna(subset=['mid_eu', 'mid_gu'])
print(f"Merged ticks: {len(merged):,}")

# Compute streaks on merged
merged['mid_eu'] = merged['mid_eu'].astype(float)
merged['mid_gu'] = merged['mid_gu'].astype(float)
streaks_eu = np.zeros(len(merged), dtype=np.int32)
streaks_gu = np.zeros(len(merged), dtype=np.int32)

mid_eu_arr = merged.mid_eu.values.astype(float)
mid_gu_arr = merged.mid_gu.values.astype(float)

dirs_eu = np.sign(np.diff(mid_eu_arr))
dirs_gu = np.sign(np.diff(mid_gu_arr))

cur = 0
for i in range(1, len(merged)):
    d = int(dirs_eu[i-1])
    if d == 1: cur = cur + 1 if cur >= 0 else 1
    elif d == -1: cur = cur - 1 if cur <= 0 else -1
    streaks_eu[i] = cur
cur = 0
for i in range(1, len(merged)):
    d = int(dirs_gu[i-1])
    if d == 1: cur = cur + 1 if cur >= 0 else 1
    elif d == -1: cur = cur - 1 if cur <= 0 else -1
    streaks_gu[i] = cur

merged['streak_eu'] = streaks_eu
merged['streak_gu'] = streaks_gu
merged['streak_eu_abs'] = abs(streaks_eu)
merged['streak_gu_abs'] = abs(streaks_gu)
merged['dir_eu'] = np.concatenate([[0], dirs_eu])
merged['dir_gu'] = np.concatenate([[0], dirs_gu])

# Compare single vs dual exhaustion
for min_s in [3, 5]:
    # Single pair: EURUSD only
    mask_single = (merged.streak_eu_abs >= min_s) & (merged.dir_eu != 0)
    idx_single = merged[mask_single].index.values
    
    # Dual: both pairs exhausted, same direction
    mask_dual = ((merged.streak_eu_abs >= min_s) & (merged.dir_eu != 0) &
                 (merged.streak_gu_abs >= min_s) & (merged.dir_gu != 0) &
                 (np.sign(merged.streak_eu) == np.sign(merged.streak_gu)))
    idx_dual = merged[mask_dual].index.values
    
    pip_mul = 10000  # EURUSD pip multiplier
    for exit_n in [5, 10, 30]:
        for label, idx in [("EURUSD_only", idx_single), ("EURUSD_GBPUSD", idx_dual)]:
            exit_idx = idx + exit_n
            valid = exit_idx < len(merged)
            ei, xi = idx[valid], exit_idx[valid]
            if len(ei) < 5: continue
            
            entry_mid = merged.mid_eu.values[ei]
            exit_mid = merged.mid_eu.values[xi]
            direction = np.sign(merged.streak_eu.values[ei]) * -1  # fade the streak
            
            gains = np.where(direction > 0, (exit_mid - entry_mid) * pip_mul,
                             (entry_mid - exit_mid) * pip_mul)
            wr = (gains > 0).mean() * 100
            avg = gains.mean()
            print(f"  streak≥{min_s} {label:20s} +{exit_n:<3d}: WR={wr:5.1f}% avg={avg:+.2f} N={len(ei):,}")

print("\n=== DONE ===")
