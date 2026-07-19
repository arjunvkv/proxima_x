"""Download Exness tick data for multiple pairs/months + run full analysis."""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from datetime import datetime, timezone
from numba import jit

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
TICK_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = ['EURJPY', 'GBPJPY', 'EURUSD']
# Download Oct-Dec 2025 (3 months, completely OOS from our 2026 M1 data)
MONTHS = [(2025, 12), (2025, 11), (2025, 10)]

@jit(nopython=True)
def rolling_beta(x, y, lb):
    n = len(x); beta = np.zeros(n)
    for i in range(lb, n):
        xw = x[i-lb:i]; yw = y[i-lb:i]
        xm = np.mean(xw); ym = np.mean(yw)
        num = np.sum((xw-xm)*(yw-ym))
        den = np.sum((xw-xm)**2)
        beta[i] = num/den if den != 0 else 0
    return beta

# ============================================================
# 1. DOWNLOAD
# ============================================================
print("=" * 70)
print("DOWNLOADING EXNESS TICK DATA")
print("=" * 70)

zips_downloaded = []
for symbol in [f'{p}_Raw_Spread' for p in PAIRS]:
    for year, month in MONTHS:
        url = f'https://ticks.ex2archive.com/ticks/{symbol}/{year}/{month:02d}/Exness_{symbol}_{year}_{month:02d}.zip'
        dest = TICK_DIR / f'{symbol}_{year}_{month:02d}.zip'
        if dest.exists():
            size = dest.stat().st_size / 1e6
            print(f'  EXISTS {dest.name} ({size:.1f} MB)')
        else:
            try:
                print(f'  Downloading {url}...', end=' ')
                urlretrieve(url, dest)
                size = dest.stat().st_size / 1e6
                print(f'{size:.1f} MB')
            except Exception as e:
                print(f'FAILED: {e}')
                continue
        zips_downloaded.append(dest)

# ============================================================
# 2. LOAD AND BUILD M1 BARS FROM TICKS
# ============================================================
print("\n" + "=" * 70)
print("BUILDING M1 BARS FROM TICKS")
print("=" * 70)

all_tick_data = {}
for pair in PAIRS:
    print(f'Loading {pair}...')
    pair_dfs = []
    for year, month in MONTHS:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{year}_{month:02d}.zip'
        if not fn.exists():
            continue
        df = pd.read_csv(fn, compression='zip',
                         names=['Exness', 'Symbol', 'Timestamp', 'Bid', 'Ask'],
                         skiprows=1, header=None, on_bad_lines='skip',
                         dtype={'Exness': str, 'Symbol': str, 'Timestamp': str, 'Bid': float, 'Ask': float})
        # Parse timestamp
        df['Timestamp'] = df['Timestamp'].str.replace('Z', '', regex=False)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['Mid'] = (df['Bid'] + df['Ask']) / 2
        pair_dfs.append(df)
    
    if pair_dfs:
        all_tick_data[pair] = pd.concat(pair_dfs, ignore_index=True)
        all_tick_data[pair] = all_tick_data[pair].sort_values('Timestamp')
        n_ticks = len(all_tick_data[pair])
        t0 = all_tick_data[pair]['Timestamp'].min()
        t1 = all_tick_data[pair]['Timestamp'].max()
        print(f'  {n_ticks:,} ticks, {t0} to {t1}')

# Build M1 bars from ticks
print('\nResampling to M1...')
m1_data = {}
for pair in PAIRS:
    df = all_tick_data[pair]
    df = df.set_index('Timestamp')
    ohlc = df['Mid'].resample('1min').ohlc()
    ohlc = ohlc.dropna()
    m1_data[pair] = ohlc
    print(f'  {pair}: {len(ohlc)} M1 bars, {ohlc.index[0]} to {ohlc.index[-1]}')

# Align all pairs
common_idx = m1_data[PAIRS[0]].index
for pair in PAIRS[1:]:
    common_idx = common_idx.intersection(m1_data[pair].index)

print(f'\nAligned: {len(common_idx)} M1 bars')
print(f'Range: {common_idx[0]} to {common_idx[-1]}')

# Build close arrays
close = {}
for pair in PAIRS:
    close[pair] = m1_data[pair].loc[common_idx, 'close'].values.astype(np.float64)

days = [t.strftime('%a') for t in common_idx]
unique_days = sorted(set(days), key=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(d))
print(f'Days: {unique_days}')
print(f'Bars per day: {[days.count(d) for d in unique_days]}')

# ============================================================
# 3. FULL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("COMBINED 3-SIGNAL ANALYSIS ON TICK-DERIVED M1 DATA")
print("=" * 70)

ej_c, gj_c, eu_c = close['EURJPY'], close['GBPJPY'], close['EURUSD']
ej_ret, gj_ret, eu_ret = np.diff(ej_c), np.diff(gj_c), np.diff(eu_c)
ns = len(ej_ret)

LB = 10
HOLD = 20

beta_eu_ej = rolling_beta(eu_ret, ej_ret, LB)
beta_ej_gj = rolling_beta(ej_ret, gj_ret, LB)
beta_eu_gj = rolling_beta(eu_ret, gj_ret, LB)

def_eu_ej = np.array([(beta_eu_ej[i]*eu_ret[i-1]-ej_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])
def_ej_gj = np.array([(beta_ej_gj[i]*ej_ret[i-1]-gj_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])
def_eu_gj = np.array([(beta_eu_gj[i]*eu_ret[i-1]-gj_ret[i-1])*100 if i>=LB else 0 for i in range(ns)])

catch_gj = np.array([np.sum(gj_ret[i:i+HOLD])*100 if i<=ns-HOLD else 0 for i in range(ns)])
catch_ej = np.array([np.sum(ej_ret[i:i+HOLD])*100 if i<=ns-HOLD else 0 for i in range(ns)])

# A. OVERALL
print("\n--- OVERALL (z thresholds) ---")
for zt in [1.0, 1.5, 2.0]:
    s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    
    configs = [
        ("EURJPY->GBPJPY", s2, catch_gj),
        ("EURUSD->GBPJPY", s3, catch_gj),
        ("1 AND 2 (EU_EJ AND EJ_GJ)", s1 & s2, catch_gj),
        ("2 AND 3 (EJ_GJ AND EU_GJ)", s2 & s3, catch_gj),
        ("ALL 3 AND", s1 & s2 & s3, catch_gj),
    ]
    print(f"\n  z > {zt:.1f}:")
    for name, sig, ca in configs:
        mask = sig & (ca != 0); idx = np.where(mask & (np.arange(ns) >= LB))[0]; n = len(idx)
        if n < 5: continue
        vals = ca[idx]; wr = np.mean(vals > 0); avg = np.mean(vals)
        wins = vals[vals>0]; losses = vals[vals<=0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {name:<30} {wr:>5.0%} {n:>5d}  avg={avg:>5.2f}p  ev={ev:>+5.2f}p")

# B. DAY BY DAY
print("\n" + "=" * 70)
print("DAY-BY-DAY: BEST CONFIGS")
print("=" * 70)

zt = 1.5
s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
sig_all = s1 & s2 & s3
sig_both = s2 & s3

for name, sig in [("ALL 3 AND", sig_all), ("EURJPY->GBPJPY AND EURUSD->GBPJPY", sig_both), ("EURJPY->GBPJPY alone", s2)]:
    print(f"\n  {name} (z>{zt:.0f}):")
    print(f"  {'Day':<6} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("  " + "-" * 38)
    total_n, all_vals = 0, []
    for d in unique_days:
        day_vals = []
        for i in range(LB, ns - HOLD):
            if days[i] == d and sig[i] and catch_gj[i] != 0:
                day_vals.append(catch_gj[i])
        n = len(day_vals)
        if n < 2:
            print(f"  {d:<6} {'--':>6} {n:>7d}")
            continue
        vals = np.array(day_vals); wr = np.mean(vals > 0); avg = np.mean(vals)
        wins = vals[vals>0]; losses = vals[vals<=0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {d:<6} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")
        total_n += n; all_vals.extend(day_vals)
    if total_n > 0:
        a = np.array(all_vals); wr_t = np.mean(a>0); avg_t = np.mean(a)
        w = a[a>0]; l = a[a<=0]
        ev_t = wr_t*np.mean(w)+(1-wr_t)*np.mean(l) if len(w)>0 and len(l)>0 else 0
        print(f"  {'TOTAL':<6} {wr_t:>5.0%} {total_n:>7d} {avg_t:>7.2f} {ev_t:>+7.2f}")
