"""Multi-day M1 + tick analysis via MT5. Saves + analyzes."""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import MetaTrader5 as mt5

PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY']

print("Connecting to MT5...")
mt5.initialize()

# Download maximum M1 history for all pairs
max_bars = 20000
print(f"Downloading up to {max_bars} M1 bars per pair...")
data = {}
for pair in PAIRS:
    rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, max_bars)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: no data")
        continue
    times = np.array([r[0] for r in rates], dtype='u8')
    ohlc = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rates], dtype='f8')
    data[pair] = {'times': times, 'ohlc': ohlc}
    print(f"  {pair}: {len(rates)} bars, {datetime.fromtimestamp(times[0])} to {datetime.fromtimestamp(times[-1])}")

# Align all pairs to common timestamps
from collections import defaultdict
all_bars_by_time = defaultdict(list)
for pair, d in data.items():
    for i, ts in enumerate(d['times']):
        rounded = int(ts / 60) * 60
        all_bars_by_time[rounded].append((pair, i, d['ohlc'][i][3]))  # close price

required_pairs = set(data.keys())
aligned_times = []
aligned_ohlc = []

for ts in sorted(all_bars_by_time.keys()):
    bars_at_ts = all_bars_by_time[ts]
    pairs_at_ts = set(b[0] for b in bars_at_ts)
    if pairs_at_ts == required_pairs:
        aligned_times.append(ts)
        row = np.full((len(PAIRS), 6), np.nan)
        bp = {b[0]: b[1] for b in bars_at_ts}
        for pi, pair in enumerate(PAIRS):
            if pair in bp:
                idx = bp[pair]
                row[pi] = data[pair]['ohlc'][idx]
        aligned_ohlc.append(row)

aligned_times = np.array(aligned_times, dtype='u8')
aligned = np.array(aligned_ohlc)  # shape: [time, pair, feature]
print(f"\nAligned: {len(aligned_times)} bars across {len(PAIRS)} pairs")
print(f"Range: {datetime.fromtimestamp(aligned_times[0])} to {datetime.fromtimestamp(aligned_times[-1])}")

# Save to parquet for future use
out_dir = Path(__file__).resolve().parents[3] / 'data' / 'temp'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'mt5_m1_9day.parquet'
import pandas as pd

# Build a flat DataFrame
records = []
for i, ts in enumerate(aligned_times):
    dt = datetime.fromtimestamp(ts)
    for pi, pair in enumerate(PAIRS):
        row = {'time': dt, 'pair': pair, 'open': aligned[i,pi,0], 'high': aligned[i,pi,1],
               'low': aligned[i,pi,2], 'close': aligned[i,pi,3], 'volume': aligned[i,pi,4]}
        records.append(row)

df = pd.DataFrame(records)
df.to_parquet(out_path)
print(f"Saved to {out_path} ({len(df)} rows)")

mt5.shutdown()

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("COMBINED 3-SIGNAL ANALYSIS ON 9-DAY M1 DATA")
print("=" * 70)

from numba import jit

@jit(nopython=True)
def rolling_beta(x, y, lb):
    n = len(x)
    beta = np.zeros(n)
    for i in range(lb, n):
        xw = x[i-lb:i]; yw = y[i-lb:i]
        xm = np.mean(xw); ym = np.mean(yw)
        num = np.sum((xw-xm)*(yw-ym))
        den = np.sum((xw-xm)**2)
        beta[i] = num/den if den != 0 else 0
    return beta

# Build close arrays by pair index
pair_idx = {p: i for i, p in enumerate(PAIRS)}
ej_c = aligned[:, pair_idx['EURJPY'], 3]
gj_c = aligned[:, pair_idx['GBPJPY'], 3]
eu_c = aligned[:, pair_idx['EURUSD'], 3]

# Day labels
days = [datetime.fromtimestamp(t).strftime('%a') for t in aligned_times]
unique_days = sorted(set(days), key=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(d))
print(f"Days: {unique_days}")
print(f"Bars per day: {[days.count(d) for d in unique_days]}")

ej_ret = np.diff(ej_c)
gj_ret = np.diff(gj_c)
eu_ret = np.diff(eu_c)
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

# ===== A. OVERALL by z-threshold =====
print("\n--- OVERALL PERFORMANCE ---")
for zt in [1.0, 1.5, 2.0, 2.5]:
    s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    
    configs = [
        ("EURJPY->GBPJPY", s2, catch_gj),
        ("EURUSD->EURJPY", s1, catch_ej),
        ("EURUSD->GBPJPY", s3, catch_gj),
        ("EURUSD->EURJPY AND EURJPY->GBPJPY", s1 & s2, catch_gj),
        ("EURJPY->GBPJPY AND EURUSD->GBPJPY", s2 & s3, catch_gj),
        ("ALL 3 AND", s1 & s2 & s3, catch_gj),
    ]
    
    print(f"\n  z > {zt:.1f}:")
    print(f"  {'Signal':<38} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("  " + "-" * 70)
    for name, sig, catch_arr in configs:
        mask = sig & (catch_arr != 0)
        idx = np.where(mask & (np.arange(ns) >= LB))[0]
        n = len(idx)
        if n < 5:
            continue
        vals = catch_arr[idx]
        wr = np.mean(vals > 0)
        avg = np.mean(vals)
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {name:<38} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")

# ===== B. DAY-BY-DAY for best configs =====
print("\n" + "=" * 70)
print("DAY-BY-DAY: EURJPY->GBPJPY + EURUSD->GBPJPY (AND)")
print("=" * 70)

for zt in [1.5, 2.0]:
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    sig = s2 & s3
    
    print(f"\n  z > {zt:.1f}:")
    print(f"  {'Day':<6} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("  " + "-" * 38)
    total_n = 0
    all_vals = []
    for d in unique_days:
        day_vals = []
        for i in range(LB, ns - HOLD):
            if days[i] == d and sig[i] and catch_gj[i] != 0:
                day_vals.append(catch_gj[i])
        n = len(day_vals)
        if n < 2:
            print(f"  {d:<6} {'--':>6} {n:>7d}")
            continue
        vals = np.array(day_vals)
        wr = np.mean(vals > 0)
        avg = np.mean(vals)
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {d:<6} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")
        total_n += n
        all_vals.extend(day_vals)
    
    if total_n > 0:
        all_arr = np.array(all_vals)
        wr_t = np.mean(all_arr > 0)
        avg_t = np.mean(all_arr)
        wins_t = all_arr[all_arr > 0]
        losses_t = all_arr[all_arr <= 0]
        ev_t = wr_t*np.mean(wins_t)+(1-wr_t)*np.mean(losses_t) if len(wins_t)>0 and len(losses_t)>0 else 0
        print(f"  {'TOTAL':<6} {wr_t:>5.0%} {total_n:>7d} {avg_t:>7.2f} {ev_t:>+7.2f}")

# ===== C. DAY-BY-DAY: ALL 3 AND =====
print("\n" + "=" * 70)
print("DAY-BY-DAY: ALL 3 AND")
print("=" * 70)

for zt in [1.5, 2.0]:
    s1 = def_eu_ej > zt * np.std(def_eu_ej[LB:])
    s2 = def_ej_gj > zt * np.std(def_ej_gj[LB:])
    s3 = def_eu_gj > zt * np.std(def_eu_gj[LB:])
    sig = s1 & s2 & s3
    
    print(f"\n  z > {zt:.1f}:")
    print(f"  {'Day':<6} {'WR':>6} {'Trades':>7} {'Avg(p)':>8} {'EV(p)':>8}")
    print("  " + "-" * 38)
    total_n = 0
    all_vals = []
    for d in unique_days:
        day_vals = []
        for i in range(LB, ns - HOLD):
            if days[i] == d and sig[i] and catch_gj[i] != 0:
                day_vals.append(catch_gj[i])
        n = len(day_vals)
        if n < 2:
            print(f"  {d:<6} {'--':>6} {n:>7d}")
            continue
        vals = np.array(day_vals)
        wr = np.mean(vals > 0)
        avg = np.mean(vals)
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        ev = wr*np.mean(wins)+(1-wr)*np.mean(losses) if len(wins)>0 and len(losses)>0 else 0
        print(f"  {d:<6} {wr:>5.0%} {n:>7d} {avg:>7.2f} {ev:>+7.2f}")
        total_n += n
        all_vals.extend(day_vals)
    
    if total_n > 0:
        all_arr = np.array(all_vals)
        wr_t = np.mean(all_arr > 0)
        avg_t = np.mean(all_arr)
        wins_t = all_arr[all_arr > 0]
        losses_t = all_arr[all_arr <= 0]
        ev_t = wr_t*np.mean(wins_t)+(1-wr_t)*np.mean(losses_t) if len(wins_t)>0 and len(losses_t)>0 else 0
        print(f"  {'TOTAL':<6} {wr_t:>5.0%} {total_n:>7d} {avg_t:>7.2f} {ev_t:>+7.2f}")
