#!/usr/bin/env python3
"""Validate strategy on fresh Dukascopy Q1 2026 data (Jan-Mar)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"

# Load and merge all 3 pairs
pair_data = {}
for p, pname in [("eurjpy","EURJPY"), ("eurusd","EURUSD"), ("gbpjpy","GBPJPY")]:
    frames = []
    for m in range(1, 4):
        f = os.path.join(DATA, f"{p}-m1-bid-2026-{m:02d}-01-2026-{m:02d}-{28 if m==2 else 31}.csv")
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        frames.append(df)
    pair_data[pname] = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"{pname}: {len(pair_data[pname])} bars, {pair_data[pname]['timestamp'].min()} — {pair_data[pname]['timestamp'].max()}")

# Align by common timestamps
common = sorted(set(pair_data["EURJPY"]["timestamp"]) & set(pair_data["EURUSD"]["timestamp"]) & set(pair_data["GBPJPY"]["timestamp"]))
tmap = {p: {t: i for i, t in enumerate(pair_data[p]["timestamp"])} for p in pair_data}
close = np.column_stack([pair_data[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
T = close.shape[0]
print(f"\nAligned bars: {T}, timestamps: {pd.to_datetime(times[0],unit='s')} — {pd.to_datetime(times[-1],unit='s')}")

rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]

MIN_IDX = 1440
LOT = 100000
costs_a = np.array([0.00008*2, 0.00005*2, 0.00010*2])
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM = 7

def pip_value_usd(p, usdjpy_rate):
    if p == 1: return 10.0
    else: return 1000.0 / usdjpy_rate

print("="*70)
print("Q1 2026 VALIDATION: P95+best_pair+H3 (trained on Oct-Dec 2015)")
print("="*70)

# Use fixed threshold from original training
MAG95 = 0.00018741

te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
te_idx = te_idx[te_idx + 3 < T - 1]
bi = np.argmax(pair_mags[te_idx], axis=1)

avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
dollars = []
for j,i in enumerate(te_idx):
    p = bi[j]
    gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p] * 2 * 1.5 * pip_value_usd(p, avg_usdjpy)
    slip = 0.5 * 2 * pip_value_usd(p, avg_usdjpy)
    if p == 1: gusd = LOT * gross
    else: gusd = LOT * gross * close[i,p] / usdjpy_proxy[i]
    dollars.append(gusd - spread - slip - ECN_COMM)

d = np.array(dollars)
n = len(d); wr = np.mean(d>0)*100
sh = np.mean(d)/(np.std(d)+1e-10)*np.sqrt(1440/3)
tpd = n/(T/1440)
print(f"  Config:        1.5x spread, 0.5p slippage, \$7 ECN comm")
print(f"  Trades:        {n}")
print(f"  Trades/day:    {tpd:.1f}")
print(f"  Win rate:      {wr:.1f}%")
print(f"  Avg per trade: \${np.mean(d):.2f}")
print(f"  Total:         \${np.sum(d):.0f}")
print(f"  Sharpe:        {sh:.2f}")
print(f"  Daily:         \${np.mean(d)*tpd:.0f}")
print(f"  Monthly (21d): \${np.mean(d)*tpd*21:.0f}")

# Monthly breakdown
ht = pd.DatetimeIndex([dt_all[ti] for ti in te_idx])
print(f"\n  Monthly:")
for m in sorted(set(ht.month)):
    mask = ht.month == m
    mp = d[mask]
    msh = np.mean(mp)/(np.std(mp)+1e-10)*np.sqrt(1440/3) if len(mp) > 5 else 0
    print(f"    {pd.to_datetime(f'2026-{m:02d}-01').strftime('%B')}: n={len(mp):4d}  WR={np.mean(mp>0)*100:.1f}%  Sharpe={msh:.2f}  Avg=\${np.mean(mp):.2f}  Tot=\${np.sum(mp):.0f}")

# Also test with rolling adaptive thresholds (no lookahead)
print(f"\n  Rolling adaptive thresholds:")
pnls_r = []
for t in range(MIN_IDX, T - 1 - 3):
    if not consensus[t]: continue
    h = hour_arr[t]
    if h < 7 or h > 21: continue
    lo = max(0, t - MIN_IDX)
    mag_t = np.percentile(avg_mag[lo:t], 95)
    if avg_mag[t] <= mag_t: continue
    p = np.argmax(pair_mags[t])
    gross = np.log(close[t+3,p]/close[t,p])*direction[t]
    spread = HALF_SPREAD_PIPS[p] * 2 * 1.5 * pip_value_usd(p, avg_usdjpy)
    slip = 0.5 * 2 * pip_value_usd(p, avg_usdjpy)
    if p == 1: gusd = LOT * gross
    else: gusd = LOT * gross * close[t,p] / usdjpy_proxy[t]
    pnls_r.append(gusd - spread - slip - ECN_COMM)

dr = np.array(pnls_r)
if len(dr) > 10:
    print(f"  n={len(dr):5d}  WR={np.mean(dr>0)*100:.1f}%  Sharpe={np.mean(dr)/(np.std(dr)+1e-10)*np.sqrt(1440/3):.2f}  Avg=\${np.mean(dr):.2f}  Tot=\${np.sum(dr):.0f}")

# Pair distribution
pair_names = ["EURJPY","EURUSD","GBPJPY"]
pair_counts = np.bincount(bi, minlength=3)
for pi, pn in enumerate(pair_names):
    print(f"  {pn}: {pair_counts[pi]} ({pair_counts[pi]/n*100:.0f}%)")

# Also run the full sensitivity grid
print(f"\n{'='*70}")
print("SENSITIVITY GRID: Q1 2026 Dukascopy Data")
print(f"{'='*70}")
print(f"{'Spread':>6s} {'Slip':>5s} {'n':>5s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>7s} {'Tot$':>8s}")
print("-"*70)
for sm in [1.0, 1.5, 2.0, 3.0]:
    for slip in [0.0, 0.5, 1.0]:
        pnls = []
        for j,i in enumerate(te_idx):
            p = bi[j]
            gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
            spread = HALF_SPREAD_PIPS[p]*2*sm*pip_value_usd(p, avg_usdjpy)
            slp = slip*2*pip_value_usd(p, avg_usdjpy)
            if p==1: gusd = LOT*gross
            else: gusd = LOT*gross*close[i,p]/usdjpy_proxy[i]
            pnls.append(gusd - spread - slp - ECN_COMM)
        dg = np.array(pnls)
        wrg = np.mean(dg>0)*100
        shg = np.mean(dg)/(np.std(dg)+1e-10)*np.sqrt(1440/3)
        print(f"{sm:5.1f}x {slip:5.1f}p {len(dg):5d} {wrg:5.1f} {shg:7.2f} {np.mean(dg):7.2f} {np.sum(dg):8.0f}   ")
