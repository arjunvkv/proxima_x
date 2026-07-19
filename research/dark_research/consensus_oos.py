#!/usr/bin/env python3
"""
OOS VALIDATION: best_pair+H3 on completely different sample (Jun-Jul 2026 MT5 data).
Tests fixed thresholds from Oct-Dec training AND rolling adaptive thresholds.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# Load training thresholds
BASE = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy","eurusd","gbpjpy"]:
    raw[p] = {k: np.load(os.path.join(BASE,f"{p}_m1_{k}.npy")) for k in ["prices","times"]}
common = sorted(set(raw["eurjpy"]["times"])&set(raw["eurusd"]["times"])&set(raw["gbpjpy"]["times"]))
idx_map = {k:{t:i for i,t in enumerate(raw[k]["times"])} for k in raw}
close_tr = np.column_stack([raw[k]["prices"][[idx_map[k][c] for c in common],3] for k in ["eurjpy","eurusd","gbpjpy"]])
rets_tr = np.diff(np.log(close_tr), axis=0)
avg_mag_tr = np.mean(np.abs(rets_tr), axis=1)
es_avg_tr = np.zeros(len(rets_tr))
for t in range(1, len(rets_tr)+1):
    s = max(0,t-50); es_avg_tr[t-1] = np.mean(np.sum(rets_tr[s:t]**2, axis=1))

# Training: compute fixed thresholds from Oct+Nov training months
dt_tr = pd.to_datetime(np.array(common,dtype=np.int64), unit="s")
tr_idx = np.where((dt_tr.month[1:].values >= 10) & (dt_tr.month[1:].values <= 11) & (np.arange(len(avg_mag_tr)) >= 1440))[0]
MAG_P95 = np.percentile(avg_mag_tr[tr_idx], 95)
MAG_P90 = np.percentile(avg_mag_tr[tr_idx], 90)
ES_P90 = np.nanpercentile(es_avg_tr[tr_idx], 90)
print(f"Training thresholds (Oct+Nov): MAG_P95={MAG_P95:.8f} MAG_P90={MAG_P90:.8f} ES_P90={ES_P90:.8f}")

# Load OOS data
df = pd.read_parquet(r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet")
pairs = ["eurjpy","eurusd","gbpjpy"]
pair_names = ["EURJPY","EURUSD","GBPJPY"]
pair_map = {p:i for i,p in enumerate(pair_names)}

piv = df.pivot_table(index="time", columns="pair", values=["open","high","low","close"])
piv.columns = [f"{c[1]}_{c[0]}" for c in piv.columns]
piv = piv.sort_index()

close_oos = np.column_stack([piv[f"{pn}_close"].values.astype(np.float64) for pn in pair_names])
times_oos = piv.index.values.astype(np.int64) // 10**9  # convert to unix seconds
T_oos = close_oos.shape[0]

rets_oos = np.diff(np.log(close_oos), axis=0)
hour_oos = pd.DatetimeIndex(piv.index).hour.values[1:]
dt_oos = pd.DatetimeIndex(piv.index)

up_oos = rets_oos > 0
consensus_oos = up_oos.all(axis=1) | (~up_oos).all(axis=1)
direction_oos = np.where(up_oos.all(axis=1), 1.0, -1.0)
avg_mag_oos = np.mean(np.abs(rets_oos), axis=1)
pair_mags_oos = np.abs(rets_oos)
costs_a = np.array([0.00008*2, 0.00005*2, 0.00010*2])

# ES
es_oos = np.zeros(T_oos-1)
for t in range(1, T_oos):
    s = max(0,t-50); es_oos[t-1] = np.mean(np.sum(rets_oos[s:t]**2, axis=1))

def backtest(mag_thresh, es_thresh, adaptive=False):
    pnls, dirs, entries = [], [], []
    for t in range(1440, T_oos - 1 - 3):
        if not consensus_oos[t]: continue
        h = hour_oos[t]
        if h < 7 or h > 21: continue
        
        if adaptive:
            lo = max(0, t-1440)
            mag_t = np.percentile(avg_mag_oos[lo:t], 95)
            es_t = np.nanpercentile(es_oos[lo:t], 90)
        else:
            mag_t = mag_thresh
            es_t = es_thresh
        
        if avg_mag_oos[t] <= mag_t: continue
        if es_oos[t] <= es_t: continue
        
        # best_pair: trade strongest mover
        bi = np.argmax(pair_mags_oos[t])
        ret = np.log(close_oos[t+3, bi] / close_oos[t, bi])
        pnl = ret * direction_oos[t] - costs_a[bi]
        pnls.append(pnl)
        dirs.append(direction_oos[t])
        entries.append(dt_oos[t])
    
    pnls = np.array(pnls)
    ht = pd.DatetimeIndex(entries)
    n = len(pnls)
    if n < 5: return 0, 0, 0, 0, 0, entries
    wr = np.mean(pnls > 0) * 100
    sh = np.mean(pnls)/(np.std(pnls)+1e-10)*np.sqrt(1440/3)
    tot = np.sum(pnls) * 10000
    return n, wr, sh, np.mean(pnls)*10000, tot, ht

print("="*70)
print("OOS VALIDATION: best_pair+H3 on Jun-Jul 2026 MT5 data")
print("="*70)

# Test 1: Fixed thresholds from Oct+Nov training
print("\n1. Fixed thresholds (trained on Oct+Nov):")
n, wr, sh, avg_p, tot, ht = backtest(MAG_P95, 0)  # no ES
print(f"   P95 mag only: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")
n, wr, sh, avg_p, tot, ht = backtest(MAG_P95, ES_P90)  # with ES
print(f"   P95 mag + P90 ES: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")
n, wr, sh, avg_p, tot, ht = backtest(MAG_P90, 0)  # P90 mag
print(f"   P90 mag only: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")

# Test 2: Rolling adaptive thresholds
print("\n2. Rolling adaptive thresholds (1440-bar window):")
n, wr, sh, avg_p, tot, ht = backtest(0, 0, adaptive=True)
print(f"   Rolling P95+P90: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")

# Test 3: In-sample optimal thresholds computed on OOS data itself
print("\n3. In-sample optimal (OOS data percentiles — best possible fit):")
oos_mag95 = np.percentile(avg_mag_oos[1440:], 95)
oos_mag90 = np.percentile(avg_mag_oos[1440:], 90)
print(f"   OOS M95={oos_mag95:.8f}  OOS M90={oos_mag90:.8f} (Training: M95={MAG_P95:.8f} M90={MAG_P90:.8f})")
n, wr, sh, avg_p, tot, ht = backtest(oos_mag95, 0)
print(f"   OOS M95 only: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")
n, wr, sh, avg_p, tot, ht = backtest(oos_mag90, 0)
print(f"   OOS M90 only: n={n:5d}  WR={wr:.1f}%  Sharpe={sh:.3f}  avg={avg_p:.2f}p  tot={tot:.0f}p")

# Weekly breakdown of the best config
print("\n4. Weekly breakdown (rolling adaptive):")
_, _, _, _, _, ht = backtest(0, 0, adaptive=True)
# Actually re-run to get pnls
pnls_all = []
for t in range(1440, T_oos - 1 - 3):
    if not consensus_oos[t]: continue
    h = hour_oos[t]
    if h < 7 or h > 21: continue
    lo = max(0, t-1440)
    if avg_mag_oos[t] <= np.percentile(avg_mag_oos[lo:t], 95): continue
    if es_oos[t] <= np.nanpercentile(es_oos[lo:t], 90): continue
    bi = np.argmax(pair_mags_oos[t])
    pnls_all.append(np.log(close_oos[t+3, bi]/close_oos[t, bi])*direction_oos[t]-costs_a[bi])

pnls_all = np.array(pnls_all)
ht_all = pd.DatetimeIndex(piv.index[1440:][:len(pnls_all)])
if len(pnls_all) > len(ht_all):
    pnls_all = pnls_all[:len(ht_all)]
elif len(ht_all) > len(pnls_all):
    ht_all = ht_all[:len(pnls_all)]
for w in sorted(set(ht_all.isocalendar().week)):
    mask = ht_all.isocalendar().week == w
    wp = pnls_all[mask]
    if len(wp) < 3: continue
    print(f"   Week {w}: n={len(wp):4d}  WR={np.mean(wp>0)*100:.1f}%  Sharpe={np.mean(wp)/(np.std(wp)+1e-10)*np.sqrt(1440/3):.3f}  tot={np.sum(wp)*10000:.0f}p")

# Also print the complete OOS stats
print(f"\n5. Complete OOS period:")
print(f"   Period: {piv.index[0]} — {piv.index[-1]}")
print(f"   Bars: {T_oos} per pair")
print(f"   Consensus events: {np.sum(consensus_oos)}")
print(f"   Total trades (rolling): {len(pnls_all)}")
print(f"   Trades/day: {len(pnls_all)/(T_oos/1440):.1f}")
