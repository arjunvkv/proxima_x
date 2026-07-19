#!/usr/bin/env python3
"""Pareto WR vs trade count + stability + bootstrap + multi-pair."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time

t0 = time.time()
base = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy","eurusd","gbpjpy"]:
    raw[p] = {k: np.load(os.path.join(base,f"{p}_m1_{k}.npy")) for k in ["prices","times"]}
common = sorted(set(raw["eurjpy"]["times"]) & set(raw["eurusd"]["times"]) & set(raw["gbpjpy"]["times"]))
# Build index maps for O(1) alignment
idx_map = {}
for k in raw:
    idx_map[k] = {t: i for i, t in enumerate(raw[k]["times"])}
close = np.column_stack([raw[k]["prices"][[idx_map[k][c] for c in common], 3] for k in ["eurjpy","eurusd","gbpjpy"]])
times = np.array(common, dtype=np.int64); T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
print(f"Data: {T} bars in {time.time()-t0:.1f}s", flush=True)

up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]

# Vectorized ES computation using cumsum
t1 = time.time()
sq = rets ** 2
cum = np.zeros((T, 3))
cum[1:] = np.cumsum(sq, axis=0)
es_full = np.zeros((T, 3))
es_full[50:] = cum[50:] - cum[:T-50]
es_full[:50] = np.nan
es_avg = np.nanmean(es_full[1:], axis=1)
print(f"ES: {time.time()-t1:.1f}s", flush=True)

costs = {"EURUSD": 0.00005*2, "EURJPY": 0.00008*2, "GBPJPY": 0.00010*2}
costs_a = np.array([0.00008*2, 0.00005*2, 0.00010*2])

# Precompute rolling percentiles via pandas rolling quantile
LB = 1440
t1 = time.time()
mag_s = pd.Series(avg_mag)
es_s = pd.Series(es_avg)
mag_pct = {}
for p in [0, 70, 80, 85, 90, 92.5, 95]:
    mag_pct[p] = mag_s.rolling(LB, min_periods=LB).quantile(p/100).values
es_pct = {}
for p in [0, 70, 80, 90]:
    es_pct[p] = es_s.rolling(LB, min_periods=LB).quantile(p/100).values
print(f"Percentiles: {time.time()-t1:.1f}s", flush=True)

def eval_config(s0, s1, mp, ep, fwd, ex):
    sess = (hour_arr >= s0) & (hour_arr <= s1)
    v = consensus.copy()
    if mp > 0: v &= (avg_mag > mag_pct[mp][:T-1])
    if ep > 0: v &= (es_avg > es_pct[ep][:T-1])
    v &= sess
    v = v[:(T-1-fwd)]
    idx = np.where(v)[0]
    if len(idx) < 10: return 0, 0, 0, 0
    if ex == "best_pair":
        bi = np.argmax(pair_mags[idx], axis=1)
        rf = np.array([np.log(close[i+fwd, bi[j]] / close[i, bi[j]]) for j, i in enumerate(idx)])
        c = costs_a[bi]
    else:
        rf = np.log(close[idx+fwd, 1] / close[idx, 1])
        c = costs["EURUSD"]
    pnl = rf * direction[idx] - c
    n = len(pnl)
    sh = np.mean(pnl)/(np.std(pnl)+1e-10)*np.sqrt(1440/fwd)
    wr = np.mean(pnl > 0)*100
    return sh, wr, n, n/((T-1-fwd-LB)/1440)

t2 = time.time()
print("="*80, flush=True)
print("PARETO FRONTIER", flush=True)
print("="*80, flush=True)
print(f"{'Config':<38s}  {'n':>5s}  {'WR%':>5s}  {'Sharpe':>6s}  {'/day':>6s}", flush=True)
print("-"*80, flush=True)
all_r = []
for mp in [0,70,80,85,90,92.5,95]:
    for ep in [0,70,80,90]:
        for fwd in [3,5,10]:
            for ex in ["EURUSD","best_pair"]:
                sh,wr,n,rate = eval_config(7,21,mp,ep,fwd,ex)
                if n < 30: continue
                all_r.append((sh,wr,n,rate,f"P{mp}m+P{ep}e+H{fwd} {ex}"))
all_r.sort(key=lambda x: -x[0])
for r in all_r[:30]:
    sh,wr,n,rate,label = r
    print(f"{label:<38s}  {n:5d}  {wr:5.1f}  {sh:6.3f}  {rate:6.1f}", flush=True)

print(f"\nEval time: {time.time()-t2:.1f}s", flush=True)

# Rest of analysis with optimized primary config
print("\n"+"="*80, flush=True)
print("HIGH WR > 60%  &  n > 500", flush=True)
print("="*80, flush=True)
for r in all_r:
    sh,wr,n,rate,label = r
    if wr >= 60 and n >= 500:
        print(f"{label:<38s}  {n:5d}  {wr:5.1f}  {sh:6.3f}  {rate:6.1f}", flush=True)

print("\n"+"="*80, flush=True)
print("BOOTSTRAP: Primary (H07-H21 P90m+P90e H5 EURUSD)", flush=True)
print("="*80, flush=True)
idx = np.where(consensus[:(T-1-5)]&(hour_arr[:(T-1-5)]>=7)&(hour_arr[:(T-1-5)]<=21)
    &(avg_mag[:(T-1-5)]>mag_pct[90][:T-1-5])&(es_avg[:(T-1-5)]>es_pct[90][:T-1-5]))[0]
pnls = np.log(close[idx+5,1]/close[idx,1])*direction[idx]-0.00005*2
N=len(pnls); ow=np.mean(pnls>0)*100; os_=np.mean(pnls)/(np.std(pnls)+1e-10)*np.sqrt(1440/5)
print(f"  n={N}, WR={ow:.1f}%, Sharpe={os_:.3f}", flush=True)
np.random.seed(42)
# Permute SIGNS to test if WR > 50% is real
perm_wr = np.array([np.mean(np.random.choice([-1,1], N) * np.abs(pnls) > 0) for _ in range(5000)])
print(f"  Sign-permutation test: p(WR>=by chance) = {np.mean(perm_wr>=np.mean(pnls>0)):.6f}", flush=True)
print(f"  95% CI perm WR: [{np.percentile(perm_wr*100,2.5):.1f}%, {np.percentile(perm_wr*100,97.5):.1f}%]", flush=True)
sub = np.array([np.mean(s:=pnls[np.random.choice(N,int(N*0.5),0)])/(np.std(s)+1e-10)*np.sqrt(1440/5) for _ in range(2000)])
print(f"  50% sub-sample: mean Sharpe={np.mean(sub):.3f}, min={np.min(sub):.3f}, p(neg)={np.mean(sub<0)*100:.2f}%", flush=True)

print("\n"+"="*80, flush=True)
print("MONTH-BY-MONTH", flush=True)
print("="*80, flush=True)
for mi,mn in [(10,"Oct"),(11,"Nov"),(12,"Dec")]:
    msk = pd.DatetimeIndex(dt_all).month[1:] == mi
    mi2 = np.where(msk[:(T-1-5)]&consensus[:(T-1-5)]&(hour_arr[:(T-1-5)]>=7)&(hour_arr[:(T-1-5)]<=21)
        &(avg_mag[:(T-1-5)]>mag_pct[90][:T-1-5])&(es_avg[:(T-1-5)]>es_pct[90][:T-1-5]))[0]
    mpn = np.log(close[mi2+5,1]/close[mi2,1])*direction[mi2]-0.00005*2
    print(f"  {mn}: n={len(mpn):5d}  WR={np.mean(mpn>0)*100:.1f}%  Sharpe={np.mean(mpn)/(np.std(mpn)+1e-10)*np.sqrt(1440/5):.3f}  tot={np.sum(mpn)*10000:.0f}p", flush=True)

print("\n"+"="*80, flush=True)
print("WALK-FORWARD: fixed thresholds from 2 months, test on 3rd", flush=True)
print("="*80, flush=True)
for tr_mn, te_mn, nm in [([10,11],12,"Oct+Nov→Dec"),([11,12],10,"Nov+Dec→Oct"),([10,12],11,"Oct+Dec→Nov")]:
    tr_msk = pd.DatetimeIndex(dt_all).month[1:].isin(tr_mn)
    tr_idx = np.where(tr_msk)[0]
    tr_idx = tr_idx[tr_idx >= LB]
    tr_m90 = np.percentile(avg_mag[tr_idx], 90)
    tr_e90 = np.nanpercentile(es_avg[tr_idx], 90)
    te_msk = pd.DatetimeIndex(dt_all).month[1:] == te_mn
    te_idx = np.where(te_msk[:(T-1-5)]&consensus[:(T-1-5)]&(hour_arr[:(T-1-5)]>=7)&(hour_arr[:(T-1-5)]<=21)
        &(avg_mag[:(T-1-5)]>tr_m90)&(es_avg[:(T-1-5)]>tr_e90))[0]
    if len(te_idx) < 5: continue
    tep = np.log(close[te_idx+5,1]/close[te_idx,1])*direction[te_idx]-0.00005*2
    print(f"  {nm}: n={len(tep):5d}  WR={np.mean(tep>0)*100:.1f}%  Sharpe={np.mean(tep)/(np.std(tep)+1e-10)*np.sqrt(1440/5):.3f}  tot={np.sum(tep)*10000:.0f}p", flush=True)

print(f"\nRuntime: {time.time()-t0:.1f}s", flush=True)
