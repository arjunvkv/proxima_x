#!/usr/bin/env python3
"""
Consensus signal optimization:
1. Test with explicit spread costs
2. Test magnitude-based filtering
3. Test ES-based filtering
4. Find optimal hold/filter combination
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

base = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"

raw = {}
for p in ["eurjpy", "eurusd", "gbpjpy"]:
    raw[p] = {
        "p": np.load(os.path.join(base, f"{p}_m1_prices.npy")),
        "t": np.load(os.path.join(base, f"{p}_m1_times.npy")),
    }
common = sorted(set(raw["eurjpy"]["t"]) & set(raw["eurusd"]["t"]) & set(raw["gbpjpy"]["t"]))

def align(key, common, field):
    tmap = {t: i for i, t in enumerate(raw[key]["t"])}
    return raw[key][field][[tmap[t] for t in common]]

close = np.column_stack([align(k, common, "p")[:, 3] for k in ["eurjpy", "eurusd", "gbpjpy"]])
times = np.array(common)
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
hour = pd.DatetimeIndex(pd.to_datetime(times, unit="s")).hour.values

# ES for filtering - align with rets dimension (T-1)
sq = rets ** 2
es_aligned = np.zeros((T, 3))
cum = np.zeros((T, 3))
cum[1:] = np.cumsum(sq, axis=0)
es_aligned[51:] = cum[51:] - cum[1:T-50]
es_aligned[:51] = np.nan
for i in range(1, T):
    if np.isnan(es_aligned[i, 0]): es_aligned[i] = es_aligned[i-1]
es_1 = es_aligned[1:]  # (T-1, 3) - aligns with consensus/rets

# Consensus
up = rets > 0
all_up = up.all(axis=1)
all_down = (~up).all(axis=1)
consensus = all_up | all_down
direction = np.where(all_up, 1.0, -1.0)

# Mean abs return at each bar (magnitude filter)
avg_mag = np.mean(np.abs(rets), axis=1)

spreads = {"EURUSD": 0.00005, "EURJPY": 0.00008, "GBPJPY": 0.00010}  # ~0.5, 1.0, 1.5 pips

print("=" * 70)
print("COST-ADJUSTED BACKTEST: Consensus Signal")
print("=" * 70)

for fwd, fwn in [(5,"M5"), (15,"M15"), (30,"M30"), (60,"H1")]:
    entries = consensus[:-fwd]
    n = entries.sum()
    if n < 10: continue
    
    dirs = direction[:-fwd][entries]
    idxs = np.where(entries)[0]
    idxs = idxs[idxs + fwd < T - 1]
    dirs = dirs[:len(idxs)]
    
    # EuroUSD only
    pnl_gross = np.array([np.log(close[i+fwd, 1] / close[i, 1]) * dirs[j] for j, i in enumerate(idxs)])
    cost_per_trade = spreads["EURUSD"] * 2  # entry + exit
    pnl_net = pnl_gross - cost_per_trade
    
    mean_g = np.mean(pnl_gross)
    mean_n = np.mean(pnl_net)
    std_n = np.std(pnl_net) + 1e-10
    sharpe_n = mean_n / std_n * np.sqrt(1440/fwd)
    wr_n = np.mean(pnl_net > 0) * 100
    tstat = mean_n / std_n * np.sqrt(len(pnl_net))
    
    print(f"\n  [{fwn}] EURUSD only — 3-pair consensus")
    print(f"    n={n:6d}  gross_mean={mean_g:+.6f}  net_mean={mean_n:+.6f}")
    print(f"    net_WR={wr_n:.1f}%  net_Sharpe={sharpe_n:.3f}  t={tstat:.3f}")
    
    # Basket of 3 pairs
    basket_gross = np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        r = np.mean(np.log(close[i+fwd] / close[i]))
        basket_gross[j] = r * dirs[j]
    
    basket_cost = sum(spreads.values()) * 2 / 3  # avg spread × 2 (entry+exit)
    basket_net = basket_gross - basket_cost
    bs = np.mean(basket_net) / (np.std(basket_net)+1e-10) * np.sqrt(1440/fwd)
    print(f"    Basket net_Sharpe={bs:.3f}")

print("\n" + "=" * 70)
print("MAGNITUDE FILTERING: Only trade when avg|return| > threshold")
print("=" * 70)

for mag_pct in [50, 60, 70, 80, 90]:
    mag_thresh = np.percentile(avg_mag, mag_pct)
    fwd = 5
    
    entries = consensus[:-fwd] & (avg_mag[:-fwd] > mag_thresh)
    n = entries.sum()
    if n < 10: continue
    
    dirs = direction[:-fwd][entries]
    idxs = np.where(entries)[0]
    idxs = idxs[idxs + fwd < T - 1]
    dirs = dirs[:len(idxs)]
    
    pnl = np.array([np.log(close[i+fwd, 1] / close[i, 1]) * dirs[j] for j, i in enumerate(idxs)])
    pnl_net = pnl - spreads["EURUSD"] * 2
    
    sharpe = np.mean(pnl_net) / (np.std(pnl_net)+1e-10) * np.sqrt(1440/fwd)
    wr = np.mean(pnl_net > 0) * 100
    print(f"  mag>P{mag_pct} (>{mag_thresh:.6f}): n={n:6d}  net_mean={np.mean(pnl_net):+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")

print("\n" + "=" * 70)
print("ES FILTERING: Only trade when ES > threshold AND consensus")
print("=" * 70)

for es_pct in [50, 60, 70, 80, 90]:
    fwd = 5
    es_avg = np.nanmean(es_1, axis=1)  # (T-1,)
    
    es_thresh = np.nanpercentile(es_avg, es_pct)
    entries = consensus[:-fwd] & (es_avg[:-fwd] > es_thresh)
    n = entries.sum()
    if n < 10: continue
    
    dirs = direction[:-fwd][entries]
    idxs = np.where(entries)[0]
    idxs = idxs[idxs + fwd < T - 1 - 1]
    dirs = dirs[:len(idxs)]
    
    pnl = np.array([np.log(close[i+fwd, 1] / close[i, 1]) * dirs[j] for j, i in enumerate(idxs)])
    pnl_net = pnl - spreads["EURUSD"] * 2
    
    sharpe = np.mean(pnl_net) / (np.std(pnl_net)+1e-10) * np.sqrt(1440/fwd)
    wr = np.mean(pnl_net > 0) * 100
    print(f"  ES > P{es_pct} (>{es_thresh:.8f}): n={n:6d}  net_mean={np.mean(pnl_net):+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")

# Also check ES×consensus interaction
print("\n" + "=" * 70)
print("ES + CONSENSUS: The combined regime filter")
print("=" * 70)
fwd = 5
for pair_idx, pname in [(0,"EURJPY"), (1,"EURUSD"), (2,"GBPJPY")]:
    es = es_1[:, pair_idx]
    sp = spreads[pname]
    
    for es_th in [50, 70, 80, 90]:
        th = np.nanpercentile(es, es_th)
        entries = consensus[:-fwd] & (es[:-fwd] > th) & (es_1[:-fwd].mean(axis=1) > np.nanpercentile(es_1[:-fwd].mean(axis=1), 50))
        n = entries.sum()
        if n < 10: continue
        
        dirs = direction[:-fwd][entries]
        idxs = np.where(entries)[0]
        idxs = idxs[idxs + fwd < T - 1]
        dirs = dirs[:len(idxs)]
        
        pnl = np.array([np.log(close[i+fwd, pair_idx] / close[i, pair_idx]) * dirs[j] for j, i in enumerate(idxs)])
        pnl_net = pnl - sp * 2
        
        sharpe = np.mean(pnl_net) / (np.std(pnl_net)+1e-10) * np.sqrt(1440/fwd)
        wr = np.mean(pnl_net > 0) * 100
        print(f"  {pname} ES>P{es_th}+cons: n={n:5d}  net_mean={np.mean(pnl_net):+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")

# FINAL: Find optimal hold/filter combo
print("\n" + "=" * 70)
print("OPTIMAL COMBO SEARCH: Magnitude + ES filtered consensus")
print("=" * 70)

best_sharpe, best_combo = -999, None
results = []
for mag_th in [50, 70, 90]:
    mag_t = np.percentile(avg_mag, mag_th)
    for es_th in [50, 70, 90]:
        es_t = np.nanpercentile(es_1.mean(axis=1), es_th)
        for fwd in [5, 15, 30, 60]:
            entries = consensus[:-fwd] & (avg_mag[:-fwd] > mag_t) & (es_1[:-fwd].mean(axis=1) > es_t)
            n = entries.sum()
            if n < 20: continue
            
            dirs = direction[:-fwd][entries]
            idxs = np.where(entries)[0]
            idxs = idxs[idxs + fwd < T - 1]
            dirs = dirs[:len(idxs)]
            
            pnl = np.array([np.log(close[i+fwd, 1] / close[i, 1]) * dirs[j] for j, i in enumerate(idxs)])
            pnl_net = pnl - spreads["EURUSD"] * 2
            
            sharpe = np.mean(pnl_net) / (np.std(pnl_net)+1e-10) * np.sqrt(1440/fwd)
            trade_rate = n / T * 1440  # per day
            wr = np.mean(pnl_net > 0) * 100
            results.append((sharpe, trade_rate, n, mag_th, es_th, fwd, wr, np.mean(pnl_net)))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = (mag_th, es_th, fwd, n, trade_rate, wr, np.mean(pnl_net))

results.sort(key=lambda x: -x[0])
print("Top 10 configurations by Sharpe:")
for sharpe, rate, n, mag, es, fwd, wr, mean_n in results[:10]:
    print(f"  P{mag}_mag+P{es}_es+H{fwd}: n={n:5d}  net_mean={mean_n:+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}  {rate:.1f}/day")

print(f"\nBest: P{best_combo[0]}_mag+P{best_combo[1]}_es+H{best_combo[2]}")
print(f"  n={best_combo[3]}, {best_combo[4]:.1f}/day, WR={best_combo[5]:.1f}%, net_mean={best_combo[6]:+.6f}, Sharpe={best_combo[0]:.3f}")
