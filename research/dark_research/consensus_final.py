#!/usr/bin/env python3
"""
FINAL: 3-Pair Consensus Signal — complete validation & strategy spec
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

base = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy", "eurusd", "gbpjpy"]:
    raw[p] = {"p": np.load(os.path.join(base, f"{p}_m1_prices.npy")),
              "t": np.load(os.path.join(base, f"{p}_m1_times.npy"))}
common = sorted(set(raw["eurjpy"]["t"]) & set(raw["eurusd"]["t"]) & set(raw["gbpjpy"]["t"]))
def align(key, common, field):
    tmap = {t: i for i, t in enumerate(raw[key]["t"])}
    return raw[key][field][[tmap[t] for t in common]]

close = np.column_stack([align(k, common, "p")[:, 3] for k in ["eurjpy", "eurusd", "gbpjpy"]])
times = np.array(common).astype(np.int64)
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)

up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)

avg_mag = np.mean(np.abs(rets), axis=1)
es_avg = np.zeros(T-1)
for t in range(1, T):
    start = max(0, t-50)
    es_avg[t-1] = np.mean(np.sum(rets[start:t]**2, axis=1))

# Strategy: P90_mag + P90_es, rolling 1440-bar lookback, H5 EURUSD
# Session: London/NY only (H07-H21 UTC)
fwd = 5
cost = 0.00005 * 2
dt_all = pd.to_datetime(times, unit="s")

all_pnls = {k: {"pnls": [], "dirs": [], "pnl_ts": []} for k in ["full", "london_ny"]}
for t in range(1440, T - 1 - fwd):
    if not consensus[t]: continue
    h = dt_all[t].hour
    
    mag90 = np.percentile(avg_mag[t-1440:t], 90)
    es90 = np.nanpercentile(es_avg[t-1440:t], 90)
    if avg_mag[t] <= mag90: continue
    if es_avg[t] <= es90: continue
    
    ret = np.log(close[t+fwd, 1] / close[t, 1])
    pnl = ret * direction[t] - cost
    
    all_pnls["full"]["pnls"].append(pnl)
    all_pnls["full"]["dirs"].append(direction[t])
    all_pnls["full"]["pnl_ts"].append(dt_all[t])
    
    if 7 <= h <= 21:
        all_pnls["london_ny"]["pnls"].append(pnl)
        all_pnls["london_ny"]["dirs"].append(direction[t])
        all_pnls["london_ny"]["pnl_ts"].append(dt_all[t])

for label in ["full", "london_ny"]:
    d = all_pnls[label]
    pnls = np.array(d["pnls"])
    dirs = np.array(d["dirs"])
    ht = pd.DatetimeIndex(d["pnl_ts"])
    
    print("="*70)
    print(f"FINAL: {label.upper().replace('_',' ')}")
    print(f"3-Pair Consensus + P90 Mag + P90 ES | H5 EURUSD")
    print("="*70)
    if len(pnls) == 0:
        print("  No trades")
        continue
    avg_ret = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    sharpe = avg_ret * np.sqrt(1440/fwd)
    print(f"  Period:      {ht[0]} — {ht[-1]}")
    print(f"  Trades:      {len(pnls)}")
    print(f"  Trades/day:  {len(pnls)/((T-1-fwd-1440)/1440):.1f}")
    print(f"  Net mean:    {np.mean(pnls):+.6f} ({np.mean(pnls)*10000:.2f} pips)")
    print(f"  Total net:   {np.sum(pnls)*10000:.1f} pips")
    print(f"  Win rate:    {np.mean(pnls>0)*100:.1f}%")
    print(f"  Max DD:      {np.min(np.cumsum(pnls))*10000:.1f} pips")
    print(f"  Sharpe:      {sharpe:.3f}")
    print(f"  T-stat:      {avg_ret * np.sqrt(len(pnls)):.3f}")
    
    print(f"\n  Monthly:")
    for m in sorted(set(ht.month)):
        mask = ht.month == m
        mp = pnls[mask]
        s = np.mean(mp)/np.std(mp)*np.sqrt(1440/fwd) if np.std(mp) > 0 else 0
        print(f"    {ht[mask][0].strftime('%b')}: n={len(mp):4d}  mean={np.mean(mp):+.6f}  WR={np.mean(mp>0)*100:.1f}%  Sharpe={s:.3f}  Σ={np.sum(mp)*10000:.0f}p")
    
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    print(f"\n  Avg win:     {np.mean(wins)*10000:.3f} pips")
    print(f"  Avg loss:    {np.mean(losses)*10000:.3f} pips")
    print(f"  Profit factor: {abs(np.sum(wins)/np.sum(losses)):.3f}")
    print(f"  Max win:     {np.max(wins)*10000:.3f} pips")
    print(f"  Max loss:    {np.min(losses)*10000:.3f} pips")
    print(f"  Longs:       {np.mean(dirs==1)*100:.1f}% (WR {np.mean(pnls[dirs==1]>0)*100:.1f}%)")
    print(f"  Shorts:      {np.mean(dirs==-1)*100:.1f}% (WR {np.mean(pnls[dirs==-1]>0)*100:.1f}%)")
