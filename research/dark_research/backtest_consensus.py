#!/usr/bin/env python3
"""Full backtest of the consensus signal: when 3/3 pairs agree → trade continuation."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

base = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
PAIRS = ["EURJPY", "EURUSD", "GBPJPY"]

# Load
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
print(f"Bars: {T}, Date: {pd.to_datetime(times[0], unit='s')} to {pd.to_datetime(times[-1], unit='s')}")

# Consensus signal: all 3 pairs move same direction
up = rets > 0
all_up = up.all(axis=1)        # 3/3 up
all_down = (~up).all(axis=1)   # 3/3 down

print(f"\nAll-up events:   {all_up.sum():6d} ({all_up.sum()/len(all_up)*100:.1f}%)")
print(f"All-down events: {all_down.sum():6d} ({all_down.sum()/len(all_down)*100:.1f}%)")
print(f"Total:           {all_up.sum()+all_down.sum():6d} ({(all_up.sum()+all_down.sum())/len(all_up)*100:.1f}%)")

# Forward returns at various horizons
for fwd, fwn in [(5,"M5"), (15,"M15"), (30,"M30"), (60,"H1")]:
    # Entries: where consensus condition is met
    entries = all_up[:-fwd] | all_down[:-fwd]
    n_trades = entries.sum()
    if n_trades < 10: continue
    
    # Direction: +1 for all-up (go long), -1 for all-down (go short)
    direction = np.where(all_up[:-fwd], 1.0, -1.0)
    
    # Forward returns for each pair
    pair_returns = np.zeros((n_trades, 3))
    trade_idx = np.where(entries)[0]
    good_trades = []
    for j, idx in enumerate(trade_idx):
        if idx + fwd >= T - 1: continue
        # Return from close[idx] (start of signal bar) to close[idx+fwd]
        r = np.log(close[idx+fwd] / close[idx])
        pair_returns[len(good_trades)] = r
        good_trades.append(j)
    
    pair_returns = pair_returns[:len(good_trades)]
    dirs = direction[trade_idx][good_trades]
    
    # Per-pair PnL
    pnl_per_pair = pair_returns * dirs[:, None]  # (n_trades, 3)
    
    # Mean return across pairs (basket approach) or pick best pair
    basket_pnl = np.mean(pnl_per_pair, axis=1)
    eurusd_pnl = pnl_per_pair[:, 1]  # Just EURUSD
    
    for name, pnl in [("Basket", basket_pnl), ("EURUSD_only", eurusd_pnl)]:
        n = len(pnl)
        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl) + 1e-10
        sharpe = mean_ret / std_ret * np.sqrt(1440 / fwd)
        wr = np.mean(pnl > 0) * 100
        tstat = mean_ret / std_ret * np.sqrt(n)
        
        print(f"\n  [{name}] {fwn}: n={n:6d}  mean={mean_ret:+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}  t={tstat:.3f}")
        
        # Profit factor (using absolute returns)
        gross_profit = np.sum(pnl[pnl > 0])
        gross_loss = abs(np.sum(pnl[pnl < 0]))
        pf = gross_profit / (gross_loss + 1e-10)
        print(f"    PF={pf:.3f}  best={np.max(pnl):+.6f}  worst={np.min(pnl):+.6f}")

# Per-session breakdown
print(f"\n--- SESSION BREAKDOWN (M5, 2/3 majority) ---")
hour_arr = pd.DatetimeIndex(pd.to_datetime(times, unit="s")).hour.values
up2 = (up[:, :2].all(axis=1)) | (up[:, 1:].all(axis=1)) | ((up[:, 0] & up[:, 2]))  # 2/3 up

# Simpler: use 2/3 majority
n_up = np.sum(rets > 0, axis=1)
maj2 = n_up >= 2  # 2/3 agree

for sname, sm in [("ASIAN", (hour_arr[1:] >= 0) & (hour_arr[1:] < 7)),
                   ("EUROPEAN", (hour_arr[1:] >= 7) & (hour_arr[1:] < 13)),
                   ("US", (hour_arr[1:] >= 13) | (hour_arr[1:] < 0))]:
    fwd = 5
    entries = maj2[:-fwd] & sm[:-fwd]
    n_trades = entries.sum()
    if n_trades < 10: continue
    
    direction = np.where(n_up[:-fwd] >= 2, 1.0, -1.0)
    trade_idx = np.where(entries)[0]
    pnls = []
    for j, idx in enumerate(trade_idx):
        if idx + fwd >= T - 1: continue
        r = np.mean(np.log(close[idx+fwd] / close[idx]))
        pnls.append(r * direction[idx])
    
    pnl = np.array(pnls)
    mean_ret = np.mean(pnl)
    sharpe = mean_ret / (np.std(pnl)+1e-10) * np.sqrt(1440/fwd)
    wr = np.mean(pnl > 0) * 100
    print(f"  {sname:8s}: n={n_trades:5d}  mean={mean_ret:+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")

# Walk-forward: split data into monthly chunks
print(f"\n--- WALK-FORWARD (M5, 3-pair consensus) ---")
monthly = [(10, "Oct"), (11, "Nov"), (12, "Dec")]
for month, mname in monthly:
    dt = pd.to_datetime(times, unit="s")
    mask = (dt.month == month) & (dt.year == 2025)
    m_idx = np.where(mask[1:])[0]  # rets indices
    if len(m_idx) < 100: continue
    
    fwd = 5
    entries_all = all_up[:-fwd] | all_down[:-fwd]
    entries = entries_all[m_idx[m_idx < len(entries_all)]]
    n = entries.sum()
    if n < 5: continue
    
    direction = np.where(all_up[m_idx[m_idx < len(entries_all)]], 1.0, -1.0)
    dirs = direction[entries.values if hasattr(entries, 'values') else entries]
    idxs = m_idx[np.where(entries)[0]]
    idxs = idxs[idxs + fwd < T - 1]
    dirs = dirs[:len(idxs)]
    
    pnls = []
    for j, idx in enumerate(idxs):
        r = np.mean(np.log(close[idx+fwd] / close[idx]))
        pnls.append(r * dirs[j])
    
    pnl = np.array(pnls)
    if len(pnl) < 3: continue
    sharpe = np.mean(pnl) / (np.std(pnl)+1e-10) * np.sqrt(1440/fwd)
    wr = np.mean(pnl > 0) * 100
    print(f"  {mname}: n={len(pnl):5d}  mean={np.mean(pnl):+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")
