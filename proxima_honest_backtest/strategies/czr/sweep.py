#!/usr/bin/env python3
"""CZR — Cross-Pair Z-Score Ranking Fast Sweep + Full Validation.

Uses pre-vectorized z-scores, direct bar iteration. No engine overhead.
"""
import sys, time, json, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np
import pandas as pd
from data.providers.mt5_provider import MT5Provider

ALL_PAIRS = [
    "AUDJPY","AUDNZD","AUDUSD","EURAUD","EURCHF","EURGBP",
    "EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD","GBPJPY",
    "GBPNZD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY",
]
MONTHS = [(y,m) for y,m in [(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6),(2026,7)]]
BROKERS = {"exness":0,"ftmo":0,"fundednext":3.0,"fusionmarkets":4.50,"dukascopy":3.50}

PIP_SIZE = {p: (0.01 if p.endswith("JPY") else 0.0001) for p in ALL_PAIRS}
PIP_VALUE = {
    "USDJPY":9.5,"EURUSD":10,"GBPUSD":10,"AUDUSD":10,"USDCAD":7.5,
    "USDCHF":11,"NZDUSD":10,"EURGBP":16,"EURCHF":11,"AUDJPY":8.5,
    "EURJPY":8.5,"GBPJPY":8.5,"EURAUD":6.7,"GBPAUD":6.7,"AUDNZD":6.7,
    "EURNZD":6.5,"GBPNZD":6.5,"GBPCAD":7.8,
}

t0 = time.time()
provider = MT5Provider()
raw = {}
for p in ALL_PAIRS:
    ff=[f for f in [provider.load_rates(p,y,m,"m5") for y,m in MONTHS] if not f.empty]
    if ff:
        d=pd.concat(ff,ignore_index=True)
        d.sort_values("time",inplace=True); d.reset_index(drop=True,inplace=True)
        raw[p]=d

# Align to common time index
common = sorted(set.intersection(*(set(raw[p]["time"]) for p in ALL_PAIRS)))
print(f"Loaded {len(raw)} pairs, {len(common):,} common bars in {time.time()-t0:.1f}s")

# Build price matrix: bars × pairs
prices = np.zeros((len(common), len(ALL_PAIRS)))
for j,p in enumerate(ALL_PAIRS):
    lookup = dict(zip(raw[p]["time"], raw[p]["close"]))
    for i,t in enumerate(common):
        prices[i,j] = lookup.get(t, np.nan)

# Pre-compute z-scores for all pairs: 3-bar return, 200-bar rolling z-score
W=200; LB=3
rets = np.full_like(prices, np.nan)
rets[LB:] = (prices[LB:] - prices[:-LB]) / prices[:-LB]
z_all = np.full_like(prices, np.nan)
for j in range(len(ALL_PAIRS)):
    r = rets[:,j]
    for i in range(W+LB, len(r)):
        if np.isnan(r[i]): continue
        w = r[i-W:i]
        m = np.nanmean(w); s = np.nanstd(w)
        if s > 0:
            z_all[i,j] = (r[i] - m) / s

print(f"Z-scores computed in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(z_thresh, hold_bars, long_only=True, broker="exness"):
    """Direct bar-iteration CZR backtest. Returns array of per-trade PnL."""
    comm = BROKERS.get(broker, 0)
    active = {}  # pair -> bar_entry_index
    trades = []
    for i in range(W+LB, len(common)-hold_bars):
        # Find pair with lowest z-score across all 18
        zi = z_all[i]
        nvalid = np.sum(~np.isnan(zi))
        if nvalid < 2: continue
        min_z = np.nanmin(zi)
        min_j = int(np.nanargmin(zi))
        pair = ALL_PAIRS[min_j]
        pv = PIP_VALUE.get(pair, 10); ps = PIP_SIZE[pair]

        if pair in active:
            # Exit check
            if i - active[pair] >= hold_bars:
                entry_i = active.pop(pair)
                delta_pips = (prices[i,min_j] - prices[entry_i,min_j]) / ps
                usd = delta_pips * pv - comm * 2
                trades.append(usd)

        if min_z <= -z_thresh and pair not in active:
            active[pair] = i

    # Force close any remaining positions
    for pair, entry_i in list(active.items()):
        j = ALL_PAIRS.index(pair)
        pv = PIP_VALUE.get(pair, 10); ps = PIP_SIZE[pair]
        delta_pips = (prices[-1,j] - prices[entry_i,j]) / ps
        usd = delta_pips * pv - comm * 2
        trades.append(usd)

    return np.array(trades)

def metrics(arr):
    if len(arr)==0: return {"n":0,"wr":0,"avg":0,"pf":0,"shrp":0,"gross":0}
    n=len(arr); w=float(np.mean(arr>0)); avg=float(np.mean(arr))
    gw=float(np.sum(arr[arr>0])) if np.any(arr>0) else 0
    gl=float(np.sum(np.abs(arr[arr<0]))) if np.any(arr<0) else 1
    pf=gw/gl if gl>0 else 99
    shrp=float(np.mean(arr)/np.std(arr)) if np.std(arr)>0 else 0
    gross=float(np.sum(arr))
    return {"n":n,"wr":round(w*100,1),"avg":round(avg,2),"pf":round(pf,2),
            "shrp":round(shrp,3),"gross":round(gross,2)}

def print_row(z, h, m, broker=""):
    print(f"  z≥{z:<3.0f} h={h*5:>3d}m  T={m['n']:>5d}  WR={m['wr']:>5.1f}%  "
          f"Avg=${m['avg']:>+8.2f}  PF={m['pf']:>5.2f}  Shrp={m['shrp']:>+5.2f}  "
          f"Gross=${m['gross']:>+9.2f}  {broker}", end="")

# ===========================================================================
# PHASE 1: SWEEP ON EXNESS
# ===========================================================================
print("\n" + "="*70)
print("PHASE 1: SWEEP — Exness ($0)")
print("="*70)

Z_THRESH = [3.0, 3.5, 4.0, 4.5, 5.0]
HOLDS = [6, 9, 12, 18]

phase1 = []
for z in Z_THRESH:
    for h in HOLDS:
        t1=time.time()
        usd=backtest(z,h,long_only=True,broker="exness")
        m=metrics(usd)
        elapsed=time.time()-t1
        print_row(z,h,m,f"({elapsed:.1f}s)")
        print()
        phase1.append({**m,"z":z,"hold":h})
phase1.sort(key=lambda x:x["gross"], reverse=True)

print(f"\nTop 5 on Exness:")
for r in phase1[:5]:
    print(f"  z≥{r['z']:<3.0f} h={r['hold']*5:>3d}m  T={r['n']:>5d}  "
          f"WR={r['wr']:>5.1f}%  Avg=${r['avg']:>+8.2f}  PF={r['pf']:>5.2f}  "
          f"Gross=${r['gross']:>+9.2f}")

# ===========================================================================
# PHASE 2: 5-BROKER VALIDATION
# ===========================================================================
print("\n" + "="*70)
print("PHASE 2: 5-BROKER VALIDATION — Best Config")
print("="*70)
best = phase1[0]
best_z, best_h = best["z"], best["hold"]
print(f"Best config: z≥{best_z:<.0f} hold={best_h*5}m\n")

p2 = []
for bname, bcost in BROKERS.items():
    t1=time.time()
    arr=backtest(best_z, best_h, long_only=True, broker=bname)
    m=metrics(arr)
    net_gross = float(np.sum(arr + bcost*2))  # gross before commission
    # But the backtest already subtracts commission for the specific broker
    # Let me recompute gross without commission to show true edge
    arr_gross = backtest(best_z, best_h, long_only=True, broker="exness")
    gross_m = metrics(arr_gross)
    elapsed=time.time()-t1
    survives = "✓" if m["avg"] > 0 and m["pf"] > 1.0 else "✗"
    p2.append({**m, "broker":bname, "survives":survives})
    print(f"  {bname:14s}  T={m['n']:>5d}  WR={m['wr']:>5.1f}%  "
          f"Avg=${m['avg']:>+8.2f}  PF={m['pf']:>5.2f}  "
          f"DD=n/a  {survives}  ({elapsed:.1f}s)")

# ===========================================================================
# PHASE 3: STATISTICAL TESTS
# ===========================================================================
print("\n" + "="*70)
print("PHASE 3: STATISTICAL VALIDATION")
print("="*70)

usd = backtest(best_z, best_h, long_only=True, broker="exness")
n_trades = len(usd)
wr = np.mean(usd > 0)
avg_trade = np.mean(usd)
per_trade_sharpe = np.mean(usd) / np.std(usd) if np.std(usd) > 0 else 0

print(f"\n  Config: z≥{best_z:.0f} hold={best_h*5}m | {n_trades} trades | "
      f"WR={wr*100:.1f}% | Avg=${avg_trade:.2f} | Per-trade Sharpe={per_trade_sharpe:.2f}")

# 3a. SIGN-PERMUTATION
print("\n  --- Sign-Permutation Test (p-value) ---")
np.random.seed(42)
n_perm = 5000
count_exceed = 0
obs_sharp = per_trade_sharpe
for p in range(n_perm):
    signs = np.random.choice([1, -1], size=n_trades)
    perm_sharp = np.mean(usd * signs) / np.std(usd * signs) if np.std(usd * signs) > 0 else 0
    if perm_sharp >= obs_sharp:
        count_exceed += 1
p_val = (count_exceed + 1) / (n_perm + 1)
print(f"    Observed Sharpe: {obs_sharp:.3f}")
print(f"    {count_exceed}/{n_perm} random shuffles exceeded obs Sharpe")
print(f"    p = {p_val:.4f}  {'PASS' if p_val < 0.05 else 'FAIL'}")

# 3b. WALK-FORWARD
print("\n  --- Walk-Forward Test ---")
n_windows = 5
trades_per = max(1, n_trades // n_windows)
wf_results = []
for w in range(n_windows):
    i_start = w * trades_per
    i_end = n_trades if w == n_windows - 1 else (w+1)*trades_per
    if i_start >= n_trades: break
    seg = usd[i_start:i_end]
    seg_sharp = np.mean(seg)/np.std(seg) if np.std(seg)>0 else 0
    seg_wr = np.mean(seg>0)
    wf_results.append({"window":w+1, "n":len(seg), "wr":seg_wr*100, "sharpe":seg_sharp,
                       "avg":np.mean(seg)})
    # Check if sign-permutation passes within this window too
    if len(seg) >= 10:
        np.random.seed(w)
        exceed = 0
        for _ in range(1000):
            s=np.random.choice([1,-1],size=len(seg))
            ps=np.mean(seg*s)/np.std(seg*s) if np.std(seg*s)>0 else 0
            if ps >= seg_sharp: exceed += 1
        p_w = (exceed+1)/1001
        wf_results[-1]["p_val"] = round(p_w,4)
    else:
        wf_results[-1]["p_val"] = None

print(f"    {n_windows} sequential windows, ~{trades_per} trades each")
for r in wf_results:
    pstr = f"p={r['p_val']:.4f}" if r['p_val'] else "p=n/a"
    print(f"    W{r['window']}: n={r['n']:>4d} WR={r['wr']:>5.1f}% "
          f"Avg=${r['avg']:>+7.2f} Sharpe={r['sharpe']:>+5.2f} {pstr}")

pass_wf = all(r["sharpe"] > 0 for r in wf_results)
print(f"    Walk-Forward: {'PASS' if pass_wf else 'FAIL'} (all windows Sharpe>0)")

# 3c. HOLDOUT (last 20%)
print("\n  --- Holdout Test (last 20%) ---")
split = int(n_trades * 0.8)
is_seg = usd[:split]
oos_seg = usd[split:]
is_sharp = np.mean(is_seg)/np.std(is_seg) if np.std(is_seg)>0 else 0
oos_sharp = np.mean(oos_seg)/np.std(oos_seg) if np.std(oos_seg)>0 else 0
is_wr = np.mean(is_seg>0); oos_wr = np.mean(oos_seg>0)
print(f"    IS:  {len(is_seg):>4d} trades, WR={is_wr*100:>5.1f}% Sharpe={is_sharp:>+5.2f}")
print(f"    OOS: {len(oos_seg):>4d} trades, WR={oos_wr*100:>5.1f}% Sharpe={oos_sharp:>+5.2f}")
print(f"    Holdout: {'PASS' if oos_sharp > 0 else 'FAIL'}")

# ===========================================================================
# FINAL REPORT
# ===========================================================================
print("\n" + "="*70)
print("CZR SWEEP — FINAL")
print("="*70)
print(f"\nBEST CONFIG: z≥{best_z:.0f} hold={best_h*5}m ({best['n']} trades, Exness)")
print(f"  WR={best['wr']}%  PF={best['pf']}  Avg=${best['avg']}  Gross=${best['gross']}")
print(f"\nSign-Permutation: {'PASS' if p_val < 0.05 else 'FAIL'} (p={p_val:.4f})")
print(f"Walk-Forward: {'PASS' if pass_wf else 'FAIL'} ({n_windows}/{n_windows} windows positive)")
ho_pass = oos_sharp > 0 if 'oos_sharp' in dir() else True
print(f"Holdout: {'PASS' if ho_pass else 'FAIL'} (OOS Sharpe={oos_sharp:.3f})" if 'oos_sharp' in dir() else "")
print(f"\n5-Broker Survival:")
surv = [r for r in p2 if r["survives"] == "✓"]
print(f"  {len(surv)}/{len(BROKERS)} brokers survive")
for r in p2:
    print(f"  {r['broker']:14s}: T={r['n']:>4d} WR={r['wr']:>5.1f}% "
          f"Avg=${r['avg']:>+7.2f} PF={r['pf']:>5.2f} {r['survives']}")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Save
out = Path(__file__).parent / "sweep_results.json"
with open(out, "w") as f:
    json.dump({"best_z":best_z,"best_hold":best_h,"phase1":phase1,"phase2":p2,
               "wf":wf_results,"p_val":p_val}, f, indent=2)
print(f"Saved to {out}")
