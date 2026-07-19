#!/usr/bin/env python3
"""Stress test: realistic adversities with proper pair-specific pip sizing."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy","eurusd","gbpjpy"]:
    raw[p] = {k: np.load(f"{BASE}/{p}_m1_{k}.npy") for k in ["prices","times"]}
common = sorted(set(raw["eurjpy"]["times"])&set(raw["eurusd"]["times"])&set(raw["gbpjpy"]["times"]))
idx_map = {k:{t:i for i,t in enumerate(raw[k]["times"])} for k in raw}
close = np.column_stack([raw[k]["prices"][[idx_map[k][c] for c in common],3] for k in ["eurjpy","eurusd","gbpjpy"]])
times=np.array(common,dtype=np.int64); T=close.shape[0]
rets=np.diff(np.log(close),axis=0)
up=rets>0; consensus=up.all(axis=1)|(~up).all(axis=1); direction=np.where(up.all(axis=1),1.0,-1.0)
avg_mag=np.mean(np.abs(rets),axis=1); pair_mags=np.abs(rets)
usdjpy_proxy=close[:,0]/close[:,1]

MIN_IDX=1440
tr_idx=np.where((np.arange(len(avg_mag))>=MIN_IDX)&consensus)[0]
mag95=np.percentile(avg_mag[tr_idx],95)
te_idx=np.where(consensus&(avg_mag>mag95))[0]
te_idx=te_idx[te_idx+3<T-1]
bi=np.argmax(pair_mags[te_idx],axis=1)
LOT=100000

# Spread costs: half-spread in pips (realistic ECN raw spreads)
# EURUSD: 0.3 pip, EURJPY: 0.5 pip, GBPJPY: 0.7 pip
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])

# Pip value in dollars per 1 lot
# EURUSD: $10/pip, EURJPY: ~$6.50/pip (depends on USDJPY), GBPJPY: ~$6.50/pip
def pip_value_usd(pair_idx, eurusd_rate=1.0, usdjpy_rate=154.0):
    if pair_idx == 1:  # EURUSD
        return 10.0
    elif pair_idx == 0:  # EURJPY
        return 1000.0 / usdjpy_rate  # ¥1000/pip ÷ USDJPY
    else:  # GBPJPY
        return 1000.0 / usdjpy_rate

def simulate(spread_mult, slippage_pips, commission_per_lot):
    """All costs in dollars per lot. Slippage in pips."""
    dollars = []
    avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
    for j,i in enumerate(te_idx):
        p=bi[j]
        gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
        # Spread cost in dollars (per lot)
        spread_usd = HALF_SPREAD_PIPS[p] * 2 * spread_mult * pip_value_usd(p, usdjpy_rate=avg_usdjpy)
        # Slippage cost in dollars (per lot)
        slip_usd = slippage_pips * 2 * pip_value_usd(p, usdjpy_rate=avg_usdjpy)
        # Gross P&L in dollars
        if p == 1:
            gross_usd = LOT * gross
        else:
            gross_usd = LOT * gross * close[i,p] / usdjpy_proxy[i]
        net_usd = gross_usd - spread_usd - slip_usd - commission_per_lot
        dollars.append(net_usd)
    d = np.array(dollars)
    n = len(d); wr = np.mean(d>0)*100
    sh = np.mean(d)/(np.std(d)+1e-10)*np.sqrt(1440/3)
    return n, wr, sh, np.mean(d), np.sum(d), n/(T/1440)

print("="*100)
print("STRESS TEST: P95+best_pair+H3 at 1 lot (pair-correct pip values)")
print("="*100)
print(f"{'Scenario':<52s} {'n':>5s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>7s} {'Tot$':>9s} {'/day':>6s}")
print("-"*100)

# Base scenarios with proper USD costs
ecn_comm = 7  # $7 round trip per lot ECN
scenarios = [
    ("Ideal (raw spread, 0 slip, 0 comm)", 1.0, 0.0, 0),
    ("Raw spread, 0 slip, ECN comm", 1.0, 0.0, ecn_comm),
    ("1x spread, 0.5p slip, ECN comm", 1.0, 0.5, ecn_comm),
    ("1.5x spread, 0.5p slip, no comm", 1.5, 0.5, 0),
    ("1.5x spread, 0.5p slip, ECN comm", 1.5, 0.5, ecn_comm),
    ("2x spread, 0.5p slip, ECN comm", 2.0, 0.5, ecn_comm),
    ("1x spread, 1.0p slip, ECN comm", 1.0, 1.0, ecn_comm),
    ("2x spread, 1.0p slip, ECN comm", 2.0, 1.0, ecn_comm),
    ("3x spread, 1.0p slip, ECN comm", 3.0, 1.0, ecn_comm),
]
for name, sm, slip, comm in scenarios:
    n, wr, sh, avg, tot, tpd = simulate(sm, slip, comm)
    print(f"{name:<52s} {n:5d} {wr:5.1f} {sh:7.2f} {avg:7.2f} {tot:9.0f} {tpd:6.1f}")

print(f"\n{'='*100}")
print("SENSITIVITY GRID: Spread × Slippage (with ECN $7 commission)")
print(f"{'='*100}")
print(f"{'Spread':>6s} {'Slip(p)':>8s} {'n':>5s} {'WR%':>6s} {'Sharpe':>7s} {'Avg$':>7s} {'Tot$':>9s} {'/day':>6s}")
print("-"*100)
for sm in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for slip in [0.0, 0.5, 1.0, 1.5, 2.0]:
        n, wr, sh, avg, tot, tpd = simulate(sm, slip, ecn_comm)
        if n > 20:
            print(f"{sm:5.1f}x {slip:7.1f}p {n:5d} {wr:5.1f} {sh:7.2f} {avg:7.2f} {tot:9.0f} {tpd:6.1f}")

print(f"\n{'='*100}")
print("BREAKEVEN spread multiplier (at 0.5p slippage, $7 comm)")
print(f"{'='*100}")
for sm in np.arange(1.0, 10.0, 0.25):
    n, wr, sh, avg, tot, tpd = simulate(sm, 0.5, ecn_comm)
    if sh < 0.5:
        print(f"  Breakeven ≈ {sm:.2f}x spread: Sharpe={sh:.2f}, WR={wr:.1f}%, Avg=${avg:.2f}")
        break

print(f"\n{'='*100}")
print("MONTE CARLO: Random slippage 0-1 pip, 1.5x spread, $7 comm")
print(f"{'='*100}")
np.random.seed(42)
mc_wrs, mc_shs, mc_avgs = [], [], []
avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
for _ in range(2000):
    pnls = []
    for j,i in enumerate(te_idx):
        p=bi[j]
        slip = np.random.uniform(0, 1.0) * pip_value_usd(p, usdjpy_rate=avg_usdjpy) * 2
        spread = HALF_SPREAD_PIPS[p] * 2 * 1.5 * pip_value_usd(p, usdjpy_rate=avg_usdjpy)
        gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
        if p==1: gusd = LOT * gross
        else: gusd = LOT * gross * close[i,p] / usdjpy_proxy[i]
        pnls.append(gusd - spread - slip - ecn_comm)
    d = np.array(pnls)
    mc_wrs.append(np.mean(d>0)*100)
    mc_shs.append(np.mean(d)/(np.std(d)+1e-10)*np.sqrt(1440/3))
    mc_avgs.append(np.mean(d))
print(f"  Mean Sharpe: {np.mean(mc_shs):.2f}")
print(f"  Min Sharpe:  {np.min(mc_shs):.2f}")
print(f"  Mean WR:     {np.mean(mc_wrs):.1f}%")
print(f"  Mean Avg$:   ${np.mean(mc_avgs):.2f}")
print(f"  p(Sharpe<0): {np.mean(np.array(mc_shs)<0)*100:.1f}%")
print(f"  p(Sharpe<1): {np.mean(np.array(mc_shs)<1)*100:.1f}%")
print(f"  p(Sharpe<2): {np.mean(np.array(mc_shs)<2)*100:.1f}%")

print(f"\n{'='*100}")
print("WORST-CASE: Every trade gets max 1.0p slippage, 2x spread, $7 comm")
print(f"{'='*100}")
n, wr, sh, avg, tot, tpd = simulate(2.0, 1.0, ecn_comm)
print(f"  n={n}  WR={wr:.1f}%  Sharpe={sh:.2f}  Avg=${avg:.2f}  Tot=${tot:.0f}  TPD={tpd:.1f}")
