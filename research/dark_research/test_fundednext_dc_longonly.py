#!/usr/bin/env python3
"""Dark Consensus LONG-ONLY on FundedNext — the best_pair always goes up after 3 bars."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(__file__)
pairs = ["eurjpy", "eurusd", "gbpjpy"]
pair_names = ["EURJPY", "EURUSD", "GBPJPY"]
data = {}
for p in pairs:
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    data[p] = df.set_index("time")

common = sorted(set(data["eurjpy"].index) & set(data["eurusd"].index) & set(data["gbpjpy"].index))
close = np.column_stack([data[p].loc[common, "close"].values for p in pairs])
spreads = np.column_stack([data[p].loc[common, "spread"].values for p in pairs])
eurusd_nnz = spreads[:, 1][spreads[:, 1] > 0]
spreads[:, 1] = np.maximum(spreads[:, 1], np.median(eurusd_nnz) if len(eurusd_nnz) > 0 else 3.0)

T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
hour_arr = pd.DatetimeIndex(common).hour.values[1:]
usdjpy_proxy = close[:, 0] / close[:, 1]

MAG95 = 0.00018741
HOLD = 3
FN_COMM = 3.0
SLIP_PIPS = 0.5

def pv(pair_idx, usdjpy):
    return 10.0 if pair_idx == 1 else 1000.0 / usdjpy

def tot_cost(pair_idx, spread_pts, usdjpy):
    sp = spread_pts / 10.0
    pip_v = pv(pair_idx, usdjpy)
    return sp * pip_v + SLIP_PIPS * pip_v + FN_COMM

# Test different configurations
configs = [
    ("P95+H07-H21+LONG", MAG95, True, True),
    ("P95+H07-H21+SHORT", MAG95, True, False),
    ("P95+0-24+LONG", MAG95, False, True),
    ("P95+0-24+SHORT", MAG95, False, False),
]

print(f"{'Config':>25s} {'n':>5s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>8s} {'Tot$':>10s}")
print("-" * 65)

for label, mag_t, use_hour, go_long in configs:
    g_list = []
    n_list = []
    for t in range(1440, T - HOLD - 1):
        if not consensus[t]: continue
        if use_hour and (hour_arr[t] < 7 or hour_arr[t] > 21): continue
        if avg_mag[t] <= mag_t: continue
        bi = int(np.argmax(pair_mags[t]))
        ep = close[t, bi]
        xp = close[t + HOLD, bi]
        # LONG only: always buy
        gross = (xp - ep) / usdjpy_proxy[t] * 100000 if bi != 1 else (xp - ep) * 100000
        cost = tot_cost(bi, spreads[t, bi], usdjpy_proxy[t])
        g_list.append(gross)
        n_list.append(gross - cost)
    g = np.array(g_list)
    n = np.array(n_list)
    if len(g) > 5:
        wr_g = np.mean(g > 0) * 100
        sh_g = np.mean(g) / (np.std(g) + 1e-10) * np.sqrt(1440/HOLD)
        avg_g = np.mean(g)
        tot_g = np.sum(g)
        wr_n = np.mean(n > 0) * 100
        sh_n = np.mean(n) / (np.std(n) + 1e-10) * np.sqrt(1440/HOLD)
        avg_n = np.mean(n)
        tot_n = np.sum(n)
        print(f"{label:>25s}: Gross: n={len(g):5d} WR={wr_g:5.1f}% Sharpe={sh_g:7.2f} {avg_g:8.2f} {tot_g:10,.0f}")
        print(f"{'':>25s}  Net:  n={len(g):5d} WR={wr_n:5.1f}% Sharpe={sh_n:7.2f} {avg_n:8.2f} {tot_n:10,.0f}")
    else:
        print(f"{label:>25s}: n={len(g)} (too few)")

# Full breakdown of the best config: LONG-ONLY with hour filter
print(f"\n{'='*70}")
print(f"DETAILED: P95 + H07-H21 + LONG ONLY")
print(f"{'='*70}")

g_list = []
n_list = []
p_list = []
sprd_list = []
dt_list = []

for t in range(1440, T - HOLD - 1):
    if not consensus[t]: continue
    if hour_arr[t] < 7 or hour_arr[t] > 21: continue
    if avg_mag[t] <= MAG95: continue
    bi = int(np.argmax(pair_mags[t]))
    ep = close[t, bi]
    xp = close[t + HOLD, bi]
    gross = (xp - ep) / usdjpy_proxy[t] * 100000 if bi != 1 else (xp - ep) * 100000
    cost = tot_cost(bi, spreads[t, bi], usdjpy_proxy[t])
    g_list.append(gross)
    n_list.append(gross - cost)
    p_list.append(bi)
    sprd_list.append((spreads[t, bi] / 10.0, usdjpy_proxy[t]))
    dt_list.append(common[t])

g = np.array(g_list)
n = np.array(n_list)
n_trades = len(g)
print(f"Total trades: {n_trades}")
print(f"Period: {common[1440]} — {common[-4]}")
print(f"Days: {(common[-4] - common[1440]).days}")

print(f"\nGross:  WR={np.mean(g>0)*100:.1f}%  Avg=${np.mean(g):.2f}  Tot=${np.sum(g):,.0f}  Sharpe={np.mean(g)/(np.std(g)+1e-10)*np.sqrt(1440/HOLD):.2f}")
print(f"Net:    WR={np.mean(n>0)*100:.1f}%  Avg=${np.mean(n):.2f}  Tot=${np.sum(n):,.0f}  Sharpe={np.mean(n)/(np.std(n)+1e-10)*np.sqrt(1440/HOLD):.2f}")

# Cost breakdown
avg_sprd_cost = np.mean([(s[0]) * pv(p_list[i], s[1]) for i, s in enumerate(sprd_list)])
avg_slip_cost = np.mean([SLIP_PIPS * pv(p_list[i], s[1]) for i, s in enumerate(sprd_list)])
print(f"\nAvg cost/trade: spread=${avg_sprd_cost:.2f} + slip=${avg_slip_cost:.2f} + comm=${FN_COMM:.2f} = ${avg_sprd_cost+avg_slip_cost+FN_COMM:.2f}")
print(f"Total costs: ${n_trades * (avg_sprd_cost+avg_slip_cost+FN_COMM):,.0f}")

# Daily
dfd = pd.DataFrame({"date": [d.date() for d in dt_list], "pnl": n})
daily = dfd.groupby("date")["pnl"].sum()
print(f"\nDaily:  days={len(daily)}  avg=${daily.mean():.2f}  WR={np.mean(daily>0)*100:.1f}%  Sharpe={daily.mean()/(daily.std()+1e-10)*np.sqrt(252):.2f}")
print(f"  Best: ${daily.max():,.2f}  Worst: ${daily.min():,.2f}")
print(f"  >=$100/day: {np.sum(daily>=100)/len(daily)*100:.1f}%")
print(f"  >=$200/day: {np.sum(daily>=200)/len(daily)*100:.1f}%")
print(f"  >=$500/day: {np.sum(daily>=500)/len(daily)*100:.1f}%")

# Pair distribution
print(f"\nPair distribution (Net):")
for pi, pn in enumerate(pair_names):
    mask = np.array(p_list) == pi
    if np.sum(mask) == 0: continue
    pnls = n[mask]
    print(f"  {pn}: n={len(pnls):4d} ({len(pnls)/n_trades*100:.0f}%)  WR={np.mean(pnls>0)*100:.1f}%  Avg=${np.mean(pnls):.2f}  Tot=${np.sum(pnls):,.0f}")
