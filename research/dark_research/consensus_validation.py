#!/usr/bin/env python3
"""
STRESS TEST: Consensus signal with rolling percentiles (no lookahead).
Plus OOS validation on 7-pair dataset (Jun-Jul 2026).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

print("=" * 70)
print("PART 1: ROLLING PERCENTILES (no lookahead bias)")
print("=" * 70)

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
times = np.array(common)
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
hour = pd.DatetimeIndex(pd.to_datetime(times, unit="s")).hour.values

# Consensus
up = rets > 0
all_up = up.all(axis=1)
all_down = (~up).all(axis=1)
consensus = all_up | all_down
direction = np.where(all_up, 1.0, -1.0)

# Rolling percentiles for magnitude
avg_mag = np.mean(np.abs(rets), axis=1)
lookback = 1440  # 1 day of M1 bars for threshold

fwd = 5
spread_cost = 0.00005 * 2  # EURUSD entry + exit
pnls = []
trade_times = []
es_avg = np.zeros(T-1)

# Compute ES on rolling basis too (no lookahead)
for t in range(1, T):
    start = max(0, t - 50)
    es_avg[t-1] = np.mean(np.sum(rets[start:t]**2, axis=1))

# Rolling backtest
for t in range(lookback, T - 1 - fwd):
    # Rolling percentile thresholds (using only data up to t)
    mag_p90 = np.percentile(avg_mag[max(0, t-lookback):t], 90)
    es_p90 = np.nanpercentile(es_avg[max(0, t-lookback):t], 90)
    
    if not consensus[t]: continue
    if avg_mag[t] <= mag_p90: continue
    if es_avg[t] <= es_p90: continue
    
    # Trade EURUSD
    ret = np.log(close[t+fwd, 1] / close[t, 1])
    pnl = ret * direction[t] - spread_cost
    pnls.append(pnl)
    trade_times.append(t)

pnls = np.array(pnls)
n = len(pnls)
if n > 0:
    mean_n = np.mean(pnls)
    std_n = np.std(pnls) + 1e-10
    sharpe = mean_n / std_n * np.sqrt(1440/fwd)
    wr = np.mean(pnls > 0) * 100
    print(f"  Rolling P90 filter: n={n:6d}  net_mean={mean_n:+.6f}  WR={wr:.1f}%  Sharpe={sharpe:.3f}")
    print(f"  Trades/day: {n/(T-1-lookback)*1440:.1f}")
    
    # Monthly breakdown
    dt_trades = pd.to_datetime(times[trade_times], unit="s")
    for month in [10, 11, 12]:
        mask = dt_trades.month == month
        mpnl = pnls[mask]
        if len(mpnl) < 3: continue
        ms = np.mean(mpnl) / (np.std(mpnl)+1e-10) * np.sqrt(1440/fwd)
        print(f"    Oct: n={len(mpnl):5d}  mean={np.mean(mpnl):+.6f}  Sharpe={ms:.3f}")
else:
    print("  No trades!")

# Now do expanding window (even more conservative - uses ALL past data)
print("\n  Expanding window (uses all past data):")
pnls2 = []
for t in range(1440, T - 1 - fwd):
    mag_p90 = np.percentile(avg_mag[:t], 90)
    es_p90 = np.nanpercentile(es_avg[:t], 90)
    
    if not consensus[t]: continue
    if avg_mag[t] <= mag_p90: continue
    if es_avg[t] <= es_p90: continue
    
    ret = np.log(close[t+fwd, 1] / close[t, 1])
    pnl = ret * direction[t] - spread_cost
    pnls2.append(pnl)

pnls2 = np.array(pnls2)
if len(pnls2) > 0:
    s = np.mean(pnls2)/(np.std(pnls2)+1e-10)*np.sqrt(1440/fwd)
    print(f"  Expanding: n={len(pnls2)}  mean={np.mean(pnls2):+.6f}  Sharpe={s:.3f}")

print("\n" + "=" * 70)
print("PART 2: OOS VALIDATION — 7-pair dataset (Jun-Jul 2026)")
print("=" * 70)

df = pd.read_parquet(r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet")
ct = pd.DatetimeIndex(df["time"].unique()).sort_values()
cl = df.pivot_table(index="time", columns="pair", values="close").values.astype(np.float64)
PAIRS7 = list(df.pair.unique())
PIDX7 = {p: i for i, p in enumerate(PAIRS7)}

T7 = cl.shape[0]
rets7 = np.diff(np.log(cl), axis=0)
hour7 = pd.DatetimeIndex(ct).hour.values[1:]

# Consensus: EURJPY, EURUSD, GBPJPY
ei, usi, gbi = PIDX7["EURJPY"], PIDX7["EURUSD"], PIDX7["GBPJPY"]
up7 = rets7[:, [ei, usi, gbi]] > 0
all_up7 = up7.all(axis=1)
all_down7 = (~up7).all(axis=1)
consensus7 = all_up7 | all_down7
direction7 = np.where(all_up7, 1.0, -1.0)

avg_mag7 = np.mean(np.abs(rets7[:, [ei, usi, gbi]]), axis=1)

# ES
sq7 = rets7[:, [ei, usi, gbi]] ** 2
es7 = np.zeros((T7, 3))
cum7 = np.zeros((T7, 3))
cum7[1:] = np.cumsum(sq7, axis=0)
es7[51:] = cum7[51:] - cum7[1:T7-50]
es7[:51] = es7[51]
es_avg7 = np.nanmean(es7[1:], axis=1)

fwd = 5
cost = 0.00005 * 2
pnls7 = []

for t in range(1440, min(T7 - 1 - fwd, len(rets7))):
    mag_p90 = np.percentile(avg_mag7[max(0, t-1440):t], 90)
    es_p90 = np.nanpercentile(es_avg7[max(0, t-1440):t], 90)
    
    if not consensus7[t]: continue
    if avg_mag7[t] <= mag_p90: continue
    if np.isnan(es_avg7[t]) or es_avg7[t] <= es_p90: continue
    
    ret = np.log(cl[t+fwd, usi] / cl[t, usi])
    pnl = ret * direction7[t] - cost
    pnls7.append(pnl)

pnls7 = np.array(pnls7)
if len(pnls7) > 0:
    s = np.mean(pnls7)/(np.std(pnls7)+1e-10)*np.sqrt(1440/fwd)
    wr = np.mean(pnls7 > 0) * 100
    print(f"  OOS Rolling: n={len(pnls7):5d}  mean={np.mean(pnls7):+.6f}  WR={wr:.1f}%  Sharpe={s:.3f}")
    print(f"  Trades/day: {len(pnls7)/(min(T7-1-fwd, len(rets7))-1440)*1440:.1f}")
else:
    print("  No trades in OOS period")

# Also try without ES filter (mag-only)
pnls7b = []
for t in range(1440, min(T7 - 1 - fwd, len(rets7))):
    mag_p90 = np.percentile(avg_mag7[max(0, t-1440):t], 90)
    if not consensus7[t]: continue
    if avg_mag7[t] <= mag_p90: continue
    ret = np.log(cl[t+fwd, usi] / cl[t, usi])
    pnl = ret * direction7[t] - cost
    pnls7b.append(pnl)

pnls7b = np.array(pnls7b)
if len(pnls7b) > 0:
    s = np.mean(pnls7b)/(np.std(pnls7b)+1e-10)*np.sqrt(1440/fwd)
    wr = np.mean(pnls7b > 0) * 100
    print(f"  OOS Mag-only: n={len(pnls7b):5d}  mean={np.mean(pnls7b):+.6f}  WR={wr:.1f}%  Sharpe={s:.3f}")

# Also check raw consensus (no filters) for comparison
pnls7c = []
for t in range(1440, min(T7 - 1 - fwd, len(rets7))):
    if not consensus7[t]: continue
    ret = np.log(cl[t+fwd, usi] / cl[t, usi])
    pnl = ret * direction7[t] - cost
    pnls7c.append(pnl)
pnls7c = np.array(pnls7c)
if len(pnls7c) > 0:
    s = np.mean(pnls7c)/(np.std(pnls7c)+1e-10)*np.sqrt(1440/fwd)
    print(f"  OOS Raw cons: n={len(pnls7c):5d}  mean={np.mean(pnls7c):+.6f}  Sharpe={s:.3f}")
