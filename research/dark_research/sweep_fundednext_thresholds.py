#!/usr/bin/env python3
"""Sweep magnitude thresholds on FundedNext data to see if any threshold works."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(__file__)
pairs = ["eurjpy", "eurusd", "gbpjpy"]
data = {}
for p in pairs:
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    data[p] = df.set_index("time")["close"]

common = sorted(set(data["eurjpy"].index) & set(data["eurusd"].index) & set(data["gbpjpy"].index))
close = np.column_stack([data[p].loc[common].values for p in pairs])
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
hour = pd.DatetimeIndex(common).hour.values[1:]
usdjpy = close[:, 0] / close[:, 1]

# Sweep thresholds
thresholds = {
    "P80": np.percentile(avg_mag, 80),
    "P85": np.percentile(avg_mag, 85),
    "P90": np.percentile(avg_mag, 90),
    "P93": np.percentile(avg_mag, 93),
    "P95": np.percentile(avg_mag, 95),
    "P97": np.percentile(avg_mag, 97),
    "P99": np.percentile(avg_mag, 99),
    "DukP95": 0.00018741,
}

print(f"{'Threshold':>10s} {'Mag':>12s} {'n':>6s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>8s} {'Tot$':>10s}")
print("-" * 62)

for label, mag_t in sorted(thresholds.items(), key=lambda x: x[1]):
    g_list = []
    for t in range(1440, len(close) - 4):
        if not consensus[t]: continue
        if hour[t] < 7 or hour[t] > 21: continue
        if avg_mag[t] <= mag_t: continue
        bi = int(np.argmax(pair_mags[t]))
        ep = close[t, bi]
        xp = close[t+3, bi]
        if bi == 1:
            gross = (xp - ep) * 100000
        else:
            gross = (xp - ep) * 100000 / usdjpy[t]
        g_list.append(gross)
    g = np.array(g_list)
    if len(g) > 5:
        wr = np.mean(g > 0) * 100
        sh = np.mean(g) / (np.std(g) + 1e-10) * np.sqrt(1440/3)
        avg = np.mean(g)
        tot = np.sum(g)
        print(f"{label:>10s} {mag_t:12.8f} {len(g):6d} {wr:5.1f} {sh:7.2f} {avg:8.2f} {tot:10,.0f}")
    else:
        print(f"{label:>10s} {mag_t:12.8f} {len(g):6d} (too few)")

# Also test without hour filter
print(f"\n--- Without hour filter (0-24 UTC) ---")
print(f"{'Threshold':>10s} {'n':>6s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>8s} {'Tot$':>10s}")
print("-" * 52)
for label, mag_t in [("P90", thresholds["P90"]), ("P95", thresholds["P95"]), ("DukP95", thresholds["DukP95"])]:
    g_list = []
    for t in range(1440, len(close) - 4):
        if not consensus[t]: continue
        if avg_mag[t] <= mag_t: continue
        bi = int(np.argmax(pair_mags[t]))
        ep = close[t, bi]
        xp = close[t+3, bi]
        if bi == 1:
            gross = (xp - ep) * 100000
        else:
            gross = (xp - ep) * 100000 / usdjpy[t]
        g_list.append(gross)
    g = np.array(g_list)
    if len(g) > 5:
        wr = np.mean(g > 0) * 100
        sh = np.mean(g) / (np.std(g) + 1e-10) * np.sqrt(1440/3)
        print(f"{label:>10s} {len(g):6d} {wr:5.1f} {sh:7.2f} {np.mean(g):8.2f} {np.sum(g):10,.0f}")

# Also test LONG-only, SHORT-only
print(f"\n--- Direction asymmetry (P95, with hour) ---")
for dlab, dcond in [("LONG", direction > 0), ("SHORT", direction < 0)]:
    g_list = []
    for t in range(1440, len(close) - 4):
        if not consensus[t]: continue
        if hour[t] < 7 or hour[t] > 21: continue
        if avg_mag[t] <= thresholds["P95"]: continue
        if not dcond[t]: continue
        bi = int(np.argmax(pair_mags[t]))
        ep = close[t, bi]
        xp = close[t+3, bi]
        if bi == 1:
            gross = (xp - ep) * 100000
        else:
            gross = (xp - ep) * 100000 / usdjpy[t]
        g_list.append(gross)
    g = np.array(g_list)
    if len(g) > 5:
        print(f"  {dlab}: n={len(g):4d} WR={np.mean(g>0)*100:.1f}% Sharpe={np.mean(g)/(np.std(g)+1e-10)*np.sqrt(1440/3):.2f} Avg=${np.mean(g):.2f} Tot=${np.sum(g):,.0f}")
