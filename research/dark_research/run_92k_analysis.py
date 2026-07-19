#!/usr/bin/env python3
"""Cross-pair analysis on 92K M1 bars (Oct-Dec 2025) for 3 pairs."""
import os, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import binomtest
import warnings; warnings.filterwarnings("ignore")

base = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
PAIRS3 = ["EURJPY", "EURUSD", "GBPJPY"]

t0 = time.time()

# Load
raw = {}
for p in ["eurjpy", "eurusd", "gbpjpy"]:
    raw[p] = {
        "p": np.load(os.path.join(base, f"{p}_m1_prices.npy")),
        "t": np.load(os.path.join(base, f"{p}_m1_times.npy")),
        "v": np.load(os.path.join(base, f"{p}_m1_volume.npy")),
    }
    print(f"  {p}: {len(raw[p]['p'])} bars")

# Common timestamps
common = sorted(set(raw["eurjpy"]["t"]) & set(raw["eurusd"]["t"]) & set(raw["gbpjpy"]["t"]))
print(f"Common timestamps: {len(common)}")

# Align all pairs to common timestamps
def align(pair_key, common, field):
    tmap = {t: i for i, t in enumerate(raw[pair_key]["t"])}
    return raw[pair_key][field][[tmap[t] for t in common]]

close = np.column_stack([align(k, common, "p")[:, 3] for k in ["eurjpy", "eurusd", "gbpjpy"]])
high  = np.column_stack([align(k, common, "p")[:, 1] for k in ["eurjpy", "eurusd", "gbpjpy"]])
low   = np.column_stack([align(k, common, "p")[:, 2] for k in ["eurjpy", "eurusd", "gbpjpy"]])
vol   = np.column_stack([align(k, common, "v") for k in ["eurjpy", "eurusd", "gbpjpy"]])
times = np.array(common)

T = close.shape[0]
hour = pd.DatetimeIndex(pd.to_datetime(times, unit="s")).hour.values
rets = np.diff(np.log(close), axis=0)
N = rets.shape[0]
print(f"Aligned: {T} bars, {N} returns")
print(f"Range: {pd.to_datetime(times[0], unit='s')} to {pd.to_datetime(times[-1], unit='s')}")

# ES proxy
sq = rets ** 2
es_aligned = np.zeros((T, 3))
cum = np.zeros((T, 3))
cum[1:] = np.cumsum(sq, axis=0)
es_aligned[51:] = cum[51:] - cum[1:T-50]
es_aligned[:51] = np.nan
for i in range(1, T):
    if np.isnan(es_aligned[i, 0]):
        es_aligned[i] = es_aligned[i-1]

delta_aligned = np.diff(es_aligned, axis=0)

print(f"\n--- A1: CROSS-PAIR ENERGY DIVERGENCE (CPED) ---")
for pA, pB in [("EURJPY","GBPJPY"), ("EURJPY","EURUSD"), ("EURUSD","GBPJPY")]:
    iA, iB = PAIRS3.index(pA), PAIRS3.index(pB)
    div = np.abs(es_aligned[:, iA] - es_aligned[:, iB]) / (es_aligned[:, iA] + es_aligned[:, iB] + 1e-10)
    for th in [0.3, 0.5, 0.7, 0.8]:
        idxs = np.where(div > th)[0]
        n_ev = len(idxs)
        if n_ev < 10: continue
        hi_more = 0
        for t in idxs:
            if t + 5 >= T: continue
            ma = abs(np.log(close[t+5, iA] / close[t, iA]))
            mb = abs(np.log(close[t+5, iB] / close[t, iB]))
            if (es_aligned[t,iA] >= es_aligned[t,iB] and ma >= mb) or \
               (es_aligned[t,iB] > es_aligned[t,iA] and mb >= ma):
                hi_more += 1
        wr = hi_more / n_ev
        pv = binomtest(hi_more, n_ev, 0.5, alternative='two-sided').pvalue
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"  [{pA}/{pB}] th={th:.1f} n={n_ev:6d} WR={wr:.1%} p={pv:.4f} {sig}")

print(f"\n--- A2: RETURN DISPERSION → VOL ---")
cs_disp = np.std(rets, axis=1)
p80, p20 = np.percentile(cs_disp, 80), np.percentile(cs_disp, 20)
for fw, fn in [(5,"M5"),(15,"M15"),(30,"M30"),(60,"H1")]:
    hi = np.where(cs_disp[:-fw] > p80)[0]
    lo = np.where(cs_disp[:-fw] < p20)[0]
    hv = np.array([np.mean(np.abs(rets[i:i+fw])) for i in hi if i+fw < N])
    lv = np.array([np.mean(np.abs(rets[i:i+fw])) for i in lo if i+fw < N])
    if len(hv) >= 5 and len(lv) >= 5:
        r = np.mean(hv)/(np.mean(lv)+1e-10)
        tst, pv = sp_stats.ttest_ind(hv, lv, equal_var=False)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"  {fn} hi={np.mean(hv):.6f} lo={np.mean(lv):.6f} ratio={r:.3f}x t={tst:.1f} p={pv:.4f} {sig}")

print(f"\n--- A3: RETURN ASYMMETRY ---")
n_up = np.sum(rets > 0, axis=1)
for th in [2, 3]:  # 2/3 or 3/3 majority
    maj = n_up >= th
    min_ = n_up <= 3 - th
    if maj.sum() < 10 or min_.sum() < 10:
        print(f"  {th}/3 maj: insufficient (maj={maj.sum()} min={min_.sum()})")
        continue
    for fw, fn in [(5,"M5"),(30,"M30"),(60,"H1")]:
        mi = np.where(maj[:-fw])[0]; ni = np.where(min_[:-fw])[0]
        mv = np.array([np.mean(np.log(close[i+fw] / close[i])) for i in mi if i+fw < T])
        nv = np.array([np.mean(np.log(close[i+fw] / close[i])) for i in ni if i+fw < T])
        if len(mv) >= 5 and len(nv) >= 5:
            tst, pv = sp_stats.ttest_ind(mv, nv, equal_var=False)
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {fn} {th}/3 maj={np.mean(mv):+.6f} min={np.mean(nv):+.6f} diff={np.mean(mv)-np.mean(nv):+.6f} t={tst:.3f} p={pv:.4f} {sig}")

print(f"\n--- A4: OHLC TAIL STRUCTURE ---")
# Upper tail = (high - close) / (high - low)
# Lower tail = (close - low) / (high - low)
# When all 3 pairs have upper tails > 0.7 (sellers dominated close), what happens?
upper_tail = (high - close) / (high - low + 1e-10)
lower_tail = (close - low) / (high - low + 1e-10)

# Extreme consensus in tails
for fw, fn in [(5,"M5"),(30,"M30")]:
    # All 3 pairs seller-dominated
    all_sell = (upper_tail > 0.7).all(axis=1)
    all_buy  = (lower_tail > 0.7).all(axis=1)
    
    if all_sell.sum() >= 10:
        idx_s = np.where(all_sell)[0]
        val_s = np.array([np.mean(np.log(close[i+fw] / close[i])) for i in idx_s if i+fw < T])
        idx_b = np.where(all_buy)[0]
        val_b = np.array([np.mean(np.log(close[i+fw] / close[i])) for i in idx_b if i+fw < T])
        if len(val_s) >= 5 and len(val_b) >= 5:
            tst, pv = sp_stats.ttest_ind(val_s, val_b, equal_var=False)
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {fn} all-sell={np.mean(val_s):+.6f} all-buy={np.mean(val_b):+.6f} diff={np.mean(val_s)-np.mean(val_b):+.6f} t={tst:.3f} p={pv:.4f} {sig} n_s={len(val_s)} n_b={len(val_b)}")

# Cross-pair tail divergence: when EURJPY has sell-tails but EURUSD has buy-tails
print(f"\n--- A5: CROSS-PAIR TAIL DIVERGENCE ---")
for (pA, pB), (iA, iB) in zip([("EURJPY","EURUSD"), ("EURJPY","GBPJPY"), ("EURUSD","GBPJPY")],
                                 [(0,1), (0,2), (1,2)]):
    div_mask = (upper_tail[:, iA] > 0.7) & (lower_tail[:, iB] > 0.7)
    cnt = div_mask.sum()
    if cnt < 10: continue
    for fw, fn in [(5,"M5"),(30,"M30")]:
        idxs = np.where(div_mask)[0]
        vals = np.array([np.mean(np.log(close[i+fw] / close[i])) for i in idxs if i+fw < T])
        if len(vals) >= 5:
            print(f"  {pA} sell/{pB} buy: {fn} ret={np.mean(vals):+.6f} WR={np.mean(vals>0):.1%} n={len(vals)}")

print(f"\n--- A6: SPREAD PROXY (H-L as vol proxy) ---")
# Use high-low range as a volatility/activity proxy
hl_range = (high - low) / close * 10000  # in pips
for pair, i in [("EURJPY",0), ("EURUSD",1), ("GBPJPY",2)]:
    hl = hl_range[:, i]
    p80 = np.percentile(hl, 80); p20 = np.percentile(hl, 20)
    for fw, fn in [(5,"M5"),(30,"M30")]:
        hi = np.where(hl[:-fw] > p80)[0]
        lo = np.where(hl[:-fw] < p20)[0]
        hv = np.array([abs(np.log(close[t+fw, i] / close[t, i])) for t in hi if t+fw < T])
        lv = np.array([abs(np.log(close[t+fw, i] / close[t, i])) for t in lo if t+fw < T])
        if len(hv) >= 5 and len(lv) >= 5:
            r = np.mean(hv)/(np.mean(lv)+1e-10)
            tst, pv = sp_stats.ttest_ind(hv, lv, equal_var=False)
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {pair:8s} {fn} hiHL_mag={np.mean(hv):.6f} loHL_mag={np.mean(lv):.6f} ratio={r:.3f}x t={tst:.2f} p={pv:.4f} {sig}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
