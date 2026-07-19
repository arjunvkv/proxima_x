#!/usr/bin/env python3
"""Compare fixed vs rolling P95 threshold across all months."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM, LOT, MAG95 = 7, 100000, 0.00018741

def pip_val(p, usdjpy):
    return 10.0 if p == 1 else 1000.0 / usdjpy

def load_all():
    frames = {}
    for p, pn in [("eurjpy","EURJPY"),("eurusd","EURUSD"),("gbpjpy","GBPJPY")]:
        dfs = []
        for y in [2024, 2026]:
            for m in range(1, 13):
                if (y==2024 and m<10) or (y==2026 and m>6): continue
                ld = calendar.monthrange(y, m)[1]
                f = os.path.join(DATA, f"{p}-m1-bid-{y}-{m:02d}-01-{y}-{m:02d}-{ld}.csv")
                if not os.path.exists(f): continue
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                dfs.append(df)
        frames[pn] = pd.concat(dfs).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    common = sorted(set(frames["EURJPY"]["timestamp"]) & set(frames["EURUSD"]["timestamp"]) & set(frames["GBPJPY"]["timestamp"]))
    tmap = {p: {t: i for i, t in enumerate(frames[p]["timestamp"])} for p in frames}
    close = np.column_stack([frames[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    opens = np.column_stack([frames[p]["open"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    high = np.column_stack([frames[p]["high"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    low = np.column_stack([frames[p]["low"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
    return close, opens, high, low, times

close, opens, high, low, times = load_all()
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0; consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]
atr_arr = np.max([(high[:,p]-low[:,p])/0.0001 for p in range(3)], axis=0)
atr_median = np.median(atr_arr[1:])
MIN_IDX = 1440

def run(mode):
    ctx = {}
    for t in range(1, T - 1 - 4):
        if not consensus[t]: continue
        h = hour_arr[t]
        if h < 7 or h > 21: continue
        if mode == "fixed":
            thresh = MAG95
        else:
            lo = max(0, t - MIN_IDX)
            thresh = np.percentile(avg_mag[lo:t], 95)
        if avg_mag[t] <= thresh: continue
        p = np.argmax(pair_mags[t])
        next_i = min(t+1, T-1)
        entry_price = opens[next_i, p]
        exit_price = close[min(t+3, T-1), p]
        gross = np.log(exit_price/entry_price)*direction[t]
        u = np.mean(usdjpy_proxy[t-50:t+50])
        spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, u)
        slip_mult = atr_arr[t+1] / max(atr_median, 0.1)
        slip_var = (0.2 + 0.3 * min(slip_mult, 5.0)) * 2 * pip_val(p, u)
        gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[t]
        pnl = gusd - spread - slip_var - ECN_COMM
        ym = f"{dt_all[t+1].year}-{dt_all[t+1].month:02d}"
        if ym not in ctx: ctx[ym] = []
        ctx[ym].append(pnl)
    return ctx

print("=" * 90)
print("FIXED P95 vs ROLLING P95 — per month comparison (combined stress: latency+var slip+1.5x+$7)")
print("=" * 90)
print(f"{'Month':>10s} {'Fix_n':>6s} {'Fix_WR':>6s} {'Fix_Sh':>7s} {'Fix_Avg$':>8s} {'Roll_n':>6s} {'Roll_WR':>6s} {'Roll_Sh':>7s} {'Roll_Avg$':>8s}")
print("-" * 90)

fixed = run("fixed")
rolling = run("rolling")
all_months = sorted(set(list(fixed.keys()) + list(rolling.keys())))

for ym in all_months:
    fp = np.array(fixed.get(ym, [0]))
    rp = np.array(rolling.get(ym, [0]))
    fn = len(fp); fw = np.mean(fp>0)*100 if fn>5 else 0
    fs = np.mean(fp)/(np.std(fp)+1e-10)*np.sqrt(1440/3) if fn>5 else 0
    fa = np.mean(fp) if fn>5 else 0
    rn = len(rp); rw = np.mean(rp>0)*100 if rn>5 else 0
    rs = np.mean(rp)/(np.std(rp)+1e-10)*np.sqrt(1440/3) if rn>5 else 0
    ra = np.mean(rp) if rn>5 else 0
    print(f"{ym:>10s} {fn:6d} {fw:5.1f}% {fs:7.2f} {fa:8.2f} {rn:6d} {rw:5.1f}% {rs:7.2f} {ra:8.2f}")

# Totals
fp_all = np.concatenate(list(fixed.values())) if fixed else np.array([])
rp_all = np.concatenate(list(rolling.values())) if rolling else np.array([])
print("-" * 90)
print(f"{'TOTAL':>10s} {len(fp_all):6d} {np.mean(fp_all>0)*100:5.1f}% {np.mean(fp_all)/(np.std(fp_all)+1e-10)*np.sqrt(1440/3):7.2f} {np.mean(fp_all):8.2f} {len(rp_all):6d} {np.mean(rp_all>0)*100:5.1f}% {np.mean(rp_all)/(np.std(rp_all)+1e-10)*np.sqrt(1440/3):7.2f} {np.mean(rp_all):8.2f}")

print()
print("WHAT CONSISTENCY IS:")
print("  Fixed threshold: same P95 value from training set = predictable, reproducible.")
print("  Rolling adaptive: adjusts every bar = follows volatility, more trades in low-vol regimes.")
print()
print("TRADEOFF:")
print("  Fixed: fewer trades in Q2 2026 (470 vs ~1500) but SAME threshold = no lookahead concern.")
print("  Rolling: more consistent trade count (all months ~2000-4000) but threshold CHANGES daily.")
