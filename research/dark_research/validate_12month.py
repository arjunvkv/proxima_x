#!/usr/bin/env python3
"""Full 12-month cross-validation: Q4 2024 + Q1+Q2 2026 Dukascopy data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"

PERIODS = [
    ("2024 Q4", (2024,10), (2024,12)),
    ("2026 Q1", (2026,1), (2026,3)),
    ("2026 Q2", (2026,4), (2026,6)),
]

HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM = 7
LOT = 100000
MAG95 = 0.00018741

def pip_val(p, usdjpy):
    if p == 1: return 10.0
    return 1000.0 / usdjpy

def load_period(period_name, start_ym, end_ym):
    pair_data = {}
    for p, pname in [("eurjpy","EURJPY"), ("eurusd","EURUSD"), ("gbpjpy","GBPJPY")]:
        frames = []
        for ym in pd.date_range(f"{start_ym[0]}-{start_ym[1]:02d}-01", f"{end_ym[0]}-{end_ym[1]:02d}-01", freq="MS"):
            m = ym.month; y = ym.year
            last_day = calendar.monthrange(y, m)[1]
            f = os.path.join(DATA, f"{p}-m1-bid-{y}-{m:02d}-01-{y}-{m:02d}-{last_day}.csv")
            if not os.path.exists(f): continue
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            frames.append(df)
        if not frames:
            print(f"  MISSING {pname} data for {period_name}")
            return None
        pair_data[pname] = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    common = sorted(set(pair_data["EURJPY"]["timestamp"]) & set(pair_data["EURUSD"]["timestamp"]) & set(pair_data["GBPJPY"]["timestamp"]))
    tmap = {p: {t: i for i, t in enumerate(pair_data[p]["timestamp"])} for p in pair_data}
    close = np.column_stack([pair_data[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    return close, np.array([int(t.timestamp()) for t in common], dtype=np.int64)

print("=" * 90)
print("FULL 12-MONTH CROSS-VALIDATION: Dukascopy Data (complete fresh sample)")
print("=" * 90)

all_rows = []

for pname, sy, ey in PERIODS:
    result = load_period(pname, sy, ey)
    if result is None:
        print(f"\n{pname}: NO DATA — skipping")
        continue
    close, times = result
    T = close.shape[0]
    rets = np.diff(np.log(close), axis=0)
    up = rets > 0; consensus = up.all(axis=1) | (~up).all(axis=1)
    direction = np.where(up.all(axis=1), 1.0, -1.0)
    avg_mag = np.mean(np.abs(rets), axis=1)
    pair_mags = np.abs(rets)
    dt_all = pd.to_datetime(times, unit="s")
    hour_arr = dt_all.hour.values[1:]
    usdjpy_proxy = close[:,0] / close[:,1]

    te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
    te_idx = te_idx[te_idx + 3 < T - 1]
    bi = np.argmax(pair_mags[te_idx], axis=1)

    avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
    ht = pd.DatetimeIndex([dt_all[ti] for ti in te_idx])

    print(f"\n{'='*70}")
    print(f"{pname}: {T:,} bars, {dt_all[0].date()} — {dt_all[-1].date()}")
    print(f"{'='*70}")

    # Realistic primary result
    pnls = []
    for j,i in enumerate(te_idx):
        p = bi[j]
        gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
        spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
        slp = 0.5*2*pip_val(p, avg_usdjpy)
        if p==1: gusd = LOT*gross
        else: gusd = LOT*gross*close[i,p]/usdjpy_proxy[i]
        pnls.append(gusd - spread - slp - ECN_COMM)
    d = np.array(pnls)
    if len(d) < 5:
        print(f"  Too few trades ({len(d)})")
        continue
    wr = np.mean(d>0)*100
    sh = np.mean(d)/(np.std(d)+1e-10)*np.sqrt(1440/3)
    tpd = len(d)/(T/1440)
    print(f"  1.5x spread, 0.5p slip: n={len(d):5d}  WR={wr:.1f}%  Sharpe={sh:5.2f}  Avg=\${np.mean(d):.2f}  Tot=\${np.sum(d):.0f}  Daily=\${np.mean(d)*tpd:.0f}")
    all_rows.append((pname, len(d), wr, sh, np.mean(d), np.sum(d)))

    # Monthly breakdown
    ym_set = sorted(set((ht.year[i], ht.month[i]) for i in range(len(ht))))
    print(f"  Monthly breakdown:")
    for y, m in ym_set:
        mask = (ht.year == y) & (ht.month == m)
        mp = d[mask]
        msh = np.mean(mp)/(np.std(mp)+1e-10)*np.sqrt(1440/3) if len(mp) > 3 else 0
        label = f"{pd.to_datetime(f'{y}-{m:02d}-01').strftime('%b %Y')}"
        print(f"    {label:>9s}: n={len(mp):4d}  WR={np.mean(mp>0)*100:.1f}%  Sharpe={msh:.2f}  Avg=\${np.mean(mp):.2f}  Tot=\${np.sum(mp):.0f}")

    # Rolling adaptive (no fixed threshold)
    pnls_r = []
    for t in range(1440, T - 1 - 3):
        if not consensus[t]: continue
        h = hour_arr[t]
        if h < 7 or h > 21: continue
        mag_t = np.percentile(avg_mag[t-1440:t], 95)
        if avg_mag[t] <= mag_t: continue
        p = np.argmax(pair_mags[t])
        gross = np.log(close[t+3,p]/close[t,p])*direction[t]
        spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
        slp = 0.5*2*pip_val(p, avg_usdjpy)
        if p==1: gusd = LOT*gross
        else: gusd = LOT*gross*close[t,p]/usdjpy_proxy[t]
        pnls_r.append(gusd - spread - slp - ECN_COMM)
    dr = np.array(pnls_r)
    if len(dr) > 10:
        wrr = np.mean(dr>0)*100
        shr = np.mean(dr)/(np.std(dr)+1e-10)*np.sqrt(1440/3)
        print(f"  Rolling adaptive: n={len(dr):5d}  WR={wrr:.1f}%  Sharpe={shr:.2f}  Avg=\${np.mean(dr):.2f}  Tot=\${np.sum(dr):.0f}")

    # Pair distribution
    pair_names = ["EURJPY","EURUSD","GBPJPY"]
    pair_counts = np.bincount(bi, minlength=3)
    print(f"  Trade distribution: ", end="")
    for pi, pn in enumerate(pair_names):
        print(f"{pn} {pair_counts[pi]}({pair_counts[pi]/len(bi)*100:.0f}%)", end="  ")
    print()

    # Sensitivity grid
    print(f"  Sensitivity (0.5p slippage):")
    for sm2 in [1.0, 1.5, 2.0, 3.0]:
        row_pnls = []
        for j,i in enumerate(te_idx):
            p = bi[j]
            gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
            spread = HALF_SPREAD_PIPS[p]*2*sm2*pip_val(p, avg_usdjpy)
            slp = 0.5*2*pip_val(p, avg_usdjpy)
            if p==1: gusd = LOT*gross
            else: gusd = LOT*gross*close[i,p]/usdjpy_proxy[i]
            row_pnls.append(gusd - spread - slp - ECN_COMM)
        dr = np.array(row_pnls)
        wr2 = np.mean(dr>0)*100
        sh2 = np.mean(dr)/(np.std(dr)+1e-10)*np.sqrt(1440/3)
        print(f"    {sm2:.0f}x spread: n={len(dr):5d}  WR={wr2:.1f}%  Sharpe={sh2:5.2f}  Avg=\${np.mean(dr):.2f}")

print()
print("=" * 90)
print("COMPOSITE: All periods combined, 1.5x spread, 0.5p slippage, $7 comm")
print("=" * 90)

# Combine all periods for a single composite run
all_pnls = []
for pname, sy, ey in PERIODS:
    result = load_period(pname, sy, ey)
    if result is None: continue
    close, times = result
    T = close.shape[0]
    rets = np.diff(np.log(close), axis=0)
    up = rets > 0; consensus = up.all(axis=1) | (~up).all(axis=1)
    direction = np.where(up.all(axis=1), 1.0, -1.0)
    avg_mag = np.mean(np.abs(rets), axis=1)
    pair_mags = np.abs(rets)
    dt_all = pd.to_datetime(times, unit="s")
    hour_arr = dt_all.hour.values[1:]
    usdjpy_proxy = close[:,0] / close[:,1]
    te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
    te_idx = te_idx[te_idx + 3 < T - 1]
    bi = np.argmax(pair_mags[te_idx], axis=1)
    avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
    for j,i in enumerate(te_idx):
        p = bi[j]
        gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
        spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
        slp = 0.5*2*pip_val(p, avg_usdjpy)
        if p==1: gusd = LOT*gross
        else: gusd = LOT*gross*close[i,p]/usdjpy_proxy[i]
        all_pnls.append(gusd - spread - slp - ECN_COMM)

d_all = np.array(all_pnls)
wr_all = np.mean(d_all>0)*100
sh_all = np.mean(d_all)/(np.std(d_all)+1e-10)*np.sqrt(1440/3)
print(f"  All months combined: n={len(d_all):5d}  WR={wr_all:.1f}%  Sharpe={sh_all:5.2f}  Avg=\${np.mean(d_all):.2f}  Tot=\${np.sum(d_all):.0f}")

# Breakeven spread
print()
print("=" * 90)
print("BREAKEVEN SEARCH: at what spread multiplier does Sharpe hit 0?")
print("=" * 90)
all_pnls2 = []
for pname, sy, ey in PERIODS:
    result = load_period(pname, sy, ey)
    if result is None: continue
    close, times = result
    T = close.shape[0]
    rets = np.diff(np.log(close), axis=0)
    up = rets > 0; consensus = up.all(axis=1) | (~up).all(axis=1)
    direction = np.where(up.all(axis=1), 1.0, -1.0)
    avg_mag = np.mean(np.abs(rets), axis=1)
    pair_mags = np.abs(rets)
    dt_all = pd.to_datetime(times, unit="s")
    hour_arr = dt_all.hour.values[1:]
    usdjpy_proxy = close[:,0] / close[:,1]
    te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
    te_idx = te_idx[te_idx + 3 < T - 1]
    bi = np.argmax(pair_mags[te_idx], axis=1)
    avg_usdjpy = np.mean(usdjpy_proxy[te_idx])
    for j,i in enumerate(te_idx):
        p = bi[j]
        gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
        spread = HALF_SPREAD_PIPS[p]*2*1.0*pip_val(p, avg_usdjpy)
        slp = 0.5*2*pip_val(p, avg_usdjpy)
        if p==1: gusd = LOT*gross
        else: gusd = LOT*gross*close[i,p]/usdjpy_proxy[i]
        all_pnls2.append((gusd, spread, slp, ECN_COMM))

gross_arr = np.array([x[0] for x in all_pnls2])
spread_arr = np.array([x[1] for x in all_pnls2])
slip_arr = np.array([x[2] for x in all_pnls2])

for sm in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    pnl = gross_arr - sm*spread_arr - slip_arr - ECN_COMM
    sh = np.mean(pnl)/(np.std(pnl)+1e-10)*np.sqrt(1440/3)
    wr = np.mean(pnl>0)*100
    print(f"  {sm:.1f}x spread: Sharpe={sh:.2f}  WR={wr:.1f}%")
