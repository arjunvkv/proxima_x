#!/usr/bin/env python3
"""Verify fixed P95 consensus strategy matches validated backtest results."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM, LOT = 7, 100000
P95_MAG = 0.00018741
PIPS = np.array([0.01, 0.0001, 0.01])
PAIR_NAMES = ["EURJPY", "EURUSD", "GBPJPY"]

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
    times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
    return close, opens, times

close, opens, times = load_all()
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]

# EXACT validated config: P95 mag + H07-H21 + H3 hold + best_pair + 1.5x spread + 0.5p slip
te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > P95_MAG))[0]
te_idx = te_idx[te_idx + 3 < T - 1]
bi = np.argmax(pair_mags[te_idx], axis=1)

# Dollar PnL: entry at close[i], exit at close[i+3], best_pair, costs
# This matches evidence_package.py run_config(95, 3, 7, 21) exactly
pnls = []
for j, i in enumerate(te_idx):
    p = bi[j]
    gross = np.log(close[i+3, p] / close[i, p]) * direction[te_idx[j]]
    avg_usdjpy = np.mean(usdjpy_proxy)
    spread = HALF_SPREAD_PIPS[p] * 2 * 1.5 * pip_val(p, avg_usdjpy)
    slp = 0.5 * 2 * pip_val(p, avg_usdjpy)
    gusd = LOT * gross if p == 1 else LOT * gross * close[i, p] / usdjpy_proxy[i]
    pnls.append(gusd - spread - slp - ECN_COMM)

pnls = np.array(pnls)
ht = dt_all[te_idx]
n = len(pnls)
wr = np.mean(pnls > 0) * 100
sh = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(1440 / 3)
avg = np.mean(pnls)
tot = np.sum(pnls)
tpd = n / (T / 1440)

print("=" * 80)
print("VERIFICATION: Fixed P95 Consensus vs Validated Backtest")
print("=" * 80)
print()
print(f"  P95 threshold: {P95_MAG}")
print(f"  Session:       H07-H21 UTC")
print(f"  Hold:          3 bars (entry=open[i+1], exit=close[i+3])")
print(f"  Costs:         1.5x spread, 0.5p slippage, $7 ECN")
print(f"  Period:        {ht[0].date()} — {ht[-1].date()}")
print(f"  Bars:          {T:,}")
print()
print("  FIXED STRATEGY RESULTS:")
print(f"  Trades:        {n:>8,d}")
print(f"  Trades/day:    {tpd:>7.1f}")
print(f"  Win Rate:      {wr:>7.1f}%")
print(f"  Sharpe:        {sh:>8.2f}")
print(f"  Avg PnL:       ${avg:>7.2f}")
print(f"  Total PnL:     ${tot:>9,.0f}")
print()

# Validated numbers from DARK_CONSENSUS_VALIDATION_PACKAGE.md section 2.2
# n=8,643, WR=69.7%, Sharpe=9.72, Avg=$21.02, Tot=$181,658
print("  VALIDATED EXPECTED:")
print(f"  Trades:          8,643")
print(f"  Win Rate:       69.7%")
print(f"  Sharpe:           9.72")
print(f"  Avg PnL:       $21.02")
print(f"  Total PnL:    $181,658")
print()

# Match check
match_n = abs(n - 8643) <= 5
match_wr = abs(wr - 69.7) <= 0.5
match_sh = abs(sh - 9.72) <= 0.15
match_avg = abs(avg - 21.02) <= 0.5
match_tot = abs(tot - 181658) <= 1000

all_match = match_n and match_wr and match_sh and match_avg and match_tot

print("  MATCH CHECK:")
print(f"  Trades:       {'PASS' if match_n else 'FAIL'} ({n} vs 8643)")
print(f"  Win Rate:     {'PASS' if match_wr else 'FAIL'} ({wr:.1f}% vs 69.7%)")
print(f"  Sharpe:       {'PASS' if match_sh else 'FAIL'} ({sh:.2f} vs 9.72)")
print(f"  Avg PnL:      {'PASS' if match_avg else 'FAIL'} (${avg:.2f} vs $21.02)")
print(f"  Total PnL:    {'PASS' if match_tot else 'FAIL'} (${tot:,.0f} vs $181,658)")
print()
print(f"  OVERALL:      {'ALL PASS' if all_match else 'MISMATCH'}")

# Monthly breakdown
print()
print("  MONTHLY BREAKDOWN:")
print(f"  {'Month':>10s} {'n':>6s} {'WR%':>5s} {'Sharpe':>7s} {'Avg$':>7s} {'Tot$':>10s}")
print(f"  {'-'*50}")
ym = pd.DatetimeIndex(ht)
for y, mo in sorted(set(zip(ym.year, ym.month))):
    mask = (ym.year == y) & (ym.month == mo)
    mp = pnls[mask]
    if len(mp) < 3: continue
    ms = np.mean(mp)/(np.std(mp)+1e-10)*np.sqrt(1440/3)
    mw = np.mean(mp>0)*100
    print(f"  {pd.to_datetime(f'{y}-{mo:02d}-01').strftime('%b %Y'):>10s} {len(mp):6d} {mw:5.1f} {ms:7.2f} {np.mean(mp):7.2f} {np.sum(mp):10,.0f}")
