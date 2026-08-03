"""Better Monte Carlo: simulate 5-day challenge by resampling trades."""
import sys, numpy as np, pandas as pd
from collections import deque
from pathlib import Path
from datetime import datetime
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                                 format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect_all(ticks):
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(mid)
    min_q = deque(); max_q = deque(); ws_idx = 0; evs = []
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > 20:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = mid[ws_idx]; hp = float(max_q[0][0] - wp); lp = float(wp - min_q[0][0])
            if ts[i] - ts[ws_idx] <= 20 and (hp >= 5*0.0001 or lp >= 5*0.0001):
                if evs and evs[-1][0] >= ws_idx: continue
                d = 1 if hp >= lp else -1
                ext_idx = max_q[0][1] if d == 1 else min_q[0][1]
                evs.append((ws_idx, ext_idx, d))
    return evs

def sim(ev_list, ticks, cost, hold_s, stop_pips):
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        sp = ep - stop_pips if ed == 1 else ep + stop_pips
        hit = False
        if stop_pips > 0:
            for j in range(ei2 + 1, he):
                if (ed == 1 and bid[j] <= sp) or (ed == -1 and ask[j] >= sp):
                    hit = True; break
        xp = sp if hit else (bid[he] if ed == 1 else ask[he])
        pnls.append((xp - ep) * ed - cost)
    return np.array(pnls, dtype=np.float64)

t0 = datetime.now()
ticks = load("EURUSD", [(2025,10),(2025,11),(2025,12)])
ev_list = detect_all(ticks)
pnls = sim(ev_list, ticks, 0.00011, 30, 5 * 0.0001)
pips = pnls / 0.0001

TRADES_PER_DAY = int(len(pips) / 65)  # ~48
print(f"Trades: {len(pips)} in 65 days = {len(pips)/65:.1f}/day")
print(f"Avg: {pips.mean():.2f}p  WR: {(pips>0).mean()*100:.1f}%")

N_SIM = 100000
print(f"\n=== 5-DAY CHALLENGE (FundedNext costs, {N_SIM} sims) ===")
print(f"{'Lots':>6s} | {'Pass%':>7s} | {'Avg5d$':>8s} | {'P10':>7s} | {'P25':>7s} | {'P50':>7s} | {'P75':>7s} | {'P90':>7s} | {'BlowDay':>8s}")
print("-"*72)

for lot in [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
    results = np.zeros(N_SIM)
    has_blow_day = np.zeros(N_SIM, dtype=bool)
    for i in range(N_SIM):
        # Simulate 5 days: each day ~48 trades
        day_pnls = np.zeros(5)
        for d in range(5):
            trades = np.random.choice(pips, size=TRADES_PER_DAY, replace=True)
            day_pnls[d] = trades.sum() * lot * 10  # $ at lot scale
        results[i] = day_pnls.sum()
        has_blow_day[i] = day_pnls.min() < -1250
    
    pass_mask = (results >= 2000) & (~has_blow_day)
    pass_pct = pass_mask.mean() * 100
    prem = results.mean()
    p10, p25, p50, p75, p90 = np.percentile(results, [10, 25, 50, 75, 90])
    blow_pct = has_blow_day.mean() * 100
    print(f" {lot:>5.1f} | {pass_pct:>6.2f}% | ${prem:>+7.0f} | ${p10:>+6.0f} | ${p25:>+6.0f} | ${p50:>+6.0f} | ${p75:>+6.0f} | ${p90:>+6.0f} | {blow_pct:>7.2f}%")

print(f"\nDone: {(datetime.now()-t0).total_seconds():.1f}s")
