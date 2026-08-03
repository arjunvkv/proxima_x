"""Full Monte Carlo: impulse fade on FundedNext costs (Exness ticks)."""
import sys, time, numpy as np, pandas as pd
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
import random
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
    min_q = deque(); max_q = deque(); ws_idx = 0
    evs = []
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

t0 = time.time()
ticks = load("EURUSD", [(2025,10),(2025,11),(2025,12)])
ev_list = detect_all(ticks)
pnls = sim(ev_list, ticks, 0.00011, 30, 5 * 0.0001)
p = pnls / 0.0001
print(f"Trades: {len(pnls)} in 65d = {len(pnls)/65:.1f}/d")
print(f"Avg: {p.mean():.2f}p  WR: {(p>0).mean()*100:.1f}%")

# Daily PnL
tick_ts = ticks["ts_s"].values
daily = defaultdict(float)
ev_idx = 0
for ws_i, ext_i, ext_dir in ev_list:
    ei2 = ext_i + 1
    day_dt = datetime.utcfromtimestamp(tick_ts[ei2]).date()
    daily[day_dt] += p[ev_idx]
    ev_idx += 1

daily_arr = np.array(list(daily.values()))
print(f"\nDaily stats (1.0 lot): mean=${daily_arr.mean():.0f} std=${daily_arr.std():.0f}")
print(f"  min=${daily_arr.min():.0f} max=${daily_arr.max():.0f} pct_pos={(daily_arr>0).mean()*100:.0f}%")

# Contiguous 5-day blocks
dates_sorted = sorted(daily.keys())
contig = []
for i in range(len(dates_sorted) - 4):
    if (dates_sorted[i+4] - dates_sorted[i]).days == 4:
        s = sum(daily[dates_sorted[i+j]] for j in range(5))
        contig.append(s)

contig = np.array(contig)
print(f"\nContiguous 5-day blocks: {len(contig)}")
print(f"  mean=${contig.mean():.0f} min=${contig.min():.0f} max=${contig.max():.0f}")

# Monte Carlo
N_SIM = 50000
print(f"\n=== 5-DAY CHALLENGE PASS RATE (FundedNext costs, {N_SIM} sims) ===")
print(f"{'Lots':>6s} | {'Pass%':>7s} | {'AvgPnL':>8s} | {'Blow%':>7s}")
print("-"*36)
for lot in [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    at_risk = 0
    n_pass = 0
    for _ in range(N_SIM):
        b = random.choice(contig) if len(contig) > 0 else 0
        net = b * lot
        if net >= 2000:
            n_pass += 1
        # Daily loss check: worst single day at this lot size
    # Daily loss check (conservative: any single day could occur in a 5-day block)
    any_blow_day = (abs(daily_arr.min()) * lot) > 1250
    blow_pct = ">95%" if any_blow_day else "<5%"
    pass_pct = n_pass / N_SIM * 100
    premium = (contig * lot).mean()
    print(f" {lot:>5.1f} | {pass_pct:>6.2f}% | ${premium:>+7.0f} | {blow_pct:>7s}")

# Find optimal lot size (max pass rate without blowing daily limit)
print("\n=== OPTIMAL LOT SIZE ===")
for lot in [x*0.25 for x in range(1, 41)]:
    worst_day_loss = abs(daily_arr.min()) * lot
    if worst_day_loss > 1250:
        break
    n_pass = 0
    for _ in range(N_SIM):
        b = random.choice(contig) if len(contig) > 0 else 0
        if b * lot >= 2000:
            n_pass += 1
    pass_pct = n_pass / N_SIM * 100
    print(f"  {lot:.2f} lots: pass={pass_pct:.1f}% worst_day=${worst_day_loss:.0f} (limit=$1250)")

print(f"\nDone: {time.time()-t0:.1f}s")
