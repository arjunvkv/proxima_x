"""Rigorous overfit/lookahead validation.

Tests:
1. Walk-forward on Exness: Oct-Nov (train) vs Dec (test)
2. Weekly breakdown on FundedNext: should be consistently positive
3. Cost sensitivity: how much cost kills the edge
4. Random label shuffle (sign test): does PnL survive label randomization?
5. Days to $2K: distribution of completion time
"""
import sys, time, numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

FNT_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")
EX_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")
PIP = 0.0001
PIP_USD = 10.0
SPREAD_PRICE = 0.8 * PIP
COMM_PRICE = 3.0 / PIP_USD * PIP
COST = SPREAD_PRICE + COMM_PRICE

def load_fundednext(pair):
    d = np.load(str(FNT_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

def load_exness(pair, months):
    dfs = []
    for y, m in months:
        p = EX_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                                 format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    ts = t["Ts"].astype(np.int64).values // 10**9
    bid = t["B"].values.astype(np.float64)
    ask = t["A"].values.astype(np.float64)
    return ts, bid, ask

def detect(ts, mid, window_s=20, detect_pips=5):
    n = len(mid)
    min_q, max_q = deque(), deque()
    ws_idx = 0
    evs = []
    thresh = detect_pips * PIP
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > window_s:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = mid[ws_idx]
            hp = float(max_q[0][0] - wp)
            lp = float(wp - min_q[0][0])
            span = ts[i] - ts[ws_idx]
            if span <= window_s and (hp >= thresh or lp >= thresh):
                if evs and evs[-1][0] >= ws_idx: continue
                ext_idx = max_q[0][1] if hp >= lp else min_q[0][1]
                d = 1 if hp >= lp else -1
                evs.append((ws_idx, ext_idx, d))
    return evs

def sim_trades(ev_list, ts, bid, ask, hold_s=30, stop_pips_abs=10*PIP):
    n = len(ts)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        stop_price = ep - stop_pips_abs if ed == 1 else ep + stop_pips_abs
        stop_hit = False
        if stop_pips_abs > 0:
            for j in range(ei2 + 1, he):
                if ed == 1 and bid[j] <= stop_price: stop_hit = True; break
                if ed == -1 and ask[j] >= stop_price: stop_hit = True; break
        xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - COST
        pnls.append(pnl)
    return np.array(pnls, dtype=np.float64)

t0 = time.time()

# ═══ 1. WALK-FORWARD ON EXNESS ═══
print("═" * 60)
print("1. WALK-FORWARD: Exness EURUSD")
print("═" * 60)

ex_samples = {
    "Oct (train)": [(2025,10)],
    "Nov (train)": [(2025,11)],
    "Dec (test)":  [(2025,12)],
    "Oct+Nov (train)": [(2025,10),(2025,11)],
    "All 3mo":    [(2025,10),(2025,11),(2025,12)],
}
for label, months in ex_samples.items():
    ts, bid, ask = load_exness("EURUSD", months)
    mid = (bid + ask) / 2.0
    evs = detect(ts, mid)
    pnls = sim_trades(evs, ts, bid, ask, 30, 10*PIP)
    if len(pnls) > 0:
        p = pnls / PIP
        wr = (p > 0).mean() * 100
        avg = p.mean()
        gross = p.sum()
        n = len(p)
    else:
        wr = avg = gross = n = 0
    print(f"  {label:>15s}: {n:>4d}t  WR={wr:>5.1f}%  avg={avg:>+7.2f}p  gross={gross:>+8.1f}p")

# ═══ 2. WEEKLY BREAKDOWN ON FUNDEDNEXT ═══
print("\n" + "═" * 60)
print("2. WEEKLY BREAKDOWN: FundedNext EURUSD")
print("═" * 60)

ts_fn, bid_fn, ask_fn = load_fundednext("EURUSD")
mid_fn = (bid_fn + ask_fn) / 2.0
evs_fn = detect(ts_fn, mid_fn)

# Tag each trade with its week
from datetime import datetime, timezone
import calendar

pnls_fn = sim_trades(evs_fn, ts_fn, bid_fn, ask_fn, 30, 10*PIP)
pips_fn = pnls_fn / PIP

# Get entry timestamp for each trade
trade_times = []
for ws_i, ext_i, ext_dir in evs_fn:
    t = ts_fn[ext_i + 1]
    trade_times.append(t)
trade_times = np.array(trade_times)

# Group by ISO week
week_of = {}
for i, t in enumerate(trade_times):
    d = datetime.fromtimestamp(t, tz=timezone.utc).date()
    iso = d.isocalendar()
    wk = f"{iso[0]}-W{iso[1]:02d}"
    week_of.setdefault(wk, []).append(pips_fn[i])

for wk in sorted(week_of.keys()):
    arr = np.array(week_of[wk])
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    gross = arr.sum()
    n = len(arr)
    # Get date range for this week
    wk_dates = []
    for i, t in enumerate(trade_times):
        d = datetime.fromtimestamp(t, tz=timezone.utc).date()
        iso = d.isocalendar()
        wk2 = f"{iso[0]}-W{iso[1]:02d}"
        if wk2 == wk:
            wk_dates.append(d)
    dr = f"{min(wk_dates).strftime('%m/%d')}-{max(wk_dates).strftime('%m/%d')}" if wk_dates else "?"
    print(f"  {wk} ({dr}): {n:>4d}t  WR={wr:>5.1f}%  avg={avg:>+7.2f}p  gross={gross:>+8.1f}p")

# ═══ 3. COST SENSITIVITY ═══
print("\n" + "═" * 60)
print("3. COST SENSITIVITY (FundedNext EURUSD, 10p stop)")
print("═" * 60)
print(f"  {'Cost(p)':>8s} {'WR':>5s} {'Avg(p)':>8s} {'Gross(p)':>10s} {'n':>5s}")
for cost_pips in [0.0, 0.3, 0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]:
    cost_adj = cost_pips * PIP
    pnls_adj = np.array([pnl - (cost_adj - COST) for pnl in pnls_fn]) if cost_pips > 0 else pnls_fn + COST
    if cost_pips == 0:
        pnls_adj = pnls_fn + COST  # remove all costs
    elif cost_pips != 1.1:
        diff = cost_pips * PIP - COST
        pnls_adj = pnls_fn - diff
    elif cost_pips == 1.1:
        pnls_adj = pnls_fn
    
    p = pnls_adj / PIP
    wr = (p > 0).mean() * 100
    avg = p.mean()
    gross = p.sum()
    print(f"  {cost_pips:>7.1f}p: {wr:>5.1f}% {avg:>+8.2f}p {gross:>+9.1f}p {len(p):>5d}")

# ═══ 4. RANDOM LABEL SHUFFLE TEST ═══
print("\n" + "═" * 60)
print("4. SIGN RANDOMIZATION TEST (Exness - 1000 shuffles)")
print("═" * 60)

ts_ex_all, bid_ex_all, ask_ex_all = load_exness("EURUSD", [(2025,10),(2025,11),(2025,12)])
mid_ex_all = (bid_ex_all + ask_ex_all) / 2.0
evs_ex_all = detect(ts_ex_all, mid_ex_all)
pnls_ex_all = sim_trades(evs_ex_all, ts_ex_all, bid_ex_all, ask_ex_all, 30, 10*PIP)
actual_avg = pnls_ex_all.mean()
n_trades = len(pnls_ex_all)

rng = np.random.default_rng(42)
shuffled_avgs = []
for _ in range(1000):
    shuffled = pnls_ex_all * rng.choice([1, -1], size=n_trades)
    shuffled_avgs.append(shuffled.mean())
shuffled_avgs = np.array(shuffled_avgs)

pctile = np.percentile(shuffled_avgs, [5, 50, 95])
p_value = (np.abs(shuffled_avgs) >= np.abs(actual_avg)).mean()
print(f"  Actual avg PnL: ${actual_avg/PIP*PIP_USD:.4f} ({actual_avg/PIP:+.2f}p)")
print(f"  Shuffled: median={np.median(shuffled_avgs)/PIP:+.2f}p  "
      f"90% range=[{pctile[0]/PIP:+.2f}p, {pctile[2]/PIP:+.2f}p]")
print(f"  p-value (sign-test): {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# ═══ 5. DAYS TO $2K DISTRIBUTION ═══
print("\n" + "═" * 60)
print("5. DAYS TO $2K (FundedNext MC, 2.5 lots, 10p stop)")
print("═" * 60)

# Use the actual trade PnLs from FundedNext
TRADES_PER_DAY = int(len(pips_fn) / 20)
N_SIM = 50000
rng2 = np.random.default_rng(123)

days_to_target = []
failures = 0
for _ in range(N_SIM):
    cum = 0.0
    for day in range(1, 31):  # max 30 days
        day_trades = rng2.choice(pips_fn, size=TRADES_PER_DAY, replace=True)
        day_pnl = day_trades.sum() * 2.5 * PIP_USD
        cum += day_pnl
        if cum >= 2000:
            days_to_target.append(day)
            break
    else:
        failures += 1

days_arr = np.array(days_to_target)
print(f"  Out of {N_SIM} sims:")
print(f"  Hit $2K: {(N_SIM-failures)/N_SIM*100:.1f}%  Never hit: {failures/N_SIM*100:.1f}%")
print(f"  Days to $2K (of those who hit):")
print(f"    Mean: {days_arr.mean():.1f}  Median: {np.median(days_arr):.0f}")
print(f"    P10: {np.percentile(days_arr,10):.0f}  P25: {np.percentile(days_arr,25):.0f}")
print(f"    P75: {np.percentile(days_arr,75):.0f}  P90: {np.percentile(days_arr,90):.0f}")
print(f"    Hit in 5 days: {(days_arr <= 5).mean()*100:.1f}%")
print(f"    Hit in 10 days: {(days_arr <= 10).mean()*100:.1f}%")
print(f"    Hit in 20 days: {(days_arr <= 20).mean()*100:.1f}%")

# Also show different lot sizes
print()
for lot in [1.5, 2.0, 2.5, 3.0]:
    dtt = []
    fail = 0
    for _ in range(N_SIM):
        cum = 0.0
        for day in range(1, 31):
            day_trades = rng2.choice(pips_fn, size=TRADES_PER_DAY, replace=True)
            day_pnl = day_trades.sum() * lot * PIP_USD
            cum += day_pnl
            if cum >= 2000:
                dtt.append(day)
                break
        else:
            fail += 1
    dtt_arr = np.array(dtt)
    hit = (N_SIM - fail) / N_SIM * 100
    hit5 = (dtt_arr <= 5).mean() * 100 if len(dtt_arr) > 0 else 0
    med = np.median(dtt_arr) if len(dtt_arr) > 0 else 999
    print(f"  {lot:.1f} lots: {hit:.0f}% hit $2K  median={med:.0f}d  hit5d={hit5:.0f}%")

print(f"\nTotal: {time.time()-t0:.1f}s")
