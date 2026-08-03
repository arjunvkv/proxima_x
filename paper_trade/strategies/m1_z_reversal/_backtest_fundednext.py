"""Impulse fade on Exness ticks, with FundedNext costs (0.8 pip spread + $3 comm).

Equivalent cost per trade at 1.0 lot EURUSD:
- Exness spread: 0.3 pips = 0.00003
- FundedNext spread: 0.8 pips = 0.00008
- FundedNext commission: $3 = 0.00003
- Total FundedNext: 0.00011 (= 1.1 pips)

We also test a scaled lot version to see what lot size passes the 5-day challenge.
"""
import sys, time, numpy as np, pandas as pd
from collections import deque
from pathlib import Path
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

def detect_all(ticks, pip_size):
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(mid)
    configs = [(5, 20, 5 * pip_size)]
    min_q = deque(); max_q = deque(); ws_idx = 0
    events = {c: [] for c in configs}
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
            span = ts[i] - ts[ws_idx]
            for tp, w, raw in configs:
                if span > w: continue
                if hp >= raw or lp >= raw:
                    evs = events[(tp, w, raw)]
                    if evs and evs[-1][0] >= ws_idx: continue
                    if hp >= lp: ext_idx = max_q[0][1]; d = 1
                    else: ext_idx = min_q[0][1]; d = -1
                    evs.append((ws_idx, ext_idx, d))
    return events[(5, 20, 5 * pip_size)]

def sim_with_stop(ev_list, ticks, pip, cost, hold_s, stop_pips, direction="both"):
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    pnls = []; cum = 0.0; peak = -1e9; max_dd = 0.0; max_cl = 0; cl = 0
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        if direction == "short" and ed == 1: continue
        if direction == "long" and ed == -1: continue
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        stop_price = ep - stop_pips if ed == 1 else ep + stop_pips
        stop_hit = False; stop_idx = he
        if stop_pips > 0:
            for j in range(ei2 + 1, he):
                if ed == 1:
                    if bid[j] <= stop_price: stop_hit = True; stop_idx = j; break
                else:
                    if ask[j] >= stop_price: stop_hit = True; stop_idx = j; break
        xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl)
        cum += pnl
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd
        if pnl > 0: cl = 0
        else: cl += 1; max_cl = max(max_cl, cl)
    return np.array(pnls, dtype=np.float64), max_cl, max_dd

# Run
t0 = time.time()
print("Loading EURUSD ticks...")
ticks = load("EURUSD", [(2025,10),(2025,11),(2025,12)])
print(f"  {len(ticks):,} ticks ({time.time()-t0:.1f}s)")

print("\nDetecting 5p/20s events...")
ev_list = detect_all(ticks, 0.0001)
print(f"  {len(ev_list)} events")

COST_EXNESS = 0.00003
COST_FUNDEDNEXT = 0.00011  # 0.8 pip spread + $3 comm at 1.0 lot
PIP_VAL = 10.0
DAYS = 65

print("\n=== EURUSD IMPULSE FADE — COST COMPARISON ===")
print(f"{'Scenario':<20s}{'Cost':>8s}{'n':>6s}{'n/d':>5s}{'WR':>6s}{'Avg(p)':>8s}{'Gross(p)':>10s}{'MDD(p)':>8s}{'MDD[$]':>10s}{'CL':>5s}")
print(f"{'─'*86}")

for label, cost_info in [("Exness 0.3p", ("Exness", COST_EXNESS)), ("FundedNext 1.1p", ("FN", COST_FUNDEDNEXT))]:
    for stop_p in [0, 5, 7, 10, 15]:
        cost = cost_info[1]
        pnls, max_cl, max_dd = sim_with_stop(ev_list, ticks, 0.0001, cost, 30, stop_p * 0.0001, "both")
        if len(pnls) == 0: continue
        p = pnls / 0.0001; n = len(p)
        wr = (p > 0).mean() * 100; avg = p.mean(); gross = p.sum()
        dd_usd = max_dd * 10.0 / 0.0001
        print(f"{label} {stop_p:>2d}p    {cost_info[1]*100000:>5.0f}pt {n:>6d} {n/DAYS:>5.1f} {wr:>5.1f}% {avg:>+7.2f}p {gross:>+9.1f}p {max_dd/0.0001:>+7.1f}p ${dd_usd:>+8.0f} {max_cl:>4d}")

# FundedNext: what about using no stop (just hold expiry)?
print("\n\n=== FUNDEDNEXT COST — WHAT LOT SIZE FOR $2K IN 5 DAYS? ===")
for stop_p in [0, 5, 7, 10]:
    pnls, max_cl, max_dd = sim_with_stop(ev_list, ticks, 0.0001, COST_FUNDEDNEXT, 30, stop_p * 0.0001, "both")
    p = pnls / 0.0001; n = len(p)
    avg = p.mean(); gross = p.sum()
    pnl_day = gross / DAYS
    dd_usd = max_dd * 10.0 / 0.0001
    
    mult_needed = 400.0 / pnl_day if pnl_day > 0 else 999
    lot_needed = mult_needed  # backtest at 1.0 lot
    
    worst_1 = pnls.min()
    worst_3_sum = sum(sorted(pnls)[:3])
    
    print(f"{stop_p:>2d}p stop: avg={avg:+.2f}p/trade, {pnl_day:+.0f}p/d, gross={gross:+.0f}p")
    print(f"  Need {lot_needed:.1f}x lot for $400/day")
    if lot_needed < 50:
        lot = max(0.75, min(lot_needed, 10))
        lot_ppd = pnl_day * lot * PIP_VAL
        lot_3w = worst_3_sum * lot * PIP_VAL
        print(f"  At {lot:.1f} lots: ${lot_ppd:.0f}/day, 3 worst losses = ${lot_3w:.0f} ({'SAFE' if abs(lot_3w)<1250 else 'BLOWN' if abs(lot_3w)<2000 else 'FATAL'})")
        # Find lot size that stays within $1,250 daily max (3 worst losses)
        safe_lot = min(1250 / abs(worst_3_sum * PIP_VAL), 10) if worst_3_sum != 0 else 10
        safe_ppd = pnl_day * safe_lot * PIP_VAL
        print(f"  Safe lot (3-loss limit): {safe_lot:.1f} → ${safe_ppd:.0f}/day, {2000/max(safe_ppd,1):.0f} days to $2K")
    else:
        print(f"  Impossible — negative avg PnL")

# Maximum feasible check: what is the absolute max daily PnL at 1.0 lot?
print("\n\n=== FUNDEDNEXT — DAILY PnL SIMULATION (5p stop, 1.0 lot) ===")
pnls, max_cl, max_dd = sim_with_stop(ev_list, ticks, 0.0001, COST_FUNDEDNEXT, 30, 5 * 0.0001, "both")
p = pnls / 0.0001
n = len(p)
print(f"Total: {n} trades in {DAYS} days = {n/DAYS:.1f}/day")
wins = p[p > 0]; losses = p[p <= 0]
print(f"Avg win: {wins.mean():+.2f}p  Avg loss: {losses.mean():+.2f}p")
print(f"Max win: {wins.max():+.2f}p  Max loss: {losses.min():.2f}p")
print(f"Worst 3 trades: {sum(sorted(p)[:3]):.2f}p")
print(f"Max consec loss: {max_cl}")
print(f"Max DD: {max_dd/0.0001:.0f}p = ${max_dd/0.0001*10:.0f}")

# Simulate per-day PnL
daily_pnls = []
curr_day_p = 0.0
curr_day = None
for i, (ts, pnl) in enumerate(zip(ticks['ts_s'].values, [0]*len(ticks))):
    # Skip
    pass

# Instead, group trades by day
from datetime import datetime
tick_ts = ticks['ts_s'].values
trade_days = np.array([datetime.utcfromtimestamp(tick_ts[ei2+1]).day 
                        for _, ei2, _ in ev_list])
# Count unique days
unique_days = np.unique(trade_days)
print(f"\nTrading days: {len(unique_days)}")

# Group by day and sum pnls
from collections import defaultdict
daily = defaultdict(float)
trade_idx = 0
for ws_i, ext_i, ext_dir in ev_list:
    ei2 = ext_i + 1
    day = datetime.utcfromtimestamp(tick_ts[ei2]).date()
    daily[day] += p[trade_idx]
    trade_idx += 1

daily_arr = np.array(list(daily.values()))
print(f"Daily stats at 1.0 lot ($10/pip):")
print(f"  Mean:  ${daily_arr.mean():.0f}")
print(f"  Std:   ${daily_arr.std():.0f}")
print(f"  Min:   ${daily_arr.min():.0f}")
print(f"  Max:   ${daily_arr.max():.0f}")
print(f"  P25:   ${np.percentile(daily_arr,25):.0f}")
print(f"  P50:   ${np.percentile(daily_arr,50):.0f}")
print(f"  P75:   ${np.percentile(daily_arr,75):.0f}")
print(f"  P90:   ${np.percentile(daily_arr,90):.0f}")
print(f"  %positive: {(daily_arr>0).mean()*100:.0f}%")

# Simulate 5-day challenge outcomes at different lot sizes
print("\n\n=== 5-DAY CHALLENGE MONTE CARLO ===")
from random import choices
N_SIM = 10000
lot_sizes = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
for lot in lot_sizes:
    pass_5d = 0
    max_dd_daily = []
    for _ in range(N_SIM):
        # Pick 5 random days
        days = np.random.choice(daily_arr, size=min(5, len(daily_arr)), replace=False)
        net = days.sum() * lot
        max_single_loss = min(0, days.min()) * lot  # worst single day loss
        passing = net >= 2000 and abs(max_single_loss) < 1250
        if passing:
            pass_5d += 1
    print(f"  {lot:.1f} lot: {pass_5d/N_SIM*100:.1f}% pass rate (of 10K sims)")

print(f"\nDone: {time.time()-t0:.1f}s")
