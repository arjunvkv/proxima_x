"""Monte Carlo with hour filter (14-19 UTC) on FundedNext ticks.

Filtering out the toxic hours (9-12 UTC) dramatically improves results:
- WR: 58.0% → 61.3%
- Avg: +0.74p → +1.07p
"""
import sys, time, numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")
PIP = 0.0001
PIP_USD = 10.0
COST = 0.8 * PIP + 3.0 / PIP_USD * PIP

GOOD_HOURS = set(range(14, 20))  # 14:00-19:59 UTC = London+NY overlap

print(f"Good trading hours: {sorted(GOOD_HOURS)} UTC")

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

t0 = time.time()
print("Loading EURUSD...")
ts, bid, ask = load("EURUSD")
mid = (bid + ask) / 2.0

print("Detecting 5p/20s with hour filter...")
min_q, max_q = deque(), deque()
ws_idx = 0
evs = []

for i in range(len(mid)):
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
        if span <= 20 and (hp >= 5*PIP or lp >= 5*PIP):
            if evs and evs[-1][0] >= ws_idx: continue
            ext_idx = max_q[0][1] if hp >= lp else min_q[0][1]
            entry_t = ts[ext_idx + 1]
            hr = datetime.fromtimestamp(entry_t, tz=timezone.utc).hour
            if hr not in GOOD_HOURS: continue
            d_dir = 1 if hp >= lp else -1
            evs.append((ws_idx, ext_idx, d_dir, entry_t))

print(f"  {len(evs)} events (filtered)")

# Simulate all trades
pnls_pips = []
for ws_i, ext_i, ext_dir, entry_t in evs:
    ed = -ext_dir
    ei2 = ext_i + 1
    if ei2 >= len(bid) - 1: continue
    ep = ask[ei2] if ed == 1 else bid[ei2]; et = ts[ei2]
    he = int(np.searchsorted(ts, et + 30, side="right"))
    if he >= len(bid): continue
    sp = ep - 10*PIP if ed == 1 else ep + 10*PIP
    hit = False
    for j in range(ei2 + 1, he):
        if ed == 1 and bid[j] <= sp: hit = True; break
        if ed == -1 and ask[j] >= sp: hit = True; break
    xp = sp if hit else (bid[he] if ed == 1 else ask[he])
    pnl = (xp - ep) * ed - COST
    pnls_pips.append(pnl / PIP)

pnls_pips = np.array(pnls_pips)
n = len(pnls_pips)
wr = (pnls_pips > 0).mean() * 100
avg = pnls_pips.mean()
std = pnls_pips.std()
gross = pnls_pips.sum()
print(f"\nResults (hour-filtered, 10p stop):")
print(f"  {n}t  WR={wr:.1f}%  avg={avg:+.2f}p  std={std:.2f}p  gross={gross:+.0f}p")
wins = pnls_pips[pnls_pips > 0]
losses = pnls_pips[pnls_pips <= 0]
print(f"  Avg win={wins.mean():+.2f}p  Avg loss={losses.mean():+.2f}p")
print(f"  Worst 1={losses.min():.2f}p  Worst 3={sum(sorted(pnls_pips)[:3]):.2f}p")

# Weekly breakdown
weeks = defaultdict(list)
for (ws_i, ext_i, ext_dir, entry_t), pnl in zip(evs, pnls_pips):
    d = datetime.fromtimestamp(entry_t, tz=timezone.utc).date()
    iso = d.isocalendar(); wk = f"{iso[0]}-W{iso[1]:02d}"
    weeks[wk].append(pnl)

print(f"\nWeekly breakdown:")
for wk in sorted(weeks.keys()):
    arr = np.array(weeks[wk])
    dr = f"{datetime.fromtimestamp(evs[sum(len(weeks[w]) for w in sorted(weeks.keys()) if w < wk)][3] if sum(len(weeks[w]) for w in sorted(weeks.keys()) if w < wk) < len(evs) else 0, tz=timezone.utc).strftime('%m/%d')}"
    print(f"  {wk}: {len(arr):>3d}t  WR={(arr>0).mean()*100:.1f}%  avg={arr.mean():+.2f}p  gross={arr.sum():+.0f}p")

# Monte Carlo: 100K iterations
TRADES_PER_DAY = int(n / 20)  # ~38/day with filter
N_SIM = 100_000
rng = np.random.default_rng(42)

print(f"\n{'='*60}")
print(f"MONTE CARLO ({N_SIM:,} sims, ~{TRADES_PER_DAY} trades/day)")
print(f"{'='*60}")
for lot in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    pass_5d = 0
    blow_day = 0
    for _ in range(N_SIM):
        trades = rng.choice(pnls_pips, size=TRADES_PER_DAY * 5, replace=True)
        day_pnls = trades.reshape(5, TRADES_PER_DAY).sum(axis=1) * lot * PIP_USD
        if day_pnls.sum() >= 2000 and day_pnls.min() > -1250:
            pass_5d += 1
        if day_pnls.min() <= -1250:
            blow_day += 1
    print(f"  {lot:.1f} lot: {pass_5d/N_SIM*100:.1f}% pass  {blow_day/N_SIM*100:.1f}% blow-day")

# Days to $2K analysis
print(f"\n{'='*60}")
print(f"DAYS TO $2K (at 2.5 lots)")
print(f"{'='*60}")
for lot in [1.5, 2.0, 2.5, 3.0]:
    dtt = []
    fail = 0
    for _ in range(50000):
        cum = 0.0
        for day in range(1, 31):
            day_trades = rng.choice(pnls_pips, size=TRADES_PER_DAY, replace=True)
            day_pnl = day_trades.sum() * lot * PIP_USD
            cum += day_pnl
            if cum >= 2000:
                dtt.append(day)
                break
        else:
            fail += 1
    dtt_arr = np.array(dtt)
    hit = (50000 - fail) / 50000 * 100
    hit5 = (dtt_arr <= 5).mean() * 100 if len(dtt_arr) > 0 else 0
    med = np.median(dtt_arr) if len(dtt_arr) > 0 else 999
    print(f"  {lot:.1f} lots: {hit:.0f}% hit $2K  median={med:.0f}d  hit5d={hit5:.0f}%")

print(f"\nTotal: {time.time()-t0:.1f}s")
