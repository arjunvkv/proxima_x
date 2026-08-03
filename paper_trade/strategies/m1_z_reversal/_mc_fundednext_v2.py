"""Monte Carlo v2: trade-level resampling on FundedNext ticks (EURUSD).

Samples individual trades (with their actual PnL) to build simulated days,
maintaining the observed trade/day rate. This gives better stats than
resampling from only 20 daily totals.
"""
import sys, time, numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")

PIP = 0.0001
PIP_USD = 10.0
SPREAD_PRICE = 0.8 * PIP
COMM_PRICE = 3.0 / PIP_USD * PIP
COST = SPREAD_PRICE + COMM_PRICE

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

def detect(ts, mid, window_s=20, detect_pips=5):
    n = len(mid)
    min_q = deque()
    max_q = deque()
    ws_idx = 0
    evs = []
    thresh = detect_pips * PIP
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v:
            min_q.pop()
        while max_q and max_q[-1][0] <= v:
            max_q.pop()
        min_q.append((v, i))
        max_q.append((v, i))
        while ts[i] - ts[ws_idx] > window_s:
            if min_q and min_q[0][1] == ws_idx:
                min_q.popleft()
            if max_q and max_q[0][1] == ws_idx:
                max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = mid[ws_idx]
            hp = float(max_q[0][0] - wp)
            lp = float(wp - min_q[0][0])
            span = ts[i] - ts[ws_idx]
            if span <= window_s and (hp >= thresh or lp >= thresh):
                if evs and evs[-1][0] >= ws_idx:
                    continue
                ext_idx = max_q[0][1] if hp >= lp else min_q[0][1]
                d = 1 if hp >= lp else -1
                evs.append((ws_idx, ext_idx, d))
    return evs

def sim_trades(ev_list, ts, bid, ask, hold_s=30, stop_pips_abs=5*PIP):
    n = len(ts)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= n - 1:
            continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n:
            continue
        stop_price = ep - stop_pips_abs if ed == 1 else ep + stop_pips_abs
        stop_hit = False
        if stop_pips_abs > 0:
            for j in range(ei2 + 1, he):
                if ed == 1 and bid[j] <= stop_price:
                    stop_hit = True
                    break
                if ed == -1 and ask[j] >= stop_price:
                    stop_hit = True
                    break
        xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - COST
        pnls.append(pnl / PIP)  # in pips
    return np.array(pnls, dtype=np.float64)

t0 = time.time()
print("Loading EURUSD...")
ts, bid, ask = load("EURUSD")
mid = (bid + ask) / 2.0
print(f"  {len(ts):,} ticks")

print("Detecting 5p/20s...")
evs = detect(ts, mid)
print(f"  {len(evs)} events")

for stop_pips_raw in [0, 5, 7, 10]:
    stop_pips = stop_pips_raw * PIP
    pnls_pips = sim_trades(evs, ts, bid, ask, 30, stop_pips)
    n = len(pnls_pips)

    print(f"\n=== EURUSD {stop_pips_raw}p stop ({n} trades) ===")
    wr = (pnls_pips > 0).mean() * 100
    avg = pnls_pips.mean()
    gross = pnls_pips.sum()
    std = pnls_pips.std()
    print(f"WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.0f}p std={std:.2f}p")

    wins = pnls_pips[pnls_pips > 0]
    losses = pnls_pips[pnls_pips <= 0]
    print(f"Avg win={wins.mean():+.2f}p  Avg loss={losses.mean():+.2f}p")
    print(f"Worst 1={losses.min():.2f}p  Worst 3={sum(sorted(pnls_pips)[:3]):.2f}p")

    # Monte Carlo: 100K simulations, sample trades to fill 5 days
    TRADES_PER_DAY = int(n / 20)  # 20 trading days in our data
    N_SIM = 100_000
    rng = np.random.default_rng()

    for lot in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        pass_5d = 0
        blow_days = 0
        for _ in range(N_SIM):
            trades = rng.choice(pnls_pips, size=TRADES_PER_DAY * 5, replace=True)
            day_pnls = trades.reshape(5, TRADES_PER_DAY).sum(axis=1) * lot * PIP_USD
            if day_pnls.sum() >= 2000 and day_pnls.min() > -1250:
                pass_5d += 1
            if day_pnls.min() <= -1250:
                blow_days += 1
        pass_pct = pass_5d / N_SIM * 100
        blow_pct = blow_days / N_SIM * 100
        print(f"  {lot:.1f} lot: {pass_pct:.1f}% pass  {blow_pct:.1f}% blow-day")

print(f"\nTotal: {time.time()-t0:.1f}s")
