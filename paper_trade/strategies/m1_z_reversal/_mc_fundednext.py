"""Monte Carlo: challenge pass rate on FundedNext tick data (EURUSD only)."""
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
COST = SPREAD_PRICE + COMM_PRICE  # 0.00011 = 1.1 pips

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

def sim_with_trades(ev_list, ts, bid, ask, mid, hold_s=30, stop_pips=5*PIP):
    n = len(ts)
    trades = []
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
        stop_price = ep - stop_pips if ed == 1 else ep + stop_pips
        stop_hit = False
        if stop_pips > 0:
            for j in range(ei2 + 1, he):
                if ed == 1 and bid[j] <= stop_price:
                    stop_hit = True
                    break
                if ed == -1 and ask[j] >= stop_price:
                    stop_hit = True
                    break
        xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - COST
        trade_time = datetime.fromtimestamp(et, tz=timezone.utc)
        # Simulate entry details for daily grouping
        trades.append({
            "ts": et,
            "day": trade_time.date(),
            "pnl_pips": pnl / PIP,
            "pnl_usd": pnl / PIP * PIP_USD,
        })
    return trades

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
    trades = sim_with_trades(evs, ts, bid, ask, mid, 30, stop_pips)
    pnls_usd = np.array([t["pnl_usd"] for t in trades])
    pnls_pips = pnls_usd / PIP_USD

    print(f"\n=== {stop_pips_raw}p stop ({len(trades)} trades) ===")
    wr = (pnls_usd > 0).mean() * 100
    avg = pnls_usd.mean()
    gross = pnls_usd.sum()
    print(f"WR={wr:.1f}% avg=${avg:.2f} gross=${gross:.0f}")

    # Group by day
    daily = defaultdict(list)
    for t in trades:
        daily[t["day"]].append(t["pnl_usd"])
    daily_arr = np.array([sum(v) for v in daily.values()])
    print(f"Daily: mean=${daily_arr.mean():.0f} std=${daily_arr.std():.0f} "
          f"min=${daily_arr.min():.0f} max=${daily_arr.max():.0f}")
    print(f"  %pos={(daily_arr>0).mean()*100:.0f}%  worst_daily={daily_arr.min():.0f}")

    # Monte Carlo: sample 5 days from available days with replacement
    N_SIM = 100_000
    for lot in [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        pass_5d = 0
        blow_day = 0
        rng = np.random.default_rng()
        for _ in range(N_SIM):
            days = rng.choice(daily_arr, size=5, replace=True)
            net = days.sum() * lot
            max_single_loss = days.min() * lot
            if net >= 2000 and max_single_loss > -1250:
                pass_5d += 1
            if max_single_loss <= -1250:
                blow_day += 1
        pct = pass_5d / N_SIM * 100
        blow_pct = blow_day / N_SIM * 100
        print(f"  {lot:.1f} lot: {pct:.1f}% pass  blow_day={blow_pct:.1f}%")

print(f"\nTotal: {time.time()-t0:.1f}s")
