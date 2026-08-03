"""Impulse fade backtest on FundedNext real tick data (Jun 29-Jul 27 2026).

FundedNext ticks have synthetic bid/ask spread (~0.1pt) but real price action.
We use mid-price for detection and add actual spread+commission cost manually.
"""
import sys, time, numpy as np
from collections import deque
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")

PAIRS = {
    "EURUSD": {"pip": 0.0001, "spread_pips": 0.8, "pip_usd": 10.0, "direction": "both"},
    "GBPUSD": {"pip": 0.0001, "spread_pips": 0.8, "pip_usd": 10.0, "direction": "both"},
    "AUDUSD": {"pip": 0.0001, "spread_pips": 0.9, "pip_usd": 10.0, "direction": "both"},
    "NZDUSD": {"pip": 0.0001, "spread_pips": 0.8, "pip_usd": 10.0, "direction": "both"},
    "USDCAD": {"pip": 0.0001, "spread_pips": 0.8, "pip_usd": 7.8, "direction": "both"},
}

CONFIGS = [(3, 10), (4, 15), (5, 20), (6, 15), (7, 20)]
DAYS = 20

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

def detect(ts, mid, pip, detect_pips, window_s):
    n = len(mid)
    min_q = deque()
    max_q = deque()
    ws_idx = 0
    evs = []
    thresh = detect_pips * pip
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

def sim(ev_list, ts, bid, ask, mid, pip, spread_price, comm_price, hold_s, stop_pips, direction="both"):
    n = len(ts)
    pnls = []
    cost = spread_price + comm_price
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        if direction == "short" and ed == 1:
            continue
        if direction == "long" and ed == -1:
            continue
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
        stop_idx = he
        if stop_pips > 0:
            for j in range(ei2 + 1, he):
                if ed == 1:
                    if bid[j] <= stop_price:
                        stop_hit = True
                        stop_idx = j
                        break
                else:
                    if ask[j] >= stop_price:
                        stop_hit = True
                        stop_idx = j
                        break
        xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl)
    return np.array(pnls, dtype=np.float64)

t0 = time.time()

for pair, cfg in PAIRS.items():
    pip_v = cfg["pip"]
    pip_usd = cfg["pip_usd"]
    direction = cfg["direction"]
    spread_price = cfg["spread_pips"] * pip_v
    comm_price = 3.0 / pip_usd * pip_v
    total_cost = spread_price + comm_price
    total_cost_pts = total_cost / pip_v * 10000

    print(f"\n{pair}: cost={total_cost:.6f} ({total_cost_pts:.1f} pts) dir={direction}")

    ts, bid, ask = load(pair)
    mid = (bid + ask) / 2.0
    t_s = np.datetime64(int(ts[0]), "s")
    t_e = np.datetime64(int(ts[-1]), "s")
    print(f"  {len(ts):,} ticks  [{t_s} -> {t_e}]")

    for detect_pips, window_s in CONFIGS:
        evs = detect(ts, mid, pip_v, detect_pips, window_s)
        if len(evs) == 0:
            continue

        for stop_pips_raw in [0, 3, 5, 7, 10]:
            stop_pips = stop_pips_raw * pip_v
            pnls = sim(evs, ts, bid, ask, mid, pip_v, spread_price, comm_price, 30, stop_pips, direction)
            if len(pnls) == 0:
                continue
            p = pnls / pip_v
            wr = (p > 0).mean() * 100
            avg = p.mean()
            gross = p.sum()
            n = len(p)
            cum = np.cumsum(p)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd_pips = dd.max()
            n_day = n / DAYS
            stop_label = "none" if stop_pips_raw == 0 else f"{stop_pips_raw}p"
            print(f"  {detect_pips}p/{window_s}s stop={stop_label:>4s}: {n:>4d}t {n_day:>5.1f}/d WR={wr:>5.1f}% avg={avg:>+6.2f}p gross={gross:>+8.2f}p MDD={max_dd_pips:>+7.1f}p")

print(f"\nTotal: {time.time()-t0:.1f}s")
