"""Compare sim vs sim_with_stop (no stop) to find bug."""
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

# Original sim (verbatim from _fast_density.py)
def sim_original(ev_list, ticks, pip, cost, hold):
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64)
    n = len(ts)
    if len(ev_list) == 0: return np.array([])
    pnl = np.full(len(ev_list), np.nan)
    for ev_i, (_, ei, ed0) in enumerate(ev_list):
        ed = -ed0
        ei2 = ei + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold, side="right"))
        if he >= n: continue
        xp = bid[he] if ed == 1 else ask[he]
        pnl[ev_i] = (xp - ep) * ed - cost
    return pnl[~np.isnan(pnl)]

def sim_mine(ev_list, ticks, pip, cost, hold_s, stop_pips):
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        stop_price = ep - stop_pips * pip if ed == 1 else ep + stop_pips * pip
        et_s = ts[ei2]
        exit_ts = et_s + hold_s
        stop_hit = False; exit_idx = None
        for j in range(ei2 + 1, n):
            if ts[j] >= exit_ts:
                exit_idx = j; break
            if ed == 1:
                if bid[j] <= stop_price: stop_hit = True; exit_idx = j; break
            else:
                if ask[j] >= stop_price: stop_hit = True; exit_idx = j; break
        if exit_idx is None: continue
        xp = stop_price if stop_hit else (bid[exit_idx] if ed == 1 else ask[exit_idx])
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl)
    return np.array(pnls, dtype=np.float64)

print("Loading...")
raw = {}
for pair in ["EURUSD"]:
    raw[pair] = load(pair, [(2025,10),(2025,11),(2025,12)])
    print(f"  {pair}: {len(raw[pair]):,} ticks")

print("Detecting EURUSD 5p/20s...")
ev = detect_all(raw["EURUSD"], 0.0001)
print(f"  {len(ev)} events")

print("Simulating...")
pnl_orig = sim_original(ev, raw["EURUSD"], 0.0001, 0.00003, 30)
pnl_mine = sim_mine(ev, raw["EURUSD"], 0.0001, 0.00003, 30, 0)

p_orig = pnl_orig / 0.0001
p_mine = pnl_mine / 0.0001

print(f"\nOriginal: n={len(p_orig)} WR={(p_orig>0).mean()*100:.1f}% "
      f"avg={p_orig.mean():+.2f}p gross={p_orig.sum():+.1f}p")
print(f"Mine:     n={len(p_mine)} WR={(p_mine>0).mean()*100:.1f}% "
      f"avg={p_mine.mean():+.2f}p gross={p_mine.sum():+.1f}p")

# Check differences
match = 0; diff = 0
min_len = min(len(p_orig), len(p_mine))
for i in range(min_len):
    if abs(p_orig[i] - p_mine[i]) < 0.001:
        match += 1
    else:
        diff += 1
        if diff <= 3:
            print(f"  Diff at {i}: orig={p_orig[i]:+.2f}p mine={p_mine[i]:+.2f}p")

print(f"Match: {match}/{min_len}, Diff: {diff}/{min_len}")
