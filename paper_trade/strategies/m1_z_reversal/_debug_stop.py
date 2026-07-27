"""Debug: why stops show -0.30p loss instead of -5.30p?"""
import sys, numpy as np, pandas as pd
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

print("Loading...")
raw = load("EURUSD", [(2025,10),(2025,11),(2025,12)])

print("Detecting...")
ev = detect_all(raw, 0.0001)
print(f"{len(ev)} events")

bid = raw["B"].values.astype(np.float64)
ask = raw["A"].values.astype(np.float64)
ts = raw["ts_s"].values.astype(np.int64)
n = len(ts); pip = 0.0001; cost = 0.00003; hold_s = 30

# Trace first 20 trades with 5p stop
stop_fired = 0; hold_exit = 0; stop_not_fired = 0
overall_pnls = []

for ev_i in range(len(ev)):
    ws_i, ext_i, ext_dir = ev[ev_i]
    ed = -ext_dir
    ei2 = ext_i + 1
    if ei2 >= n - 1: continue
    ep = ask[ei2] if ed == 1 else bid[ei2]
    et = ts[ei2]
    he = int(np.searchsorted(ts, et + hold_s, side="right"))
    if he >= n: continue
    
    # 5p stop
    stop_pips = 5 * pip
    stop_price = ep - stop_pips if ed == 1 else ep + stop_pips
    fired = False; exit_idx = he; exit_at_stop = False
    
    for j in range(ei2 + 1, he):
        if ed == 1:  # long
            if bid[j] <= stop_price:
                fired = True; exit_idx = j; exit_at_stop = True; break
        else:  # short
            if ask[j] >= stop_price:
                fired = True; exit_idx = j; exit_at_stop = True; break
    
    xp = stop_price if exit_at_stop else (bid[he] if ed == 1 else ask[he])
    pnl = (xp - ep) * ed - cost
    overall_pnls.append(pnl)
    
    if exit_at_stop:
        stop_fired += 1
    else:
        hold_exit += 1
    
    if ev_i < 5:
        pnl_pips = pnl / pip
        print(f"  #{ev_i}: dir={ext_dir}→ed={ed} entry_ts={et} stop_fired={exit_at_stop}")
        print(f"         ep={ep:.5f} stop_price={stop_price:.5f} bid[ei2]={bid[ei2]:.5f} ask[ei2]={ask[ei2]:.5f}")
        print(f"         xp={xp:.5f} pnl={pnl_pips:+.2f}p")

overall = np.array(overall_pnls) / pip
print(f"\nSummary: stop_fired={stop_fired} hold_exit={hold_exit}")
print(f"WR={(overall>0).mean()*100:.1f}% avg={overall.mean():+.2f}p")
