"""Debug: trace first 5 trades step by step."""
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

print("Loading...")
raw = load("EURUSD", [(2025,10),(2025,11),(2025,12)])
print(f"  {len(raw):,} ticks")

print("Detecting...")
ev = detect_all(raw, 0.0001)
print(f"  {len(ev)} events")

bid = raw["B"].values.astype(np.float64)
ask = raw["A"].values.astype(np.float64)
ts = raw["ts_s"].values.astype(np.int64)
mid = (bid + ask) / 2.0

# Trace first 5 events
print("\nFirst 5 events with original sim:")
for ev_i in range(5):
    ws_i, ext_i, ext_dir = ev[ev_i]
    ed = -ext_dir
    ei2 = ext_i + 1
    ep_ask = ask[ei2]; ep_bid = bid[ei2]
    et = ts[ei2]
    he = int(np.searchsorted(ts, et + 30, side="right"))
    xp_bid = bid[he]; xp_ask = ask[he]
    ep = ask[ei2] if ed == 1 else bid[ei2]
    xp = bid[he] if ed == 1 else ask[he]
    pnl_orig = (xp - ep) * ed - 0.00003
    print(f"  #{ev_i}: ext_i={ext_i} dir={ext_dir}→ed={ed} et={et}, ei2={ei2} he={he} ep={ep:.5f} xp={xp:.5f} pnl={pnl_orig/0.0001:+.2f}p")
    print(f"         ts_range: entry_ts={et} exit_ts={ts[he]} delta={ts[he]-et}s ticks_between={he-ei2}")

    # Now trace my version
    exit_ts = ts[ei2] + 30
    stop_hit = False; exit_idx = None
    for j in range(ei2 + 1, len(ts)):
        if ts[j] >= exit_ts:
            exit_idx = j; break
        if ed == 1:
            if bid[j] <= 0: stop_hit = True; exit_idx = j; break  # never
    if exit_idx is not None:
        xp_mine = bid[exit_idx] if ed == 1 else ask[exit_idx]
        pnl_mine = (xp_mine - ep) * ed - 0.00003
        print(f"         MINE: exit_idx={exit_idx} exit_ts={ts[exit_idx]} delta={ts[exit_idx]-et}s pnl={pnl_mine/0.0001:+.2f}p")
