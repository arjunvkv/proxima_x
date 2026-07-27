"""Fixed: DD tracking corrected, wider stops, plus equity curve trace."""
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
    configs = []
    for tp in [3, 4, 5, 6]:
        for w in [10, 15, 20]:
            configs.append((tp, w, float(tp * pip_size)))
    if pip_size >= 0.01:
        for tp in [7, 10, 15, 20]:
            for w in [10, 15, 20]:
                configs.append((tp, w, float(tp * pip_size)))
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
    return {f"{tp}p/{w}s": events[(tp, w, raw)] for tp, w, raw in configs}

def sim_with_stop(ev_list, ticks, pip, cost, hold_s, stop_pips, direction="both"):
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    pnls = []; cum = 0.0
    peak = -1e9; max_dd = 0.0; max_cl = 0; cl = 0
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

# ═══ Run ═══
t0 = time.time()
print("Loading...", flush=True)
raw = {}
for pair in ["EURUSD", "EURJPY"]:
    raw[pair] = load(pair, [(2025,10),(2025,11),(2025,12)])
    print(f"  {pair}: {len(raw[pair]):,} ticks ({time.time()-t0:.1f}s)", flush=True)

print("Detecting...", flush=True)
dc = {}
for pair in ["EURUSD", "EURJPY"]:
    pip = 0.0001 if pair == "EURUSD" else 0.01
    dc[pair] = detect_all(raw[pair], pip)

# Validate base case
ev = dc["EURUSD"]["5p/20s"]
p, _, _ = sim_with_stop(ev, raw["EURUSD"], 0.0001, 0.00003, 30, 0, "both")
pp = p / 0.0001
assert abs(pp.mean() - 1.68) < 0.05, f"Validation failed: avg={pp.mean():.2f}"
print(f"Validation OK: WR={(pp>0).mean()*100:.1f}% avg={pp.mean():+.2f}p", flush=True)

# Full test
tests = {
    "EURUSD": ("5p/20s", 30, 0.0001, 0.00003, "both", [0, 3, 5, 7, 10, 15, 20]),
    "EURJPY": ("10p/20s", 30, 0.01, 0.006, "short", [0, 5, 7, 10, 15, 20]),
}
days = 65

for pair, (cfg, hold, pip_v, cost, direction, stops) in tests.items():
    print(f"\n{pair} — {cfg} hold={hold}s dir={direction}")
    h = f"{'Stop':>6s} | {'n':>6s} | {'n/d':>5s} | {'WR':>5s} | {'Avg':>7s}"
    h += f" | {'Gross':>8s} | {'MDD(p)':>7s} | {'MDD[$]':>8s} | {'MaxCL':>5s}"
    print(h); print(f"{'─'*68}")
    ev_list = dc[pair][cfg]
    for sp in stops:
        pnls, max_cl, max_dd = sim_with_stop(ev_list, raw[pair], pip_v, cost, hold, sp * pip_v, direction)
        if len(pnls) == 0: continue
        p = pnls / pip_v; n = len(p)
        wr = (p > 0).mean() * 100; avg = p.mean(); gross = p.sum()
        pip_val = 10.0 if pair == "EURUSD" else 9.3
        dd_usd = max_dd * 10.0 / pip_v  # convert pips to $ at 1 lot
        label = "none" if sp == 0 else f"{sp}p"
        print(f" {label:>5s} | {n:>6d} | {n/days:>5.1f} | {wr:>5.1f}% | {avg:>+7.2f}p"
              f" | {gross:>+8.1f}p | {max_dd/pip_v:>+7.1f}p | ${dd_usd:>+8.0f} | {max_cl:>5d}")

    # For EURUSD 5p stop: show win/loss distributions
    if pair == "EURUSD":
        for sp in [0, 5, 15]:
            pnls, _, _ = sim_with_stop(ev_list, raw[pair], pip_v, cost, hold, sp * pip_v, direction)
            p = pnls / pip_v
            wins = p[p > 0]; losses = p[p <= 0]
            print(f"  [{sp}p stop] wins: n={len(wins)} avg={wins.mean():+.2f}p "
                  f"p95={np.percentile(wins,95):+.2f}p" if len(wins) > 0 else "")
            print(f"  [{sp}p stop] losses: n={len(losses)} avg={losses.mean():+.2f}p "
                  f"p05={np.percentile(losses,5):+.2f}p" if len(losses) > 0 else "")

print(f"\nDone: {time.time()-t0:.1f}s", flush=True)
