"""Ultra-fast density test — single-pass detection for all threshold×window configs."""
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
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect_all(ticks, pip_size):
    """Single-pass detection for all threshold×window configs. Returns dict of (tp,ws) -> events."""
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(mid)

    # All configs to detect
    configs = []
    for tp in [3, 4, 5, 6]:
        for w in [10, 15, 20]:
            configs.append((tp, w, float(tp * pip_size)))
    if pip_size >= 0.01:  # JPY pairs: higher threshold range
        for tp in [7, 10, 15, 20]:
            for w in [10, 15, 20]:
                configs.append((tp, w, float(tp * pip_size)))

    min_q = deque(); max_q = deque()
    ws_idx = 0
    events = {c: [] for c in configs}  # (tp,w,raw) -> [(ws_idx, ext_idx, dir)]

    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))

        while ts[i] - ts[ws_idx] > 20:  # widest window needed
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

def sim(ev_list, ticks, pip, cost, holds):
    if len(ev_list) == 0: return np.array([])
    bid = ticks["B"].values.astype(np.float64); ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    n_ev = len(ev_list); n_h = len(holds)
    pnl = np.full((n_ev, n_h), np.nan)
    for ev_i, (_, ei, ed0) in enumerate(ev_list):
        ed = -ed0
        ei2 = ei + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]; et = ts[ei2]
        for hi, h in enumerate(holds):
            he = int(np.searchsorted(ts, et + h, side="right"))
            if he >= n: continue
            xp = bid[he] if ed == 1 else ask[he]
            pnl[ev_i, hi] = (xp - ep) * ed - cost
    return pnl

def micro_trend(ev_list, ticks, lookback_s=60):
    if len(ev_list) == 0: return np.array([])
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values.astype(np.int64)
    res = np.full(len(ev_list), np.nan)
    for i, (_, ei, ed0) in enumerate(ev_list):
        if ei < 5: continue
        sj = int(np.searchsorted(ts, ts[ei] - lookback_s, side="right"))
        if sj >= ei - 1: continue
        change = mid[ei] - mid[sj]
        fade = -ed0
        res[i] = 1 if change * fade > 0 else (-1 if change * fade < 0 else 0)
    return res

# ════════════════════════════
t0 = time.time()
print("Loading...", flush=True)
raw_data = {}
for pair in ["EURUSD", "EURJPY"]:
    raw_data[pair] = load(pair, [(2025,10),(2025,11),(2025,12)])
    print(f"  {pair}: {len(raw_data[pair]):,} ticks ({time.time()-t0:.1f}s)", flush=True)

print("Detecting (single-pass per pair)...", flush=True)
detect_cache = {}
for pair in ["EURUSD", "EURJPY"]:
    pip = 0.0001 if pair == "EURUSD" else 0.01
    detect_cache[pair] = detect_all(raw_data[pair], pip)
    total_ev = sum(len(v) for v in detect_cache[pair].values())
    print(f"  {pair}: {total_ev} total events across all configs ({time.time()-t0:.1f}s)", flush=True)

holds_list = [30, 60]
lookbacks_list = [30, 60]
costs = {"EURUSD": 0.00003, "EURJPY": 0.006}
pips = {"EURUSD": 0.0001, "EURJPY": 0.01}

print("Simulating...", flush=True)
all_results = []
for pair in ["EURUSD", "EURJPY"]:
    pip = pips[pair]; cost = costs[pair]
    for cfg_name, ev_list in sorted(detect_cache[pair].items()):
        if len(ev_list) < 20: continue
        tp = int(cfg_name.split("p")[0]); ws = int(cfg_name.split("/")[1].replace("s",""))
        pnl = sim(ev_list, raw_data[pair], pip, cost, holds_list)
        trends = {lb: micro_trend(ev_list, raw_data[pair], lb) for lb in lookbacks_list}
        for hi, hold in enumerate(holds_list):
            p = pnl[:, hi][~np.isnan(pnl[:, hi])] / pip
            if len(p) < 20: continue
            n = len(p); wr = (p > 0).mean() * 100
            all_results.append(("all", pair, tp, ws, hold, n, wr, p.sum(), p.mean()))
            for lb in lookbacks_list:
                tr = trends[lb]
                for tv, tn in [(1, "with"), (-1, "against")]:
                    mask = (tr == tv) & ~np.isnan(pnl[:, hi])
                    pf = pnl[:, hi][mask] / pip
                    if len(pf) < 20: continue
                    n2 = len(pf); wr2 = (pf > 0).mean() * 100
                    all_results.append((f"{tn}_{lb}s", pair, tp, ws, hold, n2, wr2, pf.sum(), pf.mean()))

print(f"\n{'='*90}", flush=True)
print("TOP 40 RESULTS (sorted by WR, n≥30)")
print(f"{'='*90}", flush=True)
hdr = f" {'Filter':>15s} {'Pair':>6s} {'Cfg':>10s} {'Hold':>4s} | {'n(3mo)':>7s} {'n/day':>6s} | {'WR':>5s} {'Gross':>9s} {'Avg':>7s}"
print(hdr, flush=True); print(f"{'─'*90}", flush=True)
all_results.sort(key=lambda r: -r[6])
for r in all_results[:40]:
    filt, pair, tp, ws, hold, n, wr, gross, avg = r
    print(f" {filt:>15s} {pair:>6s} {tp}p/{ws}s {hold:>3d}s | {n:>7d} {n/65:>6.1f} | {wr:>5.1f}% {gross:>+9.1f}p {avg:>+7.2f}p", flush=True)

print(f"\n{'─'*90}", flush=True)
print("BEST 65%+ CONFIGS")
print(f"{'─'*90}", flush=True)
for pair in ["EURUSD", "EURJPY"]:
    high = [r for r in all_results if r[1]==pair and r[6]>=65 and r[5]>=30]
    high.sort(key=lambda r: -r[5])
    print(f"\n  {pair}:")
    for r in high[:5]:
        filt, _, tp, ws, hold, n, wr, gross, avg = r
        print(f"    {filt:>15s} {tp}p/{ws}s hold={hold}s: n={n} WR={wr:.1f}% {gross:+.1f}p ({n/65:.1f}/day)", flush=True)

print(f"\nDone: {time.time()-t0:.1f}s", flush=True)
