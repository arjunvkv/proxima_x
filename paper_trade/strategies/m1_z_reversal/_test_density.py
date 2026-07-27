"""Test event density vs WR tradeoff for EURUSD across thresholds/windows."""
import sys, numpy as np, pandas as pd
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

def detect(ticks, pip_size, thresh_pips=5, thresh_sec=10):
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size; evs = []; i = 0
    skip = max(1, int(n / max(ts[-1]-ts[0], 1)) * 3)
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        if end - i < 2: i += skip; continue
        w = mid[i:end]
        hp = np.max(w) - w[0]; lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d2 == 1 else np.argmin(w))
            evs.append({"extreme_idx": ei, "direction": d2, "price_extreme": mid[ei], "Ts": ticks["Ts"].iloc[i]})
            i = max(ei, i + 1)
        else: i += skip
    return pd.DataFrame(evs)

def sim(events, ticks, pip, cost, hold_s):
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ts)
    pnls = np.full(len(events), np.nan)
    for ev_i, (_, ev) in enumerate(events.iterrows()):
        ed = -ev["direction"]
        ei = int(ev["extreme_idx"]) + 1
        if ei >= n - 1: continue
        ep = asks[ei] if ed == 1 else bids[ei]; et = ts[ei]
        hold_end = et + hold_s
        win_end = min(int(np.searchsorted(ts, hold_end, side="right")), n)
        if win_end <= ei + 1: continue
        xi = int(np.searchsorted(ts, hold_end, side="right"))
        if xi >= n: continue
        xp = bids[xi] if ed == 1 else asks[xi]
        pnls[ev_i] = (xp - ep) * ed - cost
    return pnls

t = load("EURUSD", [(2025,10),(2025,11),(2025,12)])
pip = 0.0001; cost = 0.00003

print("=" * 80, flush=True)
print("EVENT DENSITY vs WR — EURUSD Oct-Dec 2025")
print("=" * 80, flush=True)
print(f" {'Config':>18s} | {'n(3mo)':>7s} {'Days':>4s} {'n/Day':>5s} | {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s}", flush=True)
print(f" {'-'*18} | {'-'*7} {'-'*4} {'-'*5} | {'-'*5} {'-'*9} {'-'*7}", flush=True)

configs = [
    # (threshold_pips, window_sec, hold_sec, retrace_pips)
    (3, 10, 30, 0),  (4, 10, 30, 0),  (5, 10, 30, 0),  (6, 10, 30, 0),  (7, 10, 30, 0),
    (5, 15, 30, 0),  (5, 20, 30, 0),  (6, 15, 30, 0),  (7, 15, 30, 0),
    (5, 5, 30, 0),   (6, 5, 30, 0),
    (5, 10, 60, 0),  (6, 10, 60, 0),  (7, 10, 60, 0),
    # With retrace gate
    (5, 10, 30, 0.1), (5, 10, 30, 0.2),
]

for tp, ws, hold, ret in configs:
    ev = detect(t, pip, tp, ws)
    
    # Apply retrace filter
    if ret > 0 and len(ev) > 0:
        bids = t["B"].values; asks = t["A"].values; ts = t["ts_s"].values
        raw_r = ret * pip; n_ticks = len(ts)
        valid = np.ones(len(ev), dtype=bool)
        for j, (_, e) in enumerate(ev.iterrows()):
            ed = -e["direction"]; ei = int(e["extreme_idx"])
            found = False
            for k in range(ei, min(ei + 500, n_ticks)):
                px = asks[k] if ed == 1 else bids[k]
                if (ed == 1 and px <= e["price_extreme"] - raw_r) or \
                   (ed == -1 and px >= e["price_extreme"] + raw_r):
                    found = True; break
            valid[j] = found
        ev = ev[valid].reset_index(drop=True)
    
    p = sim(ev, t, pip, cost, hold)
    p = p[~np.isnan(p)] / pip
    if len(p) < 15: continue
    
    n = len(p); wr = (p > 0).mean() * 100
    tdays = ev["Ts"].dt.date.nunique() if "Ts" in ev.columns and len(ev) > 0 else 0
    label = f"{tp}p/{ws}s h={hold}s"
    if ret > 0: label += f" r={ret}"
    print(f" {label:>18s} | {n:>7d} {tdays:>4d} {n/65:>5.1f} | {wr:>5.1f}% {p.sum():>+9.1f}p {p.mean():>+7.2f}p", flush=True)

# ── Multi-config overlap analysis ──
print(f"\n{'='*80}", flush=True)
print("OVERLAP ANALYSIS — running multiple configs on same pair")
print(f"{'='*80}", flush=True)

from itertools import combinations

config_pairs = [(5,10), (6,10), (7,10), (5,15), (6,15)]
all_ev = {}
for tp, ws in config_pairs:
    ev = detect(t, pip, tp, ws)
    all_ev[(tp, ws)] = set(ev["extreme_idx"].values)

print(f"\n {'Config A':>10s} {'Config B':>10s} | {'A∩B':>6s} {'A∪B':>6s} {'Overlap%':>8s} {'A only':>6s} {'B only':>6s}", flush=True)
for a, b in combinations(config_pairs, 2):
    sa = all_ev[a]; sb = all_ev[b]
    inter = len(sa & sb); union = len(sa | sb)
    a_only = len(sa - sb); b_only = len(sb - sa)
    pct = inter / union * 100 if union > 0 else 0
    print(f" {f'{a[0]}p/{a[1]}s':>10s} {f'{b[0]}p/{b[1]}s':>10s} | {inter:>6d} {union:>6d} {pct:>7.1f}% {a_only:>6d} {b_only:>6d}", flush=True)

# ── Combined: 5p/10s + 7p/10s events (no overlap) ──
print(f"\n{'='*80}", flush=True)
print("COMBINED CONFIGS — non-overlapping events")
print(f"{'='*80}", flush=True)

for configs_combo in [
    [(5, 10, 30), (7, 10, 30)],
    [(5, 10, 30), (6, 10, 30)],
    [(6, 10, 30), (7, 10, 30)],
    [(5, 10, 30), (6, 10, 30), (7, 10, 30)],
]:
    all_ei = set()
    all_pnls = []
    label_parts = []
    for tp, ws, hold in configs_combo:
        ev = detect(t, pip, tp, ws)
        # Remove overlapping events (already in a higher config)
        new_ei = set(ev["extreme_idx"].values) - all_ei
        ev_filtered = ev[ev["extreme_idx"].isin(new_ei)].reset_index(drop=True)
        if len(ev_filtered) == 0: continue
        p = sim(ev_filtered, t, pip, cost, hold)
        p = p[~np.isnan(p)] / pip
        all_pnls.append(p)
        all_ei |= new_ei
        label_parts.append(f"{tp}p/{ws}s")
    
    all_p = np.concatenate(all_pnls) if len(all_pnls) > 0 else np.array([])
    if len(all_p) < 15: continue
    n = len(all_p); wr = (all_p > 0).mean() * 100
    label = "+".join(label_parts)
    max_l = 0; cur = 0
    for v in all_p:
        if v < 0: cur += 1
        else: cur = 0
        max_l = max(max_l, cur)
    print(f" {label:>30s}: n={n:>4d} WR={wr:>5.1f}% Gross={all_p.sum():+7.1f}p Avg={all_p.mean():+7.2f}p MaxL={max_l:>2d} ({n/65:.1f}/day)", flush=True)
