"""Confirm 30s hold performance (FundedNext min=30s)."""
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
            evs.append({"extreme_idx": ei, "direction": d2, "price_extreme": mid[ei]})
            i = max(ei, i + 1)
        else: i += skip
    return pd.DataFrame(evs)

def sim(events, ticks, pip, cost, hold_s, stop_pips=0):
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
        if stop_pips > 0:
            raw_s = stop_pips * pip
            window = bids[ei:win_end] if ed == 1 else asks[ei:win_end]
            hit = np.where(window <= ep - raw_s)[0] if ed == 1 else np.where(window >= ep + raw_s)[0]
            if len(hit) > 0 and ts[ei + hit[0]] <= hold_end:
                xi = ei + hit[0]
            else:
                xi = int(np.searchsorted(ts, hold_end, side="right"))
        else:
            xi = int(np.searchsorted(ts, hold_end, side="right"))
        if xi >= n: continue
        xp = bids[xi] if ed == 1 else asks[xi]
        pnls[ev_i] = (xp - ep) * ed - cost
    return pnls

print("=" * 70, flush=True)
print("30s HOLD CONFIRMATION (FundedNext min=30s)")
print("=" * 70, flush=True)

for pair, pip, cost, configs in [
    ("EURUSD", 0.0001, 0.00003, [(5,30),(7,30),(5,60),(7,60)]),
    ("EURJPY", 0.01, 0.006, [(10,30),(10,60)]),
]:
    t = load(pair, [(2025,10),(2025,11),(2025,12)])
    print(f"\n{pair} (Oct-Dec 2025):", flush=True)
    print(f"  {'Config':>15s} {'Hold':>5s} | {'n':>4s} {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s} {'MaxLoss':>7s}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for imp, hold in configs:
        ev = detect(t, pip, imp, 10)
        p = sim(ev, t, pip, cost, hold)
        p = p[~np.isnan(p)] / pip
        if len(p) == 0: continue
        n = len(p); w = (p > 0).sum(); wr = w/n*100
        cur = 0; max_l = 0
        for v in p:
            if v < 0: cur += 1
            else: cur = 0
            max_l = max(max_l, cur)
        print(f"  {pair} {imp:>2d}p/10s {'hold='+str(hold)+'s':>8s} | {n:>4d} {wr:>5.1f}% {p.sum():>+9.1f}p {p.mean():>+7.2f}p {max_l:>4d}", flush=True)
    
    # Monthly + walk-forward for best configs
    print(f"\n  Monthly breakdown (best configs):", flush=True)
    for imp, hold in [(7,30),(7,60)] if pair == "EURUSD" else [(10,30),(10,60)]:
        print(f"    {pair} {imp}p/10s hold={hold}s:", flush=True)
        t_oct = load(pair, [(2025,10)])
        t_nov = load(pair, [(2025,11)])
        t_dec = load(pair, [(2025,12)])
        for mlab, tm in [("Oct",t_oct),("Nov",t_nov),("Dec",t_dec)]:
            ev_m = detect(tm, pip, imp, 10)
            p_m = sim(ev_m, tm, pip, cost, hold)
            p_m = p_m[~np.isnan(p_m)] / pip
            if len(p_m) == 0: continue
            n = len(p_m); w = (p_m > 0).sum()
            print(f"      {mlab}: n={n:>3d} WR={w/n*100:.1f}% Gross={p_m.sum():+7.1f}p Avg={p_m.mean():+7.2f}p", flush=True)
