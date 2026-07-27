"""Fast validation of best FundedNext configs — optimized detect."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_META = {
    "EURUSD": {"pip": 0.0001, "cost": 0.00003},
    "EURJPY": {"pip": 0.01, "cost": 0.006},
}

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

def detect_fast(ticks, pip_size, thresh_pips=5, thresh_sec=10):
    """Optimized detection: advance by ~1 second skip when no event found."""
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size; evs = []; i = 0
    
    # Compute skip step: number of ticks in ~1 second
    total_sec = ts[-1] - ts[0] if n > 1 else 1
    tick_rate = n / max(total_sec, 1)
    skip = max(1, int(tick_rate * 5))  # ~5 seconds of ticks (10x faster)
    
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        if end - i < 2: i += skip; continue
        w = mid[i:end]
        hp = np.max(w) - w[0]; lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d2 == 1 else np.argmin(w))
            evs.append({"time": ts[i], "extreme_idx": ei, "direction": d2, "price_extreme": mid[ei]})
            i = max(ei, i + 1)  # skip to extreme
        else:
            i += skip
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

def analyze(pair, imp, hold, stop, months_data, pip, cost):
    evs = {}
    for mlab in ["Oct","Nov","Dec"]:
        evs[mlab] = detect_fast(months_data[mlab], pip, imp, 10)
    
    ps = {}
    for mlab in ["Oct","Nov","Dec"]:
        p = sim(evs[mlab], months_data[mlab], pip, cost, hold, stop)
        ps[mlab] = p[~np.isnan(p)] / pip
    
    t_train = pd.concat([months_data["Oct"], months_data["Nov"]], ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t_train["ts_s"] = t_train["Ts"].astype(np.int64) // 10**9
    ev_train = detect_fast(t_train, pip, imp, 10)
    p_train = sim(ev_train, t_train, pip, cost, hold, stop)
    ps["Train"] = p_train[~np.isnan(p_train)] / pip
    ps["Test"] = ps["Dec"]
    
    print(f"\n  {pair} {imp}p/10s hold={hold}s stop={stop}p:")
    for mlab in ["Oct","Nov","Dec"]:
        p = ps[mlab]
        if len(p) == 0: continue
        n = len(p); wr = (p > 0).mean() * 100
        print(f"      {mlab}: n={n:>3d} WR={wr:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p")
    for mlab in ["Train","Test"]:
        p = ps[mlab]
        if len(p) == 0: continue
        n = len(p); wr = (p > 0).mean() * 100
        print(f"      {mlab}: n={n:>3d} WR={wr:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p")
    
    all_p = np.concatenate([ps[m] for m in ["Oct","Nov","Dec"] if len(ps[m]) > 0])
    if len(all_p) > 0:
        cur = 0; cur_dd = 0; max_l = 0; max_dd = 0
        for v in all_p:
            if v < 0: cur += 1; cur_dd += abs(v)
            else: cur = 0; cur_dd = 0
            max_l = max(max_l, cur); max_dd = max(max_dd, cur_dd)
        print(f"      Worst streak: {max_l} losses ({max_dd:.1f}p)")


print("=" * 80, flush=True)
print("BEST PLAN — FundedNext Validation (optimized detect)", flush=True)
print("=" * 80, flush=True)

loaded = {}
for pair in ["EURUSD", "EURJPY"]:
    loaded[pair] = {}
    for mlab, months in [("Oct",[(2025,10)]),("Nov",[(2025,11)]),("Dec",[(2025,12)])]:
        print(f"  Loading {pair} {mlab}...", flush=True)
        loaded[pair][mlab] = load(pair, months)
        print(f"    {len(loaded[pair][mlab]):,} ticks", flush=True)

# ── EURUSD ──
print(f"\n{'='*80}")
print("EURUSD — best FundedNext configs (hold≥60s, no stop)")
print(f"{'='*80}")
for imp, hold, stop in [(7, 60, 0), (7, 120, 0), (5, 60, 0), (5, 120, 0)]:
    analyze("EURUSD", imp, hold, stop, loaded["EURUSD"], 0.0001, 0.00003)

# ── EURJPY ──
print(f"\n{'='*80}")
print("EURJPY — best FundedNext configs (hold≥60s, no stop)")
print(f"{'='*80}")
for imp, hold, stop in [(10, 180, 0), (10, 120, 0), (10, 60, 0)]:
    analyze("EURJPY", imp, hold, stop, loaded["EURJPY"], 0.01, 0.006)

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")
