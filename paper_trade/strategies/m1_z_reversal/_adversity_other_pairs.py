"""Fast adversity tests for EURJPY and GBPJPY.
Optimizations: stride-based impulse detection, cached months, no redundant loads."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
import pandas as pd
import numpy as np
from pathlib import Path

D = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def load_month(pair, m):
    y, mm = m
    f = D / f"{pair}_Raw_Spread_{y}_{mm:02d}.zip"
    d = pd.read_csv(f, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                    dtype={"Ts": str, "B": np.float64, "A": np.float64})
    d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    d = d.dropna(subset=["Ts"])
    d["ts_s"] = d["Ts"].astype(np.int64) // 10**9
    return d


def fi_fast(t, ip=5, is_=10):
    """Two-pointer with stride. end advances by 1 for correctness, i strides."""
    m = (t["B"].values + t["A"].values) / 2
    ts = t["ts_s"].values; n = len(t); stride = 5
    events = []; i = 0; end = 0
    while i < n:
        if i >= end: end = i + 1
        while end < n and ts[end] <= ts[i] + is_:
            end += 1
        if end - i >= 2:
            w = m[i:end]
            hp = (np.max(w) - m[i]) * 10000
            lp = abs((np.min(w) - m[i]) * 10000)
            if max(hp, lp) >= ip:
                d = 1 if hp >= lp else -1
                ei = i + (np.argmax(w) if d == 1 else np.argmin(w))
                events.append({"d": d, "ei": ei})
                i = ei
                continue
        i += stride
    return pd.DataFrame(events)


def sf(ev, t, h=15, fl=False):
    b = t["B"].values; a = t["A"].values; ts = t["ts_s"].values; n = len(t)
    tr = []
    for _, e in ev.iterrows():
        ed = e["d"] if fl else -e["d"]
        ei = int(e["ei"]) + 1
        if ei >= n: continue
        ep = a[ei] if ed == 1 else b[ei]
        xi = int(np.searchsorted(ts, ts[ei] + h, side="right"))
        if xi >= n: continue
        xp = b[xi] if ed == 1 else a[xi]
        tr.append({"pnl": (xp - ep) * 10000 * ed})
    return pd.DataFrame(tr) if tr else pd.DataFrame(columns=["pnl"])


cache = {}
def get_events(t, k, ip, is_):
    key = (k, ip, is_)
    if key not in cache:
        cache[key] = fi_fast(t, ip, is_)
    return cache[key]


configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]
pairs = ["EURJPY", "GBPJPY"]
months = [(2025, 10), (2025, 11), (2025, 12)]

for pair in pairs:
    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# {pair}")
    print(f"{'#'*60}")

    # Load each month ONCE, concatenate for all/train
    month_data = {}
    for m in months:
        month_data[m] = load_month(pair, m)
        print(f"  Loaded {m[1]}mo: {len(month_data[m]):,} ticks")
    
    all_t = pd.concat([month_data[m] for m in months], ignore_index=True).sort_values("ts_s").reset_index(drop=True)
    train_t = pd.concat([month_data[(2025,10)], month_data[(2025,11)]], ignore_index=True).sort_values("ts_s").reset_index(drop=True)
    print(f"  Total: {len(all_t):,} ticks")

    # 1: Monthly breakdown
    print("--- 1: BREAKDOWN ---")
    for m in months:
        t = month_data[m]
        for ip, is_, h in configs:
            ev = get_events(t, f"{pair}_{m}", ip, is_)
            tr = sf(ev, t, h)
            if len(tr) < 5: continue
            wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
            f = " ✓" if wr >= 0.60 else (" !! FAIL" if wr < 0.50 else "")
            print(f"  {m[1]:>2d}mo {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p{f}")

    # 2: Direction reversal
    print("--- 2: DIRECTION ---")
    for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
        ev = get_events(all_t, f"{pair}_All", ip, is_)
        tf = sf(ev, all_t, h); tm = sf(ev, all_t, h, fl=True)
        wf = (tf["pnl"] > 0).mean() if len(tf) > 0 else 0
        wm = (tm["pnl"] > 0).mean() if len(tm) > 0 else 0
        gap = wf - wm
        v = "✓ directional" if gap > 0.15 else ("??" if gap > 0.05 else "!! NO")
        print(f"  {ip}p/{is_}s: Fade WR={wf:.1%} Mom WR={wm:.1%} gap={gap:+.0%} {v}")

    # 3: Walk-forward Oct+Nov -> Dec
    print("--- 3: WALK-FORWARD ---")
    best_wr, best_cfg = 0, None
    for ip, is_, h in configs:
        ev = get_events(train_t, f"{pair}_Train", ip, is_)
        tr = sf(ev, train_t, h)
        if len(tr) < 20: continue
        wr = (tr["pnl"] > 0).mean()
        if wr > best_wr: best_wr, best_cfg = wr, (ip, is_, h)
        print(f"  Train {ip}p/{is_}s h{h}s: n={len(tr)} WR={wr:.1%} avg={tr['pnl'].mean():+.2f}p")
    if best_cfg:
        ip, is_, h = best_cfg
        print(f"  Best: {ip}p/{is_}s WR={best_wr:.1%}")
        t_dec = month_data[(2025, 12)]
        ev = get_events(t_dec, f"{pair}_Dec", ip, is_)
        tr = sf(ev, t_dec, h)
        if len(tr) > 0:
            wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
            v = "✓ HOLDS" if wr > 0.55 else ("!! FALLS" if wr < 0.50 else "? marginal")
            print(f"  Test Dec: n={len(tr)} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p {v}")

    # 4: Full period
    print("--- 4: FULL PERIOD ---")
    for ip, is_, h in configs:
        ev = get_events(all_t, pair, ip, is_)
        tr = sf(ev, all_t, h)
        if len(tr) < 10: continue
        wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
        print(f"  {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p")

    print(f"  ({time.time()-t0:.0f}s)")
