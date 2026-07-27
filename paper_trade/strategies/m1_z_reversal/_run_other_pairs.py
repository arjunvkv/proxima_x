"""Fast adversity tests for EURJPY and GBPJPY - run this directly."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
import pandas as pd
import numpy as np
from pathlib import Path

D = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def ld(p, months):
    dfs = []
    for y, mm in months:
        t0 = time.time()
        f = D / f"{p}_Raw_Spread_{y}_{mm:02d}.zip"
        d = pd.read_csv(f, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
        print(f"  {f.name}: {len(d):,} rows in {time.time()-t0:.0f}s", flush=True)
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


def fi(t, ip=5, is_=10):
    m = (t["B"].values + t["A"].values) / 2
    ts = t["ts_s"].values; n = len(t)
    stride = 5; ev = []; i = 0; end = 0
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
                ev.append({"d": d, "ei": ei})
                i = ei
                continue
        i += stride
    return pd.DataFrame(ev)


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


# Preload cache
cache = {}
def ge(t, k, ip, is_):
    key = (k, ip, is_)
    if key not in cache:
        t0 = time.time()
        cache[key] = fi(t, ip, is_)
        print(f"    detect {ip}p/{is_}s: {len(cache[key])} events in {time.time()-t0:.0f}s", flush=True)
    return cache[key]


months = [(2025, 10), (2025, 11), (2025, 12)]
configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]

for pair in ["EURJPY", "GBPJPY"]:
    print(f"\n# {pair}", flush=True)
    print(f"Loading {pair}...", flush=True)
    t0 = time.time()
    md = {}
    for m in months:
        md[m] = ld(pair, [m])
    all_t = pd.concat([md[m] for m in months], ignore_index=True).sort_values("ts_s").reset_index(drop=True)
    train_t = pd.concat([md[(2025, 10)], md[(2025, 11)]], ignore_index=True).sort_values("ts_s").reset_index(drop=True)
    print(f"  Total: {len(all_t):,} ticks in {time.time()-t0:.0f}s", flush=True)

    # 1: Monthly breakdown
    print("--- 1: Breakdown ---", flush=True)
    for m in months:
        for ip, is_, h in configs:
            t1 = time.time()
            ev = ge(md[m], f"{pair}_{m[1]}mo", ip, is_)
            tr = sf(ev, md[m], h)
            if len(tr) < 5: continue
            wr = (tr["pnl"] > 0).mean()
            avg = tr["pnl"].mean()
            g = tr["pnl"].sum()
            tag = " ✓" if wr >= 0.60 else (" !! FAIL" if wr < 0.50 else "")
            print(f"  {m[1]:>2d}mo {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p{tag} [{time.time()-t1:.0f}s]", flush=True)

    # 2: Direction reversal
    print("--- 2: Direction ---", flush=True)
    for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
        ev = ge(all_t, f"{pair}_all", ip, is_)
        tf = sf(ev, all_t, h)
        tm = sf(ev, all_t, h, fl=True)
        wf = (tf["pnl"] > 0).mean() if len(tf) > 0 else 0
        wm = (tm["pnl"] > 0).mean() if len(tm) > 0 else 0
        gap = wf - wm
        v = "✓" if gap > 0.15 else ("??" if gap > 0.05 else "!!")
        print(f"  {ip}p/{is_}s: Fade WR={wf:.1%} Mom WR={wm:.1%} gap={gap:+.0%} {v}", flush=True)

    # 3: Walk-forward
    print("--- 3: Walk-forward Oct+Nov -> Dec ---", flush=True)
    best_wr, best_cfg = 0, None
    for ip, is_, h in configs:
        ev = ge(train_t, f"{pair}_train", ip, is_)
        tr = sf(ev, train_t, h)
        if len(tr) < 20: continue
        wr = (tr["pnl"] > 0).mean()
        if wr > best_wr:
            best_wr, best_cfg = wr, (ip, is_, h)
        print(f"  Train {ip}p/{is_}s h{h}s: n={len(tr)} WR={wr:.1%} avg={tr['pnl'].mean():+.2f}p", flush=True)
    if best_cfg:
        ip, is_, h = best_cfg
        print(f"  Best: {ip}p/{is_}s WR={best_wr:.1%}", flush=True)
        ev = ge(md[(2025, 12)], f"{pair}_Dec", ip, is_)
        tr = sf(ev, md[(2025, 12)], h)
        if len(tr) > 0:
            wr = (tr["pnl"] > 0).mean()
            avg = tr["pnl"].mean()
            g = tr["pnl"].sum()
            v = "✓ HOLDS" if wr > 0.55 else ("!! FAILS" if wr < 0.50 else "? marginal")
            print(f"  Test Dec: n={len(tr)} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p {v}", flush=True)

    # 4: Full period
    print("--- 4: Full period ---", flush=True)
    for ip, is_, h in configs:
        t1 = time.time()
        ev = ge(all_t, pair, ip, is_)
        tr = sf(ev, all_t, h)
        if len(tr) < 10: continue
        wr = (tr["pnl"] > 0).mean()
        avg = tr["pnl"].mean()
        g = tr["pnl"].sum()
        print(f"  {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p{g:+.1f}p [{time.time()-t1:.0f}s]", flush=True)

    print(f"  Pair done in {time.time()-t0:.0f}s", flush=True)

print("All done!", flush=True)
