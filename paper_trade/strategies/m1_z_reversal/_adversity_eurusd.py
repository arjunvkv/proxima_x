"""EURUSD-only adversity runner - fast, focused."""
import sys
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import pandas as pd
import numpy as np
from pathlib import Path
import time

D = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def ld(p, m):
    dfs = []
    for y, mm in m:
        f = D / f"{p}_Raw_Spread_{y}_{mm:02d}.zip"
        print(f"    Loading {f.name}...")
        t0 = time.time()
        d = pd.read_csv(f, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
        print(f"      {len(d):,} rows in {time.time()-t0:.1f}s")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


def fi(t, ip=5, is_=10):
    m = (t["B"].values + t["A"].values) / 2
    ts = t["ts_s"].values
    n = len(t); ev = []; i = 0
    while i < n:
        e = min(int(np.searchsorted(ts, ts[i] + is_, side="right")), n)
        w = m[i:e]
        if len(w) < 2: i += 1; continue
        hp = (np.max(w) - w[0]) * 10000
        lp = abs((np.min(w) - w[0]) * 10000)
        if max(hp, lp) >= ip:
            d = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d == 1 else np.argmin(w))
            ev.append({"t": ts[i], "et": ts[ei], "ip": max(hp, lp), "d": d,
                       "ps": w[0], "pe": m[ei], "ei": ei})
            i = ei
        else:
            i += 1
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
        pnl = (xp - ep) * ed
        tr.append({"pnl_pips": pnl * 10000})
    return pd.DataFrame(tr)


def ef(ev, t, rw=15):
    m = (t["B"].values + t["A"].values) / 2
    ts = t["ts_s"].values; n = len(t)
    ff = []
    for _, e in ev.iterrows():
        ei = int(e["ei"]); et = ts[ei]; ep = e["pe"]; sp = e["ps"]
        d = int(e["d"])
        we = min(int(np.searchsorted(ts, et + rw, side="right")), n)
        wm = m[ei:we]; wt = ts[ei:we]
        if len(wm) < 2: ff.append(None); continue
        imp = ep - sp
        if abs(imp) < 1e-10: ff.append({"r5": 0, "r10": 0, "r15": 0}); continue
        def _r(lb):
            idx = np.searchsorted(wt, et + lb, side="right")
            if idx >= len(wm): idx = -1
            c = wm[idx]
            if d == 1: rv = (ep - c) / imp
            else: rv = (c - ep) / abs(imp)
            return float(np.clip(rv, -2, 2))
        ff.append({"r5": _r(5), "r10": _r(10), "r15": _r(15)})
    return ff


print("=" * 70)
print("ADVERSITY TESTS — EURUSD only")
print("=" * 70)

months = [(2025, 10), (2025, 11), (2025, 12)]
configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]
pair = "EURUSD"

# Preload all months
print(f"\nLoading {pair} data...")
t_all = ld(pair, months)
print(f"  Total: {len(t_all):,} ticks")
t_months = {}
for m in months:
    t_months[m] = ld(pair, [m])
    print(f"  {m[1]}mo: {len(t_months[m]):,} ticks")
t_train = ld(pair, [(2025, 10), (2025, 11)])
print(f"  Train: {len(t_train):,} ticks")

# ── 1. Monthly breakdown ──
print(f"\n{'='*70}")
print("1: Monthly breakdown")
print('=' * 70)
for m in months:
    t = t_months[m]
    for ip, is_, h in configs:
        ev = fi(t, ip, is_)
        tr = sf(ev, t, h)
        if len(tr) < 5: continue
        wr = (tr["pnl_pips"] > 0).mean()
        avg = tr["pnl_pips"].mean()
        g = tr["pnl_pips"].sum()
        f = " ✓" if wr >= 0.60 else (" !! FAIL" if wr < 0.50 else "")
        print(f"  {m[1]:>2d}mo {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p{f}")

# ── 2. Retrace gate ──
print(f"\n{'='*70}")
print("2: Retrace gate (>=0.1) stability")
print('=' * 70)
for m in months:
    t = t_months[m]
    for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
        ev = fi(t, ip, is_)
        tr = sf(ev, t, h)
        ff = ef(ev, t)
        g = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
        if len(g) >= 5:
            g_df = pd.DataFrame(g)
            print(f"  {m[1]:>2d}mo {ip}p/{is_}s gated: n={len(g_df):>4d} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p")
        else:
            print(f"  {m[1]:>2d}mo {ip}p/{is_}s gated: {len(g)} trades (too few)")

# ── 3. Direction reversal ──
print(f"\n{'='*70}")
print("3: Direction reversal (momentum vs fade)")
print('=' * 70)
for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
    ev = fi(t_all, ip, is_)
    tf = sf(ev, t_all, h)
    tm = sf(ev, t_all, h, fl=True)
    wf = (tf["pnl_pips"] > 0).mean() if len(tf) > 0 else 0
    wm = (tm["pnl_pips"] > 0).mean() if len(tm) > 0 else 0
    af = tf["pnl_pips"].mean() if len(tf) > 0 else 0
    am = tm["pnl_pips"].mean() if len(tm) > 0 else 0
    gap = wf - wm
    v = "✓ directional edge" if gap > 0.15 else ("?? weak" if gap > 0.05 else "!! NO DIRECTION")
    print(f"  {ip}p/{is_}s: Fade WR={wf:.1%}({af:+.2f}p) Mom WR={wm:.1%}({am:+.2f}p) gap={gap:+.0%} {v}")

# ── 4. Random baseline ──
print(f"\n{'='*70}")
print("4: Random entry baseline")
print('=' * 70)
for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
    ev = fi(t_all, ip, is_)
    tr = sf(ev, t_all, h)
    if len(tr) < 20: continue
    rng = np.random.RandomState(42)
    rp = tr["pnl_pips"].values.copy()
    for i in range(len(rp)): rp[i] *= (1 if rng.rand() > 0.5 else -1)
    wr_rand = (rp > 0).mean()
    wr_real = (tr["pnl_pips"] > 0).mean()
    print(f"  {ip}p/{is_}s: Real WR={wr_real:.1%} Random WR={wr_rand:.1%} edge={wr_real-wr_rand:+.0%} {'✓' if wr_real-wr_rand>0.08 else '??'}")

# ── 5. Walk-forward Oct+Nov -> Dec ──
print(f"\n{'='*70}")
print("5: Walk-forward train Oct+Nov, test Dec")
print('=' * 70)
best_wr, best_cfg = 0, None
for ip, is_, h in configs:
    ev = fi(t_train, ip, is_)
    tr = sf(ev, t_train, h)
    if len(tr) < 20: continue
    wr = (tr["pnl_pips"] > 0).mean()
    if wr > best_wr: best_wr, best_cfg = wr, (ip, is_, h)
    print(f"  Train {ip}p/{is_}s h{h}s: n={len(tr)} WR={wr:.1%} avg={tr['pnl_pips'].mean():+.2f}p")
if best_cfg:
    ip, is_, h = best_cfg
    print(f"\n  Best on train: {ip}p/{is_}s h{h}s WR={best_wr:.1%}")
    t_dec = t_months[(2025, 12)]
    ev = fi(t_dec, ip, is_)
    tr = sf(ev, t_dec, h)
    if len(tr) > 0:
        wr = (tr["pnl_pips"] > 0).mean()
        avg = tr["pnl_pips"].mean()
        g = tr["pnl_pips"].sum()
        v = "✓ HOLDS" if wr > 0.55 else ("!! DEGRADES" if wr < 0.50 else "? marginal")
        print(f"  Test Dec: n={len(tr)} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p {v}")
        ff = ef(ev, t_dec)
        g2 = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
        if len(g2) >= 5:
            g_df = pd.DataFrame(g2)
            print(f"  Test Dec gated: n={len(g_df)} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p")

# ── 6. Full period ──
print(f"\n{'='*70}")
print("6: Full period (Oct-Dec) summary")
print('=' * 70)
for ip, is_, h in configs:
    ev = fi(t_all, ip, is_)
    tr = sf(ev, t_all, h)
    if len(tr) < 10: continue
    wr = (tr["pnl_pips"] > 0).mean()
    avg = tr["pnl_pips"].mean()
    g = tr["pnl_pips"].sum()
    print(f"  {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p")
    ff = ef(ev, t_all)
    g2 = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
    if len(g2) >= 10:
        g_df = pd.DataFrame(g2)
        print(f"    gated(ret>=0.1): n={len(g_df):>4d} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p gross={g_df['pnl_pips'].sum():+.1f}p")

print("\n✓ EURUSD adversity tests complete")
