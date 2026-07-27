"""Fast adversity testing - caches impulses to avoid redundant computation."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
import pandas as pd
import numpy as np
from pathlib import Path

D = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def ld(p, months):
    dfs = []
    for y, mm in months:
        f = D / f"{p}_Raw_Spread_{y}_{mm:02d}.zip"
        d = pd.read_csv(f, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


class ImpulseCache:
    """Cache impulse events so each (ticks_id, ip, is_) is computed once."""
    def __init__(self):
        self._cache = {}

    def get(self, ticks, ticks_id, ip, is_):
        key = (ticks_id, ip, is_)
        if key not in self._cache:
            self._cache[key] = self._find(ticks, ip, is_)
        return self._cache[key]

    def _find(self, t, ip, is_):
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
                ev.append({"d": d, "ei": ei, "ps": w[0], "pe": m[ei], "ip": max(hp, lp)})
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
        tr.append({"pnl": (xp - ep) * 10000 * ed})
    return pd.DataFrame(tr) if tr else pd.DataFrame(columns=["pnl"])


def ef(ev, t, rw=15):
    m = (t["B"].values + t["A"].values) / 2
    ts = t["ts_s"].values; n = len(t)
    ff = []
    for _, e in ev.iterrows():
        ei = int(e["ei"]); et = ts[ei]; ep = e["pe"]; sp = e["ps"]; d = int(e["d"])
        we = min(int(np.searchsorted(ts, et + rw, side="right")), n)
        wm = m[ei:we]; wt = ts[ei:we]
        if len(wm) < 2: ff.append(None); continue
        imp = ep - sp
        if abs(imp) < 1e-10: ff.append({"r5": 0}); continue
        def _r(lb):
            idx = np.searchsorted(wt, et + lb, side="right")
            if idx >= len(wm): idx = -1
            c = wm[idx]
            return float(np.clip(((ep - c) / imp) if d == 1 else ((c - ep) / abs(imp)), -2, 2))
        ff.append({"r5": _r(5)})
    return ff


def row(r): return {"pnl": r["pnl"]}


def run_all_tests(pair_label, all_t, oct_t, nov_t, dec_t, train_t, cache):
    configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]
    
    # 1: Monthly breakdown
    print(f"\n{'='*60}")
    print(f"1: MONTHLY BREAKDOWN — {pair_label}")
    print('=' * 60)
    for name, t in [("Oct", oct_t), ("Nov", nov_t), ("Dec", dec_t)]:
        for ip, is_, h in configs:
            ev = cache.get(t, f"{pair_label}_{name}", ip, is_)
            tr = sf(ev, t, h)
            if len(tr) < 5: continue
            wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
            f = " ✓" if wr >= 0.60 else (" !! FAIL" if wr < 0.50 else "")
            print(f"  {name:>4s} {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p{f}")

    # 2: Retrace gate
    print(f"\n{'='*60}")
    print(f"2: RETRACE GATE — {pair_label}")
    print('=' * 60)
    for name, t in [("Oct", oct_t), ("Nov", nov_t), ("Dec", dec_t)]:
        for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
            ev = cache.get(t, f"{pair_label}_{name}", ip, is_)
            tr = sf(ev, t, h)
            ff = ef(ev, t)
            g = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
            if len(g) >= 5:
                g_df = pd.DataFrame(g)
                print(f"  {name:>4s} {ip}p/{is_}s gated: n={len(g_df):>4d} WR={(g_df['pnl']>0).mean():.1%} avg={g_df['pnl'].mean():+.2f}p")
            else:
                print(f"  {name:>4s} {ip}p/{is_}s gated: {len(g)} trades (too few)")

    # 3: Direction reversal
    print(f"\n{'='*60}")
    print(f"3: DIRECTION REVERSAL — {pair_label}")
    print('=' * 60)
    for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
        ev = cache.get(all_t, f"{pair_label}_All", ip, is_)
        tf = sf(ev, all_t, h); tm = sf(ev, all_t, h, fl=True)
        wf = (tf["pnl"] > 0).mean() if len(tf) > 0 else 0
        wm = (tm["pnl"] > 0).mean() if len(tm) > 0 else 0
        af = tf["pnl"].mean() if len(tf) > 0 else 0
        am = tm["pnl"].mean() if len(tm) > 0 else 0
        gap = wf - wm
        v = "✓ directional" if gap > 0.15 else ("?? weak" if gap > 0.05 else "!! NO DIRECTION")
        print(f"  {ip}p/{is_}s: Fade WR={wf:.1%}({af:+.2f}p) Mom WR={wm:.1%}({am:+.2f}p) gap={gap:+.0%} {v}")

    # 4: Random baseline
    print(f"\n{'='*60}")
    print(f"4: RANDOM BASELINE — {pair_label}")
    print('=' * 60)
    for ip, is_, h in [(5, 10, 15), (7, 10, 15)]:
        ev = cache.get(all_t, f"{pair_label}_All", ip, is_)
        tr = sf(ev, all_t, h)
        if len(tr) < 20: continue
        rng = np.random.RandomState(42)
        rp = tr["pnl"].values.copy()
        for i in range(len(rp)): rp[i] *= (1 if rng.rand() > 0.5 else -1)
        wr_rand = (rp > 0).mean(); wr_real = (tr["pnl"] > 0).mean()
        print(f"  {ip}p/{is_}s: Real WR={wr_real:.1%} Random WR={wr_rand:.1%} edge={wr_real-wr_rand:+.0%} {'✓' if wr_real-wr_rand>0.08 else '??'}")

    # 5: Walk-forward
    print(f"\n{'='*60}")
    print(f"5: WALK-FORWARD Oct+Nov -> Dec — {pair_label}")
    print('=' * 60)
    best_wr, best_cfg = 0, None
    for ip, is_, h in configs:
        ev = cache.get(train_t, f"{pair_label}_Train", ip, is_)
        tr = sf(ev, train_t, h)
        if len(tr) < 20: continue
        wr = (tr["pnl"] > 0).mean()
        if wr > best_wr: best_wr, best_cfg = wr, (ip, is_, h)
        print(f"  Train {ip}p/{is_}s h{h}s: n={len(tr)} WR={wr:.1%} avg={tr['pnl'].mean():+.2f}p")
    if best_cfg:
        ip, is_, h = best_cfg
        print(f"  Best: {ip}p/{is_}s h{h}s WR={best_wr:.1%}")
        ev = cache.get(dec_t, f"{pair_label}_Dec", ip, is_)
        tr = sf(ev, dec_t, h)
        if len(tr) > 0:
            wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
            v = "✓ HOLDS" if wr > 0.55 else ("!! DEGRADES" if wr < 0.50 else "? marginal")
            print(f"  Test Dec: n={len(tr)} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p {v}")
            ff = ef(ev, dec_t)
            g2 = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
            if len(g2) >= 5:
                g_df = pd.DataFrame(g2)
                print(f"  Test Dec gated: n={len(g_df)} WR={(g_df['pnl']>0).mean():.1%} avg={g_df['pnl'].mean():+.2f}p")

    # 6: Full period
    print(f"\n{'='*60}")
    print(f"6: FULL PERIOD — {pair_label}")
    print('=' * 60)
    for ip, is_, h in configs:
        ev = cache.get(all_t, f"{pair_label}_All", ip, is_)
        tr = sf(ev, all_t, h)
        if len(tr) < 10: continue
        wr = (tr["pnl"] > 0).mean(); avg = tr["pnl"].mean(); g = tr["pnl"].sum()
        print(f"  {ip}p/{is_}s h{h}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p")
        ff = ef(ev, all_t)
        g2 = [tr.iloc[i].to_dict() for i, f in enumerate(ff) if f is not None and f["r5"] >= 0.1 and i < len(tr)]
        if len(g2) >= 10:
            g_df = pd.DataFrame(g2)
            print(f"    gated(ret>=0.1): n={len(g_df):>4d} WR={(g_df['pnl']>0).mean():.1%} avg={g_df['pnl'].mean():+.2f}p gross={g_df['pnl'].sum():+.1f}p")


# ── Main ──
t0 = time.time()
cache = ImpulseCache()

for pair in ["EURUSD", "EURJPY", "GBPJPY"]:
    print(f"\n\n{'#'*70}")
    print(f"# {pair}")
    print(f"{'#'*70}")
    print(f"Loading {pair}...")
    all_t = ld(pair, [(2025, 10), (2025, 11), (2025, 12)])
    oct_t = ld(pair, [(2025, 10)])
    nov_t = ld(pair, [(2025, 11)])
    dec_t = ld(pair, [(2025, 12)])
    train_t = ld(pair, [(2025, 10), (2025, 11)])
    print(f"Loaded ({len(all_t):,} ticks)")
    run_all_tests(pair, all_t, oct_t, nov_t, dec_t, train_t, cache)

print(f"\nAll done in {time.time()-t0:.1f}s")
