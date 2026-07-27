"""Adversity test runner — preloads all data once, runs each test sequentially."""
import sys, os
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import pandas as pd
import numpy as np
from pathlib import Path

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def load_ticks(pair="EURUSD", months=None):
    if months is None:
        months = [(2025, 10)]
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t


def find_impulses(ticks, impulse_pips=5, impulse_sec=10):
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values
    n = len(ticks)
    events = []
    i = 0
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + impulse_sec, side="right")), n)
        w = mid[i:end]
        if len(w) < 2:
            i += 1
            continue
        high = np.max(w); low = np.min(w)
        hp = (high - w[0]) * 10000
        lp = abs((low - w[0]) * 10000)
        if max(hp, lp) >= impulse_pips:
            d = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d == 1 else np.argmin(w))
            events.append({"time": ts[i], "extreme_time": ts[ei],
                "impulse_pips": max(hp, lp), "impulse_sec": ts[ei] - ts[i],
                "direction": d, "price_start": w[0], "price_extreme": mid[ei],
                "event_idx": i, "extreme_idx": ei})
            i = ei
        else:
            i += 1
    return pd.DataFrame(events)


def simulate_fade(events, ticks, hold_sec=15, flip_dir=False):
    bids = ticks["B"].values; asks = ticks["A"].values
    ts = ticks["ts_s"].values; n = len(ticks)
    trades = []
    for _, ev in events.iterrows():
        ed = ev["direction"] if flip_dir else -ev["direction"]
        ei = int(ev["extreme_idx"]) + 1
        if ei >= n: continue
        ep = asks[ei] if ed == 1 else bids[ei]
        xi = int(np.searchsorted(ts, ts[ei] + hold_sec, side="right"))
        if xi >= n: continue
        xp = bids[xi] if ed == 1 else asks[xi]
        pnl = (xp - ep) * ed
        trades.append({"entry_time": ts[ei], "entry_price": ep, "exit_price": xp,
            "pnl_pips": pnl * 10000, "direction": ed,
            "impulse_pips": ev["impulse_pips"], "hold_sec": ts[xi] - ts[ei]})
    return pd.DataFrame(trades)


def extract_features(events, ticks, reaction_window=15):
    mid = (ticks["B"].values + ticks["A"].values) / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    features = []
    for _, ev in events.iterrows():
        eidx = int(ev["extreme_idx"])
        et = ts[eidx]
        ep = ev["price_extreme"]
        sp = ev["price_start"]
        imp = ev["impulse_pips"] / 10000
        d = int(ev["direction"])
        we = min(int(np.searchsorted(ts, et + reaction_window, side="right")), n)
        wm = mid[eidx:we]
        wt = ts[eidx:we]
        if len(wm) < 2:
            features.append(None); continue
        ret5 = _ret(wm, wt, 5, et, sp, ep, d)
        ret10 = _ret(wm, wt, 10, et, sp, ep, d)
        ret15 = _ret(wm, wt, 15, et, sp, ep, d)
        features.append({"retrace_5s": ret5, "retrace_10s": ret10, "retrace_15s": ret15})
    return features


def _ret(wm, wt, lb, et, sp, ep, d):
    idx = np.searchsorted(wt, et + lb, side="right")
    if idx >= len(wm): idx = -1
    c = wm[idx]; imp = ep - sp
    if abs(imp) < 1e-10: return 0.0
    if d == 1: r = (ep - c) / imp
    else: r = (c - ep) / abs(imp)
    return float(np.clip(r, -2, 2))


print("=" * 70)
print("ADVERSITY TEST SUITE")
print("=" * 70)

# Preload
print("\nPreloading tick data...")
months = [(2025, 10), (2025, 11), (2025, 12)]
pairs = ["EURUSD", "EURJPY", "GBPJPY"]
configs = [(5, 10, 15), (7, 10, 15), (5, 5, 15), (3, 5, 15)]
data = {}
for pair in pairs:
    data[(pair, "all")] = load_ticks(pair, months)
    print(f"  {pair} all: {len(data[(pair,'all')]):,}")
    for m in months:
        data[(pair, m)] = load_ticks(pair, [m])
    data[(pair, "train")] = load_ticks(pair, [(2025, 10), (2025, 11)])
print("Preload done.\n")

# TEST 1: Monthly breakdown
print("-" * 70)
print("TEST 1: Monthly breakdown")
print("-" * 70)
for pair in pairs:
    print(f"\n  {pair}:")
    for m in months:
        t = data[(pair, m)]
        for ip, isec, hold in configs:
            ev = find_impulses(t, ip, isec)
            tr = simulate_fade(ev, t, hold)
            if len(tr) < 5: continue
            wr = (tr["pnl_pips"] > 0).mean()
            avg = tr["pnl_pips"].mean()
            gross = tr["pnl_pips"].sum()
            f = ""
            if wr < 0.50: f = " !! FAIL"
            elif wr >= 0.60: f = " ✓"
            print(f"    {m[1]:>2d}mo {ip}p/{isec}s h{hold}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={gross:+.1f}p{f}")

# TEST 2: Retrace gate
print(f"\n{'-'*70}")
print("TEST 2: Retrace gate (>=0.1) across months")
print('-' * 70)
for pair in pairs:
    print(f"\n  {pair}:")
    for m in months:
        t = data[(pair, m)]
        for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
            ev = find_impulses(t, ip, isec)
            tr = simulate_fade(ev, t, hold)
            feats = extract_features(ev, t)
            g = []
            for f, (_, trade) in zip(feats, tr.iterrows()):
                if f is not None and f["retrace_5s"] >= 0.1:
                    g.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
            if len(g) >= 5:
                g_df = pd.DataFrame(g)
                print(f"    {m[1]:>2d}mo {ip}p/{isec}s gated: n={len(g_df):>4d} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p")
            else:
                print(f"    {m[1]:>2d}mo {ip}p/{isec}s gated: {len(g)} trades (too few)")

# TEST 3: Direction reversal
print(f"\n{'-'*70}")
print("TEST 3: Direction reversal (momentum vs fade)")
print('-' * 70)
for pair in pairs:
    t = data[(pair, "all")]
    print(f"\n  {pair}:")
    for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
        ev = find_impulses(t, ip, isec)
        tf = simulate_fade(ev, t, hold)
        tm = simulate_fade(ev, t, hold, flip_dir=True)
        wf = (tf["pnl_pips"] > 0).mean() if len(tf) > 0 else 0
        wm = (tm["pnl_pips"] > 0).mean() if len(tm) > 0 else 0
        af = tf["pnl_pips"].mean() if len(tf) > 0 else 0
        am = tm["pnl_pips"].mean() if len(tm) > 0 else 0
        gap = wf - wm
        v = "✓ directional edge" if gap > 0.15 else ("?? weak" if gap > 0.05 else "!! NO DIRECTION")
        print(f"    {ip}p/{isec}s: Fade WR={wf:.1%}({af:+.2f}p) Mom WR={wm:.1%}({am:+.2f}p) gap={gap:+.0%} {v}")

# TEST 4: Random baseline
print(f"\n{'-'*70}")
print("TEST 4: Random entry baseline")
print('-' * 70)
for pair in pairs[:1]:
    t = data[(pair, "all")]
    print(f"\n  {pair}:")
    for ip, isec, hold in [(5, 10, 15), (7, 10, 15)]:
        ev = find_impulses(t, ip, isec)
        tr = simulate_fade(ev, t, hold)
        if len(tr) < 20: continue
        rng = np.random.RandomState(42)
        rp = tr["pnl_pips"].values.copy()
        for i in range(len(rp)): rp[i] *= (1 if rng.rand() > 0.5 else -1)
        wr_rand = (rp > 0).mean()
        wr_real = (tr["pnl_pips"] > 0).mean()
        print(f"    {ip}p/{isec}s: Real WR={wr_real:.1%} Random WR={wr_rand:.1%} edge={wr_real-wr_rand:+.0%} {'✓' if wr_real-wr_rand>0.08 else '??'}")

# TEST 5: Walk-forward Oct+Nov -> Dec
print(f"\n{'-'*70}")
print("TEST 5: Walk-forward train Oct+Nov, test Dec")
print('-' * 70)
for pair in pairs:
    train = data[(pair, "train")]
    test = data[(pair, (2025, 12))]
    print(f"\n  {pair}: train={len(train):,} test={len(test):,}")
    best_wr, best_cfg = 0, None
    for ip, isec, hold in configs:
        ev = find_impulses(train, ip, isec)
        tr = simulate_fade(ev, train, hold)
        if len(tr) < 20: continue
        wr = (tr["pnl_pips"] > 0).mean()
        if wr > best_wr:
            best_wr, best_cfg = wr, (ip, isec, hold)
    if best_cfg:
        ip, isec, hold = best_cfg
        print(f"    Best on train: {ip}p/{isec}s hold={hold}s WR={best_wr:.1%}")
        evt = find_impulses(test, ip, isec)
        trt = simulate_fade(evt, test, hold)
        if len(trt) > 0:
            wr = (trt["pnl_pips"] > 0).mean()
            avg = trt["pnl_pips"].mean()
            g = trt["pnl_pips"].sum()
            v = "✓ HOLDS" if wr > 0.55 else ("!! DEGRADES" if wr < 0.50 else "? marginal")
            print(f"    Test Dec: n={len(trt)} WR={wr:.1%} avg={avg:+.2f}p gross={g:+.1f}p {v}")
            feats = extract_features(evt, test)
            g2 = []
            for f, (_, trade) in zip(feats, trt.iterrows()):
                if f is not None and f["retrace_5s"] >= 0.1:
                    g2.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
            if len(g2) >= 5:
                g_df = pd.DataFrame(g2)
                print(f"    Test Dec gated: n={len(g_df)} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p")

# TEST 6: Full period summary
print(f"\n{'-'*70}")
print("TEST 6: Full period (Oct-Dec) summary")
print('-' * 70)
for pair in pairs:
    t = data[(pair, "all")]
    print(f"\n  {pair} ({len(t):,} ticks):")
    for ip, isec, hold in configs:
        ev = find_impulses(t, ip, isec)
        tr = simulate_fade(ev, t, hold)
        if len(tr) < 10: continue
        wr = (tr["pnl_pips"] > 0).mean()
        avg = tr["pnl_pips"].mean()
        gross = tr["pnl_pips"].sum()
        print(f"    {ip}p/{isec}s hold={hold}s: n={len(tr):>4d} WR={wr:.1%} avg={avg:+.2f}p gross={gross:+.1f}p")
        feats = extract_features(ev, t)
        g = []
        for f, (_, trade) in zip(feats, tr.iterrows()):
            if f is not None and f["retrace_5s"] >= 0.1:
                g.append(trade.to_dict() if hasattr(trade, 'to_dict') else trade)
        if len(g) >= 10:
            g_df = pd.DataFrame(g)
            print(f"           gated(ret>=0.1): n={len(g_df):>4d} WR={(g_df['pnl_pips']>0).mean():.1%} avg={g_df['pnl_pips'].mean():+.2f}p gross={g_df['pnl_pips'].sum():+.1f}p")

print("\nDone.")
