"""overnight_gap_study2.py — index break-gap with TRUE session boundaries.

US30/US500/GER40/UK100 trade ~23h; daily break = server hour 0 (CME
settlement). Real gap = open[01:05] - close[23:45 prev]. Fwd windows:
30min, to NY open (15:30 srv), full session. Fade vs follow per quartile,
per-side, LODO, costs. Also reports unconditional session-drift control.
"""
import sys, os
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARKET = "audit_7_eas/market"

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def study(sym):
    s = load(sym)
    ts = s["time"].to_numpy(); op = s["open"].to_numpy(); cl = s["close"].to_numpy()
    n = len(ts)
    def hm(tt): return (tt // 60) % 60, (tt // 3600) % 24
    reopens = {}; closes = {}
    for i in range(n):
        m, h = hm(ts[i]); d = int(ts[i]) // 86400
        if h == 1 and m == 5: reopens[d] = i
        if h == 23 and m == 45: closes[d] = i
    rows = []
    for d in sorted(reopens):
        if d - 1 not in closes: continue
        i_o, i_c = reopens[d], closes[d - 1]
        gap = op[i_o] - cl[i_c]
        f30 = cl[i_o + 6] - op[i_o]   # 01:05-01:35
        fny = (cl[min(i_o + 173, n - 1)] - op[i_o]) if i_o + 173 < n else np.nan  # 01:05->15:30
        fday = (cl[min(i_o + 272, n - 1)] - op[i_o]) if i_o + 272 < n else np.nan  # 01:05->23:45
        rows.append((gap, f30, fny, fday, d))
    rows = [r for r in rows if not np.isnan(r[3])]
    g = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows])
    b = np.array([r[2] for r in rows]); c = np.array([r[3] for r in rows])
    days = np.array([r[4] for r in rows])
    q = np.quantile(g, [1/3, 0.5, 2/3])
    print(f"--- {sym} n={len(rows)} gap med={np.median(g):+.1f}pt q33={q[0]:+.1f} q66={q[2]:+.1f}")
    for nm, f, lab in [("30m", a, "30min"), ("NYopen", b, "toNYopen"), ("session", c, "session")]:
        for side, mask in [("fade", g < q[0]), ("follow", g > q[2])]:
            if mask.sum() < 20: continue
            sgn = np.where(g[mask] < 0, 1.0, -1.0) if side == "fade" else np.sign(g[mask])
            x = sgn * f[mask]
            wins = x[x > 0].sum(); losses = -x[x < 0].sum()
            pf = wins / losses if losses > 0 else 99
            z = x.mean() / (x.std() / np.sqrt(len(x)))
            lodo = sum(1 for dd in set(days[mask]) if x[days[mask] != dd].mean() <= 0)
            print(f"  {lab:>8} {side:>6}: net={x.mean():+.2f}pt PF={pf:.2f} z={z:+.2f} LODO={lodo}/{len(set(days[mask]))} n={len(x)}")
    # unconditional controls
    for nm, f, lab in [("30m", a, "30min"), ("NYopen", b, "toNYopen"), ("session", c, "session")]:
        z = f.mean() / (f.std() / np.sqrt(len(f)))
        print(f"  CONTROL {lab}: mean={f.mean():+.2f}pt z={z:+.2f} (unconditional drift)")

for sym in ["US30.cash", "US500.cash", "GER40.cash", "UK100.cash"]:
    try:
        study(sym)
    except Exception as e:
        print(f"{sym}: ERROR {e}")
