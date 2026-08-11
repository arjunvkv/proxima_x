"""overnight_gap_study.py — US-index overnight gap (server 20:00 close -> 13:30 open).

US30/US500/GER40/UK100. Gap = open[13:30] - close[20:00 prev] in points. Fade
hypothesis (gap overshoot) vs follow (overnight news momentum), per-quartile,
per-side, LODO, costs (measured spreads + $3/lot). Research-only.
"""
import sys, os
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARKET = "audit_7_eas/market"
META = {"US30.cash": {"pt": 1.0, "spread": 21.0},
        "US500.cash": {"pt": 0.1, "spread": 6.0},
        "GER40.cash": {"pt": 0.1, "spread": 11.3},
        "UK100.cash": {"pt": 0.1, "spread": 7.5}}

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def study(sym):
    s = load(sym)
    ts = s["time"].to_numpy(); op = s["open"].to_numpy(); cl = s["close"].to_numpy()
    n = len(ts)
    def hour(tt): return (tt // 3600) % 24
    def minute(tt): return (tt // 60) % 60
    opens = {}   # day -> index of 13:30 bar
    closes = {}  # day -> index of 20:00 bar
    for i in range(n):
        h, m = hour(ts[i]), minute(ts[i])
        d = int(ts[i]) // 86400
        if h == 13 and m == 30:
            opens[d] = i
        if h == 20 and m == 0:
            closes[d] = i
    days = sorted(opens)
    rows = []
    for k, d in enumerate(days):
        if d not in closes:
            continue
        i_o, i_c = opens[d], closes[d]
        gap = op[i_o] - cl[i_c]                       # points
        f30 = cl[i_o + 6] - op[i_o]                   # first 30 min move
        f120 = cl[i_o + 24] - op[i_o]                 # first 2h move
        fday = cl[min(i_o + 78, n - 1)] - op[i_o]     # rest of session
        rows.append((gap, f30, f120, fday, d))
    rows.sort()
    g = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows])
    b = np.array([r[2] for r in rows]); c = np.array([r[3] for r in rows])
    days = np.array([r[4] for r in rows])
    q = np.quantile(g, [0.25, 0.5, 0.75])
    print(f"--- {sym} n={len(rows)} gap median={np.median(g):+.1f}pt q25={q[0]:+.1f} q75={q[2]:+.1f}")
    for nm, f, lab in [("30m", a, "30min"), ("2h", b, "2h"), ("day", c, "day")]:
        for side, mask in [("fade", g < q[0]), ("follow", g > q[2]), ("all", np.ones(len(g), bool))]:
            if mask.sum() < 40:
                continue
            sgn = np.where(g[mask] < 0, 1.0, -1.0) if side == "fade" else np.sign(g[mask])
            x = sgn * f[mask]
            wins = x[x > 0].sum(); losses = -x[x < 0].sum()
            pf = wins / losses if losses > 0 else 99
            z = x.mean() / (x.std() / np.sqrt(len(x)))
            lodo = sum(1 for dd in set(days[mask]) if x[days[mask] != dd].mean() <= 0)
            print(f"  {lab} {side:>6}: net={x.mean():+.2f}pt PF={pf:.2f} z={z:+.2f} LODO={lodo}/{len(set(days[mask]))} n={len(x)}")

for sym in META:
    try:
        study(sym)
    except Exception as e:
        print(f"{sym}: ERROR {e}")
