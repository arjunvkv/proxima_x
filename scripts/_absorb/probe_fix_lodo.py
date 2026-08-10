"""scripts/_absorb/probe_fix_lodo.py — leave-one-day-out on the money cells.

Per-side (long/short) T2 cells that look tradeable in probe_fix_sides.py have
n ~9-21. The absorption lesson: verify none is 1-2 freak days. For each money
cell, remove every day in turn and recompute the per-side mean in pips;
report min/max/sign-flips vs the full mean.
"""
from __future__ import annotations
import json, os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, "results", "ticks")
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
POINT = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "EURJPY": 1e-2,
         "GBPJPY": 1e-2, "AUDUSD": 1e-4, "USDCAD": 1e-4}
PRE_MIN = 30
ENTRY_GAP = 5
Q75 = 0.75

# (family, F, H, side) — the cells with any claim to an edge
MONEY = [("tokyo", 3, 15, "short"), ("tokyo", 3, 60, "short"),
         ("wmr", 17, 15, "short"), ("wmr", 17, 60, "long"),
         ("tokyo", 3, 60, "long"), ("wmr", 17, 30, "long")]


def events(ts, op, cl, day, F, H):
    out = []
    pre0 = F * 3600 - 60 * PRE_MIN
    entry_ts = F * 3600 + 60 * ENTRY_GAP
    for d in np.unique(day):
        d0 = int(d) * 86400
        m = (ts >= d0 + pre0) & (ts < d0 + F * 3600) & (day == d)
        idx = np.where(m)[0]
        if len(idx) < 5 or ts[idx[-1]] - ts[idx[0]] != 60 * len(idx) - 60:
            continue
        pv = cl[idx[-1]] - op[idx[0]]
        if pv == 0:
            continue
        e = np.searchsorted(ts, d0 + entry_ts)
        if e >= len(ts) or ts[e] != d0 + entry_ts:
            continue
        x = e + H - 1
        if x >= len(ts) or ts[x] - ts[e] != 60 * (H - 1) or day[x] != day[e]:
            continue
        out.append((int(d), pv, cl[x] - op[e]))
    return out


def main():
    bars = {s: pl.read_parquet(os.path.join(TICKS, f"{s}.pqt")) for s in UNIVERSE}
    res = {}
    for fam, F, H, side in MONEY:
        for s in UNIVERSE:
            df = bars[s]
            ev = events(df["ts"].to_numpy().astype("int64"),
                        df["open"].to_numpy().astype("float64"),
                        df["close"].to_numpy().astype("float64"),
                        df["ts"].to_numpy().astype("int64") // 86400, F, H)
            if not ev:
                continue
            thr = np.quantile(np.abs([e[1] for e in ev]), Q75)
            sel = [(d, post) for d, pv, post in ev
                   if (pv > 0 if side == "long" else pv < 0)
                   and abs(pv) >= thr]
            if len(sel) < 4:
                continue
            posts = np.array([x[1] for x in sel]) / POINT[s]  # pips
            full = float(posts.mean())
            days = {}
            for d, post in sel:
                m = np.array([x[0] for x in sel]) != d
                days[int(d)] = round(float(posts[m].mean()), 2)
            vals = list(days.values())
            flips = sum(1 for v in vals if (v > 0) != (full > 0))
            res[f"{fam}_F{F}_H{H}_{side}_{s}"] = {
                "n": len(sel), "full_mean_pips": round(full, 2),
                "lodo_min": min(vals), "lodo_max": max(vals),
                "flips": flips}
            print(f"{fam}_F{F}_H{H} {side:>5} {s:>7} n={len(sel):>2} "
                  f"mean={full:+7.2f}  LODO[{min(vals):+7.2f},{max(vals):+7.2f}] "
                  f"flips={flips}/{len(vals)}")
    with open(os.path.join(HERE, "results", "probe_fix_lodo.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()