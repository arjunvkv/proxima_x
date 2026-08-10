"""scripts/_absorb/probe_fix_sides.py — per-pair, per-side economics in pips.

The A/B study's pooled sign x post z is short-side-dominated. The engine's
legacy path (session_momentum) is BUY-only, so the shippable long side is what
matters — and JPY-cross spreads are 2.2-3.0 (typical) / 5.7-7.1 (measured)
pips. This probe prints per-pair LONG-only and SHORT-only means in pips for
T1 (all days) and T2 (|pre-move| >= pair-day q75) at the primary fix hours,
so the verdict rests on per-pair economics, not mixed-scale pooling.
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
        out.append((pv, cl[x] - op[e]))
    return out


def main():
    bars = {s: pl.read_parquet(os.path.join(TICKS, f"{s}.pqt")) for s in UNIVERSE}
    res = {"cells": {}}
    for fam, F, Hs in [("tokyo", 3, [15, 30, 60]), ("wmr", 17, [15, 30, 60])]:
        for H in Hs:
            key = f"{fam}_F{F}_H{H}"
            print(f"\n== {key} ==")
            print(f"{'pair':>7} | " + " ".join(
                f"{lab:>9}" for lab in
                ["long_T1", "short_T1", "long_T2", "short_T2", "nL", "nS"]))
            row = {}
            for s in UNIVERSE:
                df = bars[s]
                ev = events(df["ts"].to_numpy().astype("int64"),
                            df["open"].to_numpy().astype("float64"),
                            df["close"].to_numpy().astype("float64"),
                            df["ts"].to_numpy().astype("int64") // 86400, F, H)
                if not ev:
                    continue
                pvs = np.array([e[0] for e in ev])
                posts = np.array([e[1] for e in ev])
                thr = np.quantile(np.abs(pvs), Q75)
                pts = POINT[s]
                ld = posts[pvs > 0]
                sd = posts[pvs < 0]
                lt2 = posts[(pvs > 0) & (np.abs(pvs) >= thr)]
                st2 = posts[(pvs < 0) & (np.abs(pvs) >= thr)]
                vals = dict(
                    long_T1=round(float(ld.mean()) / pts, 2) if len(ld) else None,
                    short_T1=round(float(sd.mean()) / pts, 2) if len(sd) else None,
                    long_T2=round(float(lt2.mean()) / pts, 2) if len(lt2) else None,
                    short_T2=round(float(st2.mean()) / pts, 2) if len(st2) else None,
                    nL=int(len(ld)), nS=int(len(sd)))
                row[s] = vals
                print(f"{s:>7} | " + " ".join(
                    f"{'-' if vals[k] is None else vals[k]:>9}"
                    for k in ["long_T1", "short_T1", "long_T2", "short_T2",
                              "nL", "nS"]))
            res["cells"][key] = row
    with open(os.path.join(HERE, "results", "probe_fix_sides.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()