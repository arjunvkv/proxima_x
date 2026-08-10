"""scripts/_absorb/probe_fix_signs.py — reconcile fix_study signal vs engine signal.

The grid gate (fix_gate.py, engine session_momentum BUY-only) is deeply
negative where the M1 study (fix_study.py) found strong continuation. Two
candidate causes, probed on the SAME M1 cache:

 A) asymmetry: the sign x post mean is carried by the SHORT side; the
    long-only subset (which the legacy engine path trades) may be null/negative.
 B) window: the engine measures direction close-to-close THROUGH F+5 (M5 bar
    at hour F closes 5 min after the fix, i.e. includes the post-fix spike),
    while the study's direction ends at F. If the first 5 min after the fix
    move OPPOSITE the pre-fix drift (unwind spike), the engine enters
    systematically wrong.

For each family/primary F and hold H: long-subset mean, short-subset mean,
signed-mean (T1), and the F+5-window variant of each.
"""
from __future__ import annotations
import json, os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, "results", "ticks")
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
PRE_MIN = 30
ENTRY_GAP = 5


def load(sym):
    df = pl.read_parquet(os.path.join(TICKS, f"{sym}.pqt"))
    return {"ts": df["ts"].to_numpy().astype("int64"),
            "open": df["open"].to_numpy().astype("float64"),
            "close": df["close"].to_numpy().astype("float64")}


def probe(ts, op, cl, day, F, H, end_at_fplus5):
    """Return list of (pre_move, post_return) events for the fix hour F."""
    out = []
    sign_ts = F * 3600 + (300 if end_at_fplus5 else 0)  # direction window end
    pre0 = F * 3600 - 60 * PRE_MIN
    entry_ts = F * 3600 + 60 * ENTRY_GAP
    for d in np.unique(day):
        d0 = int(d) * 86400
        m = (ts >= d0 + pre0) & (ts < d0 + sign_ts) & (day == d)
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
    bars = {s: load(s) for s in UNIVERSE}
    js = {"hypotheses": ["A: long vs short asymmetry", "B: F+5 window"],
          "cells": {}}
    for fam, F, Hs in [("tokyo", 3, [15, 30, 60]), ("wmr", 17, [15, 30, 60])]:
        for H in Hs:
            key = f"{fam}_F{F}_H{H}"
            js["cells"][key] = {}
            for win in (False, True):
                all_e = []
                for s in UNIVERSE:
                    b = bars[s]
                    all_e += [(pv, post) for pv, post in
                              probe(b["ts"], b["open"], b["close"],
                                    b["ts"] // 86400, F, H, win)]
                if not all_e:
                    continue
                pvs = np.array([e[0] for e in all_e])
                posts = np.array([e[1] for e in all_e])
                long_m = posts[pvs > 0].mean()
                short_m = posts[pvs < 0].mean()
                signed = float((np.sign(pvs) * posts).mean())
                n = len(all_e)
                row = {"n": n, "long_only_mean": round(float(long_m), 8),
                       "short_only_mean": round(float(short_m), 8),
                       "signed_T1_mean": round(signed, 8)}
                print(f"{key} win_to_F+5={win}: n={n} "
                      f"long={long_m:+.6f} short={short_m:+.6f} "
                      f"signed={signed:+.6f}")
                js["cells"][key][str(win)] = row
    with open(os.path.join(HERE, "results", "probe_fix_signs.json"), "w") as f:
        json.dump(js, f, indent=2)


if __name__ == "__main__":
    main()