"""scripts/_absorb/fix_study.py — fixing-window A/B study (research only).

Pre-registered hypotheses (FTMO cache, 7 majors x 60d, 1-min bars, server clock):

  The WMR/ECB/Tokyo benchmark fixes are scheduled institutional-flow events.
  Dealers hedge inventory in the run-up and unwind after -> classic signature:
  price pressure INTO the fix window, then short-horizon REVERSAL after it
  (see "Foreign Exchange Fixings and Returns around the Clock", JF; Osler 2005
  on stop cascades; the fix-manipulation literature). This battery asks whether
  that signature is (a) present at all on this feed and (b) mechanically
  tradeable.

  T0  unconditional: mean post-fix return vs random-minute null.
  T1  conditional:   sign(pre-fix move) x post-fix return  (continuation +,
                      reversal -) vs random-sign null.
  T2  strong-flow:   same as T1 but only days where |pre-fix move| >= q75
                      of the pair's distribution.

  Clock: cache hours are server time = UTC+2 (verified empirically: activity
  step-up at server 09h = 07:00 UTC London open; daily break at server 00h =
  22:00 UTC; peak 15-17h = London/NY overlap). August: London is BST, so the
  WMR 16:00 fix = 15:00 UTC = 17:00 server; ECB 14:15 CET = 14:15 server;
  Tokyo 09:55 JST = 02:55 server. A prespecified offset grid around each fix
  is reported in full (honest neighborhood, matched nulls, no pick).

  Entry: first bar at/after F+5min, exit H bars later (same-day guard).
  Costs: realistic (typical busy spread) and stress (closed-market measured).

Run:  unset PYTHONPATH && ./.venv/Scripts/python.exe -u scripts/_absorb/fix_study.py
"""
from __future__ import annotations
import json, os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, "results", "ticks")
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
HORIZONS = [5, 15, 30, 60]           # minutes held after entry (F+5)
PRE_MIN = 30                         # minutes of pre-fix move measurement
ENTRY_GAP = 5                        # enter at F+5 (skip the fix print minute)
N_ITER = 500
SEED = 7
Q75 = 0.75
Z_LODO = 2.5                         # LODO cutoff for candidate cells

# server-clock fix hypotheses (UTC+2, summer) + prespecified offset grid
FIXES = {
    "wmr_london":   {"primary": 17, "grid": [15, 16, 17, 18, 19]},
    "ecb":          {"primary": 14, "grid": [12, 13, 14, 15, 16]},
    "tokyo":        {"primary": 3,  "grid": [1, 2, 3, 4, 5]},
}
POINT = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "EURJPY": 1e-2,
         "GBPJPY": 1e-2, "AUDUSD": 1e-4, "USDCAD": 1e-4}
SPREAD_TYP = {"EURUSD": 0.8, "USDJPY": 1.2, "GBPUSD": 1.5, "AUDUSD": 1.1,
              "EURJPY": 2.2, "GBPJPY": 3.0, "USDCAD": 1.8}
SPREAD_STRESS = {"EURUSD": 0.6, "USDJPY": 3.5, "GBPUSD": 1.3, "AUDUSD": 0.9,
                 "EURJPY": 5.7, "GBPJPY": 7.1, "USDCAD": 1.6}


def load_minutes(sym: str) -> dict:
    df = pl.read_parquet(os.path.join(TICKS, f"{sym}.pqt"))
    return {
        "ts": df["ts"].to_numpy().astype("int64"),
        "open": df["open"].to_numpy().astype("float64"),
        "close": df["close"].to_numpy().astype("float64"),
    }


def summarize(arr) -> dict:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": n, "mean": round(float(arr.mean()), 8),
            "sd": round(float(arr.std(ddof=1)), 8)}


def hour_activity(bars_map: dict) -> dict:
    """Clock verification: pooled mean |1-min return| by server hour."""
    out = {}
    for b in bars_map.values():
        ts, cl = b["ts"], b["close"]
        hrs = ts[1:] // 3600 % 24
        r1 = np.abs(np.diff(cl))
        for h in range(24):
            m = hrs == h
            if m.sum():
                out.setdefault(h, []).append(float(r1[m].mean()))
    prof = {h: float(np.mean(v)) for h, v in sorted(out.items())}
    peak = sorted(prof, key=lambda k: -prof[k])[:3]
    return {"profile": prof, "peak_hours": peak,
            "peak_utc_equiv": [int(p) - 2 for p in peak]}


def day_pool(bars_map: dict, s: str, H: int) -> np.ndarray:
    """Random-minute forward returns for pair s (same fill convention, same-day
    guarded): enter at bar open, exit H bars later at close."""
    b = bars_map[s]
    ts, op, cl = b["ts"], b["open"], b["close"]
    day = ts // 86400
    out = []
    for i in range(len(ts) - H):
        j = i + H - 1
        if ts[j] - ts[i] == 60 * (H - 1) and day[j] == day[i]:
            out.append(cl[j] - op[i])
    return np.asarray(out, dtype=float)


def fix_events(bars_map: dict, fam: str, F: int, H: int, tier: str,
               pools: dict) -> tuple[list, dict, np.ndarray, np.ndarray]:
    """Per-symbol + pooled gated post-fix returns for one (family, F, H, tier).

    Returns (per_symbol, pooled_summary, gated_arr, day_arr).
    """
    per, gated, days = [], [], []
    f_open = F * 3600
    pre0 = F * 3600 - 60 * PRE_MIN
    entry_ts = F * 3600 + 60 * ENTRY_GAP
    for s, b in bars_map.items():
        ts, op, cl = b["ts"], b["open"], b["close"]
        day = ts // 86400
        # per-day |pre-fix move| for the T2 threshold (pair-level across days)
        pm = {}
        for d in np.unique(day):
            d0 = int(d) * 86400
            f_open = d0 + F * 3600
            pre0 = f_open - 60 * PRE_MIN
            entry_ts = f_open + 60 * ENTRY_GAP
            m = (ts >= pre0) & (ts < f_open) & (day == d)
            idx = np.where(m)[0]
            if len(idx) >= 5 and ts[idx[-1]] - ts[idx[0]] == 60 * len(idx) - 60:
                pm[d] = cl[idx[-1]] - op[idx[0]]
        thr = np.quantile(list(pm.values()), Q75) if pm else 0.0
        vals, dts = [], []
        for d, mv in pm.items():
            if tier == "T2" and abs(mv) < abs(thr):
                continue
            if mv == 0:
                continue
            entry_ts = int(d) * 86400 + F * 3600 + 60 * ENTRY_GAP
            e = np.searchsorted(ts, entry_ts, side="left")
            if e >= len(ts) or ts[e] != entry_ts:
                continue
            x = e + H - 1
            if x >= len(ts) or ts[x] - ts[e] != 60 * (H - 1) or day[x] != day[e]:
                continue
            post = cl[x] - op[e]
            if tier == "T0":
                vals.append(post)
            else:
                vals.append(post * np.sign(mv))
            dts.append(d)
        per.append({"symbol": s, "n": len(vals),
                    "mean": round(float(np.mean(vals)), 8) if vals else None,
                    "pips": round(float(np.mean(vals)) / POINT[s], 3)
                    if vals else None})
        gated.extend(vals)
        days.extend(dts)
    return per, summarize(gated), np.array(gated, dtype=float), np.array(days)


def null_stats(pool: np.ndarray, n_sig: int, signed: bool) -> dict:
    rng = np.random.default_rng(SEED)
    if len(pool) == 0 or n_sig == 0 or len(pool) < n_sig:
        return {"mean": None, "sd": None}
    means = np.empty(N_ITER)
    for k in range(N_ITER):
        idx = rng.integers(0, len(pool), size=n_sig)
        a = pool[idx]
        if signed:
            a = a * rng.choice([-1.0, 1.0], size=n_sig)
        means[k] = float(a.mean())
    return {"mean": round(float(means.mean()), 8),
            "sd": round(float(means.std(ddof=1)), 8)}


def lodo(gated: np.ndarray, days: np.ndarray) -> dict:
    out = {}
    for d in np.unique(days):
        m = days != d
        if m.sum() > 0:
            out[str(int(d))] = round(float(gated[m].mean()), 8)
    return out


def main() -> None:
    bars_map = {s: load_minutes(s) for s in UNIVERSE
                if os.path.exists(os.path.join(TICKS, f"{s}.pqt"))}
    print(f"loaded {len(bars_map)} pairs")
    clock = hour_activity(bars_map)
    print("CLOCK peak_hours(server/utc-equiv):",
          clock["peak_hours"], clock["peak_utc_equiv"])
    pools = {s: {H: day_pool(bars_map, s, H) for H in HORIZONS}
             for s in bars_map}
    results = {"clock": clock, "fixes": {k: v["primary"] for k, v in FIXES.items()},
               "tiers": ["T0", "T1", "T2"], "cells": {}}
    for fam, cfg in FIXES.items():
        print(f"\n== {fam} (primary F={cfg['primary']}, grid={cfg['grid']}) ==")
        print(f"{'F':>2} {'H':>3} | " + "  ".join(
            f"T{t} mean/z(n)" for t in [0, 1, 2]))
        for F in cfg["grid"]:
            for H in HORIZONS:
                # pool across pairs for the null, matched to this H
                pool_all = np.concatenate([pools[s][H] for s in bars_map])
                key = f"{fam}_F{F}_H{H}"
                cell = {}
                for tier in ["T0", "T1", "T2"]:
                    per, summ, gated, days = fix_events(
                        bars_map, fam, F, H, tier, pools)
                    n_sig = summ["n"]
                    nul = null_stats(pool_all, n_sig, signed=(tier != "T0"))
                    z = ((summ["mean"] - nul["mean"]) / nul["sd"]
                         if summ["mean"] is not None and nul["sd"] else None)
                    l = lodo(gated, days) if (z is not None and abs(z) >= Z_LODO
                                              and len(days) > 2) else None
                    cell[tier] = {"n": n_sig, "mean": summ["mean"],
                                  "z": round(float(z), 2) if z is not None else None,
                                  "lodo": l,
                                  "per_symbol": per}
                results["cells"][key] = cell
                row = []
                for t in [0, 1, 2]:
                    m, z, n = cell[f"T{t}"]["mean"], cell[f"T{t}"]["z"], cell[f"T{t}"]["n"]
                    ms = "-" if m is None else f"{m:+.6f}"
                    zs = "-" if z is None else f"{z:+.1f}"
                    row.append(f"{ms}/{zs}({n})")
                print(f"{F:>2} {H:>3} | " + "  ".join(f"{r:>20}" for r in row))
    # per-pair pips table for the PRIMARY cells (T1/T2) only
    print("\n== per-pair T1/T2 gated mean (pips) @ primary F ==")
    for fam, cfg in FIXES.items():
        F = cfg["primary"]
        print(f"  {fam} F={F}: " + "  ".join(f"{s:>6}" for s in bars_map))
        for H in HORIZONS:
            for t in [1, 2]:
                c = results["cells"][f"{fam}_F{F}_H{H}"][f"T{t}"]
                cells_ = [f"{'--':>6}" if p["pips"] is None else f"{p['pips']:+.2f}"
                          for p in c["per_symbol"]]
                print(f"    T{t} H{H:>2}: " + "  ".join(cells_))
    out_path = os.path.join(HERE, "results", "fix_study.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()