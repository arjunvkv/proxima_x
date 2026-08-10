"""scripts/_absorb/fix_gate.py — grid gate for the fixing-window candidate (research only).

Engine-grade test of the fix-study finding (continuation of the pre-fix move in
JPY crosses at the true fix hours, see fix_study.py) using the REAL engine path:

  rule = session_momentum (legacy byte-parity path, BUY-long continuation),
  lookback = 6 M5 bars (30 min — the pre-fix direction window, ends at the
  fix-hour bar close), sessions = [F] (server-clock fix hour), fill_bar = 1
  -> entry at F+5 bar OPEN (identical to the study's ENTRY_GAP=5),
  exit = sl_tp_hold, hold_bars {3,6,12} min, stop_first True (MT5 conservative;
  jpy SL/TP 35/45 pips engine defaults) — plus a hold_only reference arm to
  isolate how much of the raw edge dies to the engine's stop-first execution.

Costs: commission 3.0/lot + spread maps (measured = FTMO-demo worst-case,
typical = busy-session). Universe = JPY crosses (the stratum that carried the
finding), top_n=3 (trades all three). Neighbor-hour grid cells serve as
controls (fix study showed ~null there).

Apples-to-apples: bars sliced to the exact study window (minute-cache range),
so the gate evaluates the SAME 43 trading days as fix_study.py.

Run:  unset PYTHONPATH && ./.venv/Scripts/python.exe -u scripts/_absorb/fix_gate.py
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict

from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

HERE = os.path.dirname(os.path.abspath(__file__))
JUPY = ["USDJPY", "EURJPY", "GBPJPY"]

SPREAD_MEASURED = {"EURUSD": 0.60, "USDJPY": 3.50, "GBPUSD": 1.30, "AUDUSD": 0.90,
                   "EURJPY": 5.70, "GBPJPY": 7.10, "USDCAD": 1.60}
SPREAD_TYPICAL = {"EURUSD": 0.8, "USDJPY": 1.2, "GBPUSD": 1.5, "AUDUSD": 1.1,
                  "EURJPY": 2.2, "GBPJPY": 3.0, "USDCAD": 1.8}

F_GRID = {"tokyo_fix": [1, 2, 3, 4, 5], "wmr_fix": [15, 16, 17, 18, 19]}
HOLDS = {15: 3, 30: 6, 60: 12}  # minutes -> M5 bars


def window() -> tuple[int, int]:
    w = json.load(open(os.path.join(HERE, "results", "ticks", "_window.json")))
    return int(w["from_s"]), int(w["to_s"])


def spec(name: str, F: int, hold_bars: int, mode: str) -> StrategySpec:
    return StrategySpec.from_dict({
        "name": name, "universe": JUPY,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_momentum", "lookback": 6, "pick": "n_best",
                   "top_n": 3, "side": "both", "fill_bar": 1},
        "exit": {"mode": mode, "hold_bars": hold_bars, "stop_first": True},
        "sessions": [F], "base_lot": 0.15,
    })


def summarize(trades, label, spread_map, lot):
    day = defaultdict(float)
    for t in trades:
        day[t["entry_ts"] // 86400] += t["net"]
    tot = sum(day.values())
    pos = sum(v for v in day.values() if v > 0)
    neg = -sum(v for v in day.values() if v < 0)
    n = len(trades)
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1
    gross = sum(t["gross_usd"] for t in trades) if trades else 0.0
    comm = sum(t["commission"] for t in trades) if trades else 0.0
    spr = sum(t.get("spread", 0.0) for t in trades) if trades else 0.0
    mad = max(day.values()) if day else 0.0
    mid = min(day.values()) if day else 0.0
    return {"label": label, "trades": n, "gross": round(gross, 2),
            "comm": round(comm, 2), "spread": round(spr, 2),
            "net": round(tot, 2), "avg_per_trade": round(tot / n, 2) if n else None,
            "pf": round(pos / neg, 2) if neg > 0 else None,
            "wr": round(sum(1 for t in trades if t["net"] > 0) / n, 3) if n else None,
            "stop_first_hits": reasons["SL"] + reasons["SL-sto"],
            "tp_hits": reasons["TP"], "hold_exits": reasons["HOLD"],
            "green_days": sum(1 for v in day.values() if v > 0), "total_days": len(day),
            "worst_day": round(mid, 2), "best_day": round(mad, 2)}


def lodo_days(trades) -> dict:
    day = defaultdict(list)
    for t in trades:
        day[t["entry_ts"] // 86400].append(t["net"])
    base = sum(map(sum, day.values()))
    out = {}
    for d, nets in day.items():
        out[str(int(d))] = round(base - sum(nets), 2)
    return out


def main() -> None:
    w0, w1 = window()
    bars = {s: [b for b in build_bars_map(JUPY)[s] if w0 <= b["ts"] <= w1]
            for s in JUPY}
    print(f"window {w0}..{w1} bars:", {s: len(b) for s, b in bars.items()})
    results = {"window": [w0, w1], "cells": {}}
    for fam, hours in F_GRID.items():
        print(f"\n== {fam} ==")
        for F in hours:
            for hmin, hb in HOLDS.items():
                key = f"{fam}_F{F}_H{hmin}"
                row = {}
                for mode in ("sl_tp_hold", "hold_only"):
                    for spr_lab, spmap in [("MEAS", SPREAD_MEASURED),
                                           ("TYP", SPREAD_TYPICAL)]:
                        t = run_strategy(bars, spec(f"{key}_{mode}", F, hb, mode),
                                         volume=0.15, commission_per_lot=3.0,
                                         spread_pips_map=spmap)
                        s = summarize(t, f"{key} {mode} {spr_lab}", spmap, 0.15)
                        lodo = lodo_days(t) if len(t) >= 3 else None
                        if lodo:
                            vals = list(lodo.values())
                            s["lodo"] = {"min": min(vals), "max": max(vals),
                                         "flips": sum(1 for v in vals
                                                      if (v < 0) != (s["net"] < 0))}
                        row[f"{mode}_{spr_lab}"] = s
                results["cells"][key] = row
                c = row["sl_tp_hold_MEAS"]
                c2 = row["hold_only_TYP"]
                print(f" F{F:>2} H{hmin:>2} | "
                      f"SL/TP MEAS n={c['trades']:>3} net={c['net']:>8.2f} "
                      f"avg={c['avg_per_trade'] if c['avg_per_trade'] is not None else 0:>7.2f} "
                      f"PF={c['pf'] if c['pf'] is not None else 0:>5.2f} "
                      f"SLhits={c['stop_first_hits']:>3} | "
                      f"holdonly TYP net={c2['net']:>8.2f}")
    out_path = os.path.join(HERE, "results", "fix_gate.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()