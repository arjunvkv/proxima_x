"""scripts/_absorb/fix_gate2.py — same-tape engine embodiment, raw + costed.

fix_gate.py ran on the audit M5 cache; the A/B study ran on the M1 tick cache.
This reruns the engine on M1-RESAMPLED M5 (same tape as the finding) and prints
RAW pnl (points, no costs) alongside costed net, for the two primary fix hours
over all holds — settle exactly what the engine-embodied long-continuation
edge is worth before the verdict.

M1 -> M5: group by ts//300; open first, high max, low min, close last.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proxima_ops.backtest import StrategySpec, run_strategy

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, "results", "ticks")
JUPY = ["USDJPY", "EURJPY", "GBPJPY"]
POINT = {"USDJPY": 1e-2, "EURJPY": 1e-2, "GBPJPY": 1e-2}
SPREAD_TYPICAL = {"USDJPY": 1.2, "EURJPY": 2.2, "GBPJPY": 3.0}
SPREAD_MEASURED = {"USDJPY": 3.5, "EURJPY": 5.7, "GBPJPY": 7.1}
HOLDS = {15: 3, 30: 6, 60: 12}


def resample_m5(sym: str) -> list[dict]:
    df = pl.read_parquet(os.path.join(TICKS, f"{sym}.pqt")).sort("ts")
    ts = df["ts"].to_numpy().astype("int64")
    op = df["open"].to_numpy().astype("float64")
    hi = df["high"].to_numpy().astype("float64")
    lo = df["low"].to_numpy().astype("float64")
    cl = df["close"].to_numpy().astype("float64")
    g = ts // 300
    bars = []
    for k in np.unique(g):
        m = g == k
        bars.append({"ts": int(ts[m][0]), "open": float(op[m][0]),
                     "high": float(hi[m].max()), "low": float(lo[m].min()),
                     "close": float(cl[m][-1])})
    return bars


def spec(name, F, hold_bars):
    return StrategySpec.from_dict({
        "name": name, "universe": JUPY, "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_momentum", "lookback": 6, "pick": "n_best",
                   "top_n": 3, "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold_bars, "stop_first": True},
        "sessions": [F], "base_lot": 0.15})


def main():
    bars = {s: resample_m5(s) for s in JUPY}
    print("M5 bars (resampled from M1):", {s: len(b) for s, b in bars.items()})
    res = {}
    for fam, F in [("tokyo", 3), ("wmr", 17)]:
        for hmin, hb in HOLDS.items():
            key = f"{fam}_F{F}_H{hmin}"
            raw = run_strategy(bars, spec(f"{key}", F, hb), volume=0.15, raw=True)
            typ = run_strategy(bars, spec(f"{key}", F, hb), volume=0.15,
                               commission_per_lot=3.0, spread_pips_map=SPREAD_TYPICAL)
            meas = run_strategy(bars, spec(f"{key}", F, hb), volume=0.15,
                                commission_per_lot=3.0, spread_pips_map=SPREAD_MEASURED)
            n = len(raw)
            raw_pips = {}
            for s in JUPY:
                pts = [t["pnl_pts"] for t in raw if t["symbol"] == s]
                raw_pips[s] = round(float(np.mean(pts)) / POINT[s], 2) if pts else None
            net_typ = sum(t["net"] for t in typ)
            net_meas = sum(t["net"] for t in meas)
            res[key] = {"n": n, "raw_pips": raw_pips,
                        "raw_avg_pts": round(float(np.mean([t["pnl_pts"] for t in raw])), 4),
                        "net_typ": round(net_typ, 2), "net_meas": round(net_meas, 2)}
            assert n == len(typ) == len(meas), "trade-count mismatch"
            print(f"{key:>18} n={n:>3} raw_pips={raw_pips} "
                  f"net_typ={net_typ:>8.2f} net_meas={net_meas:>8.2f}")
    with open(os.path.join(HERE, "results", "fix_gate2.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()