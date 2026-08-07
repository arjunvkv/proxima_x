"""JOURNAL round 2 — validate NON-hour-0 journal FX strategies (distinct market-
structure classes, none of which fade the Tokyo0 / UTC hour-0 open) through the
full engine battery: gate + determinism + purple REAL-EDGE + shuffle + split +
walk-forward windowing. Ships only battery-proven cells; any hour-0 cell is
flagged a CONFound, never shipped.

Output: validation_journal_round2.json
"""
from __future__ import annotations
import sys, os, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, determinism, split_by_ts,
                                  walk_forward)
from proxima_ops.backtest.feed import build_bars_map

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "validation_journal_round2.json")
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
COMMISSION = 3.5

# static monthly carry proxy: +1 long, -1 short, 0 flat. Carry-clock builds a
# long/short basket (high-yield vs low-yield) carried through the US-hours block.
CARRY_DIR = {"EURUSD": 0, "USDJPY": 1, "GBPUSD": 1, "AUDUSD": 1, "EURJPY": 1,
             "GBPJPY": 1, "EURAUD": -1, "EURNZD": -1, "GBPAUD": -1, "GBPNZD": -1,
             "GBPCAD": 1, "AUDNZD": 0, "USDCAD": 1, "NZDUSD": 1, "EURGBP": -1,
             "EURCHF": 1, "USDCHF": 1, "AUDJPY": 1}

JOURNAL_CIT = {
  "carry_clock": "Krohn-Mueller-Whelan 2024 JF (intraday carry premia)",
  "intraday_momentum_london": "Gao-Han-Li-Zhou 2018 JFE (first-half->last-half hr)",
  "day_of_week_usd": "Khademalomoom-Narayan 2020 EMR (day-of-week USD)",
  "vol_compress_fade": "Andersen-Bollerslev 1998 / vol-gate construction",
  "lead_lag": "Basnarkov et al. 2020 Physica A (EUR lead-lag)",
  "thin_market_fade": "quiet-hours micro-structure fade (ILM 1998)",
}

# (rule, grid, kw, nonjpy_sltp, jpy_sltp)
JOURNAL = [
    ("carry_clock",
     {"sessions": [[13,14,15,16,17,18,19,20], [14,15,16,17,18,19,20]],
      "lookback": [6], "top_n": [6, 8, 10], "side": ["both"],
      "hold": [96, 168], "per_hour": False},
     {"direction_map": CARRY_DIR}, (0.0030, 0.0040), (0.30, 0.40)),
    ("intraday_momentum_london",
     {"sessions": [[15], [8, 15]], "lookback": [12], "top_n": [5, 8],
      "side": ["both"], "hold": [6, 12], "per_hour": False},
     {}, (0.0025, 0.0035), (0.25, 0.35)),
    ("day_of_week_usd",
     {"sessions": [[11,12,13,14,15,16,17,18], [7,8,9,10,11,12,13,14,15,16]],
      "lookback": [6], "top_n": [5, 8, 12], "side": ["both"],
      "hold": [6, 12], "per_hour": True, "weekdays": [[0, 1]]},
     {}, (0.0025, 0.0030), (0.25, 0.30)),
    ("vol_compress_fade",
     {"sessions": [[8,9,10,11,12], [13,14,15,16]], "lookback": [36, 60],
      "top_n": [3, 5, 8], "side": ["both"], "hold": [12, 24], "per_hour": True},
     {}, (0.0025, 0.0035), (0.25, 0.35)),
    ("lead_lag",
     {"sessions": [[8,9,10,11], [13,14,15,16,17]], "lookback": [6],
      "top_n": [4, 6, 8], "side": ["both"], "hold": [6, 12], "per_hour": True},
     {}, (0.0020, 0.0030), (0.20, 0.30)),
    ("thin_market_fade",
     {"sessions": [[2,3,4], [1,2,3,4,5], [4,5]], "lookback": [24],
      "top_n": [3, 5, 8], "side": ["both"], "hold": [6, 12], "per_hour": True},
     {}, (0.0018, 0.0025), (0.18, 0.25)),
]

JOURNAL_MAP = {r: (g, o, n, j) for r, g, o, n, j in JOURNAL}

_G = {}
def _init_worker():
    _G["bars"] = build_bars_map(UNIVERSE)
    _G["det"] = {}

def _mk_spec(rule, sess, lb, tn, side, hold, per_hour, wds, nonjpy, jpy):
    ss = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": tn,
          "side": side, "fill_bar": 1 if rule != "intraday_momentum_london" else 6,
          "per_hour": per_hour}
    if rule == "carry_clock":
        ss["direction_map"] = CARRY_DIR
    spec = {"name": rule, "universe": UNIVERSE,
            "feed": {"kind": "bar", "timeframe": "M5"}, "signal": ss,
            "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True,
                     "jpy_sl_tp": jpy, "non_jpy_sl_tp": nonjpy},
            "sessions": sess, "base_lot": 0.15}
    if wds is not None:
        spec["weekdays"] = wds
    return spec

def _maybe_determinism(rule):
    if rule not in _G["det"]:
        g, _, n, j = JOURNAL_MAP[rule]
        spec = StrategySpec.from_dict(_mk_spec(rule, g["sessions"][0], g["lookback"][0],
                                           g["top_n"][0], g["side"][0], g["hold"][0],
                                           g["per_hour"],
                                           (g.get("weekdays") or [None])[0], n, j))
        counts = set()
        for _ in range(3):
            u = run_strategy(_G["bars"], spec, volume=0.15,
                            commission_per_lot=COMMISSION)
            counts.add(len(u))
        _G["det"][rule] = len(counts) == 1
    return _G["det"][rule]

def process_cell(d) -> dict:
    rule = d["rule"]
    spec = StrategySpec.from_dict(_mk_spec(rule, d["sessions"], d["lookback"], d["top_n"],
                                       d["side"], d["hold"], d["per_hour"], d["weekdays"],
                                       d["nonjpy"], d["jpy"]))
    bars = _G["bars"]
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=COMMISSION)
    m = metrics(usd)
    g = gate(m, lot=0.15)
    d_ok = _maybe_determinism(rule)
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=0.15,
                                                       commission_per_lot=COMMISSION),
                         m["expectancy"] / 0.15, iters=5)
    tr, va = split_by_ts(usd)
    wf = walk_forward(usd, train_size=120, test_size=60, lot=0.15)
    vm = metrics(va)
    sess = d["sessions"]
    hour0 = sess is not None and 0 in sess
    passed = (g["passed"] and d_ok and purple == "REAL-EDGE" and not hour0
              and wf["stable"] and vm["net_pnl"] > 0)
    return {"rule": rule, "sessions": sess, "weekdays": d["weekdays"],
            "label": JOURNAL_CIT[rule], "lookback": d["lookback"], "top_n": d["top_n"],
            "side": d["side"], "hold": d["hold"], "trades": m["trades"],
            "win_rate": round(m["win_rate"], 4),
            "profit_factor": round(m["profit_factor"], 2), "net": round(m["net_pnl"], 2),
            "exp_lot": g["expectancy_per_lot"], "max_dd": round(m["max_drawdown"], 2),
            "trades_day": round(m["trades"] / 200.0, 1), "gate": g["passed"],
            "determinism": d_ok, "purple": purple,
            "wf_share": wf.get("positive_share", 0.0), "wf_stable": wf.get("stable", False),
            "val_net": round(vm["net_pnl"], 2), "val_trades": vm["trades"],
            "val_pf": round(vm["profit_factor"], 2), "hour0_confound": hour0,
            "PASSES_BATTERY": passed}

def _cells(rule):
    out = []
    g, _, n, j = JOURNAL_MAP[rule]
    for sess, lb, tn, side, hold, wds in itertools.product(
            g["sessions"], g["lookback"], g["top_n"], g["side"], g["hold"],
            g.get("weekdays") or [None]):
        out.append({"rule": rule, "sessions": sess, "lookback": lb, "top_n": tn,
                    "side": side, "hold": hold, "per_hour": g["per_hour"],
                    "weekdays": wds, "nonjpy": n, "jpy": j})
    return out

def main() -> int:
    cs = []
    for rule in JOURNAL_MAP:
        cs += _cells(rule)
    print(f"round2 grid cells: {len(cs)}")
    res = {}
    n_workers = 8 if (os.cpu_count() or 1) >= 8 else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(process_cell, cs)):
            key = (r["rule"], tuple(r["sessions"]) if r["sessions"] else None,
                   r["top_n"], r["side"], r["hold"])
            res[key] = r
            print(f"[{i+1}/{len(cs)}] {r['rule']:<24} sess={str(r['sessions'])[:30]:<32} "
                  f"t={r['trades']:5d} PF={r['profit_factor']:.2f} net=${r['net']:>8,.0f} "
                  f"purple={r['purple']:<9} conf0={r['hour0_confound']} PASS={r['PASSES_BATTERY']}",
                  flush=True)
    json.dump(list(res.values()), open(OUT, "w"), indent=2)
    ships = [r for r in res.values() if r["PASSES_BATTERY"]]
    print(f"\n{len(ships)} of {len(cs)} pass FULL battery non-hour-0 across {len(JOURNAL)} rules")
    for rule in JOURNAL_MAP:
        sl = [r for r in ships if r["rule"] == rule]
        if sl:
            best = max(sl, key=lambda x: x["net"])
            print(f"  {rule:<22}: {len(sl)} cells pass; best net=${best['net']:,.0f} "
                  f"PF={best['profit_factor']} WR={best['win_rate']} {best['trades']}t "
                  f"{best['trades_day']}/day")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())