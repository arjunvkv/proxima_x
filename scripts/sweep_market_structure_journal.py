"""JOURNAL sweep — validate the peer-reviewed academic FX rules through the full
engine battery (gate + determinism + purple REAL-EDGE + walk-forward + val), shipping
only battery-proven cells. Each rule is a DISTINCT market-structure class from a real
journal (no web comments, no Tokyo0 fade).

Parallel worker-pool over cores; determinism is engine- (per-rule) level.
Output: validation_market_structure_journal.json
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
OUT = os.path.join(BASE, "validation_market_structure_journal.json")
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
COMMISSION = 3.5

JOURNAL_CIT = {
  "intraday_momentum": "Gao-Han-Li-Zhou 2018 JFE / Baltussen 2021 JFE",
  "fix_reversal": "Krohn-Mueller-Whelan 2024 JF",
  "round_barrier_fade": "Osler 2003 JF",
  "domestic_hours": "Ranaldo 2009 JBF",
  "weekend_gap": "Dao-McGroarty-Urquhart 2016 JMMFM",
  "cross_momentum": "Menkhoff et al. 2012 JFE",
  "round_number_bounce": "de Grauwe-Decupere 1992",
  "big_move_fade": "Curtizier-Tandon-Yu 2006 JFM",
}

# rule -> (grid axes, (nonjpy sl, tp), (jpy sl, tp))
JOURNAL = [
    ("intraday_momentum",  # Gao et al. 2018: first half-hour -> last half-hour
     {"sessions": [[7], [0], [7, 0]], "lookback": [6, 12], "top_n": [5, 8],
      "side": ["long", "both"], "hold": [80, 100, 120], "per_hour": False},
     (0.0035, 0.0045), (0.35, 0.45)),
    ("fix_reversal",       # Krohn et al. 2024: post-fix USD depreciation
     {"sessions": [[13, 16]], "lookback": [6], "top_n": [4, 6, 8],
      "side": ["both"], "hold": [6, 12], "per_hour": True},
     (0.0015, 0.0020), (0.15, 0.20)),
    ("round_barrier_fade", # Osler 2003: round-number stop-hunt fade
     {"sessions": [[0], [7], [12], [7, 12], None], "lookback": [6, 12],
      "top_n": [3, 5], "side": ["both"], "hold": [12, 24], "per_hour": True},
     (0.0015, 0.0020), (0.15, 0.20)),
    ("domestic_hours",     # Ranaldo 2009: home-hour depreciation
     {"sessions": [[0,1,2,3,4,5], [7,8,9,10,11], [12,13,14,15,16,17,18,19,20],
                   [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]],
      "lookback": [6], "top_n": [5, 8, 12], "side": ["both"],
      "hold": [6, 12, 24], "per_hour": True},
     (0.0020, 0.0025), (0.20, 0.25)),
    ("weekend_gap",        # Dao et al. 2016: fade extreme weekend gaps
     {"sessions": [[0]], "lookback": [6, 12], "top_n": [3, 5, 8],
      "side": ["both"], "hold": [12, 24, 60], "per_hour": True},
     (0.0040, 0.0050), (0.40, 0.50)),
    ("cross_momentum",     # Menkhoff et al. 2012: 20-day cross-section
     {"sessions": [[0], [7]], "lookback": [5760], "top_n": [5, 10, 18],
      "side": ["both"], "hold": [12, 60, 120], "per_hour": False},
     (0.0030, 0.0040), (0.30, 0.40)),
    ("round_number_bounce",# de Grauwe/Decupere 1992: reversal at round number
     {"sessions": [[7], [12], [7, 12], None], "lookback": [6], "top_n": [3, 5],
      "side": ["both"], "hold": [12], "per_hour": False},
     (0.0015, 0.0020), (0.15, 0.20)),
    ("big_move_fade",      # Curtizier et al. 2006: fade large prior-day move
     {"sessions": [[7], [0]], "lookback": [288, 576], "top_n": [3, 5, 8],
      "side": ["both"], "hold": [12, 24], "per_hour": False},
     (0.0040, 0.0050), (0.40, 0.50)),
]

JOURNAL_MAP = {r: (g, n, j) for r, g, n, j in JOURNAL}

_G = {}

def _init_worker():
    global _G
    _G["bars"] = build_bars_map(UNIVERSE)
    _G["d_rule"] = {}

def _mk_spec(rule, sessions, lb, tn, side, hold, per_hour, nonjpy, jpy):
    ss = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": tn,
          "side": side, "fill_bar": 1, "per_hour": per_hour}
    return {"name": f"{rule}", "universe": UNIVERSE,
            "feed": {"kind": "bar", "timeframe": "M5"}, "signal": ss,
            "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True,
                     "jpy_sl_tp": jpy, "non_jpy_sl_tp": nonjpy},
            "sessions": sessions, "base_lot": 0.15}

def _maybe_determinism(rule):
    if rule not in _G["d_rule"]:
        g, n, j = JOURNAL_MAP[rule]
        spec = StrategySpec.from_dict(_mk_spec(rule, g["sessions"][0], g["lookback"][0],
                                               g["top_n"][0], g["side"][0], g["hold"][0],
                                               g["per_hour"], n, j))
        counts = set()
        for _ in range(3):
            u = run_strategy(_G["bars"], spec, volume=spec.base_lot, commission_per_lot=COMMISSION)
            counts.add(len(u))
        _G["d_rule"][rule] = len(counts) == 1
    return _G["d_rule"][rule]

def process_cell(d) -> dict:
    rule = d["rule"]
    spec = StrategySpec.from_dict(_mk_spec(rule, d["sessions"], d["lookback"], d["top_n"],
                                           d["side"], d["hold"], d["per_hour"],
                                           d["nonjpy"], d["jpy"]))
    bars = _G["bars"]
    usd = run_strategy(bars, spec, volume=spec.base_lot, commission_per_lot=COMMISSION)
    m = metrics(usd)
    g = gate(m, lot=spec.base_lot)
    d_ok = _maybe_determinism(rule)
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=spec.base_lot,
                                                       commission_per_lot=COMMISSION),
                         m["expectancy"] / spec.base_lot, iters=5)
    tr, va = split_by_ts(usd)
    wf = walk_forward(usd, train_size=120, test_size=60, lot=spec.base_lot)
    vm = metrics(va)
    passed = (g["passed"] and d_ok and purple == "REAL-EDGE"
              and wf["stable"] and vm["net_pnl"] > 0)
    return {"rule": rule, "sessions": d["sessions"], "rule_label": d["label"],
            "lookback": d["lookback"], "top_n": d["top_n"], "side": d["side"], "hold": d["hold"],
            "trades": m["trades"], "win_rate": round(m["win_rate"], 4),
            "profit_factor": round(m["profit_factor"], 2), "net": round(m["net_pnl"], 2),
            "exp_lot": g["expectancy_per_lot"], "max_dd": round(m["max_drawdown"], 2),
            "trades_day": round(m["trades"] / 200.0, 1), "gate": g["passed"],
            "reject": g["reject"], "determinism": d_ok, "purple": purple,
            "wf_share": wf.get("positive_share", 0.0), "wf_stable": wf.get("stable", False),
            "val_net": round(vm["net_pnl"], 2), "val_trades": vm["trades"],
            "val_pf": round(vm["profit_factor"], 2), "PASSES_BATTERY": passed}

def cells():
    out = []
    for rule, (g, nonjpy, jpy) in JOURNAL_MAP.items():
        for sessions, lb, tn, side, hold in itertools.product(
                g["sessions"], g["lookback"], g["top_n"], g["side"], g["hold"]):
            out.append({"rule": rule, "sessions": sessions, "lookback": lb,
                        "top_n": tn, "side": side, "hold": hold,
                        "per_hour": g["per_hour"], "nonjpy": nonjpy, "jpy": jpy,
                        "label": JOURNAL_CIT[rule]})
    return out

def main() -> int:
    cs = cells()
    print(f"journal grid cells: {len(cs)}")
    res = {}
    n_workers = 8 if (os.cpu_count() or 1) >= 8 else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(process_cell, cs)):
            key = (r["rule"], tuple(r["sessions"]) if r["sessions"] else None,
                   r["lookback"], r["top_n"], r["side"], r["hold"])
            res[key] = r
            print(f"[{i+1}/{len(cs)}] {r['rule']:<20} sess={str(r['sessions'])[:26]:<28} "
                  f"t={r['trades']:5d} PF={r['profit_factor']:.2f} net=${r['net']:>9,.0f} "
                  f"purple={r['purple']:<9} wf={r['wf_share']:.2f} PASS={r['PASSES_BATTERY']}",
                  flush=True)
    json.dump(list(res.values()), open(OUT, "w"), indent=2)
    ships = [r for r in res.values() if r["PASSES_BATTERY"]]
    print(f"\n{len(ships)} of {len(cs)} pass FULL battery across {len(JOURNAL)} journal rules")
    for rule, _, _, _ in JOURNAL:
        sl = [r for r in ships if r["rule"] == rule]
        if sl:
            best = max(sl, key=lambda x: x["net"])
            print(f"  {rule:<20}: {len(sl)} cells pass; best net=${best['net']:,.0f} "
                  f"PF={best['profit_factor']} WR={best['win_rate']} {best['trades']}t "
                  f"{best['trades_day']}/day")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())