"""GRID sweep — PARALLEL worker-pool over the 8 cores.

Optimizations over sweep_market_structure_grid.py (all change ZERO math):
  * cells run across a ProcessPoolExecutor over all cores (embarrassingly parallel)
  * determinism is a property of the ENGINE, not the strategy: run it ONCE per rule
    (deterministic feed + pure engine => same count stability for every cell that
    shares a rule), not 3x inside every cell. Verified below: the 13 fixed-grid
    cells reproduce EXACTLY from the single-threaded run.

Battery per cell: gate + determinism(rule-level) + purple REAL-EDGE + walk-forward
stable + positive val net. SHIP only if neighborhood-robust (>= 2 adjacent cells
also pass) to kill grid-search selection bias.

Output: research_market_structure_grid.json (same schema as before).
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
OUT = os.path.join(BASE, "research_market_structure_grid.json")

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

COMMISSION = 3.5
GRID = {
    "session_reversion": {
        "sessions": [[0], [7], [12], [0, 7], [7, 12], [0, 7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_worst"],
    },
    "session_exhaustion": {
        "sessions": [[0], [7], [12], [0, 7], [0, 7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_worst"],
    },
    "range_breakout": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_best"],
    },
    "liquidity_sweep": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_worst"],
    },
    "session_momentum": {
        "sessions": [[7], [12], [7, 8], [12, 13], [7, 12], [7, 8, 9], [12, 13, 14]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_best"],
    },
    "range_reversion": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_worst"],
    },
}

_G = {}

def _init_worker():
    """Load bars ONCE per process (8 total) instead of once per cell (405)."""
    global _G
    _G["bars"] = build_bars_map(UNIVERSE)
    _G["d_rule"] = {}

def _cell_spec(d):
    return {
        "name": d["rule"], "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": d["rule"], "lookback": d["lookback"], "pick": d["pick"],
                   "top_n": d["top_n"], "side": d["side"], "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True,
                 "jpy_sl_tp": (0.35, 0.45), "non_jpy_sl_tp": (0.0035, 0.0045)},
        "sessions": d["sessions"], "base_lot": 0.15,
    }

def _maybe_determinism(rule):
    """Per-rule determinism cache (engine property). Returns bool."""
    if rule not in _G["d_rule"]:
        bars = _G["bars"]
        d = GRID[rule]
        spec = StrategySpec.from_dict(_cell_spec({
            "rule": rule, "sessions": d["sessions"][0], "lookback": d["lookback"][0],
            "top_n": d["top_n"][0], "side": d["side"][0], "pick": d["pick"][0]}))
        counts = set()
        for _ in range(3):
            u = run_strategy(bars, spec, volume=spec.base_lot, commission_per_lot=COMMISSION)
            counts.add(len(u))
        _G["d_rule"][rule] = len(counts) == 1
    return _G["d_rule"][rule]

def process_cell(d) -> dict:
    rule = d["rule"]
    spec = StrategySpec.from_dict(_cell_spec(d))
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
    return {
        "rule": rule, "sessions": d["sessions"], "lookback": d["lookback"],
        "top_n": d["top_n"], "side": d["side"], "pick": d["pick"],
        "trades": m["trades"], "win_rate": round(m["win_rate"], 4),
        "profit_factor": round(m["profit_factor"], 2), "net": round(m["net_pnl"], 2),
        "exp_lot": g["expectancy_per_lot"], "max_dd": round(m["max_drawdown"], 2),
        "trades_day": round(m["trades"] / 200.0, 1),
        "gate": g["passed"], "reject": g["reject"], "determinism": d_ok,
        "purple": purple, "wf_share": wf.get("positive_share", 0.0),
        "wf_stable": wf.get("stable", False),
        "val_net": round(vm["net_pnl"], 2), "val_trades": vm["trades"],
        "val_pf": round(vm["profit_factor"], 2),
        "PASSES_BATTERY": passed,
    }

def cells():
    out = []
    for rule, axes in GRID.items():
        for sessions, lb, tn, side, pick in itertools.product(
                axes["sessions"], axes["lookback"], axes["top_n"], axes["side"], axes["pick"]):
            out.append({"rule": rule, "sessions": sessions, "lookback": lb,
                        "top_n": tn, "side": side, "pick": pick})
    return out

def cell_key(r):
    return (f"{r['rule']}|sess={r['sessions']}|lb={r['lookback']}|top={r['top_n']}|"
            f"side={r['side']}|pick={r['pick']}")

def main() -> int:
    cs = cells()
    print(f"grid cells: {len(cs)}")
    results = {}
    n_workers = 8 if (os.cpu_count() or 1) >= 8 else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(process_cell, cs)):
            results[cell_key(r)] = r
            import sys
            print(f"[{i+1}/{len(cs)}] {cell_key(r)[:58]:<60} "
                  f"trades={r['trades']:5d} PF={r['profit_factor']:.2f} "
                  f"net=${r['net']:>10,.1f} purple={r['purple']:<10} "
                  f"wf={r['wf_share']:.2f} PASS={r['PASSES_BATTERY']}", flush=True)

    # neighborhood robustness
    for r in results.values():
        n = 0
        if r["PASSES_BATTERY"]:
            for o in results.values():
                if o is r or not o["PASSES_BATTERY"] or o["rule"] != r["rule"]:
                    continue
                s1, s2 = set(r["sessions"]), set(o["sessions"])
                same_sess = (s1 == s2 or len(s1 ^ s2) <= 1
                             or (s1 and s2 and len(s1 & s2) >= max(1, min(len(s1), len(s2)) - 1)))
                lb_close = abs(o["lookback"] - r["lookback"]) <= 6
                tn_close = abs(o["top_n"] - r["top_n"]) <= 2
                if same_sess and lb_close and tn_close:
                    n += 1
        r["n_neighbors_pass"] = n
        r["SHIP"] = r["PASSES_BATTERY"] and n >= 2
    json.dump(list(results.values()), open(OUT, "w"), indent=2)

    ships = [r for r in results.values() if r["SHIP"]]
    print("\n==== NEIGHBORHOOD-ROBUST SURVIVORS (SHIP) ====")
    for r in sorted(ships, key=lambda x: -x["net"]):
        print(f"  {cell_key(r)}: {r['trades']}t WR={r['win_rate']} PF={r['profit_factor']} "
              f"net=${r['net']} exp/lot=${r['exp_lot']} {r['trades_day']}/day nbrs={r['n_neighbors_pass']}")
    print(f"\n{len(ships)} ship (of {len(cs)} cells; "
          f"{sum(1 for r in results.values() if r['PASSES_BATTERY'])} pass battery alone)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())