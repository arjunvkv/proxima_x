"""GRID sweep — search parameter space per market-structure archetype, but ship only
battery-survivors that are NEIGHBORHOOD-ROBUST (adjacent parameterizations also pass),
so picking the best grid point cannot be curve-fitting.

Battery per candidate (identical to sweep_market_structure.py):
  gate + determinism + purple REAL-EDGE + walk-forward stable + positive val net.

ROBUSTNESS: a candidate's config lives in a grid (rule x sessions x lookback x
top_n x side). It ships only if >= 2 OTHER cells in its immediate neighborhood
(same rule; sessions/lookback/top_n within +/-1 step) ALSO pass the battery.
This kills the classic "grid-search then report the winner" selection bias.

Output: research_market_structure_grid.json with every candidate + verdicts.
"""
from __future__ import annotations
import sys, os, json, itertools
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

COMMISSION = 3.5   # audit-conservative, same as validated baseline

# --- grid axes per archetype (rule) ---
GRID = {
    # mean-reversion fade of session overextension (VWAP-style anchor)
    "session_reversion": {
        "sessions": [[0], [7], [12], [0, 7], [7, 12], [0, 7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_worst"],
    },
    # cross-sectional return fade (Tokyo baseline rule)
    "session_exhaustion": {
        "sessions": [[0], [7], [12], [0, 7], [0, 7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_worst"],
    },
    # open-range / level breakout (momentum)
    "range_breakout": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_best"],
    },
    # liquidity sweep / stop-hunt rejection
    "liquidity_sweep": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_worst"],
    },
    # first-bar session momentum (kill-zone direction ride)
    "session_momentum": {
        "sessions": [[7], [12], [7, 8], [12, 13], [7, 12], [7, 8, 9], [12, 13, 14]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["long", "both"],
        "pick": ["n_best"],
    },
    # fade extension beyond trailing range (ORB fade)
    "range_reversion": {
        "sessions": [[7], [12], [7, 12]],
        "lookback": [6, 12, 24], "top_n": [3, 5, 8], "side": ["both"],
        "pick": ["n_worst"],
    },
}

def cell_id(spec) -> str:
    s = spec["signal"]
    return (f"{spec['name']}|sess={spec['sessions']}|lb={s['lookback']}|"
            f"top={s['top_n']}|side={s['side']}|pick={s['pick']}")

def run_candidate(bars, spec, out) -> dict:
    usd = run_strategy(bars, spec, volume=spec.base_lot, commission_per_lot=COMMISSION)
    m = metrics(usd)
    g = gate(m, lot=spec.base_lot)
    d_ok = determinism(lambda: run_strategy(bars, spec, volume=spec.base_lot,
                                            commission_per_lot=COMMISSION))
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=spec.base_lot,
                                                       commission_per_lot=COMMISSION),
                         m["expectancy"] / spec.base_lot, iters=5)
    tr, va = split_by_ts(usd)
    wf = walk_forward(usd, train_size=120, test_size=60, lot=spec.base_lot)
    vm = metrics(va)
    passed = (g["passed"] and d_ok and purple == "REAL-EDGE"
              and wf["stable"] and vm["net_pnl"] > 0)
    out["name"] = spec.name
    out["trades"] = m["trades"]; out["win_rate"] = round(m["win_rate"], 4)
    out["profit_factor"] = round(m["profit_factor"], 2)
    out["net"] = round(m["net_pnl"], 2); out["exp_lot"] = g["expectancy_per_lot"]
    out["max_dd"] = round(m["max_drawdown"], 2)
    out["trades_day"] = round(m["trades"] / 200.0, 1)
    out["gate"] = g["passed"]; out["reject"] = g["reject"]
    out["determinism"] = d_ok; out["purple"] = purple
    out["wf_share"] = wf.get("positive_share", 0.0)
    out["wf_stable"] = wf.get("stable", False)
    out["val_net"] = round(vm["net_pnl"], 2); out["val_trades"] = vm["trades"]
    out["val_pf"] = round(vm["profit_factor"], 2)
    out["PASSES_BATTERY"] = passed
    return out

def neighbors_pass(cell: dict, results_by_id: dict) -> int:
    """Count battery-passing cells in the immediate neighborhood: same rule,
    sessions equal OR one step away, lookback and top_n within one grid step."""
    cid = cell_id(cell)
    c = results_by_id[cid]
    if not c["PASSES_BATTERY"]:
        return 0
    n = 0
    for oid, o in results_by_id.items():
        if oid == cid or not o["PASSES_BATTERY"]:
            continue
        if o["name"] != cell["name"]:
            continue
        # sessions within one shared hour or adjacent list
        s1, s2 = set(cell["sessions"]), set(o["sessions"])
        same_sess = (s1 == s2 or len(s1 ^ s2) <= 1
                     or (s1 and s2 and len(s1 & s2) >= max(1, min(len(s1), len(s2)) - 1)))
        if not same_sess:
            continue
        lb_close = abs(o["lookback"] - cell["signal"]["lookback"]) <= 6
        tn_close = abs(o["top_n"] - cell["signal"]["top_n"]) <= 2
        if lb_close and tn_close:
            n += 1
    return n

def main() -> int:
    bars = build_bars_map(UNIVERSE)
    cells = []
    for rule, axes in GRID.items():
        for sessions, lb, tn, side, pick in itertools.product(
                axes["sessions"], axes["lookback"], axes["top_n"], axes["side"], axes["pick"]):
            cells.append({
                "name": f"{rule}", "universe": UNIVERSE,
                "feed": {"kind": "bar", "timeframe": "M5"},
                "signal": {"rule": rule, "lookback": lb, "pick": pick,
                           "top_n": tn, "side": side, "fill_bar": 1},
                "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True,
                         "jpy_sl_tp": (0.35, 0.45), "non_jpy_sl_tp": (0.0035, 0.0045)},
                "sessions": sessions, "base_lot": 0.15,
            })
    print(f"grid cells: {len(cells)}")
    results = {}
    for idx, cell in enumerate(cells):
        spec = StrategySpec.from_dict(cell)
        out = {"cell": cell_id(cell)}
        run_candidate(bars, spec, out)
        results[out["cell"]] = out
        print(f"[{idx+1}/{len(cells)}] {out['cell'][:60]:<62} "
              f"trades={out['trades']:5d} PF={out['profit_factor']:.2f} "
              f"net=${out['net']:>10,.1f} purple={out['purple']:<10} "
              f"wf={out['wf_share']:.2f} PASS={out['PASSES_BATTERY']}")
    # neighborhood robustness
    for cell in cells:
        cid = cell_id(cell)
        results[cid]["n_neighbors_pass"] = neighbors_pass(cell, results)
        results[cid]["SHIP"] = results[cid]["PASSES_BATTERY"] and results[cid]["n_neighbors_pass"] >= 2
    json.dump(list(results.values()), open(OUT, "w"), indent=2)

    ships = [r for r in results.values() if r["SHIP"]]
    print("\n==== NEIGHBORHOOD-ROBUST BATTERY SURVIVORS (SHIP) ====")
    for r in sorted(ships, key=lambda x: -x["net"]):
        print(f"  {r['cell']}: {r['trades']}t WR={r['win_rate']} PF={r['profit_factor']} "
              f"net=${r['net']} exp/lot=${r['exp_lot']} {r['trades_day']}/day "
              f"nbrs={r['n_neighbors_pass']}")
    print(f"\n{len(ships)} ship (of {len(cells)} cells; {sum(1 for r in results.values() if r['PASSES_BATTERY'])} pass battery alone)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())