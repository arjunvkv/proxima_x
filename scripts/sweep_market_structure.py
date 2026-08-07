"""Sweep — validate every web-derived market-structure archetype through the engine's
full anti-lookahead / anti-overfit battery; report ONLY proven survivors.

For each candidate spec: raw run -> metrics -> gate -> determinism -> purple
(shuffle-edge) -> train/val -> walk-forward. A candidate ships only if it passes
the WHOLE battery (gate + determinism + REAL-EDGE + stable walk-forward + positive
out-of-sample val). Compared against the proven Tokyo_H0 baseline (which must still
pass as a sanity check for the harness).

The web research (see fx_market_structure_strategies.md) is deliberately NOT
trusted: most retail "edges" conflict or die to costs. Only battery-PASSING
survivors are reported. Every strategy below is a *real market-structure* rule
(open-range breakout, kill-zone momentum, liquidity-sweep rejection, session
reversion, prior-range S/R).
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, determinism, split_by_ts,
                                  walk_forward)
from proxima_ops.backtest.feed import build_bars_map

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "research_market_structure_results.json")

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

# Same conservative per-lot commission the audit gate used (keep comparable to the
# validated baseline curve).
COMMISSION = 3.5

# --------------------------------------------------------------------------
# Candidate specs — each maps 1:1 to a web-derived market-structure archetype.
# sessions = UTC hours that fire. exit per symbol via jpy/non_jpy distances.
# --------------------------------------------------------------------------
def cand(rule, sessions, name, pick="n_worst", side="long", lookback=6, top_n=5,
         hold=12, jpy=(0.35,0.45), nonjpy=(0.0035,0.0045), comment=""):
    return {
        "name": name, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": rule, "lookback": lookback, "pick": pick,
                   "top_n": top_n, "side": side, "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True,
                 "jpy_sl_tp": jpy, "non_jpy_sl_tp": nonjpy},
        "sessions": sessions, "base_lot": 0.15, "comment": comment,
    }

CANDIDATES = [
    # --- baseline must pass (sanity) ---
    cand("session_exhaustion", [0], "BASELINE_Tokyo_h0_fade",
         comment="audit-validated Tokyo hour-0 fade"),
    # --- archetype: open-range / breakout ---
    cand("range_breakout", [7], "London_AsianRangeBreakout",
         side="both", comment="break of trailing London-open range (ORB)"),
    cand("range_breakout", [12], "NY_OpenRangeBreakout",
         side="both", comment="NY open-range breakout"),
    # --- archetype: kill-zone momentum (London/NY session) ---
    cand("session_momentum", [7], "London_KillZone_Momentum",
         pick="n_best", side="long", comment="ride London-open first-bar direction"),
    cand("session_momentum", [12], "NY_KillZone_Momentum",
         pick="n_best", side="long", comment="ride NY-open first-bar direction"),
    cand("session_momentum", [7], "London_KillZone_Both",
         pick="n_best", side="both", comment="London-open momentum long+short"),
    # --- archetype: liquidity sweep / stop-hunt rejection ---
    cand("liquidity_sweep", [7], "London_SweepRejection",
         side="both", comment="wick beyond trailing low/high then reclaim"),
    cand("liquidity_sweep", [12], "NY_SweepRejection",
         side="both", comment="NY sweep-rejection"),
    # --- archetype: session mean-reversion (VWAP-style) ---
    cand("session_reversion", [0], "Tokyo_H0_SessionReversion",
         side="long", comment="fade overextension vs session-avg anchor"),
    cand("session_reversion", [7], "London_SessionReversion",
         side="both", comment="VWAP-style reversion at London open"),
    cand("session_reversion", [12], "NY_SessionReversion",
         side="both", comment="VWAP-style reversion at NY open"),
    # --- archetype: fade the open-range extreme ---
    cand("range_reversion", [7], "London_ORB_Fade",
         side="both", comment="fade extension beyond trailing range"),
    cand("range_reversion", [12], "NY_ORB_Fade",
         side="both", comment="fade extension beyond trailing NY range"),
]

def run_candidate(bars, spec) -> dict:
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
    return {
        "name": spec.name, "trades": m["trades"], "win_rate": round(m["win_rate"], 4),
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

def main() -> int:
    bars = build_bars_map(UNIVERSE)
    results = []
    for c in CANDIDATES:
        spec = StrategySpec.from_dict(c)
        res = run_candidate(bars, spec)
        results.append(res)
        flag = "PASS" if res["PASSES_BATTERY"] else "FAIL"
        print(f"[{flag}] {res['name']:<32} trades={res['trades']:5d} WR={res['win_rate']:.3f} "
              f"PF={res['profit_factor']:.2f} net=${res['net']:>10,.1f} exp/lot=${res['exp_lot']:>7,.1f} "
              f"trades/day={res['trades_day']:>4.1f} purple={res['purple']} wf={res['wf_share']:.2f} "
              f"val={res['val_net']:>9,.0f}")
    json.dump(results, open(OUT, "w"), indent=2)
    passed = [r for r in results if r["PASSES_BATTERY"]]
    print("\n==== BATTERY SURVIVORS ====")
    for r in passed:
        print(f"  {r['name']}: {r['trades']}t, WR {r['win_rate']}, PF {r['profit_factor']}, "
              f"net ${r['net']}, exp/lot ${r['exp_lot']}, {r['trades_day']}/day")
    print(f"\n{len(passed)}/{len(results)} pass the whole battery.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())