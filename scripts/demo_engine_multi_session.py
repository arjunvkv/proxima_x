"""Demo: emit live manifest for Tokyo_H0 + run a multi-session hypothesis spec
through the generalized engine. Reports events/day and edge metrics per spec."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, walk_forward, split_by_ts)
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.liveport import emit_live_manifest

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

FTMO_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"

def base_spec(name, sessions, top_n=5, side="long"):
    return StrategySpec.from_dict({
        "name": name, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_exhaustion", "lookback": 6, "pick": "n_worst",
                   "top_n": top_n, "side": side, "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True},
        "sessions": sessions, "base_lot": 0.15,
    })

def report(bars, spec, volume=0.15):
    usd = run_strategy(bars, spec, volume=volume)
    m = metrics(usd)
    g = gate(m, lot=volume)
    days = len({t["entry_ts"] // 86400 for t in usd}) if usd else 1
    wf = walk_forward(usd, train_size=300, test_size=100, lot=volume)
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=volume),
                         m["expectancy"] / volume if m["trades"] else 0.0, iters=5)
    tr, va = split_by_ts(usd)
    vm = metrics(va)
    return {"spec": spec.name, "trades": m["trades"],
            "events_per_day": round(m["trades"] / days, 2),
            "win_rate": m["win_rate"], "pf": m["profit_factor"],
            "net": m["net_pnl"], "exp_lot": g["expectancy_per_lot"],
            "max_dd": m["max_drawdown"], "gate": "PASS" if g["passed"] else "REJECT",
            "val_net": vm["net_pnl"], "val_pf": vm["profit_factor"],
            "wf_stable": wf.get("stable", False), "purple": purple}

def main():
    bars = build_bars_map(UNIVERSE)
    specs = [base_spec("tokyo_h0", [0]),
             base_spec("tokyo+london+ny", [0, 7, 12]),
             base_spec("all_3_plus", [0, 7, 12, 13])]
    emit_live_manifest(specs[0], "proxima_ops/state/tokyo_h0_live.json",
                       FTMO_TERMINAL, account="FTMO-Demo")
    print(json.dumps([report(bars, s) for s in specs], indent=2))
    print("manifest: proxima_ops/state/tokyo_h0_live.json")

if __name__ == "__main__":
    main()
