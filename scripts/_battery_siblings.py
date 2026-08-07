"""Full battery on session_momentum[14-19] hold12 — the cleanest additive candidate
(all-BUY trend continuation, 8/8 neighbors, GIS. Confirm full battery + best config."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
    split_by_ts, walk_forward, purple_edge, determinism)
from proxima_ops.backtest.feed import build_bars_map
from collections import Counter

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

def mk(rule, sess, top=5, lb=50, hold=12):
    return StrategySpec.from_dict({"name":rule,"universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},
        "signal":{"rule":rule,"lookback":lb,"pick":"n_worst","top_n":top,"side":"both",
                  "fill_bar":1,"per_hour":False},
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":(0.40,0.60),"non_jpy_sl_tp":(0.004,0.006)},
        "sessions":sess,"base_lot":0.15})

def full(label, rule, sess, top=5, lb=50, hold=12):
    spec = mk(rule, sess, top, lb, hold)
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    m = metrics(usd)
    g = gate(m, lot=0.15)
    run = lambda: run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    rb = lambda bm: run_strategy(bm, spec, volume=0.15, commission_per_lot=3.5)
    det = determinism(run, 3)
    purp = purple_edge(bars, rb, m["expectancy"]/0.15, iters=5)
    wf = walk_forward(usd)
    _, va = split_by_ts(usd)
    vp = metrics(va)["net_pnl"] if va else 0.0
    sides = Counter(t["side"] for t in usd)
    flags = []
    if g["passed"]: flags.append("GATE")
    if det: flags.append("DET")
    if purp == "REAL-EDGE": flags.append("REAL")
    if wf["positive_share"] >= 0.75: flags.append("WF")
    if vp > 0: flags.append("VAL")
    ok = all(x in flags for x in ["GATE","DET","REAL","WF","VAL"])
    print(f"[{'PASS' if ok else 'x'}] {rule} {sess} hold{hold} WR {m['win_rate']:.3f} "
          f"PF {m['profit_factor']:.2f} net {m['net_pnl']:.0f} exp {m['net_pnl']/len(usd):.1f} "
          f"{'/'.join(flags)} val={vp:.0f} sides={dict(sides)} wf_share={wf['positive_share']:.2f}")
    return ok

import inspect
print("wf keys:", list(walk_forward.__doc__.split() if walk_forward.__doc__ else []))
full("momentum", "session_momentum", [14,15,16,17,18,19], 5, 50, 12)
full("momentum h24", "session_momentum", [14,15,16,17,18,19], 5, 50, 24)
full("momentum smaller top4", "session_momentum", [14,15,16,17,18,19], 4, 50, 12)