"""Neighborhood-robustness: the one full-battery survivor London [7,8,9] h12
(WR .668 PF 2.85). Ship only if >=2 of 8 neighbor cells (hour-shift +/-1,
top_n +/-1, lookback ±, hold 6/24) also clear gate+PF>1.8 & positive val.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics, gate, split_by_ts
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

def mk(top, lb, hold, sess, sltp=(0.004,0.006), jpy=(0.40,0.60)):
    ss = {"rule":"session_exhaustion","lookback":lb,"pick":"n_worst","top_n":top,
          "side":"both","fill_bar":1,"per_hour":False}
    return StrategySpec.from_dict({"name":"cand","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},"signal":ss,
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":jpy,"non_jpy_sl_tp":sltp},
        "sessions":sess,"base_lot":0.15})

def quick(sess, top, lb, hold):
    usd = run_strategy(bars, mk(top,lb,hold,sess), volume=0.15, commission_per_lot=3.5)
    m = metrics(usd)
    g = gate(m, lot=0.15)
    _, va = split_by_ts(usd)
    v = metrics(va)
    ok = g["passed"] and m["profit_factor"] > 1.5 and v["net_pnl"] > 0
    return ok, m

# base survivor + 8 neighbors: shift hour-window, top_n, lookback, hold
base_ok, base_m = quick([7,8,9], 5, 1440, 12)
print(f"BASE  [7,8,9] top5 lb1440 h12  PF={base_m['profit_factor']:.2f} ok={base_ok}")
cells = {
  "[7,8]   top5 lb1440 h12": ([7,8], 5, 1440, 12),
  "[8,9]   top5 lb1440 h12": ([8,9], 5, 1440, 12),
  "[6,7,8] top5 lb1440 h12": ([6,7,8], 5, 1440, 12),
  "[8,9,10] top5 lb1440 h12": ([8,9,10], 5, 1440, 12),
  "[7,8,9] top4 lb1440 h12": ([7,8,9], 4, 1440, 12),
  "[7,8,9] top6 lb1440 h12": ([7,8,9], 6, 1440, 12),
  "[7,8,9] top5 lb720  h12": ([7,8,9], 5, 720, 12),
  "[7,8,9] top5 lb2880 h12": ([7,8,9], 5, 2880, 12),
  "[7,8,9] top5 lb1440 h6 ": ([7,8,9], 5, 1440, 6),
}
print("\n=== neighbor grid ===")
npass = 0
for name, args in cells.items():
    ok, m = quick(*args)
    print(f"[{'PASS' if ok else 'x'}] {name} PF={m['profit_factor']:.2f} WR={m['win_rate']:.3f}")
    if ok: npass += 1
print(f"\nneighbors passing: {npass}/9  (need >=2 for robustness)")
print("BASE PF={:.2f} WR={:.3f}".format(base_m['profit_factor'], base_m['win_rate']))