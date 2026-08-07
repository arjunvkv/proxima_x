"""Battery-verify the improved carry_clock config: V0 map, top_n=8, hold=96
(sessions 14-20). Compares against committed top_n=10 winner. Full battery:
gate + determinism + purple + train/val + walk-forward.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, split_by_ts, walk_forward)
from proxima_ops.backtest.feed import build_bars_map

CARRY_V0 = {"EURUSD": 0, "USDJPY": 1, "GBPUSD": 1, "AUDUSD": 1, "EURJPY": 1,
            "GBPJPY": 1, "EURAUD": -1, "EURNZD": -1, "GBPAUD": -1, "GBPNZD": -1,
            "GBPCAD": 1, "AUDNZD": 0, "USDCAD": 1, "NZDUSD": 1, "EURGBP": -1,
            "EURCHF": 1, "USDCHF": 1, "AUDJPY": 1}
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)
SESS = [14,15,16,17,18,19,20]

def mk(top, hold):
    ss = {"rule":"carry_clock","lookback":6,"pick":"n_worst","top_n":top,"side":"both",
          "fill_bar":1,"per_hour":False,"direction_map":CARRY_V0}
    return StrategySpec.from_dict({"name":"carry","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},"signal":ss,
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":(0.30,0.40),"non_jpy_sl_tp":(0.0030,0.0040)},
        "sessions":SESS,"base_lot":0.15})

def battery(name, top, hold):
    spec = mk(top, hold)
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    m = metrics(usd)
    g = gate(m, lot=0.15)
    # determinism: 3 runs same count
    cnts = {len(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)) for _ in range(3)}
    det = len(cnts) == 1
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=0.15,
                       commission_per_lot=3.5), m["expectancy"]/0.15, iters=5)
    tr, va = split_by_ts(usd)
    vm = metrics(va)
    wf = walk_forward(usd, train_size=120, test_size=60, lot=0.15)
    ok = (g["passed"] and det and purple=="REAL-EDGE" and wf["stable"] and vm["net_pnl"]>0)
    print(f"\n[{name}] top={top} hold={hold}")
    print(f"  gate:{g['passed']} det:{det} purple:{purple} wf:{wf['stable']} val_net>0:{vm['net_pnl']>0}")
    print(f"  t={m['trades']} WR={m['win_rate']:.4f} PF={m['profit_factor']:.4f} net=${m['net_pnl']:,} "
          f"exp/lot=${m['expectancy']:.2f} dd={m['max_drawdown']:,}")
    print(f"  val_t={vm['trades']} val_PF={vm['profit_factor']:.3f} val_net=${vm['net_pnl']:,} "
          f"wf_share={wf['positive_share']:.3f}")
    print(f"  >>> FULL-BATTERY: {'PASS' if ok else 'FAIL'}\n")
    return ok

w_prev = battery("committed", 10, 96)
w_new = battery("improved ", 8, 72)
battery("improved2", 8, 96)

print("\n=== WEEKDAY STACK battery-verify (top8/h72, Wed-Fri) ===")
def mk_wd(wd):
    ss = {"rule":"carry_clock","lookback":6,"pick":"n_worst","top_n":8,"side":"both",
          "fill_bar":1,"per_hour":False,"direction_map":CARRY_V0}
    return StrategySpec.from_dict({"name":"carry","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},"signal":ss,
        "exit":{"mode":"sl_tp_hold","hold_bars":72,"stop_first":True,
                "jpy_sl_tp":(0.30,0.40),"non_jpy_sl_tp":(0.0030,0.0040)},
        "sessions":SESS,"base_lot":0.15, **({"weekdays":wd} if wd else {})})

def battery2(name, spec):
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    m = metrics(usd)
    g = gate(m, lot=0.15)
    cnts = {len(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)) for _ in range(3)}
    det = len(cnts) == 1
    purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=0.15,
                       commission_per_lot=3.5), m["expectancy"]/0.15, iters=5)
    tr, va = split_by_ts(usd)
    vm = metrics(va)
    wf = walk_forward(usd, train_size=120, test_size=60, lot=0.15)
    ok = (g["passed"] and det and purple=="REAL-EDGE" and wf["stable"] and vm["net_pnl"]>0)
    print(f"[{name}] gate:{g['passed']} det:{det} purple:{purple} wf:{wf['stable']} val_net>0:{vm['net_pnl']>0}")
    print(f"  t={m['trades']} WR={m['win_rate']:.4f} PF={m['profit_factor']:.4f} net=${m['net_pnl']:,} "
          f"exp/lot=${m['expectancy']:.2f} dd={m['max_drawdown']:,} | val_PF={vm['profit_factor']:.3f} "
          f"val_net=${vm['net_pnl']:,} wf_share={wf['positive_share']:.3f}  >>> {'PASS' if ok else 'FAIL'}")
    return ok

battery2("carry WF(2,3,4)", mk_wd([2,3,4]))