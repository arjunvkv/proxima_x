"""Post-Tokyo cascade (UTC 2-3) sweep + full battery on best configs.
Distinct from hour-0 core and from carry (different anchor, hours 2-3).
Core-quality targets: PF>1.8, WR>0.55, positive val, wf stable.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, split_by_ts, walk_forward)
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

def mk(top, lb, hold, sess, sltp=(0.004,0.006), jpy=(0.40,0.60)):
    ss = {"rule":"session_exhaustion","lookback":lb,"pick":"n_worst","top_n":top,
          "side":"both","fill_bar":1,"per_hour":False}
    return StrategySpec.from_dict({"name":"cascade","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},"signal":ss,
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":jpy,"non_jpy_sl_tp":sltp},
        "sessions":sess,"base_lot":0.15})

def battery(name, spec):
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
    print(f"[{name}] gate:{g['passed']} det:{det} purple:{purple} wf:{wf['stable']} val>0:{vm['net_pnl']>0}")
    print(f"  t={m['trades']} WR={m['win_rate']:.4f} PF={m['profit_factor']:.4f} "
          f"net=${m['net_pnl']:,.0f} exp=${m['expectancy']:.2f} dd={m['max_drawdown']:,.0f}")
    print(f"  val_t={vm['trades']} val_PF={vm['profit_factor']:.3f} val_net=${vm['net_pnl']:,.0f} "
          f"wf_share={wf['positive_share']:.3f}  >>> {'PASS' if ok else 'FAIL'}\n")
    return ok

print("=== coarse grid (gate only) ===")
for sess in [[2,3],[1,2,3],[2,3,4]]:
    for top in [4,6,8]:
        for lb in [1440,2880]:
            for hold in [12,24]:
                u = run_strategy(bars, mk(top,lb,hold,sess), volume=0.15, commission_per_lot=3.5)
                me = metrics(u); g = gate(me, lot=0.15)
                if g["passed"]:
                    print(f"PASS sess={sess} top={top} lb={lb} h={hold:<3} t={me['trades']:4d} "
                          f"WR={me['win_rate']:.3f} PF={me['profit_factor']:.3f} net=${me['net_pnl']:,.0f}")

print("\n=== battery of best configs ===")
battery("cascade [2,3] top5 lb1440 h12", mk(5,1440,12,[2,3]))
battery("cascade [2,3] top6 lb1440 h12", mk(6,1440,12,[2,3]))
battery("cascade [2,3] top4 lb1440 h24", mk(4,1440,24,[2,3]))
battery("cascade [1,2,3] top5 lb1440 h12", mk(5,1440,12,[1,2,3]))
battery("cascade [2,3] top5 lb1440 h48", mk(5,1440,48,[2,3]))
print("=== strongest [2,3,4] configs, full battery ===")
battery("cascade [2,3,4] top6 lb2880 h24", mk(6,2880,24,[2,3,4]))
battery("cascade [2,3,4] top6 lb2880 h12", mk(6,2880,12,[2,3,4]))
battery("cascade [2,3,4] top8 lb2880 h24", mk(8,2880,24,[2,3,4]))
battery("cascade [2,3,4] top8 lb1440 h24", mk(8,1440,24,[2,3,4]))