import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,"GBPJPY":3.0,
          "EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,"GBPCAD":3.2,"AUDNZD":2.6,
          "USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,"EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
bars = build_bars_map(UNIVERSE)
t0 = time.time()
spec = StrategySpec.from_dict({"name":"test","universe":UNIVERSE,
    "feed":{"kind":"bar","timeframe":"M5"},
    "signal":{"rule":"range_reversion","lookback":50,"pick":"n_worst","top_n":5,
              "side":"both","fill_bar":1},
    "exit":{"mode":"sl_tp_hold","hold_bars":12,"stop_first":True},
    "sessions":[7,8,9],"base_lot":0.15})
tr = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0, spread_pips_map=SPREAD)
print(f"{len(tr)} trades in {time.time()-t0:.2f}s  net ${sum(t['net'] for t in tr):,.0f}")
