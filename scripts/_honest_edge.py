"""Purple-shuffle edge check WITH spread+commission (the honest version):
re-run each leg 12x on day-shuffled tapes, same costs, and compare honest net
vs the shuffled distribution. Edge survives only if everywhere > shuffled mean.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD_TYPICAL = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
                  "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
                  "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
                  "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
CONFIGS = {
    "tokyo":   {"sessions": [0],                "lookback": 6,   "top_n": 3, "hold_bars": 12, "lot": 0.35},
    "cascade": {"sessions": [2,3,4],            "lookback": 1440,"top_n": 8, "hold_bars": 24, "lot": 0.09},
    "london":  {"sessions": [7,8,9],            "lookback": 1440,"top_n": 5, "hold_bars": 12, "lot": 0.15},
    "usfade":  {"sessions": [14,15,16,17,18,19],"lookback": 50,  "top_n": 5, "hold_bars": 24, "lot": 0.30},
}
def make_spec(name, c):
    return StrategySpec.from_dict({
        "name": name, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_exhaustion", "lookback": c["lookback"], "pick": "n_worst",
                   "top_n": c["top_n"], "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": c["hold_bars"], "stop_first": True},
        "sessions": c["sessions"], "base_lot": 0.15,
    })

bars = build_bars_map(UNIVERSE)
def net_of(trades): return sum(t["net"] for t in trades)

def shuffle_bars(bm, seed):
    rnd = random.Random(seed)
    out = {}
    for sym, bars in bm.items():
        days = defaultdict(list)
        for b in bars: days[b["ts"] // 86400].append(b)
        ds = list(days.keys()); rnd.shuffle(ds)
        out[sym] = [b for d in ds for b in days[d]]
    return out

for name, c in CONFIGS.items():
    spec = make_spec(name, c)
    real = run_strategy(bars, spec, volume=c["lot"], commission_per_lot=3.0,
                        spread_pips_map=SPREAD_TYPICAL)
    real_net = net_of(real)
    shuf = [net_of(run_strategy(shuffle_bars(bars, 1000+i), spec, volume=c["lot"],
                                commission_per_lot=3.0, spread_pips_map=SPREAD_TYPICAL))
            for i in range(12)]
    mean_s = sum(shuf)/len(shuf); sd_s = (sum((x-mean_s)**2 for x in shuf)/len(shuf))**0.5
    z = (real_net - mean_s)/sd_s if sd_s else float("inf")
    above = sum(1 for x in shuf if x < real_net)
    print(f"{name:<9} REAL ${real_net:>8,.0f} | shuffled n={len(shuf)} mean ${mean_s:>7,.0f} "
          f"sd ${sd_s:>6,.0f} | z={z:>5.1f}  {above}/{len(shuf)} shuffles below")