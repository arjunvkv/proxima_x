"""Is run_strategy mutating the shared bars dicts? Check:
1. fresh run -> net
2. run again on SAME bars object (no shuffle) -> same net?
3. after 12 shuffles on shared-dict copies -> fresh run still same?
4. deep-compare a bar dict before/after a run"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
bars = build_bars_map(UNIVERSE)

def make(w, lb=1440, top=8, hold=24, ph=False):
    sig = {"rule": "session_reversion", "lookback": lb, "pick": "n_worst",
           "top_n": top, "side": "both", "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": "session_reversion", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

spec = make([17, 18, 19])

def net_of(usd):
    return sum(t["net"] for t in usd), len(usd)

# 1+2: two runs on the SAME bars object
n1 = net_of(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                         spread_pips_map=SPREAD))
n2 = net_of(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                         spread_pips_map=SPREAD))
print(f"run1 {n1}  run2(same bars) {n2}  equal={n1==n2}")

# 3: snapshot bars, run, compare
import copy
snap = copy.deepcopy(bars)
run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
             spread_pips_map=SPREAD)
mutated = []
for s in UNIVERSE:
    for i, (a, b) in enumerate(zip(snap[s], bars[s])):
        if a != b:
            mutated.append((s, i, a, b))
            if len(mutated) > 3:
                break
    if len(mutated) > 3:
        break
print(f"bars mutated by run_strategy: {len(mutated)} instances "
      f"(first: {mutated[:2] if mutated else None})")

# 4: shuffle-loop on shallow copies (like the vetter), then fresh run
import random
rng = random.Random(42)
for _ in range(12):
    sh = {s: b[:] for s, b in bars.items()}
    for s in sh:
        rng.shuffle(sh[s])
    run_strategy(sh, spec, volume=0.15, commission_per_lot=3.0,
                 spread_pips_map=SPREAD)
n3 = net_of(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                         spread_pips_map=SPREAD))
print(f"after 12 shallow-copy shuffles: fresh run {n3}  (was {n1})")