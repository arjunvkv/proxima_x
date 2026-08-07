"""Position-sizing / allocation search for the core book.

Per-edge UTC-daily PnL (computed at 0.15 lot) is scaled by a weight w_i
(relative to 0.15 lot; PnL scales linearly with lots in this engine). We search
the weight simplex for combos that RAISE positive-day coverage and risk-adjusted
return vs the baseline (1,1,1). Then map the winner to a $25k funded account.
"""
import sys, os
from itertools import product
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map
from collections import defaultdict
import statistics as st

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)
CARRY_V0 = {"EURUSD":0,"USDJPY":1,"GBPUSD":1,"AUDUSD":1,"EURJPY":1,"GBPJPY":1,
            "EURAUD":-1,"EURNZD":-1,"GBPAUD":-1,"GBPNZD":-1,"GBPCAD":1,"AUDNZD":0,
            "USDCAD":1,"NZDUSD":1,"EURGBP":-1,"EURCHF":1,"USDCHF":1,"AUDJPY":1}

def tokyo(): return StrategySpec.from_dict({"name":"t","universe":UNIVERSE,
  "feed":{"kind":"bar","timeframe":"M5"},"signal":{"rule":"weekend_gap","lookback":6,
  "pick":"n_worst","top_n":3,"side":"both","fill_bar":1,"per_hour":False},
  "exit":{"mode":"sl_tp_hold","hold_bars":12,"stop_first":True,"jpy_sl_tp":(0.5,0.7),
          "non_jpy_sl_tp":(0.005,0.007)},"sessions":[0],"base_lot":0.15})
def cascade(): return StrategySpec.from_dict({"name":"c","universe":UNIVERSE,
  "feed":{"kind":"bar","timeframe":"M5"},"signal":{"rule":"session_exhaustion","lookback":1440,
  "pick":"n_worst","top_n":8,"side":"both","fill_bar":1,"per_hour":False},
  "exit":{"mode":"sl_tp_hold","hold_bars":24,"stop_first":True,"jpy_sl_tp":(0.4,0.6),
          "non_jpy_sl_tp":(0.004,0.006)},"sessions":[2,3,4],"base_lot":0.15})
def carry(): return StrategySpec.from_dict({"name":"k","universe":UNIVERSE,
  "feed":{"kind":"bar","timeframe":"M5"},"signal":{"rule":"carry_clock","lookback":6,
  "pick":"n_worst","top_n":8,"side":"both","fill_bar":1,"per_hour":False,"direction_map":CARRY_V0},
  "exit":{"mode":"sl_tp_hold","hold_bars":72,"stop_first":True,"jpy_sl_tp":(0.3,0.4),
          "non_jpy_sl_tp":(0.003,0.004)},"sessions":[14,15,16,17,18,19,20],
  "weekdays":[2,3,4],"base_lot":0.15})

specs = {"t":tokyo(),"c":cascade(),"k":carry()}
day0 = min(bars[s][0]["ts"]//86400 for s in UNIVERSE)
per = {}
for n, spec in specs.items():
    tr = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    d = defaultdict(float)
    for t in tr: d[t["entry_ts"]//86400 - day0] += t["net"]
    per[n] = dict(d)

grid = [0.0, 0.5, 1.0, 1.5, 2.0]
best = []
for wt, wc, wk in product(grid, grid, grid):
    if wt == 0 and wc == 0 and wk == 0: continue
    days = sorted(set(per["t"]) | set(per["c"]) | set(per["k"]))
    vals = [wt*per["t"].get(d,0) + wc*per["c"].get(d,0) + wk*per["k"].get(d,0) for d in days]
    pos = sum(1 for v in vals if v > 0)
    cov = pos/len(days)
    mean = sum(vals)/len(vals)
    worst = min(vals)
    # longest down streak
    cur = mstreak = 0
    for v in vals:
        cur = cur+1 if v < 0 else 0
        mstreak = max(mstreak, cur)
    sd = st.pstdev(vals)
    sharpe = mean/sd if sd else 0.0
    tot = sum(vals)
    best.append((cov, mean, sharpe, worst, mstreak, tot, wt, wc, wk))

best.sort(key=lambda x: (-x[0], -x[1]))   # maximize coverage then mean
print("=== top by daily coverage (cov, mean/day, sharpe, worst, down>seq, net, [wt wc wk]) ===")
for b in best[:12]:
    print(f"cov={b[0]:.3f} mean=${b[1]:7.2f} sharpe={b[2]:.3f} worst=${b[3]:7.0f} "
          f"dseq={b[4]} tot=${b[5]:8,.0f} w(t,c,k)=({b[6]},{b[7]},{b[8]})")

# also best by sharpe with coverage > 0.80
print("\n=== best-TQ by sharpe among daily-coverage>0.80 ===")
good = [b for b in best if b[0] >= 0.80]
good.sort(key=lambda x: -x[2])
for b in good[:6]:
    print(f"cov={b[0]:.3f} sharpe={b[2]:.3f} mean=${b[1]:7.2f} worst=${b[3]:7.0f} "
          f"tot=${b[5]:8,.0f} w=( {b[6]},{b[7]},{b[8]})")