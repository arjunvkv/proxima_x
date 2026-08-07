"""Core combined book -> measure DAILY net distribution (not just 200-day total).

Buckets every trade by its UTC trading day and sums all three edges per day, so
we can answer 'fraction of days green, worst day, longest down streak' - the
real measure of a daily-positive core. Also shows each sub-edge separately.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map
import statistics as st

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

CARRY_V0 = {"EURUSD": 0, "USDJPY": 1, "GBPUSD": 1, "AUDUSD": 1, "EURJPY": 1,
            "GBPJPY": 1, "EURAUD": -1, "EURNZD": -1, "GBPAUD": -1, "GBPNZD": -1,
            "GBPCAD": 1, "AUDNZD": 0, "USDCAD": 1, "NZDUSD": 1, "EURGBP": -1,
            "EURCHF": 1, "USDCHF": 1, "AUDJPY": 1}

def tokyo():
    return StrategySpec.from_dict({"name":"tokyo","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},
        "signal":{"rule":"weekend_gap","lookback":6,"pick":"n_worst","top_n":3,
                  "side":"both","fill_bar":1,"per_hour":False},
        "exit":{"mode":"sl_tp_hold","hold_bars":12,"stop_first":True,
                "jpy_sl_tp":(0.50,0.70),"non_jpy_sl_tp":(0.0050,0.0070)},
        "sessions":[0],"base_lot":0.15})

def cascade():
    return StrategySpec.from_dict({"name":"cascade","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},
        "signal":{"rule":"session_exhaustion","lookback":1440,"pick":"n_worst",
                  "top_n":8,"side":"both","fill_bar":1,"per_hour":False},
        "exit":{"mode":"sl_tp_hold","hold_bars":24,"stop_first":True,
                "jpy_sl_tp":(0.40,0.60),"non_jpy_sl_tp":(0.0040,0.0060)},
        "sessions":[2,3,4],"base_lot":0.15})

def carry():
    return StrategySpec.from_dict({"name":"carry","universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},
        "signal":{"rule":"carry_clock","lookback":6,"pick":"n_worst","top_n":8,
                  "side":"both","fill_bar":1,"per_hour":False,"direction_map":CARRY_V0},
        "exit":{"mode":"sl_tp_hold","hold_bars":72,"stop_first":True,
                "jpy_sl_tp":(0.30,0.40),"non_jpy_sl_tp":(0.0030,0.0040)},
        "sessions":[14,15,16,17,18,19,20],"weekdays":[2,3,4],"base_lot":0.15})

specs = {"tokyo":tokyo(), "cascade":cascade(), "carry":carry()}
trades = {}
for name, spec in specs.items():
    trades[name] = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    me = metrics(trades[name])
    print(f"{name:<8} t={me['trades']:5d} WR={me['win_rate']:.3f} PF={me['profit_factor']:.3f} "
          f"net=${me['net_pnl']:>8,.0f} dd={me['max_drawdown']:,.0f}")

# bucket by UTC day (entry_ts//86400)
def by_day(tl):
    d = defaultdict(float)
    for t in tl:
        d[t["entry_ts"] // 86400] += t["net"]
    return d

per = {n: by_day(tl) for n, tl in trades.items()}

def describe(name_pairs, n_days):
    days = [d for d in range(n_days) if any(p.get(d, 0) != 0 for _, p in name_pairs)]
    vals = [sum(p.get(d, 0.0) for _, p in name_pairs) for d in days]
    if not vals:
        print("  (no active days)")
        return
    pos = sum(1 for x in vals if x > 0)
    neg = sum(1 for x in vals if x < 0)
    z = sum(1 for x in vals if x == 0)
    worst = cur = 0
    for x in vals:
        cur = cur + 1 if x < 0 else 0
        worst = max(worst, cur)
    print(f"  active-days={len(vals)}  positive={pos} ({100*pos/len(vals):.1f}%)  "
          f"negative={neg} zero={z}  worst-day=${min(vals):,.0f}  "
          f"longest-down={worst}  mean=${st.mean(vals):,.2f}  "
          f"median=${st.median(vals):,.2f}")

# number of UTC days in tape (approx from any symbol's first/last)
firstts = {s: bars[s][0]["ts"] for s in UNIVERSE}
lastts = {s: bars[s][-1]["ts"] for s in UNIVERSE}
day0 = min(v // 86400 for v in firstts.values())
dayN = max(v // 86400 for v in lastts.values())
ndays = dayN - day0 + 1
zon = day0
print(f"\ntotal UTC days in tape: {ndays}")
per = {n: {d - zon: v for d, v in dd.items()} for n, dd in per.items()}

print("\n=== core = tokyo + cascade (the 0-4 UTC book) ===")
describe([("t",per["tokyo"]),("c",per["cascade"])], ndays)
print("\n=== core + carry satellite (full-day book) ===")
describe([("t",per["tokyo"]),("c",per["cascade"]),("k",per["carry"])], ndays)
print("\n=== tokyo alone ===")
describe([("t",per["tokyo"])], ndays)
print("\n=== cascade alone ===")
describe([("c",per["cascade"])], ndays)
print("\n=== carry alone ===")
describe([("k",per["carry"])], ndays)