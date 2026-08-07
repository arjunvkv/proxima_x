"""Long-run evidence for usfade [14-19] top5 lb50 hold24:
1) per-window walk-forward expectancies (regime stability across 200d)
2) OOS val profit factor + IS->OOS decay
3) standalone daily-green rate vs book (tokyo+cascade+london)
4) tail: worst window, worst day"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, split_by_ts,
    walk_forward)
from proxima_ops.backtest.feed import build_bars_map
from collections import defaultdict

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

def mk(rule, sess, top, lb, hold):
    return StrategySpec.from_dict({"name":rule,"universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},
        "signal":{"rule":rule,"lookback":lb,"pick":"n_worst","top_n":top,"side":"both",
                  "fill_bar":1,"per_hour":False},
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":(0.40,0.60),"non_jpy_sl_tp":(0.004,0.006)},
        "sessions":sess,"base_lot":0.15})

# ---- the candidate ----
usfade = mk("session_exhaustion", [14,15,16,17,18,19], 5, 50, 24)
u = run_strategy(bars, usfade, volume=0.15, commission_per_lot=3.5)
m = metrics(u)
tr, va = split_by_ts(u)
mt, mv = metrics(tr), metrics(va)
print("=== usfade [14-19] top5 lb50 h24 ===")
print(f"trades {m['trades']}  IS {mt['trades']} / OOS {mv['trades']}")
print(f"IS  PF {mt['profit_factor']:.2f} net {mt['net_pnl']:.0f}")
print(f"OOS PF {mv['profit_factor']:.2f} net {mv['net_pnl']:.0f}  (decay {(mt['profit_factor']-mv['profit_factor'])/mt['profit_factor']*100:.0f}%)")
wf = walk_forward(u, train_size=300, test_size=100, lot=0.15)
print(f"walk-fwd: {wf}")

# per-window detail
print("\nper-window walk-forward:")
for i, w in enumerate(wf.get("windows", [])):
    print(f"  win{i}: exp {w.get('exp', w.get('expectancy',0)):.2f} net {w.get('net',0):.0f} "
          f"pf {w.get('pf',0):.2f} wr {w.get('wr',0):.2f}")

# ---- daily green rate standalone vs book ----
def daily_net(usd):
    d = defaultdict(float)
    for t in usd:
        d[t["entry_ts"] // 86400] += t["net"]
    return d

du = daily_net(u)
green_u = sum(1 for v in du.values() if v > 0)
print(f"\nusfade standalone: {green_u}/{len(du)} green days ({green_u/len(du)*100:.1f}%)  "
      f"net {sum(du.values()):.0f}  worst-day {min(du.values()):.0f}")

book_specs = [
    ("tokyo", mk("session_exhaustion", [0], 3, 6, 12)),
    ("cascade", mk("session_exhaustion", [2,3,4], 8, 1440, 24)),
    ("london", mk("session_exhaustion", [7,8,9], 5, 1440, 12)),
]
db = defaultdict(float)
for nm, spec in book_specs:
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    for t in usd:
        db[t["entry_ts"] // 86400] += t["net"]
green_b = sum(1 for v in db.values() if v > 0)
print(f"book (3 legs @0.15): {green_b}/{len(db)} green days ({green_b/len(db)*100:.1f}%)  "
      f"net {sum(db.values()):.0f}  worst-day {min(db.values()):.0f}")

db4 = defaultdict(float, db)
for t in u:
    db4[t["entry_ts"] // 86400] += t["net"]
green_b4 = sum(1 for v in db4.values() if v > 0)
print(f"book + usfade (4 legs @0.15): {green_b4}/{len(db4)} green days "
      f"({green_b4/len(db4)*100:.1f}%)  net {sum(db4.values()):.0f}  worst-day {min(db4.values()):.0f}")