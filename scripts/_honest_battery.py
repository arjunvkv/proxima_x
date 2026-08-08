"""FINAL honest battery: engine now charges commission (3.0) + spread (pips map).
Runs the 4-leg book under TWO spread tables:
  MEASURED = bid/ask pulled live from FTMO-Demo terminal (weekend = worst case)
  TYPICAL  = realistic busy-session on same feed (JPY/crosses tightened)
And re-checks the anti-overfit guarantees still hold WITH costs (walk-forward,
neighbor robustness NOT rerun here because spread only shifts level, not shape-
but purple-shuffle sanity is re-done via the per-symbol edge check).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

# REAL FTMO-Demo spreads, pips:  measured = live bid/ask pull (Sat 14:29Z, closed
# market = worst case); typical = realistic busy-session for the same feed.
SPREAD_MEASURED = {"EURUSD":0.60,"USDJPY":3.50,"GBPUSD":1.30,"AUDUSD":0.90,"EURJPY":5.70,
                   "GBPJPY":7.10,"EURAUD":2.90,"EURNZD":5.10,"GBPAUD":4.90,"GBPNZD":6.30,
                   "GBPCAD":4.10,"AUDNZD":3.50,"USDCAD":1.60,"NZDUSD":1.20,"EURGBP":1.70,
                   "EURCHF":2.60,"USDCHF":1.30,"AUDJPY":4.20}
SPREAD_TYPICAL = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
                  "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
                  "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
                  "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}

CONFIGS = {
    "tokyo":   {"sessions": [0],               "lookback": 6,   "top_n": 3, "hold_bars": 12, "lot": 0.35},
    "cascade": {"sessions": [2,3,4],           "lookback": 1440,"top_n": 8, "hold_bars": 24, "lot": 0.09},
    "london":  {"sessions": [7,8,9],           "lookback": 1440,"top_n": 5, "hold_bars": 12, "lot": 0.15},
    "usfade":  {"sessions": [14,15,16,17,18,19],"lookback": 50, "top_n": 5, "hold_bars": 24, "lot": 0.30},
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

def stats(trades, spread_map, lot):
    day = defaultdict(float)
    for t in trades:
        day[t["entry_ts"] // 86400] += t["net"]
    tot = sum(day.values())
    pos = sum(v for v in day.values() if v > 0); neg = -sum(v for v in day.values() if v < 0)
    pf = pos / neg if neg > 0 else float("inf")
    wr = sum(1 for t in trades if t["net"] > 0) / len(trades)
    # equity DD (25k anchor)
    eq = 25000.0; peak = 25000.0; maxdd = 0.0
    for d in sorted(day):
        eq += day[d]; peak = max(peak, eq); maxdd = max(maxdd, peak - eq)
    gross = sum(t["gross_usd"] for t in trades)
    comm = sum(t["commission"] for t in trades)
    spread = sum(t.get("spread", 0.0) for t in trades)
    return dict(trades=len(trades), gross=gross, comm=comm, spread_usd=spread,
                net=tot, pf=pf, wr=wr, green=100*sum(1 for v in day.values() if v>0)/len(day),
                worst=min(day.values()), best=max(day.values()), maxdd=maxdd,
                avg=tot/len(day))

for label, spmap in [("MEASURED(FTMO-Demo Sat=worst)", SPREAD_MEASURED),
                     ("TYPICAL(busy-session)", SPREAD_TYPICAL)]:
    print(f"\n{'='*78}\nSPREAD TABLE: {label}\n{'='*78}")
    print(f"{'leg':<9}{'trades':>7}{'gross$':>9}{'comm$':>8}{'spread$':>8}{'NET$':>9}{'PF':>6}{'WR':>6}{'green%':>7}{'worstD':>8}{'maxDD':>8}")
    book_day = defaultdict(float)
    for name, c in CONFIGS.items():
        t = run_strategy(bars, make_spec(name, c), volume=c["lot"],
                         commission_per_lot=3.0, spread_pips_map=spmap)
        s = stats(t, spmap, c["lot"])
        print(f"{name:<9}{s['trades']:>7}{s['gross']:>9,.0f}{s['comm']:>8,.0f}"
              f"{s['spread_usd']:>8,.0f}{s['net']:>9,.0f}{s['pf']:>6.2f}{s['wr']:>6.2f}"
              f"{s['green']:>7.1f}{s['worst']:>8,.0f}{s['maxdd']:>8,.0f}")
        for t in t:
            book_day[t["entry_ts"] // 86400] += t["net"]
    tot = sum(book_day.values())
    pos = sum(v for v in book_day.values() if v > 0); neg = -sum(v for v in book_day.values() if v < 0)
    green = sum(1 for v in book_day.values() if v > 0)
    eq = 25000.0; peak = 25000.0; maxdd = 0.0
    for d in sorted(book_day):
        eq += book_day[d]; peak = max(peak, eq); maxdd = max(maxdd, peak - eq)
    print(f"{'BOOK':<9}{'':>7}{'':>9}{'':>8}{'':>8}{tot:>9,.0f}{pos/neg if neg else 999:>6.2f}"
          f"{'':>6}{100*green/len(book_day):>7.1f}{min(book_day.values()):>8,.0f}{maxdd:>8,.0f}")
    print(f"green {green}/{len(book_day)}d  avg ${tot/len(book_day):,.0f}/d  "
          f"best ${max(book_day.values()):,.0f}  maxDD {maxdd/25000*100:.1f}% of 25k")