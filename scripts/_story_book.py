"""Per-leg plain-English story numbers with REAL FTMO spreads.
Spread source A: pulled from the live FTMO-Demo terminal (Saturday snapshot = worst case).
Spread source B: realistic busy-session norms for the SAME FTMO feed (JPY/crosses tighten
                 ~40-55% during London/NY liquidity vs the closed-weekend quotation).
Engine = session_exhaustion -> _run_legacy, live lots, both-leg commission @ 3.0/lot.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.pnl import pip_value_usd
from collections import defaultdict

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

# REAL measured (FTMO-Demo bid/ask, Sat 14:29Z):  pips
SPREAD_MEASURED = {"EURUSD":0.60,"USDJPY":3.50,"GBPUSD":1.30,"AUDUSD":0.90,"EURJPY":5.70,
                   "GBPJPY":7.10,"EURAUD":2.90,"EURNZD":5.10,"GBPAUD":4.90,"GBPNZD":6.30,
                   "GBPCAD":4.10,"AUDNZD":3.50,"USDCAD":1.60,"NZDUSD":1.20,"EURGBP":1.70,
                   "EURCHF":2.60,"USDCHF":1.30,"AUDJPY":4.20}
# Realistic typical busy-session (London/NY silos the legs actually trade);
# majors near the measured level, JPY/cross tighter than the weekend quote
SPREAD_TYPICAL = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
                  "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
                  "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
                  "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}

CONFIGS = {
    "tokyo":   {"sessions": [0],          "lookback": 6,    "top_n": 3, "hold_bars": 12, "lot": 0.35},
    "cascade": {"sessions": [2,3,4],      "lookback": 1440, "top_n": 8, "hold_bars": 24, "lot": 0.09},
    "london":  {"sessions": [7,8,9],      "lookback": 1440, "top_n": 5, "hold_bars": 12, "lot": 0.15},
    "usfade":  {"sessions": [14,15,16,17,18,19], "lookback": 50, "top_n": 5, "hold_bars": 24, "lot": 0.30},
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

def spread_cost(r, lot, table):
    return table[r["symbol"]] * pip_value_usd(r["symbol"], r["entry"]) * lot

def per_leg(name, c, table):
    raw = run_strategy(bars, make_spec(name, c), volume=c["lot"], raw=True)
    usd = run_strategy(bars, make_spec(name, c), volume=c["lot"], commission_per_lot=3.0)
    day = defaultdict(float)
    for t, r in zip(usd, raw):
        day[r["entry_ts"] // 86400] += t["net"] - spread_cost(r, c["lot"], table)
    tot = sum(day.values())
    pos = sum(v for v in day.values() if v > 0); neg = -sum(v for v in day.values() if v < 0)
    pf = pos / neg if neg > 0 else float("inf")
    wins = sum(1 for t, r in zip(usd, raw) if t["net"] - spread_cost(r, c["lot"], table) > 0)
    return dict(trades=len(raw), net_engine=sum(t["net"] for t in usd),
                spread=sum(spread_cost(r, c["lot"], table) for r in raw),
                honest=tot, pf=pf, wr=wins/len(raw), worst=min(day.values()),
                green=100*sum(1 for v in day.values() if v>0)/len(day))

for label, table in [("MEASURED(Sat,worst)", SPREAD_MEASURED), ("TYPICAL(live)", SPREAD_TYPICAL)]:
    print(f"\n=== spread table: {label} ===")
    print(f"{'leg':<9}{'trades':>7}{'eng_net':>9}{'spread$':>9}{'honest':>9}{'PF':>6}{'WR':>6}{'green%':>7}{'worst':>8}")
    book_day = defaultdict(float)
    for name, c in CONFIGS.items():
        r = per_leg(name, c, table)
        print(f"{name:<9}{r['trades']:>7}{r['net_engine']:>9,.0f}{r['spread']:>9,.0f}{r['honest']:>9,.0f}"
              f"{r['pf']:>6.2f}{r['wr']:>6.2f}{r['green']:>7.1f}{r['worst']:>8,.0f}")
        raw = run_strategy(bars, make_spec(name, c), volume=c["lot"], raw=True)
        usd = run_strategy(bars, make_spec(name, c), volume=c["lot"], commission_per_lot=3.0)
        for t, rr in zip(usd, raw):
            book_day[rr["entry_ts"] // 86400] += t["net"] - spread_cost(rr, c["lot"], table)
    tot = sum(book_day.values())
    pos = sum(v for v in book_day.values() if v > 0); neg = -sum(v for v in book_day.values() if v < 0)
    green = sum(1 for v in book_day.values() if v > 0)
    # peak-to-trough DD of the honest cumulative curve (vs 25k baseline)
    eq = 25000.0; peak = 25000.0; maxdd = 0.0; maxdd_at = ""
    for d in sorted(book_day):
        eq += book_day[d]; peak = max(peak, eq)
        dd = peak - eq
        if dd > maxdd:
            maxdd = dd; maxdd_at = f"dd{maxdd:,.0f}@{eq:,.0f}eq peak{peak:,.0f}"
    print(f"{'BOOK':<9}{'':>7}{'':>9}{'':>9}{tot:>9,.0f}{pos/neg if neg else 999:>6.2f}"
          f"{'':>6}{100*green/len(book_day):>7.1f}{min(book_day.values()):>8,.0f}"
          f"  green {green}/{len(book_day)}d avg ${tot/len(book_day):,.0f}/d  maxDD ${maxdd:,.0f} ({maxdd/25000*100:.1f}%)")