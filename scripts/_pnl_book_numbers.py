"""HONEST book numbers: engine net minus realistic spread per round trip,
at the LIVE lot sizes (tokyo 0.35 / cascade 0.09 / london 0.15 / usfade 0.30).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.pnl import pip_value_usd
from collections import defaultdict

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
# realistic FTMO-style spread in PIPs per pair (mid-day, non-news)
SPREADS_PIPS = {"EURUSD":0.7,"USDJPY":0.8,"GBPUSD":1.1,"AUDUSD":0.9,"EURJPY":1.2,
                "GBPJPY":1.8,"EURAUD":1.8,"EURNZD":2.2,"GBPAUD":2.0,"GBPNZD":2.6,
                "GBPCAD":2.0,"AUDNZD":1.8,"USDCAD":1.0,"NZDUSD":1.2,"EURGBP":0.9,
                "EURCHF":1.0,"USDCHF":1.0,"AUDJPY":1.1}
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

def spread_cost(sym, entry, lot):
    return SPREADS_PIPS[sym] * pip_value_usd(sym, entry) * lot

bars = build_bars_map(UNIVERSE)
print(f"{'leg':<9}{'lot':>5}{'trades':>7}{'net':>9}{'spread':>8}{'HONEST':>9}{'PF':>6}{'WR':>6}{'worstD':>8}")
all_days = defaultdict(float)
mixed = defaultdict(lambda: defaultdict(float))  # day -> leg -> honest pnl
for name, c in CONFIGS.items():
    raw = run_strategy(bars, make_spec(name, c), volume=c["lot"], raw=True)
    usd = run_strategy(bars, make_spec(name, c), volume=c["lot"], commission_per_lot=3.5)
    day_pnl = defaultdict(float)
    for t, r in zip(usd, raw):
        net0 = t["net"]  # engine net (gross - both-leg commission)
        spd = spread_cost(r["symbol"], r["entry"], c["lot"])
        honest = net0 - spd
        day_pnl[r["entry_ts"] // 86400] += honest
        mixed[r["entry_ts"] // 86400][name] += honest
    wins = sum(1 for t, r in zip(usd, raw) if t["net"] - spread_cost(r["symbol"], r["entry"], c["lot"]) > 0)
    tot = sum(day_pnl.values())
    pos = sum(v for v in day_pnl.values() if v > 0); neg = -sum(v for v in day_pnl.values() if v < 0)
    pf = pos / neg if neg > 0 else float("inf")
    for d, v in day_pnl.items(): all_days[d] += v
    print(f"{name:<9}{c['lot']:>5}{len(raw):>7}{sum(t['net'] for t in usd):>9,.0f}"
          f"{sum(spread_cost(r['symbol'], r['entry'], c['lot']) for r in raw):>8,.0f}"
          f"{tot:>9,.0f}{pf:>6.2f}{wins/len(raw):>6.2f}{min(day_pnl.values()):>8,.0f}")
print("-" * 62)
print(f"BOOK honest total (live lots, post-spread): ${sum(all_days.values()):,.0f} over {len(all_days)} days")
green = sum(1 for v in all_days.values() if v > 0)
print(f"green days: {green}/{len(all_days)} ({100*green/len(all_days):.1f}%)  "
      f"avg/day ${sum(all_days.values())/len(all_days):,.2f}  "
      f"worst ${min(all_days.values()):,.0f}  best ${max(all_days.values()):,.0f}")
# worst-day co-occurrence: which legs bled the same day?
worst5 = sorted(all_days.items(), key=lambda kv: kv[1])[:5]
print("worst days breakdown (legs):")
for d, v in worst5:
    print(f"  day {d}: ${v:,.0f}  {dict(mixed[d])}")