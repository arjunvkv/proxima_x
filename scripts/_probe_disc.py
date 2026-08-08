"""RESOLVE the 8,291 vs 5,353 discrepancy for session_reversion [17,19] lb1440 top8 h24.
Re-run the identical spec through BOTH scripts' code paths and diff the trade list."""
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

def run(bars_):
    return run_strategy(bars_, make([17, 18, 19]), volume=0.15,
                        commission_per_lot=3.0, spread_pips_map=SPREAD)

usd = run(bars)
n = len(usd)
net = sum(t["net"] for t in usd)
wins = [t for t in usd if t["net"] > 0]
gw = sum(t["gross_usd"] for t in wins)
gl = -sum(t["gross_usd"] for t in usd if t["net"] < 0)
print(f"RUN1 (fresh spec): {n}t net ${net:,.0f} PF {gw/gl:.2f}")

# re-run, twice, check determinism + entry-hour distribution
for k in range(2):
    u2 = run(bars)
    print(f"RUN{k+2}: {len(u2)}t net ${sum(t['net'] for t in u2):,.0f} "
          f"first-hour {u2[0]['entry_ts']//3600} last-hour {u2[-1]['entry_ts']//3600}")

# entry-hour histogram of the trades (server hour)
from collections import Counter
hours = Counter((t["entry_ts"] // 3600) % 24 for t in run(bars))
print("entry-hour histogram:", dict(sorted(hours.items())))