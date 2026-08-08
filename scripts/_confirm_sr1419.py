"""Confirm the full battery for candidate session_reversion 14-19 lb1440 top8 h24
(the direct usfade-band replacement): walk-forward + tokyo overlap + month split."""
import sys, os, random, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import walk_forward, metrics

SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

def make_sr(w, lb=1440, top=8, hold=24, ph=False):
    sig = {"rule": "session_reversion", "lookback": lb, "pick": "n_worst",
           "top_n": top, "side": "both", "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": "session_reversion", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def make_tokyo():
    sig = {"rule": "session_exhaustion", "lookback": 6, "pick": "n_worst",
           "top_n": 3, "side": "both", "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": "tokyo", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True},
        "sessions": [0], "base_lot": 0.15})

spec = make_sr([14, 15, 16, 17, 18, 19])
usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                   spread_pips_map=SPREAD)
m = metrics(usd)
print(f"sr[14-19]: {m['trades']}t net {m['net_pnl']:,.0f} PF {m['profit_factor']:.2f} "
      f"WR {m['win_rate']:.3f} DD {m['max_drawdown']:,.0f}")

wf = walk_forward([t for t in usd], train_size=300, test_size=100)
print(f"walk-forward: {wf['n_windows']} windows, positive_share {wf['positive_share']} "
      f"stable={wf['stable']}")
for w in wf["windows"]:
    print(f"  win@{w['test_start']:>5} n={w['n']:>3} net ${w['net']:>7,.0f} "
          f"WR {w['wr']:.3f} PF {w['pf']:.2f}")

tokyo_ids = {(t["entry_ts"]//86400, t["symbol"], t["side"])
             for t in run_strategy(bars, make_tokyo(), volume=0.15,
                                   commission_per_lot=3.0, spread_pips_map=SPREAD)}
ids = {(t["entry_ts"]//86400, t["symbol"], t["side"]) for t in usd}
inter = len(ids & tokyo_ids)
union = len(ids | tokyo_ids)
print(f"tokyo overlap: {inter}/{union} J={inter/union:.3f}")

# month breakdown
import datetime
by_m = {}
for t in usd:
    k = datetime.datetime.fromtimestamp(t["entry_ts"], datetime.UTC).strftime("%Y-%m")
    by_m.setdefault(k, []).append(t["net"])
print("months:")
for k in sorted(by_m):
    nets = by_m[k]
    print(f"  {k}: {len(nets):>3}t net {sum(nets):>8,.0f} "
          f"WR {sum(1 for x in nets if x>0)/len(nets):.2f}")