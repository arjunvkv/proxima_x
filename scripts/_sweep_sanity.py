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

def make(rule, sessions, side="both", lb=50, top=5, hold=24):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": sessions, "base_lot": 0.15})

for rule, win in [("vol_compress_fade", [0, 1]), ("intraday_momentum_london", [0]),
                  ("lead_lag", [0]), ("weekend_gap", [0])]:
    spec = make(rule, win)
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                       spread_pips_map=SPREAD)
    wins = [t for t in usd if t["net"] > 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in usd if t["net"] < 0)
    print(f"{rule:<26} {win} {len(usd):>4}t  WR {len(wins)/len(usd):.3f}  "
          f"PF {gw/gl if gl else 99:.2f}  net ${sum(t['net'] for t in usd):,.0f}")
    # first 3 trade identities (symbol, side, entry_ts)
    ids = [(t["symbol"], t["side"], t["entry_ts"] // 3600) for t in usd[:3]]
    print(f"    first: {ids}")