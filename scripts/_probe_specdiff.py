"""Which spec build gives 5,353 and which gives 8,291? Diff the constructions."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY", "EURAUD",
            "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD",
            "EURGBP", "EURCHF", "USDCHF", "AUDJPY"]
SPREAD_T = {"EURUSD": 0.8, "USDJPY": 1.2, "GBPUSD": 1.5, "AUDUSD": 1.1,
            "EURJPY": 2.2, "GBPJPY": 3.0, "EURAUD": 2.8, "EURNZD": 3.4,
            "GBPAUD": 3.6, "GBPNZD": 4.4, "GBPCAD": 3.2, "AUDNZD": 2.6,
            "USDCAD": 1.8, "NZDUSD": 1.4, "EURGBP": 1.2, "EURCHF": 1.8,
            "USDCHF": 1.4, "AUDJPY": 1.8}

bars = build_bars_map(UNIVERSE)
entries = json.load(open(os.path.join(os.path.dirname(__file__),
                                      "_sweep1b_survivors.json")))
s = [x for x in entries if x["rule"] == "session_reversion"
     and x["window"] == "17-19" and x["lb"] == 1440 and x["top"] == 8
     and x["hold"] == 24][0]
win = [int(v) for v in s["window"].split("-")]
print("entry:", s["rule"], s["window"], "lb", s["lb"], "top", s["top"],
      "hold", s["hold"], "ph", s["per_hour"])

def make(rule, w, lb, top, hold, ph, side="both"):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

spec = make(s["rule"], [17, 18, 19], s["lb"], s["top"], s["hold"], s["per_hour"])
for ph in (False, 0, True, None):
    sp = make(s["rule"], [17, 18, 19], s["lb"], s["top"], s["hold"], ph)
    usd = run_strategy(bars, sp, volume=0.15, commission_per_lot=3.0,
                       spread_pips_map=SPREAD_T)
    net = sum(t["net"] for t in usd)
    wins = [t for t in usd if t["net"] > 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in usd if t["net"] < 0)
    print(f"ph={ph!r}: {len(usd)}t net ${net:,.0f} PF {gw/gl:.2f}")