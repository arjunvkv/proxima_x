"""Close the coverage gap: 4 real engine rules never swept (range_reversion,
intraday_momentum, round_number_bounce, carry_clock) + range_breakout short-
side complement. Full window set, BOTH spreads, cheap gate."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD_T = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
            "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
            "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
            "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
SPREAD_M = {"EURUSD":0.60,"USDJPY":3.50,"GBPUSD":1.30,"AUDUSD":0.90,"EURJPY":5.70,
            "GBPJPY":7.10,"EURAUD":2.90,"EURNZD":5.10,"GBPAUD":4.90,"GBPNZD":6.30,
            "GBPCAD":4.10,"AUDNZD":3.50,"USDCAD":1.60,"NZDUSD":1.20,"EURGBP":1.70,
            "EURCHF":2.60,"USDCHF":1.30,"AUDJPY":4.20}
WINDOWS = {
    "h0":[0],"h1":[1],"h2":[2],"h3":[3],"h4":[4],"h5":[5],"h6":[6],"h7":[7],
    "h8":[8],"h9":[9],"h10":[10],"h11":[11],"h12":[12],"h13":[13],"h14":[14],
    "h15":[15],"h16":[16],"h17":[17],"h18":[18],"h19":[19],"h20":[20],"h21":[21],
    "h22":[22],"h23":[23],
    "0-1":[0,1],"2-3":[2,3],"4-5":[4,5],"6-8":[6,7,8],"7-9":[7,8,9],"9-11":[9,10,11],
    "12-13":[12,13],"14-16":[14,15,16],"17-19":[17,18,19],"14-19":[14,15,16,17,18,19],
}
RULES = {
    "range_reversion":    {"lb":50, "top":5, "hold":24, "side":"both"},
    "intraday_momentum":  {"lb":50, "top":5, "hold":24, "side":"both"},
    "round_number_bounce":{"lb":50, "top":5, "hold":24, "side":"both"},
    "carry_clock":        {"lb":50, "top":5, "hold":24, "side":"both"},
    "range_reversion_short": {"lb":50, "top":5, "hold":24, "side":"short"},
    "intraday_momentum_short":{"lb":50, "top":5, "hold":24, "side":"short"},
}

def make_spec(rule, sess, lb, top, hold, side):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": sess, "base_lot": 0.15})

def bench(usd):
    if not usd:
        return None
    wins = [t for t in usd if t["net"] > 0]
    losses = [t for t in usd if t["net"] < 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in losses)
    return {"trades": len(usd), "wr": len(wins)/len(usd),
            "pf": (gw/gl) if gl else (99.9 if gw else 0.0),
            "net": sum(t["net"] for t in usd)}

bars = build_bars_map(UNIVERSE)
t0 = time.time()
survivors, lines = [], []
for rule, p in RULES.items():
    real_rule = rule.replace("_short", "")
    for wname, w in WINDOWS.items():
        spec = make_spec(real_rule, w, p["lb"], p["top"], p["hold"], p["side"])
        usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                           spread_pips_map=SPREAD_T)
        m = bench2 = None
        m = bench(usd)
        if not m or m["trades"] < 12:
            continue
        m2 = bench(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                                spread_pips_map=SPREAD_M))
        net2 = m2["net"] if m2 else -1e9
        ok = m["pf"] > 1.2 and m["net"] > 0 and net2 > 0
        tag = "  SURVIVES-both" if ok else ""
        lines.append(f"{rule:<24}{wname:<8} {m['trades']:>4} WR {m['wr']:.3f} "
                     f"PF {m['pf']:6.2f} net ${m['net']:>8,.0f} "
                     f"worst ${net2:>8,.0f}{tag}")
        if ok:
            survivors.append({"rule": rule, "window": wname, "m": m,
                              "worst_net": net2})
print("\n".join(lines[-140:]))
print(f"\n--- survivors: {len(survivors)} ({time.time()-t0:.0f}s)")
for s in survivors:
    print(f"  SURVIVOR: {s['rule']} {s['window']} PF {s['m']['pf']:.2f} "
          f"WR {s['m']['wr']:.3f} net ${s['m']['net']:.0f} "
          f"worst ${s['worst_net']:.0f}")
json.dump(survivors, open(os.path.join(os.path.dirname(__file__),
                                       "_sweep_gap_survivors.json"), "w"), indent=2)