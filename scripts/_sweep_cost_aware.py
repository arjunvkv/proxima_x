"""COST-AWARE all-edges × all-window sweep (replacement hunt).

Every engine rule (signed + legacy) × every hour/block window × side, charged
REAL FTMO spread (typical busy-session, then re-checked at measured worst-case)
+ $3.0/lot/side commission, 0.15-lot basis. Survivors = positive net at BOTH
spread regimes with PF > 1.2. Stage 2 (shuffle-with-costs + walk-forward) runs
on this shortlist.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD_TYPICAL = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
                  "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
                  "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
                  "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
SPREAD_MEASURED = {"EURUSD":0.60,"USDJPY":3.50,"GBPUSD":1.30,"AUDUSD":0.90,"EURJPY":5.70,
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
    "session_reversion":   {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "range_momentum":      {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "range_breakout":      {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "liquidity_sweep":     {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "session_open_breakout":{"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "big_move_fade":       {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "thin_market_fade":    {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "vol_compress_fade":   {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "weekend_gap":         {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "round_barrier_fade":  {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "fix_reversal":        {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "domestic_hours":      {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "day_of_week_usd":     {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "cross_momentum":      {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "intraday_momentum_london": {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "lead_lag":            {"lb":50, "top":5, "hold":24, "side":"both", "legacy":False},
    "session_exhaustion":  {"lb":48, "top":5, "hold":24, "side":"both", "legacy":True},
    "session_momentum":    {"lb":48, "top":5, "hold":24, "side":"both", "legacy":True},
    "session_reversion_short": {"lb":50, "top":5, "hold":24, "side":"short", "legacy":False},
    "range_breakout_short": {"lb":50, "top":5, "hold":24, "side":"short", "legacy":False},
}

def make_spec(rule, sess, params):
    sig = {"rule": rule, "lookback": params["lb"], "pick": "n_worst",
           "top_n": params["top"], "side": params["side"], "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": params["hold"], "stop_first": True},
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
survivors = []
count = 0
lines = []
for rule, params in RULES.items():
    spec_rule = rule.replace("_short", "")
    for wname, w in WINDOWS.items():
        spec = make_spec(spec_rule, w, params)
        spec.signal.rule = spec_rule  # legacy + short renames resolve here
        usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                           spread_pips_map=SPREAD_TYPICAL)
        m = bench(usd)
        if not m or m["trades"] < 12:
            continue
        count += 1
        m2 = bench(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                                spread_pips_map=SPREAD_MEASURED))
        net2 = m2["net"] if m2 else -1e9
        gate_ok = m["pf"] > 1.2 and m["net"] > 0
        worst_ok = net2 > 0
        if gate_ok and worst_ok:
            survivors.append({"rule": rule, "window": wname, "m": m, "worst_net": net2})
            tag = "  SURVIVES-both"
        elif gate_ok:
            tag = "  (typical only)"
        else:
            tag = ""
        lines.append(f"{rule:<26}{wname:<8} {m['trades']:>4}  WR {m['wr']:.3f}  "
                     f"PF {m['pf']:6.2f}  net ${m['net']:>8,.0f}  "
                     f"worst ${net2:>8,.0f}{tag}")
print("\n".join(lines[-120:]))
print(f"\n--- {count} cells with trades; BOTH-spread survivors: {len(survivors)} "
      f"({time.time()-t0:.0f}s)")
for s in survivors:
    print(f"  SURVIVOR: {s['rule']} {s['window']}  PF {s['m']['pf']:.2f} "
          f"WR {s['m']['wr']:.3f} net ${s['m']['net']:.0f} worst-net ${s['worst_net']:.0f}")
with open(os.path.join(os.path.dirname(__file__), "_sweep_cost_survivors.json"), "w") as f:
    json.dump(survivors, f, indent=2)