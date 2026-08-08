"""Stage 1b: focused param grid at the DEAD windows (2-4, 7-9, 9-11, 12-16, 17-19)
for the rule families that show any signal anywhere, plus per_hour bucketing.
The coarse sweep used ONE config (lb50/top5/hold24) for all rules; this widens
lb/top/hold to the regions the surviving legs actually use (tokyo lb6, cascade/
london lb1440, usfade lb50) — cheap gate at BOTH spreads; survivors -> JSON.
"""
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

WINDOWS = {"2-4":[2,3,4],"7-9":[7,8,9],"9-11":[9,10,11],"12-13":[12,13],
           "14-16":[14,15,16],"17-19":[17,18,19]}
GRID = {
    # rule -> list of (lb, top, hold, per_hour)
    "session_exhaustion": [(6,3,12,False),(50,5,24,False),(300,5,12,False),
                           (1440,5,12,False),(1440,8,24,False),(6,3,24,False),
                           (1440,5,24,True),(50,5,24,True),(300,8,24,False)],
    "session_reversion":  [(50,5,24,False),(300,5,12,False),(1440,5,12,False),
                           (50,5,12,True),(1440,8,24,False)],
    "big_move_fade":      [(50,5,24,False),(300,5,12,False),(1440,5,24,False),
                           (50,5,24,True)],
    "weekend_gap":        [(50,5,24,False),(300,5,12,False),(1440,5,24,False)],
    "lead_lag":           [(50,5,24,False),(300,5,12,False),(1440,5,24,False)],
    "range_reversion":    [(50,5,24,False),(300,5,12,False),(1440,5,24,False)],
    "liquidity_sweep":    [(50,5,24,False),(1440,5,24,False)],
}

def make_spec(rule, sess, lb, top, hold, per_hour, side="both"):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1, "per_hour": per_hour}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def rec(usd):
    if not usd or len(usd) < 12:
        return None
    wins = [t for t in usd if t["net"] > 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in usd if t["net"] < 0)
    return {"trades": len(usd), "wr": len(wins)/len(usd),
            "pf": (gw/gl if gl else (99.9 if gw else 0.0)),
            "net": sum(t["net"] for t in usd)}

bars = build_bars_map(UNIVERSE)
t0 = time.time()
survivors = []
lines = []
for rule, cells in GRID.items():
    for (lb, top, hold, ph) in cells:
        for wname, w in WINDOWS.items():
            spec = make_spec(rule, w, lb, top, hold, ph)
            m1 = rec(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                                  spread_pips_map=SPREAD_T))
            if not m1 or m1["pf"] <= 1.05 or m1["net"] <= 0:
                continue
            m2 = rec(run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                                  spread_pips_map=SPREAD_M))
            net2 = m2["net"] if m2 else -1e9
            ok = net2 > 0 and m1["pf"] > 1.2
            tag = "  both-spread" if ok else ""
            lines.append(f"{rule:<20}{wname:<6}(lb{lb:<4} top{top} h{hold:<2}"
                         f" ph={int(ph)}) WR {m1['wr']:.3f} PF {m1['pf']:.2f} "
                         f"net ${m1['net']:>7,.0f} worst ${net2:>7,.0f}{tag}")
            if ok:
                survivors.append({"rule": rule, "window": wname, "lb": lb,
                                  "top": top, "hold": hold, "per_hour": ph,
                                  "m": m1, "worst_net": net2})
print("\n".join(lines))
print(f"\n--- {time.time()-t0:.0f}s  BOTH-spread survivors at dead windows: {len(survivors)}")
for s in survivors:
    print(f"  {s['rule']} {s['window']} lb{s['lb']} top{s['top']} h{s['hold']} "
          f"ph={int(s['per_hour'])}  PF {s['m']['pf']:.2f} WR {s['m']['wr']:.3f} "
          f"net ${s['m']['net']:.0f} worst ${s['worst_net']:.0f}")
json.dump(survivors, open(os.path.join(os.path.dirname(__file__),
                                       "_sweep1b_survivors.json"), "w"), indent=2)