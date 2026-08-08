"""Probe: does session_reversion produce the SAME trades at different windows?
Uniform high results across all windows = degenerate/mask or trend artifact.
Also print trade identity fingerprints + overlap vs the book's tokyo leg."""
import sys, os, hashlib
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

def make(rule, w, lb, top, hold, side="both", ph=False):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def ident(usd):
    """Day-level identity set: (day, symbol, side) per trade."""
    return {(t["entry_ts"] // 86400, t["symbol"], t["side"]) for t in usd}

def rec(usd):
    wins = [t for t in usd if t["net"] > 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in usd if t["net"] < 0)
    return len(usd), len(wins)/len(usd), (gw/gl if gl else 99.9), sum(t["net"] for t in usd)

print("=== session_reversion lb1440 top8 h24 at each window ===")
sets = {}
for wname, w in [("2-4",[2,3,4]),("7-9",[7,8,9]),("9-11",[9,10,11]),
                 ("14-16",[14,15,16]),("17-19",[17,18,19])]:
    spec = make("session_reversion", w, 1440, 8, 24)
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                       spread_pips_map=SPREAD)
    n, wr, pf, net = rec(usd)
    ids = ident(usd)
    sets[wname] = ids
    print(f"{wname}: {n:>4}t WR {wr:.3f} PF {pf:6.2f} net ${net:>8,.0f} "
          f"distinct-day-sym {len(ids)}")

print("\n=== pairwise overlap (Jaccard on day-sym-side) ===")
names = list(sets)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = sets[names[i]], sets[names[j]]
        inter = len(a & b)
        union = len(a | b)
        print(f"  {names[i]} vs {names[j]}: {inter}/{union} = {inter/union:.3f}")

print("\n=== does per-hour bucketing change the fill? ===")
for wname, w in [("17-19",[17,18,19])]:
    for ph in (False, True):
        spec = make("session_reversion", w, 1440, 8, 24, ph=ph)
        usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                           spread_pips_map=SPREAD)
        n, WR, pf, net = rec(usd)
        print(f"  {wname} ph={int(ph)}: n={n} WR {WR:.3f} PF {pf:.2f} net ${net:,.0f}")