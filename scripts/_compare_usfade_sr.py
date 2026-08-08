"""Final comparison for the replacement report:
session_reversion lb1440 top8 h24 in the US band [14-19] (x2: sub-bands) vs the
incumbent usfade leg — full honest metrics at both spreads + DD for sizing."""
import sys, os, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import metrics

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

def make_usfade(w):
    sig = {"rule": "session_momentum", "lookback": 50, "pick": "n_worst",
           "top_n": 5, "side": "both", "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": "usfade", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": 24, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def full(spec, spread):
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                       spread_pips_map=spread)
    m = metrics(usd)
    return usd, m

def shuffle_z(spec, spread, iters=12, seed=42):
    rng = random.Random(seed)
    usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                       spread_pips_map=spread)
    real_m = sum(t["net"] for t in usd) / len(usd) if usd else 0.0
    means = []
    for _ in range(iters):
        sh = {s: b[:] for s, b in bars.items()}
        for s in sh:
            rng.shuffle(sh[s])
        u = run_strategy(sh, spec, volume=0.15, commission_per_lot=3.0,
                         spread_pips_map=spread)
        if u:
            means.append(sum(x["net"] for x in u) / len(u))
    sm = sum(means) / len(means)
    sd = st.stdev(means) if len(means) > 1 else 0.0
    below = sum(1 for mm in means if mm < real_m)
    return (real_m - sm) / sd if sd > 0 else 0.0, below, len(means)

print("=== candidate: session_reversion lb1440 top8 h24 ===")
for tag, wname, w in [("14-19", "14-19", [14, 15, 16, 17, 18, 19]),
                      ("17-19", "17-19", [17, 18, 19]),
                      ("14-16", "14-16", [14, 15, 16])]:
    spec = make_sr([int(x) for x in wname.split("-")] if False else w)
    ut, mt = full(spec, SPREAD_T)
    um, mm = full(spec, SPREAD_M)
    zt, bt, _ = shuffle_z(spec, SPREAD_T)
    print(f"  sr[{wname}: {mt['trades']:>4}t  TYP net ${mt['net_pnl']:>8,.0f} "
          f"PF {mt['profit_factor']:5.2f} WR {mt['win_rate']:.3f} DD ${mt['max_drawdown']:>6,.0f}")
    print(f"           MEAS net ${mm['net_pnl']:>8,.0f} PF {mm['profit_factor']:5.2f} "
          f"| shuffle-z {zt:+.2f} ({bt}/12)")

print("\n=== incumbent: usfade (session_fade 14-19 lb50 top5 h24) ===")
us = make_usfade([14, 15, 16, 17, 18, 19])
ut, mt = full(us, SPREAD_T)
um, mm_ = full(us, SPREAD_M)
zt, bec, _ = shuffle_z(us, SPREAD_T)
print(f"  usfade: {mt['trades']:>4}t  TYP {mt['net_pnl']:>8,.0f} PF "
      f"{mt['profit_factor']:5.2f} WR {mt['win_rate']:.3f} DD ${mt['max_drawdown']:>6,.0f}")
print(f"           MEAS {mm_['net_pnl']:>8,.0f} PF {mm_['profit_factor']:5.2f} "
      f"| shuffle z {zt:+.2f} {bec}/12")