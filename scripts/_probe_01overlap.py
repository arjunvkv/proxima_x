"""Are the 0-1 'survivors' (weekend_gap/lead_lag/big_move_fade/session_reversion)
actually DIFFERENT trades from the tokyo flagship, or the same dance?
Jaccard on (day, symbol, side). Also their network: do the survivors overlap
EACH OTHER? If all 0-1 winners are the same few symbols per day -> one edge."""
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

def make(rule, w, lb=50, top=5, hold=24, side="both", ph=False, pick="n_worst"):
    sig = {"rule": rule, "lookback": lb, "pick": pick, "top_n": top,
           "side": side, "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def run(spec):
    return run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                        spread_pips_map=SPREAD)

def ident(usd):
    return {(t["entry_ts"] // 86400, t["symbol"], t["side"]) for t in usd}

CAND = {
    "tokyo_flagship": make("session_exhaustion", [0], 6, 3, 12),
    "weekend_gap":    make("weekend_gap", [0, 1], 50, 5, 24),
    "lead_lag":       make("lead_lag", [0, 1], 50, 5, 24),
    "big_move_fade":  make("big_move_fade", [0, 1], 50, 5, 24),
    "session_rev":    make("session_reversion", [0, 1], 50, 5, 24),
    "exhaustion01":   make("session_exhaustion", [0, 1], 50, 5, 24),
    "momentum01":     make("session_momentum", [0, 1], 50, 5, 24),
}
sets = {}
nets = {}
for name, spec in CAND.items():
    usd = run(spec)
    sets[name] = ident(usd)
    nets[name] = sum(t["net"] for t in usd)
    print(f"{name:<18} {len(usd):>4}t  net ${nets[name]:>8,.0f}  distinct {len(sets[name])}")

names = list(CAND)
print("\n== pairwise Jaccard (day-sym-side) ==")
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = sets[names[i]], sets[names[j]]
        inter = len(a & b)
        union = len(a | b)
        jac = inter / union if union else 0.0
        # how much of the SMALLER set is contained in the bigger?
        cont = inter / min(len(a), len(b)) if min(len(a), len(b)) else 0.0
        mark = "  <-- HIGH" if (jac > 0.35 or cont > 0.5) else ""
        print(f"  {names[i]:<16} vs {names[j]:<16}: J={jac:.2f} cont={cont:.2f}{mark}")