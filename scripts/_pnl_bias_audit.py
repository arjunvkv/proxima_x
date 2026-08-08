"""PnL-bias audit: engine fills at bar OPEN with NO bid/ask spread model.
A live BUY pays ask (=open+spread), closes at bid (=open-spread) -> the engine
is optimistic by ~1 spread per round trip. Measure the book's edges after a
realistic per-pair spread haircut (FTMO demo spreads, pips)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.pnl import pip_size, pip_value_usd
from collections import defaultdict

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

# realistic FTMO-Demo spread in pips (mid, non-news)
SPREADS_PIPS = {"EURUSD":0.7,"USDJPY":0.8,"GBPUSD":1.1,"AUDUSD":0.9,"EURJPY":1.2,
               "GBPJPY":1.8,"EURAUD":1.8,"EURNZD":2.2,"GBPAUD":2.0,"GBPNZD":2.6,
               "GBPCAD":2.0,"AUDNZD":1.8,"USDCAD":1.0,"NZDUSD":1.2,"EURGBP":0.9,
               "EURCHF":1.0,"USDCHF":1.0,"AUDJPY":1.1}

CONFIGS = {
    "tokyo":   {"sessions": [0],             "lookback": 6,    "top_n": 3, "hold_bars": 12},
    "cascade": {"sessions": [2, 3, 4],       "lookback": 1440, "top_n": 8, "hold_bars": 24},
    "london":  {"sessions": [7, 8, 9],       "lookback": 1440, "top_n": 5, "hold_bars": 12},
    "usfade":  {"sessions": [14,15,16,17,18,19], "lookback": 50, "top_n": 5, "hold_bars": 24},
}

def make_spec(name, c):
    return StrategySpec.from_dict({
        "name": name, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_exhaustion", "lookback": c["lookback"], "pick": "n_worst",
                   "top_n": c["top_n"], "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": c["hold_bars"], "stop_first": True},
        "sessions": c["sessions"], "base_lot": 0.15,
    })

def spread_cost(sym, entry, volume):
    """USD cost of one round-trip spread at rng lots (pips x pip-value)."""
    return SPREADS_PIPS[sym] * pip_value_usd(sym, entry) * volume

bars = build_bars_map(UNIVERSE)
print(f"{'leg':<10}{'gross':>10}{'comm':>8}{'net':>10}{'spread$':>10}{'net-spd':>10}{'pf':>6}{'pf-spd':>8}")
tot = {}
for name, c in CONFIGS.items():
    tr = run_strategy(bars, make_spec(name, c), volume=0.15, raw=True)
    usd = run_strategy(bars, make_spec(name, c), volume=0.15, commission_per_lot=3.5)
    spd = sum(spread_cost(t["symbol"], t["entry"], 0.15) for t in tr)
    net_spd = sum(t["net"] for t in usd) - spd
    m = metrics(usd)
    pf_spd = (sum(t["net"] for t in usd if t["net"]>0) / max(1e-9, -sum(t["net"] for t in usd if t["net"]<0) + spd))
    print(f"{name:<10}{sum(t['gross_usd'] for t in usd):>10,.0f}{sum(t['net'] for t in usd):>10,.0f}"
          f"{sum(t['net'] for t in usd):>10,.0f}{spd:>10,.0f}{net_spd:>10,.0f}"
          f"{m['profit_factor']:>6.2f}{pf_spd:>8.2f}")
    tot[name] = net_spd
print(f"\nbook net after spread haircut: {sum(tot.values()):,.0f} "
      f"(was {sum(sum(t['net'] for t in run_strategy(bars, make_spec(n,c), volume=0.15, commission_per_lot=3.5)) for n,c in CONFIGS.items()):,.0f})")