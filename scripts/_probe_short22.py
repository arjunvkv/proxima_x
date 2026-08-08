"""Deep-dive intraday_momentum_short h22: 24-shuffle × 3 seeds, neighbors 21/23,
trade identity (what does it SHORT?), month-by-month, and the h22 vs h21/h23 profile.
The engine's intraday_momentum is LONG the leading symbol; short side = fade it."""
import sys, os, json, random, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import metrics

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
bars = build_bars_map(UNIVERSE)

def make(w, lb=50, top=5, hold=24):
    sig = {"rule": "intraday_momentum", "lookback": lb, "pick": "n_worst",
           "top_n": top, "side": "short", "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": "intraday_momentum", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def run(bars_):
    return run_strategy(bars_, make([22]), volume=0.15, commission_per_lot=3.0,
                        spread_pips_map=SPREAD)

usd = run(bars)
m = metrics(usd)
print(f"h22: {m['trades']}t net ${m['net_pnl']:,.0f} PF {m['profit_factor']:.2f} "
      f"WR {m['win_rate']:.3f} DD ${m['max_drawdown']:,.0f}")
print("sides:", Counter(t["side"] for t in usd))
print("symbols:", dict(Counter(t["symbol"] for t in usd).most_common(8)))

# month-by-month
import datetime
by_month = {}
for t in usd:
    dt = datetime.datetime.utcfromtimestamp(t["entry_ts"])
    k = f"{dt.year}-{dt.month:02d}"
    by_month.setdefault(k, []).append(t["net"])
for k in sorted(by_month):
    nets = by_month[k]
    pos = sum(1 for x in nets if x > 0)
    print(f"  {k}: {len(nets):>3}t net ${sum(nets):>8,.0f} WR {pos/len(nets):.2f}")

print("\n24-shuffle × 3 seeds:")
rng = random.Random(42)
for seed in (42, 7, 2024):
    rng.seed(seed)
    real_m = sum(t["net"] for t in usd) / len(usd)
    means = []
    for _ in range(24):
        sh = {s: b[:] for s, b in bars.items()}
        for s in sh:
            rng.shuffle(sh[s])
        u = run(sh)
        if u:
            means.append(sum(x["net"] for x in u) / len(u))
    sm, sd = sum(means)/len(means), st.stdev(means)
    below = sum(1 for mm in means if mm < real_m)
    print(f"  seed {seed}: shuff ${sm:,.0f}±${sd:,.0f} z {(real_m-sm)/sd:+.2f} "
          f"{below}/24 below")

print("\nneighbors h21 / h23 (same rule, short side):")
for h in (21, 23, 20):
    u = run_strategy(bars, make([h]), volume=0.15, commission_per_lot=3.0,
                     spread_pips_map=SPREAD)
    mm = metrics(u)
    print(f"  h{h}: {mm['trades']}t net {mm['net_pnl']:>9,.0f} PF "
          f"{mm['profit_factor']:.2f} WR {mm['win_rate']:.3f}")