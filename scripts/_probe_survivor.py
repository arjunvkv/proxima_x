"""Focus: the single 2b PASS (session_reversion 17-19 lb1440 top8 h24).
Question: is this a REAL replacement edge, or the day-bucket artifact passing
one window's shuffle by luck?

A: 24-shuffle z (2 seeds) — stable?
B: economics by window (2-4,7-9,9-11,12-13,14-16,17-19) — does it LOCALIZE?
   A real clock edge should be meaningfully better at its own window.
C: hourly breakdown inside 17-19 (which hours actually drive P&L)
D: trades/day + correlation with tokyo daily PnL (book-level value)
"""
import sys, os, json, random, statistics as st
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
WINS = {"2-4":[2,3,4],"7-9":[7,8,9],"9-11":[9,10,11],"12-13":[12,13],
        "14-16":[14,15,16],"17-19":[17,18,19]}
bars = build_bars_map(UNIVERSE)

def make(w, lb=1440, top=8, hold=24, ph=False):
    sig = {"rule": "session_reversion", "lookback": lb, "pick": "n_worst",
           "top_n": top, "side": "both", "fill_bar": 1, "per_hour": ph}
    return StrategySpec.from_dict({
        "name": "session_reversion", "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def run(bars_, spec, spread=SPREAD):
    return run_strategy(bars_, spec, volume=0.15, commission_per_lot=3.0,
                        spread_pips_map=spread)

print("=== A: 24-shuffle z, 2 seeds (17-19) ===")
spec = make([17, 18, 19])
usd = run(bars, spec)
real_m = sum(t["net"] for t in usd) / len(usd)
for seed in (42, 7):
    rng = random.Random(seed)
    means = []
    for _ in range(24):
        sh = {s: b[:] for s, b in bars.items()}
        for s in sh:
            rng.shuffle(sh[s])
        u = run(sh, spec)
        if u:
            means.append(sum(x["net"] for x in u) / len(u))
    sm, sd = sum(means)/len(means), st.stdev(means)
    below = sum(1 for m_ in means if m_ < real_m)
    print(f"  seed {seed}: shuff-mean ${sm:,.0f} sd ${sd:,.0f} "
          f"z {(real_m-sm)/sd:+.2f} {below}/24 below")

print("\n=== B: economics by window (same spec, only window changes) ===")
nets = {}
for wname, w in WINS.items():
    usd_ = run(bars, make(w))
    m_ = metrics(usd_)
    nets[wname] = m_["net_pnl"]
    print(f"  {wname:<6} {m_['trades']:>4}t net ${m_['net_pnl']:>9,.0f} "
          f"PF {m_['profit_factor']:5.2f} WR {m_['win_rate']:.3f}")
med = st.median(list(nets.values()))
print(f"  median {med:,.0f}  own-window {nets['17-19']:,.0f} "
      f"({nets['17-19']/med:.2f}x median)")

print("\n=== C: hourly split inside 17-19 (per_hour=1) ===")
for h in (17, 18, 19):
    ph = make([h], ph=True)
    usd_h = run(bars, ph)
    m_h = metrics(usd_h)
    print(f"  h{h}: {m_h['trades']:>4}t net ${m_h['net_pnl']:>9,.0f} "
          f"PF {m_h['profit_factor']:5.2f}")

print("\n=== D: daily PnL overlap vs tokyo (book additivity) ===")
tokyo = make([0], 6, 3, 12)
usd_t = run(bars, tokyo)
usd_17 = run(bars, make([17, 18, 19]))
def daily(usd):
    d = {}
    for t in usd:
        k = t["entry_ts"] // 86400
        d[k] = d.get(k, 0) + t["net"]
    return d
dt, d17 = daily(usd_t), daily(usd_17)
days = sorted(set(dt) | set(d17))
corr_days = [x for x in days if x in dt and x in d17]
import math
if len(corr_days) > 10:
    xs = [dt[k] for k in corr_days]; ys = [d17[k] for k in corr_days]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / len(xs)
    sx = (sum((x-mx)**2 for x in xs)/len(xs)) ** 0.5
    sy = (sum((y-my)**2 for y in ys)/len(ys)) ** 0.5
    r = cov / (sx*sy) if sx and sy else 0
    print(f"  tokyo days {len(dt)}, 17-19 days {len(d17)}, {len(corr_days)} common")
    print(f"  daily-PnL correlation r = {r:+.3f}")
else:
    print(f"  too few common days: {len(corr_days)}")