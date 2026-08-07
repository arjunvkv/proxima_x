"""Sustainability audit for carry_clock (top8/h72, Wed-Fri).

Aggregates to DATE level (near-independent bets; 8 pairs same-day same-direction
are correlated), computes t-stats, stability across halves, and cost drag.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map
import statistics as st

CARRY_V0 = {"EURUSD": 0, "USDJPY": 1, "GBPUSD": 1, "AUDUSD": 1, "EURJPY": 1,
            "GBPJPY": 1, "EURAUD": -1, "EURNZD": -1, "GBPAUD": -1, "GBPNZD": -1,
            "GBPCAD": 1, "AUDNZD": 0, "USDCAD": 1, "NZDUSD": 1, "EURGBP": -1,
            "EURCHF": 1, "USDCHF": 1, "AUDJPY": 1}
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

spec = StrategySpec.from_dict({"name":"carry","universe":UNIVERSE,
    "feed":{"kind":"bar","timeframe":"M5"},
    "signal":{"rule":"carry_clock","lookback":6,"pick":"n_worst","top_n":8,"side":"both",
              "fill_bar":1,"per_hour":False,"direction_map":CARRY_V0},
    "exit":{"mode":"sl_tp_hold","hold_bars":72,"stop_first":True,
            "jpy_sl_tp":(0.30,0.40),"non_jpy_sl_tp":(0.0030,0.0040)},
    "sessions":[14,15,16,17,18,19,20],"weekdays":[2,3,4],"base_lot":0.15})

usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
m = metrics(usd)
print(f"metrics: t={m['trades']} WR={m['win_rate']:.4f} PF={m['profit_factor']:.4f} "
      f"net=${m['net_pnl']:,.2f} exp/trade=${m['expectancy']:.3f} dd=${m['max_drawdown']:,}")

nets = [t["net"] for t in usd]

# 1. per-trade t-stat (naive, over-counts correlation)
mean = st.mean(nets); sd = st.stdev(nets); n = len(nets)
t_per = mean / (sd / (n ** 0.5))

# 2. date-aggregated t-stat (near-independent daily PnL; a trading day ~ UTC date of entry)
byd = defaultdict(float)
for t in usd:
    byd[t["entry_ts"] // 86400] += t["net"]
dvals = list(byd.values())
dm = st.mean(dvals); dsd = st.stdev(dvals); dn = len(dvals)
t_daily = dm / (dsd / (dn ** 0.5))

def autocorr(a, lag=1):
    a = list(a); x = a[:-lag]; y = a[lag:]
    if st.stdev(x) == 0 or st.stdev(y) == 0:
        return 0.0
    return st.covariance(x, y) / (st.stdev(x) * st.stdev(y))

ac1 = autocorr(dvals)

print(f"\nper-trade: N={n} t-stat={t_per:.2f}   (inflated by cross-sectional correlation)")
print(f"daily:     N={dn} mean=${dm:.2f} sd=${dsd:.2f} t-stat={t_daily:.2f} autocorr(1)={ac1:.2f}")

# 3. stability: first half vs second half of trading days
d1 = dvals[:dn // 2]; d2 = dvals[dn // 2:]
print(f"\nfirst-half  days={len(d1)} mean=${st.mean(d1):.2f} win-days={sum(1 for v in d1 if v>0)}/{len(d1)}")
print(f"second-half days={len(d2)} mean=${st.mean(d2):.2f} win-days={sum(1 for v in d2 if v>0)}/{len(d2)}")

# 4. cost drag: gross vs commission
gross = sum(t["gross_usd"] for t in usd)
comm = sum(t["commission"] for t in usd)
print(f"\ngross PnL=${gross:,.2f}  commission=${comm:,.2f}  "
      f"cost={100*comm/max(gross,1e-9):.1f}% of gross  net=${m['net_pnl']:,.2f}")

# 5. worst runs
wins = [1 if v > 0 else 0 for v in dvals]
worst = cur = 0
for w in wins:
    cur = cur + 1 if w == 0 else 0
    worst = max(worst, cur)
print(f"worst losing-day streak: {worst} consecutive down days")

# 6. per-month net (regime dependence)
from collections import OrderedDict
bym = OrderedDict()
for t in usd:
    ym = t["entry_ts"] // 86400 // 30
    bym.setdefault(ym, 0.0)
    bym[ym] += t["net"]
mo = list(bym.values())
print(f"\nmonths={len(mo)}  positive months={sum(1 for v in mo if v>0)}/{len(mo)}  "
      f"mean=${st.mean(mo):,.2f}  min=${min(mo):,.2f}  max=${max(mo):,.2f}")