"""LIVE-FAITHFUL verification of session_reversion [14-19] lb1440 top8 h24.

Simulates EXACTLY what run_core_book_live.py will do:
- poll at hour 14 (first session hour): take the FIRST closed M5 bar of hour 14
- score = (anchor - close)/anchor*100, anchor = trailing mean of typical price
  over lb=1440 closed bars BEFORE the signal bar (engine _session_avg_price)
- rank |score| desc across symbols, top_n=8, unique symbol per day
- side = BUY if score>=0 (discount) else SELL (premium)  [engine _signal_side]
- fill at NEXT bar open; exit sl_tp_hold hold=24, stop_first,
  jpy (0.35,0.45) / non-jpy (0.0035,0.0045) = engine ExitSpec defaults
- cost: typical FTMO busy spread + $3.0/lot/side commission, SAME engine pnl.py
- battery: 12x shuffle WITH costs, walk-forward, month split
"""
import sys, os, random, statistics as st, datetime
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import metrics, walk_forward
from proxima_ops.backtest.engine import simulate_exit
from proxima_ops.backtest.spec import StrategySpec
from proxima_ops.backtest.pnl import trade_to_usd

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

SPEC = StrategySpec.from_dict({
    "name": "sr_livecheck", "universe": UNIVERSE,
    "feed": {"kind": "bar", "timeframe": "M5"},
    "signal": {"rule": "session_reversion", "lookback": 1440, "pick": "n_worst",
               "top_n": 8, "side": "both", "fill_bar": 1},
    "exit": {"mode": "sl_tp_hold", "hold_bars": 24, "stop_first": True,
             "jpy_sl_tp": (0.35, 0.45), "non_jpy_sl_tp": (0.0035, 0.0045)},
    "sessions": [14, 15, 16, 17, 18, 19], "base_lot": 0.30})

SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
LB, TOP, HOLD, HOUR = 1440, 8, 24, 14

def bar_hour(ts): return (ts // 3600) % 24

def anchor_price(bars, i, lb):
    seg = bars[max(0, i - lb):i]
    if not seg:
        return bars[i]["open"]
    return sum((b["high"] + b["low"] + b["close"]) / 3.0 for b in seg) / len(seg)

def first_signal_idx(bars_map, sym, day):
    bars = bars_map[sym]
    for i in range(LB, len(bars) - 1):
        b = bars[i]
        if b["ts"] // 86400 == day and bar_hour(b["ts"]) == HOUR:
            return i
    return None

def live_signal_day(bars_map, day):
    """Greedy causal pick: first closed bar of hour HOUR per symbol, rank |score|."""
    cands = []
    for sym, bars in bars_map.items():
        sig = first_signal_idx(bars_map, sym, day)
        if sig is None:
            continue
        b = bars[sig]
        anc = anchor_price(bars, sig, LB)
        score = (anc - b["close"]) / anc * 100.0 if anc else 0.0
        cands.append((abs(score), score, sym, sig))
    cands.sort(key=lambda x: -x[0])
    return [{"symbol": c[2], "score": c[1], "sig": c[3],
             "side": "BUY" if c[1] >= 0 else "SELL"} for c in cands[:TOP]]

def run_book(bars_map, volume=0.30, comm=3.0):
    days = sorted({b["ts"] // 86400 for bars in bars_map.values() for b in bars})
    trades = []
    for day in days:
        for p in live_signal_day(bars_map, day):
            sym = p["symbol"]
            bars = bars_map[sym]
            sig = p["sig"]
            if sig + 1 >= len(bars):
                continue
            raw = simulate_exit(bars, sig + 1, p["side"], SPEC, sym)
            t = trade_to_usd({**raw, "entry_ts": bars[sig + 1]["ts"],
                              "exit_ts": raw["exit_ts"], "side": p["side"]},
                             volume, tick_value_map=None,
                             commission_per_lot=comm, spread_pips_map=SPREAD)
            trades.append(t)
    return trades

bars = build_bars_map(UNIVERSE)
usd = run_book(bars)
m = metrics(usd)
print(f"LIVE-FAITHFUL sr[{HOUR}] lb{LB} top{TOP} h24 @0.30 (typical spread + $3 comm):")
print(f"  {m['trades']}t net ${m['net_pnl']:,.0f} PF {m['profit_factor']:.2f} "
      f"WR {m['win_rate']:.3f} DD ${m['max_drawdown']:,.0f}")
print("  sides:", dict(Counter(t["side"] for t in usd)))
print("  symbols:", dict(Counter(t["symbol"] for t in usd).most_common(6)))

by_m = {}
for t in usd:
    k = datetime.datetime.fromtimestamp(t["entry_ts"], datetime.UTC).strftime("%Y-%m")
    by_m.setdefault(k, []).append(t["net"])
for k in sorted(by_m):
    nets = by_m[k]
    print(f"  {k}: {len(nets):>3}t net {sum(nets):>8,.0f} "
          f"WR {sum(1 for x in nets if x>0)/len(nets):.2f}")

wf = walk_forward(usd, train_size=300, test_size=100)
print(f"  walk-forward: {wf['n_windows']} windows, positive {wf['positive_share']:.2f} "
      f"stable={wf['stable']}")

rng = random.Random(42)
real_m = sum(t["net"] for t in usd) / len(usd)
means = []
for _ in range(12):
    sh = {s: b[:] for s, b in bars.items()}
    for s in sh:
        rng.shuffle(sh[s])
    u = run_book(sh)
    if u:
        means.append(sum(x["net"] for x in u) / len(u))
sm, sd = sum(means)/len(means), st.stdev(means)
below = sum(1 for mm in means if mm < real_m)
print(f"  shuffle-with-costs: shuffle ${sm:,.2f}±${sd:,.2f}/t z={(real_m-sm)/sd:+.2f} "
      f"{below}/4 below (real ${real_m:,.2f}/t)")