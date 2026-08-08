"""Causal variant: enter at LAST session hour (19). All hours' first-bar
candidates known by hour-19 first-bar close; rank |score| across the whole
day's 6-hour pool, top 8, fill at hour-19 next-bar open. This IS causal and
IS what a live worker can do (poll at hour 19). Battery it fully."""
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
    "name": "sr_ent19", "universe": UNIVERSE,
    "feed": {"kind": "bar", "timeframe": "M5"},
    "signal": {"rule": "session_reversion", "lookback": 1440, "pick": "n_worst",
               "top_n": 8, "side": "both", "fill_bar": 1},
    "exit": {"mode": "sl_tp_hold", "hold_bars": 24, "stop_first": True,
             "jpy_sl_tp": (0.35, 0.45), "non_jpy_sl_tp": (0.0035, 0.0045)},
    "sessions": [19], "base_lot": 0.30})
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.8,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
LB, TOP, ENTRY_HOUR = 1440, 8, 19

def bar_hour(ts): return (ts // 3600) % 24

def anchor_price(bars, i, lb):
    seg = bars[max(0, i - lb):i]
    if not seg:
        return bars[i]["open"]
    return sum((b["high"] + b["low"] + b["close"]) / 3.0 for b in seg) / len(seg)

def first_bar_idx(bars, day, hour):
    for i in range(LB, len(bars) - 1):
        b = bars[i]
        if b["ts"] // 86400 == day and bar_hour(b["ts"]) == hour:
            return i
    return None

def day_pool(bars_map, day):
    """All session-hour first-bar candidates for the day (14-19), known by 20:00."""
    cands = []
    for sym, bars in bars_map.items():
        for hour in range(14, 20):
            sig = first_bar_idx(bars, day, hour)
            if sig is None:
                continue
            b = bars[sig]
            anc = anchor_price(bars, sig, LB)
            score = (anc - b["close"]) / anc * 100.0 if anc else 0.0
            cands.append((abs(score), score, sym, sig, hour))
    return cands

def run_book(bars_map, volume=0.30, comm=3.0):
    days = sorted({b["ts"] // 86400 for bars in bars_map.values() for b in bars})
    trades = []
    for day in days:
        pool = day_pool(bars_map, day)
        pool.sort(key=lambda x: -x[0])
        opened: set[str] = set()
        for _, score, sym, sig, hour in pool:
            if len(opened) >= TOP or sym in opened:
                continue
            opened.add(sym)
            bars = bars_map[sym]
            # causal: enter at hour-19 FIRST bar closed -> next bar open (i.e.,
            # the first bar with hour 19, +1). All day info known by then.
            eidx = first_bar_idx(bars, day, ENTRY_HOUR)
            if eidx is None or eidx + 1 >= len(bars):
                continue
            side = "BUY" if score >= 0 else "SELL"
            entry_idx = eidx + 1
            raw = simulate_exit(bars, entry_idx, side, SPEC, sym)
            t = trade_to_usd({**raw, "entry_ts": bars[entry_idx]["ts"],
                              "exit_ts": raw["exit_ts"], "side": side},
                             volume, commission_per_lot=comm, spread_pips_map=SPREAD)
            trades.append(t)
    return trades

bars = build_bars_map(UNIVERSE)
usd = run_book(bars)
m = metrics(usd)
print(f"CAUSAL-19 sr[14-19] lb{LB} top{TOP} @0.30 enter@20:00 (typical spread):")
print(f"  {m['trades']}t net ${m['net_pnl']:,.0f} PF {m['profit_factor']:.2f} "
      f"WR {m['win_rate']:.3f} DD ${m['max_drawdown']:,.0f}")
print("  sides:", dict(Counter(t["side"] for t in usd)))

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
for _ in range(8):
    sh = {s: b[:] for s, b in bars.items()}
    for s in sh:
        rng.shuffle(sh[s])
    u = run_book(sh)
    if u:
        means.append(sum(x["net"] for x in u) / len(u))
sm, sd = sum(means)/len(means), st.stdev(means)
below = sum(1 for mm in means if mm < real_m)
print(f"  shuffle-with-costs: shuffle {sm:+.2f}±{sd:+.2f}/t z={(real_m-sm)/sd:+.2f} "
      f"{below}/{len(means)} below (real {real_m:+.2f}/t)")