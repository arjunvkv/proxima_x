"""Last causal variant test: per-hour threshold entry for session_reversion.

At the FIRST closed bar of each session hour 14-19, enter every symbol whose
|score| >= thr at next-bar open (side by score sign), hold-24 SL/TP. Fully
causal: no pooling of future hours, no retroactive ranking.
"""
import sys, os, random, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import metrics
from proxima_ops.backtest.engine import simulate_exit
from proxima_ops.backtest.spec import StrategySpec
from proxima_ops.backtest.pnl import trade_to_usd

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPEC = StrategySpec.from_dict({
    "name": "sr_thr", "universe": UNIVERSE,
    "feed": {"kind": "bar", "timeframe": "M5"},
    "signal": {"rule": "session_reversion", "lookback": 1440, "pick": "n_worst",
               "top_n": 18, "side": "both", "fill_bar": 1},
    "exit": {"mode": "sl_tp_hold", "hold_bars": 24, "stop_first": True,
             "jpy_sl_tp": (0.35, 0.45), "non_jpy_sl_tp": (0.0035, 0.0045)},
    "sessions": [14], "base_lot": 0.30})
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
LB = 1440

def bar_hour(ts): return (ts // 3600) % 24

def anchor_price(bars, i, lb):
    seg = bars[max(0, i - lb):i]
    if not seg:
        return bars[i]["open"]
    return sum((b["high"] + b["low"] + b["close"]) / 3.0 for b in seg) / len(seg)

def first_sig(bars, day, hour):
    for i in range(LB, len(bars) - 1):
        b = bars[i]
        if b["ts"] // 86400 == day and bar_hour(b["ts"]) == hour:
            return i
    return None

def run_thr(bars_map, thr, volume=0.30, comm=3.0):
    trades = []
    days = sorted({b["ts"] // 86400 for bars in bars_map.values() for b in bars})
    for day in days:
        for hour in range(14, 20):
            for sym, bars_i in bars_map.items():
                sig = first_sig(bars_i, day, hour)
                if sig is None:
                    continue
                b = bars_i[sig]
                anc = anchor_price(bars_i, sig, LB)
                score = (anc - b["close"]) / anc * 100.0 if anc else 0.0
                if abs(score) < thr:
                    continue
                side = "BUY" if score >= 0 else "SELL"
                entry_idx = sig + 1
                if entry_idx >= len(bars_i):
                    continue
                raw = simulate_exit(bars_i, entry_idx, side, SPEC, sym)
                t = trade_to_usd({**raw, "entry_ts": bars_i[entry_idx]["ts"],
                                  "exit_ts": raw["exit_ts"], "side": side},
                                 volume, commission_per_lot=comm, spread_pips_map=SPREAD)
                trades.append(t)
    return trades

bars = build_bars_map(UNIVERSE)
print("causal per-hour threshold entry (signal at first bar of each session hour):")
for thr in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    usd = run_thr(bars, thr)
    m = metrics(usd)
    print(f"  thr {thr:>4.2f}: {m['trades']:>5}t net {m['net_pnl']:>9,.0f} "
          f"PF {m['profit_factor']:5.2f} WR {m['win_rate']:.3f} DD {m['max_drawdown']:>8,.0f}")