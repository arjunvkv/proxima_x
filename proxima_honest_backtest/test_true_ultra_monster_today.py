"""
Test True Ultra Monster Breakout Logic Today (Aug 4, 2026)
Evaluates evaluate_ultra_monster() with true breakout & single best pair selection.
"""

import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
ENGINE_DIR = Path(__file__).resolve().parent.parent.parent / "proxima_alpha_engine"
sys.path.insert(0, str(ENGINE_DIR))

from strategies.ultra_monster import evaluate_ultra_monster

if not mt5.initialize():
    print("❌ Could not connect to local MT5")
    sys.exit(1)

UNIVERSE = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","EURJPY","GBPJPY","EURAUD"]
config = {
    "triggers": [0, 30],
    "lookback_bars": 12,
    "min_range_pips": 6.0,
    "universe": UNIVERSE,
    "lot": 1.20,
    "hold_bars": 3
}

rates_dict = {}
for p in UNIVERSE:
    rates = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M5, 0, 300)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        rates_dict[p] = df

# Get all half-hour timestamps today
all_times = set()
for p, df in rates_dict.items():
    df_today = df[df.index >= "2026-08-04 00:00:00"]
    all_times.update(df_today.index.tolist())

sorted_times = sorted([t for t in all_times if t.minute in [0, 30]])
print(f"Testing {len(sorted_times)} half-hour breakout boundaries today (Aug 4, 2026)...")

signals_today = []
for t in sorted_times:
    sigs = evaluate_ultra_monster(rates_dict, t, config)
    if sigs:
        for s in sigs:
            s['trigger_time'] = t
            signals_today.append(s)

print("\n" + "="*80)
print(f"TRUE ULTRA MONSTER SIGNALS TODAY (AUG 4, 2026): {len(signals_today)}")
print("="*80)

trades = []
for idx, sig in enumerate(signals_today, 1):
    pair = sig['pair']
    side = sig['side']
    t_entry = sig['trigger_time']

    df_p = rates_dict[pair]
    loc = df_p.index.get_loc(t_entry)
    open_p = df_p.iloc[loc]['open']
    
    exit_loc = min(loc + 3, len(df_p) - 1)
    close_p = df_p.iloc[exit_loc]['close']
    exit_t = df_p.index[exit_loc]

    pip_unit = 0.01 if "JPY" in pair else 0.0001
    pips = (close_p - open_p) / pip_unit if side == "BUY" else (open_p - close_p) / pip_unit
    pip_val = 6.8 if "JPY" in pair else 10.0
    gross_pnl = pips * pip_val * 1.20
    comm = 3.60
    net_pnl = gross_pnl - comm

    is_win = net_pnl > 0
    w_str = "WIN 🟢" if is_win else "LOSS 🔴"

    trades.append(net_pnl)

    print(f"{idx:02d}. [{t_entry.strftime('%H:%M UTC')}] {pair:7s} {side:4s} | Entry: {open_p:.5f} -> Exit @ {exit_t.strftime('%H:%M')}: {close_p:.5f} | Pips: {pips:+5.1f}p | Net PnL: ${net_pnl:+7.2f} | {w_str}")

print("\n" + "-"*80)
wins = sum(1 for pnl in trades if pnl > 0)
total_net = sum(trades)
wr = (wins / len(trades) * 100) if trades else 0.0
print(f"TRUE ULTRA MONSTER SUMMARY TODAY:")
print(f"  Total Trades:   {len(trades)}")
print(f"  Wins / Losses:  {wins} / {len(trades) - wins}")
print(f"  Win Rate:       {wr:.1f}%")
print(f"  Total Net PnL:  ${total_net:+8.2f}")
print("="*80)

mt5.shutdown()
