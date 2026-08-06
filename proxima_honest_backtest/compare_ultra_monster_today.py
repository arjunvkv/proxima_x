"""
Compare Today's Ultra Monster Backtest Signals vs Live VPS Recorded Trades (Aug 4, 2026)
"""

import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("C:/Trading/Agentic_Trading/proxima_x")
ENGINE_DIR = Path("C:/Trading/Agentic_Trading/proxima_alpha_engine")
sys.path.insert(0, str(BASE_DIR))
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

all_times = set()
for p, df in rates_dict.items():
    df_today = df[df.index >= "2026-08-04 00:00:00"]
    all_times.update(df_today.index.tolist())

sorted_times = sorted([t for t in all_times if t.minute in [0, 30]])
print(f"=================================================================================")
print(f"TODAY'S ULTRA MONSTER LOCAL MT5 BACKTEST (AUG 4, 2026: 00:00 UTC -> {sorted_times[-1].strftime('%H:%M UTC')})")
print(f"=================================================================================")

bt_trades = []
for t in sorted_times:
    sigs = evaluate_ultra_monster(rates_dict, t, config)
    if sigs:
        for s in sigs:
            pair = s['pair']
            side = s['side']
            df_p = rates_dict[pair]
            if t in df_p.index:
                loc = df_p.index.get_loc(t)
                open_p = df_p.iloc[loc]['open']
                exit_loc = min(loc + 3, len(df_p) - 1)
                close_p = df_p.iloc[exit_loc]['close']
                exit_t = df_p.index[exit_loc]

                pip_unit = 0.01 if "JPY" in pair else 0.0001
                pips = (close_p - open_p) / pip_unit if side == "BUY" else (open_p - close_p) / pip_unit
                pip_val = 6.8 if "JPY" in pair else 10.0
                gross_usd = pips * pip_val * 1.20
                comm = 3.60
                net_pnl = gross_usd - comm

                bt_trades.append({
                    "time": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": pair,
                    "side": side,
                    "entry_p": open_p,
                    "exit_p": close_p,
                    "exit_t": exit_t.strftime("%H:%M:%S"),
                    "pips": round(pips, 1),
                    "net_pnl": round(net_pnl, 2),
                    "is_win": net_pnl > 0
                })

for idx, tr in enumerate(bt_trades, 1):
    w_str = "WIN 🟢" if tr['is_win'] else "LOSS 🔴"
    print(f"{idx:02d}. [{tr['time'][11:16]} UTC] {tr['pair']:7s} {tr['side']:4s} | Entry: {tr['entry_p']:.5f} -> Exit @ {tr['exit_t'][:5]}: {tr['exit_p']:.5f} | Pips: {tr['pips']:+5.1f}p | Net: ${tr['net_pnl']:+7.2f} | {w_str}")

print("\n" + "="*80)
print(f"BACKTEST SUMMARY TODAY: {len(bt_trades)} Trades | Net PnL: ${sum(t['net_pnl'] for t in bt_trades):+8.2f}")
print("="*80)

mt5.shutdown()
