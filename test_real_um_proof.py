import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from audit_ultra_monster_weekly_monthly_proofs import run_ultra_monster_backtest, load_and_align, PAIRS_ALL

print("=== RUNNING PROVEN ULTRA MONSTER (ROLLING ORB) AUDIT ===")
raw, pre_align = load_and_align()
pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
for i, p in enumerate(raw.keys()):
    pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
times = pd.to_datetime(df_all.index)

close_mat = df_all[[p for p in PAIRS_ALL]].values
open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values

hours = times.hour.values
minutes = times.minute.values

df_trades = run_ultra_monster_backtest(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, min_range_pips=6.0, trigger_mins=[0, 30], hold_bars=3)

wins = df_trades[df_trades["net_pnl"] > 0]
losses = df_trades[df_trades["net_pnl"] <= 0]
wr = len(wins) / len(df_trades) * 100
pf = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())

print(f"Total Trades : {len(df_trades):,}")
print(f"Win Rate     : {wr:.2f}% 🟢")
print(f"Profit Factor: {pf:.2f} 🚀")
print(f"Net PnL      : +${df_trades['net_pnl'].sum():,.2f}")
