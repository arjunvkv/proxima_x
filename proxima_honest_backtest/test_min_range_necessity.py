"""
Empirical Test: Is min_range_pips needed for Ultra Monster?
Compares 7-Month Backtest Performance WITH vs WITHOUT min_range_pips filter.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("C:/Trading/Agentic_Trading/proxima_x")
ENGINE_DIR = Path("C:/Trading/Agentic_Trading/proxima_alpha_engine")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ENGINE_DIR))

from proxima_honest_backtest.data.providers.mt5_provider import MT5Provider
from strategies.ultra_monster import evaluate_ultra_monster

UNIVERSE = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","EURJPY","GBPJPY","EURAUD"]

def run_test(min_range):
    provider = MT5Provider()
    raw = {}
    for p in UNIVERSE:
        frames = [provider.load_rates(p, y, m, "m5") for y, m in [(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6),(2026,7)]]
        frames = [f for f in frames if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
            d['time'] = pd.to_datetime(d['time'])
            d.set_index('time', inplace=True)
            raw[p] = d

    config = {
        "triggers": [0, 30],
        "lookback_bars": 12,
        "min_range_pips": min_range,
        "universe": UNIVERSE,
        "lot": 1.20,
        "hold_bars": 3
    }

    # Common half-hour timestamps
    all_times = set()
    for p, df in raw.items():
        all_times.update(df.index.tolist())

    sorted_times = sorted([t for t in all_times if t.minute in [0, 30]])

    trades = []
    for t in sorted_times:
        sigs = evaluate_ultra_monster(raw, t, config)
        if sigs:
            for sig in sigs:
                pair = sig['pair']
                side = sig['side']
                df_p = raw[pair]
                if t in df_p.index:
                    loc = df_p.index.get_loc(t)
                    open_p = df_p.iloc[loc]['open']
                    exit_loc = min(loc + 3, len(df_p) - 1)
                    close_p = df_p.iloc[exit_loc]['close']

                    pip_unit = 0.01 if "JPY" in pair else 0.0001
                    pips = (close_p - open_p) / pip_unit if side == "BUY" else (open_p - close_p) / pip_unit
                    pip_val = 6.8 if "JPY" in pair else 10.0
                    gross_usd = pips * pip_val * 1.20
                    comm = 3.60  # $3.60/lot FTMO comm
                    net_pnl = gross_usd - comm
                    trades.append(net_pnl)

    n_trades = len(trades)
    n_wins = sum(1 for pnl in trades if pnl > 0)
    wr = (n_wins / n_trades * 100) if n_trades else 0.0
    net_pnl = sum(trades)
    gross_win = sum(pnl for pnl in trades if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trades if pnl < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0

    return n_trades, n_wins, wr, net_pnl, pf

print("=================================================================================")
print("EMPIRICAL TEST: IS min_range_pips NEEDED FOR ULTRA MONSTER? (7-MONTH MT5 DATA)")
print("=================================================================================")

print("\nRunning Test 1: WITH min_range_pips = 6.0...")
n1, w1, wr1, pnl1, pf1 = run_test(6.0)

print("Running Test 2: WITHOUT min_range_pips (min_range_pips = 0.0)...")
n2, w2, wr2, pnl2, pf2 = run_test(0.0)

print("\n" + "="*85)
print(f"RESULTS COMPARISON (7 MONTHS DATA)")
print("="*85)
print(f"  Configuration             | Total Trades | Wins / Losses | Win Rate | Profit Factor | Net PnL (FTMO)")
print("-" * 85)
print(f"  WITH min_range = 6.0p     | {n1:12d} | {w1:4d} / {n1-w1:4d}   | {wr1:7.1f}% | {pf1:13.2f} | ${pnl1:+10.2f}")
print(f"  WITHOUT min_range (0.0p)  | {n2:12d} | {w2:4d} / {n2-w2:4d}   | {wr2:7.1f}% | {pf2:13.2f} | ${pnl2:+10.2f}")
print("="*85)
