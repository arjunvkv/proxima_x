"""
Empirical 7-Month Backtest of Ultra Monster using loc - 1 (Completed Bar Close)
Compares performance over 7 months of MT5 data.
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

def main():
    print("=================================================================================")
    print("EMPIRICAL 7-MONTH BACKTEST OF ULTRA MONSTER WITH loc - 1 (COMPLETED BAR CLOSE)")
    print("=================================================================================")

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
        "min_range_pips": 6.0,
        "universe": UNIVERSE,
        "lot": 1.20,
        "hold_bars": 3
    }

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

    print("\n" + "="*85)
    print(f"7-MONTH BACKTEST RESULTS WITH loc - 1")
    print("="*85)
    print(f"  Total Trades:     {n_trades}")
    print(f"  Wins / Losses:    {n_wins} / {n_trades - n_wins}")
    print(f"  Win Rate:         {wr:.1f}%")
    print(f"  Profit Factor:    {pf:.2f}")
    print(f"  Total Net PnL:    ${net_pnl:+10.2f}")
    print("="*85)

if __name__ == "__main__":
    main()
