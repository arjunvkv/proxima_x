"""
Micro-Bleed Analysis: Does min_range_pips < 4.0 cause micro-bleeds during quiet hours?
Analyzes trade breakdown by 60m range tier over 7 months of MT5 data.
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
    print("MICRO-BLEED & NOISE ANALYSIS FOR ULTRA MONSTER (7 MONTHS MT5 DATA)")
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
        "min_range_pips": 0.0,
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

                    # Calculate range size
                    window = df_p.iloc[loc-12:loc]
                    rng_pips = (window['high'].max() - window['low'].min()) / (0.01 if "JPY" in pair else 0.0001)

                    pip_unit = 0.01 if "JPY" in pair else 0.0001
                    pips = (close_p - open_p) / pip_unit if side == "BUY" else (open_p - close_p) / pip_unit
                    pip_val = 6.8 if "JPY" in pair else 10.0
                    gross_usd = pips * pip_val * 1.20
                    comm = 3.60  # $3.60/lot FTMO comm
                    net_pnl = gross_usd - comm
                    trades.append({"pnl": net_pnl, "range": rng_pips, "pips": pips})

    df_tr = pd.DataFrame(trades)

    # Bin trades by 60m range tier
    tier1 = df_tr[df_tr['range'] < 4.0]
    tier2 = df_tr[(df_tr['range'] >= 4.0) & (df_tr['range'] < 6.0)]
    tier3 = df_tr[df_tr['range'] >= 6.0]

    def stats(sub, name):
        n = len(sub)
        w = len(sub[sub['pnl'] > 0])
        wr = (w / n * 100) if n else 0.0
        pnl = sub['pnl'].sum()
        avg_pnl = (pnl / n) if n else 0.0
        return f"  {name:25s} | Trades: {n:5d} | Wins: {w:4d} | WR: {wr:5.1f}% | Total Net: ${pnl:+10.2f} | Avg/Trade: ${avg_pnl:+6.2f}"

    print(stats(tier1, "Tier 1: Range < 4.0p"))
    print(stats(tier2, "Tier 2: Range 4.0p - 6.0p"))
    print(stats(tier3, "Tier 3: Range >= 6.0p"))
    print("-" * 85)
    print(stats(df_tr, "TOTAL PORTFOLIO"))
    print("=" * 85)

if __name__ == "__main__":
    main()
