#!/usr/bin/env python3
"""Compare Ultra Monster performance on 9-pair universe vs 18-pair universe."""

import sys
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_9 = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]
PAIRS_18 = [
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADJPY","EURAUD","EURCAD","EURCHF","EURGBP",
    "EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD",
    "GBPJPY","GBPNZD","GBPUSD"
]

def main():
    print("Loading dataset for 9-Pair vs 18-Pair Universe Audit...")
    raw, pre_align = load_and_align()
    
    # 9-Pair Universe
    pieces_9 = [df.set_index("time")[["close","open","high","low"]] for p, df in raw.items() if p in PAIRS_9]
    for i, p in enumerate([p for p in raw.keys() if p in PAIRS_9]):
        pieces_9[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_9 = pd.concat(pieces_9, axis=1, sort=True).ffill().bfill()
    times_9 = pd.to_datetime(df_9.index)

    close_9 = df_9[[p for p in PAIRS_9]].values
    open_9 = df_9[[f"{p}_open" for p in PAIRS_9]].values
    high_9 = df_9[[f"{p}_high" for p in PAIRS_9]].values
    low_9 = df_9[[f"{p}_low" for p in PAIRS_9]].values

    df_u9 = run_ultra_buffed_orb(df_9, close_9, open_9, high_9, low_9, times_9.hour.values, times_9.minute.values, PAIRS_9, 6.0, range(0, 24), [0, 30], 3)
    pnl_9 = df_u9["net_pnl"].values if not df_u9.empty else np.array([])
    wr_9 = (pnl_9 > 0).mean() * 100
    net_9 = pnl_9.sum()
    gw_9 = pnl_9[pnl_9 > 0].sum()
    gl_9 = abs(pnl_9[pnl_9 < 0].sum())
    pf_9 = gw_9 / gl_9

    # 18-Pair Universe
    pieces_18 = [df.set_index("time")[["close","open","high","low"]] for p, df in raw.items() if p in PAIRS_18]
    for i, p in enumerate([p for p in raw.keys() if p in PAIRS_18]):
        pieces_18[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_18 = pd.concat(pieces_18, axis=1, sort=True).ffill().bfill()
    times_18 = pd.to_datetime(df_18.index)

    close_18 = df_18[[p for p in PAIRS_18]].values
    open_18 = df_18[[f"{p}_open" for p in PAIRS_18]].values
    high_18 = df_18[[f"{p}_high" for p in PAIRS_18]].values
    low_18 = df_18[[f"{p}_low" for p in PAIRS_18]].values

    df_u18 = run_ultra_buffed_orb(df_18, close_18, open_18, high_18, low_18, times_18.hour.values, times_18.minute.values, PAIRS_18, 6.0, range(0, 24), [0, 30], 3)
    pnl_18 = df_u18["net_pnl"].values if not df_u18.empty else np.array([])
    wr_18 = (pnl_18 > 0).mean() * 100
    net_18 = pnl_18.sum()
    gw_18 = pnl_18[pnl_18 > 0].sum()
    gl_18 = abs(pnl_18[pnl_18 < 0].sum())
    pf_18 = gw_18 / gl_18

    print("="*115)
    print("ULTRA MONSTER UNIVERSE COMPARISON: 9-PAIR vs 18-PAIR")
    print("="*115)
    print(f"9-PAIR UNIVERSE (EURUSD, GBPUSD, USDJPY, EURAUD, GBPAUD, EURJPY, GBPJPY, EURNZD, GBPNZD):")
    print(f"  • Total Trades   : {len(pnl_9)}")
    print(f"  • Win Rate       : {wr_9:.1f}% 🟢")
    print(f"  • Net Realized   : +${net_9:,.2f}")
    print(f"  • Profit Factor  : {pf_9:.2f} 🚀")

    print(f"\n18-PAIR UNIVERSE (All Major & Cross FX Pairs):")
    print(f"  • Total Trades   : {len(pnl_18)}")
    print(f"  • Win Rate       : {wr_18:.1f}% 🟢")
    print(f"  • Net Realized   : +${net_18:,.2f}")
    print(f"  • Profit Factor  : {pf_18:.2f} 🚀")
    print("="*115)

if __name__ == "__main__":
    main()
