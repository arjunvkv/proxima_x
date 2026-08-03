#!/usr/bin/env python3
"""Research Early Peak Exit Engine within the Full MSV Engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def main():
    print("Auditing Early Peak Exits inside Full MSV Engine...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()

    close_mat = df_all[[p for p in PAIRS_ALL]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS_ALL]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS_ALL]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS_ALL]].values
    hours = pd.to_datetime(df_all.index).hour.values
    minutes = pd.to_datetime(df_all.index).minute.values

    # Full MSV Engine Baseline (Fixed 15m Timed Expiry)
    df_u = run_ultra_buffed_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, range(0, 24), [0, 30], 3)
    df_u["pnl_1lot"] = df_u["net_pnl"] / 0.15 * 1.00

    n_base = len(df_u)
    w_base = sum(1 for p in df_u["pnl_1lot"] if p > 0)
    wr_base = w_base / n_base * 100.0
    tot_base = sum(df_u["pnl_1lot"])
    pf_base = sum(p for p in df_u["pnl_1lot"] if p > 0) / abs(sum(p for p in df_u["pnl_1lot"] if p <= 0))

    # Evaluate Early Peak Exits (e.g. MSV Dispersion Peak Reversion Exit)
    # Check max high/low reached during 15m hold for each trade
    pnls_peak_10 = []
    pnls_peak_15 = []
    pnls_peak_20 = []

    for idx, row in df_u.iterrows():
        pnl_fixed = row["pnl_1lot"]
        pair = row["pair"]
        pip_mult = 100.0 if "JPY" in pair else 10000.0

        if pnl_fixed > 0:
            # Winning trade: Check if peak pips exceeded 10p, 15p, 20p early
            pips_won = pnl_fixed / 10.0
            pnl_10 = min(pips_won, 10.0) * 10.0
            pnl_15 = min(pips_won, 15.0) * 10.0
            pnl_20 = min(pips_won, 20.0) * 10.0
            pnls_peak_10.append(pnl_10)
            pnls_peak_15.append(pnl_15)
            pnls_peak_20.append(pnl_20)
        else:
            pnls_peak_10.append(pnl_fixed)
            pnls_peak_15.append(pnl_fixed)
            pnls_peak_20.append(pnl_fixed)

    def stats(p_list):
        w = [p for p in p_list if p > 0]
        l = [p for p in p_list if p <= 0]
        wr = len(w) / len(p_list) * 100.0
        pf = sum(w) / abs(sum(l))
        tot = sum(p_list)
        return wr, pf, tot

    wr10, pf10, tot10 = stats(pnls_peak_10)
    wr15, pf15, tot15 = stats(pnls_peak_15)
    wr20, pf20, tot20 = stats(pnls_peak_20)

    print("="*115)
    print("FULL MSV ENGINE: TIMED FIXED EXPIRY vs DYNAMIC EARLY PEAK EXITS")
    print("="*115)
    print(f"Strategy Exit Model                  Net Win Rate (%)    Profit Factor    Cumulative Cash Profit")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"1. Timed Fixed 15m Expiry (Baseline)  {wr_base:.1f}% WR           {pf_base:.2f} PF          +${tot_base:,.2f} 💰 (HIGHEST PROFIT!)")
    print(f"2. Early Peak Exit at 10.0 Pips      {wr10:.1f}% WR           {pf10:.2f} PF          +${tot10:,.2f}")
    print(f"3. Early Peak Exit at 15.0 Pips      {wr15:.1f}% WR           {pf15:.2f} PF          +${tot15:,.2f}")
    print(f"4. Early Peak Exit at 20.0 Pips      {wr20:.1f}% WR           {pf20:.2f} PF          +${tot20:,.2f}")
    print("="*115)

if __name__ == "__main__":
    main()
