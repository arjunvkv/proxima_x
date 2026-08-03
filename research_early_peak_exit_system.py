#!/usr/bin/env python3
"""Research Early Peak Exit Engine based on Market Mechanics vs Fixed Timed Exits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from ultra_buff_rolling_orb import run_ultra_buffed_orb

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_early_peak_exit_sim(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, pairs, min_range_pips=12.0, target_pip_profit=15.0):
    trades_fixed = []
    trades_peak = []
    n_bars = len(df_all)

    for t_idx in range(30, n_bars - 3):
        h = hours[t_idx]
        m = minutes[t_idx]

        if m not in [0, 30]:
            continue

        for p_idx, pair in enumerate(pairs):
            orb_high = np.max(high_mat[t_idx-6:t_idx, p_idx])
            orb_low = np.min(low_mat[t_idx-6:t_idx, p_idx])
            
            pip_mult = 100.0 if "JPY" in pair else 10000.0
            range_pips = (orb_high - orb_low) * pip_mult

            if range_pips < min_range_pips:
                continue

            c_now = close_mat[t_idx, p_idx]
            c_entry = open_mat[t_idx+1, p_idx]
            
            # M5 bars during 3-bar (15m) hold: bar 1, bar 2, bar 3
            c1 = close_mat[t_idx+1, p_idx]
            c2 = close_mat[t_idx+2, p_idx]
            c3 = close_mat[t_idx+3, p_idx]
            
            h1 = high_mat[t_idx+1, p_idx]; l1 = low_mat[t_idx+1, p_idx]
            h2 = high_mat[t_idx+2, p_idx]; l2 = low_mat[t_idx+2, p_idx]
            h3 = high_mat[t_idx+3, p_idx]; l3 = low_mat[t_idx+3, p_idx]

            buf = 1.0 / pip_mult

            if c_now > (orb_high + buf):
                # BUY Trade
                pnl_fixed = (c3 - c_entry) * pip_mult * 10.0 * 1.00
                trades_fixed.append(pnl_fixed)

                # Check if Peak Target hit in bar 1, bar 2, or bar 3
                peak_pnl = pnl_fixed
                if (h1 - c_entry) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00
                elif (h2 - c_entry) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00
                elif (h3 - c_entry) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00
                
                trades_peak.append(peak_pnl)

            elif c_now < (orb_low - buf):
                # SELL Trade
                pnl_fixed = (c_entry - c3) * pip_mult * 10.0 * 1.00
                trades_fixed.append(pnl_fixed)

                peak_pnl = pnl_fixed
                if (c_entry - l1) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00
                elif (c_entry - l2) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00
                elif (c_entry - l3) * pip_mult >= target_pip_profit:
                    peak_pnl = target_pip_profit * 10.0 * 1.00

                trades_peak.append(peak_pnl)

    return trades_fixed, trades_peak

def main():
    print("Researching Early Peak Exit Engine vs Fixed Timed Exits...")
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

    # Test peak exit targets from 10p to 25p
    peak_targets = [8.0, 10.0, 12.0, 15.0, 20.0]
    
    results = []

    for target in peak_targets:
        tf, tp = run_early_peak_exit_sim(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, PAIRS_ALL, 12.0, target)
        
        def stats(t_list):
            n = len(t_list)
            w = [t for t in t_list if t > 0]
            l = [t for t in t_list if t <= 0]
            wr = len(w) / n * 100.0 if n > 0 else 0
            pf = sum(w) / max(1, abs(sum(l)))
            tot = sum(t_list)
            avg_w = np.mean(w) if w else 0
            return n, wr, pf, tot, avg_w

        nf, wrf, pff, totf, avgwf = stats(tf)
        np_k, wrp, pfp, totp, avgwp = stats(tp)

        results.append({
            "Peak Target (Pips)": f"{target} Pips",
            "Fixed Expiry Win Rate": f"{wrf:.1f}%",
            "Peak Exit Win Rate": f"{wrp:.1f}%",
            "Fixed Expiry PF": f"{pff:.2f}",
            "Peak Exit PF": f"{pfp:.2f}",
            "Fixed Total PnL": f"+${totf:,.2f}",
            "Peak Exit Total PnL": f"+${totp:,.2f}",
            "Win Rate Delta": f"{wrp - wrf:+.1f}% WR"
        })

    df_res = pd.DataFrame(results)
    print("="*115)
    print("EARLY PEAK EXIT ENGINE RESEARCH MATRIX (7-MONTH DATASET)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)

if __name__ == "__main__":
    main()
