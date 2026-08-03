#!/usr/bin/env python3
"""Quantitative Conditioning & Optimization for Rolling Hourly ORB."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_conditioned_rolling_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, pairs_subset, min_range_pips=15.0, allowed_hours=range(7, 17), hold_bars=3):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    pair_indices = [PAIRS_ALL.index(p) for p in pairs_subset]

    in_pos = [False] * len(PAIRS_ALL)
    exit_bar = [0] * len(PAIRS_ALL)
    entry_pr = [0.0] * len(PAIRS_ALL)
    direction = [0] * len(PAIRS_ALL)
    
    trades = []

    for t in range(13, n_bars):
        # Exits
        for p_i in pair_indices:
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "pair": PAIRS_ALL[p_i], "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Entry Trigger: At top of hour (minutes == 0) & within allowed session hours
        if minutes[t] == 0 and hours[t] in allowed_hours:
            for p_i in pair_indices:
                if in_pos[p_i]: continue
                
                # Prior 1-hour range (12 M5 bars back)
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_now = close_mat[t, p_i]
                
                # Volatility gate: range must be >= min_range_pips
                range_pips = (h_prev - l_prev) * 10000.0 if "JPY" not in PAIRS_ALL[p_i] else (h_prev - l_prev) * 100.0
                if range_pips < min_range_pips:
                    continue

                if c_now > h_prev:
                    in_pos[p_i] = True
                    direction[p_i] = 1 # BUY hourly breakout
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif c_now < l_prev:
                    in_pos[p_i] = True
                    direction[p_i] = -1 # SELL hourly breakout
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def main():
    t0 = time.time()
    print("Loading M5 dataset for Rolling Hourly ORB Conditioning Audit...")
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

    print("\n" + "="*95)
    print("ROLLING HOURLY ORB — QUANTITATIVE CONDITIONING & DRAWDOWN MINIMIZATION")
    print("="*95)

    top_crosses = ["GBPAUD", "GBPNZD", "EURAUD", "GBPJPY"]
    
    sweep_rows = []
    for min_pips in [10.0, 15.0, 20.0]:
        for window_name, h_range in [("Full Day (07-20 UTC)", range(7, 21)), ("London+US Peak (07-16 UTC)", range(7, 17)), ("London Open Only (07-11 UTC)", range(7, 12))]:
            for hold_b in [3, 6, 12]:
                df_c = run_conditioned_rolling_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes, top_crosses, min_pips, h_range, hold_b)
                if df_c.empty: continue
                pnls = df_c["net_pnl"].values
                n_t = len(pnls)
                wins = sum(1 for p in pnls if p > 0)
                gw = sum(p for p in pnls if p > 0)
                gl = abs(sum(p for p in pnls if p < 0))
                net = sum(pnls)
                wr = wins / n_t * 100.0
                pf = gw / gl if gl > 0 else 0.0
                avg_w = net / n_t
                trades_per_day = n_t / 154.0

                # Max Drawdown calculation
                cum_pnl = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum_pnl)
                dd = peak - cum_pnl
                max_dd = np.max(dd) if len(dd) > 0 else 0.0

                sweep_rows.append({
                    "Min Range": f"{min_pips:.0f}p",
                    "Session Window": window_name,
                    "Hold": f"{hold_b*5}m",
                    "Trades": n_t,
                    "Trades/Day": f"{trades_per_day:.1f}",
                    "Win Rate": f"{wr:.1f}%",
                    "Net PnL": f"+${net:.2f}",
                    "PF": round(pf, 2),
                    "Max DD": f"${max_dd:.2f}",
                    "Avg$/Trade": f"+${avg_w:.2f}"
                })

    df_res = pd.DataFrame(sweep_rows)
    print(df_res.sort_values(by="PF", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
