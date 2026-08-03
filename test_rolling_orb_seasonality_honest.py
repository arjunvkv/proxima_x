#!/usr/bin/env python3
"""Rolling Hourly ORB & Intraday Seasonality — 5-Broker Honest Backtest Audit."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def run_rolling_hourly_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = [False] * len(PAIRS)
    exit_bar = [0] * len(PAIRS)
    entry_pr = [0.0] * len(PAIRS)
    direction = [0] * len(PAIRS)
    
    trades = []
    hold_bars = 6 # 30m hold

    for t in range(25, n_bars):
        # Exits
        for p_i in range(len(PAIRS)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "pair": PAIRS[p_i], "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Entry Trigger: At top of every hour (minutes == 0)
        if minutes[t] in [0, 5]:
            for p_i in range(len(PAIRS)):
                if in_pos[p_i]: continue
                # Prior 1-hour range (12 bars back)
                h_prev = np.max(high_mat[t-12:t, p_i])
                l_prev = np.min(low_mat[t-12:t, p_i])
                c_now = close_mat[t, p_i]
                
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

def run_intraday_seasonality(df_all, close_mat, open_mat, hours, minutes):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = [False] * len(PAIRS)
    exit_bar = [0] * len(PAIRS)
    entry_pr = [0.0] * len(PAIRS)
    direction = [0] * len(PAIRS)
    
    trades = []
    hold_bars = 12 # 60m hold

    # Intraday Seasonality Tendency Map (Hour UTC -> Pair -> Direction +1 BUY, -1 SELL)
    # 15:00 UTC (US Lunch Lull Reversion on JPY crosses)
    # 20:00 UTC (US Equity Close Flow on AUD/NZD pairs)
    season_map = {
        15: {"EURJPY": -1, "GBPJPY": -1, "USDJPY": -1}, # Sell JPY extensions at 15:00 UTC
        20: {"EURAUD": 1, "GBPAUD": 1}                 # Buy AUD cross dips at 20:00 UTC
    }

    for t in range(25, n_bars):
        # Exits
        for p_i in range(len(PAIRS)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "pair": PAIRS[p_i], "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Entry Trigger
        h_curr = hours[t]
        m_curr = minutes[t]
        if h_curr in season_map and m_curr in [0, 5]:
            targets = season_map[h_curr]
            for p_name, dir_val in targets.items():
                p_i = PAIRS.index(p_name)
                if not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = dir_val
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def main():
    t0 = time.time()
    print("Loading M5 dataset for Rolling Hourly ORB & Intraday Seasonality Audit...")
    raw, pre_align = load_and_align()
    pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in PAIRS]].values
    high_mat = df_all[[f"{p}_high" for p in PAIRS]].values
    low_mat = df_all[[f"{p}_low" for p in PAIRS]].values

    hours = times.hour.values
    minutes = times.minute.values

    print("\n" + "="*85)
    print("5-BROKER HONEST BACKTEST AUDIT: ROLLING HOURLY ORB & INTRADAY SEASONALITY")
    print("="*85)

    df_r_orb = run_rolling_hourly_orb(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes)
    df_seas = run_intraday_seasonality(df_all, close_mat, open_mat, hours, minutes)

    for name, df_t in [("Rolling Hourly ORB", df_r_orb), ("Intraday Seasonality", df_seas)]:
        pnls = df_t["net_pnl"].values if not df_t.empty else np.array([])
        n_t = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        net = sum(pnls)
        wr = wins / n_t * 100.0 if n_t > 0 else 0.0
        pf = gw / gl if gl > 0 else 0.0
        avg_w = net / n_t if n_t > 0 else 0.0

        print(f"\n--- {name} ---")
        print(f"  Total Trades Audited : {n_t}")
        print(f"  Net Win Rate        : {wr:.1f}%")
        print(f"  Total Net PnL       : +${net:.2f}" if net > 0 else f"  Total Net PnL       : -${abs(net):.2f}")
        print(f"  Profit Factor       : {pf:.2f}")
        print(f"  Average $/Trade     : +${avg_w:.2f}")
        print(f"  Status              : {'PASS' if net > 0 and pf > 1.2 else 'FAIL'}")

if __name__ == "__main__":
    main()
