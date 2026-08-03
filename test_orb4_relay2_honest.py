#!/usr/bin/env python3
"""ORB Breakout Ride (#4) & Session Momentum Relay (#2) — 5-Broker Honest Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY"]

def run_orb_breakout_ride(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = [False] * len(PAIRS)
    exit_bar = [0] * len(PAIRS)
    entry_pr = [0.0] * len(PAIRS)
    direction = [0] * len(PAIRS)
    
    trades = []
    hold_bars = 12 # 60m hold

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
                trades.append({"time": pd.to_datetime(df_all.index[t]), "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Trigger: London Open ORB Breakout Ride (07:00 UTC)
        # ORB range measured from 06:30 to 07:00 UTC (6 bars)
        if hours[t] == 7 and minutes[t] in [0, 5]:
            for p_i in range(len(PAIRS)):
                orb_high = np.max(high_mat[t-6:t, p_i])
                orb_low = np.min(low_mat[t-6:t, p_i])
                c_curr = close_mat[t, p_i]
                
                # If breakout above ORB high -> BUY momentum ride
                if c_curr > orb_high and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = 1 # BUY momentum
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif c_curr < orb_low and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = -1 # SELL momentum
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def run_session_momentum_relay(df_all, close_mat, open_mat, hours, minutes):
    sim = ExecutionSimulator("fundednext")
    n_bars = len(df_all)
    in_pos = [False] * len(PAIRS)
    exit_bar = [0] * len(PAIRS)
    entry_pr = [0.0] * len(PAIRS)
    direction = [0] * len(PAIRS)
    
    trades = []
    hold_bars = 18 # 90m hold

    for t in range(75, n_bars):
        # Exits
        for p_i in range(len(PAIRS)):
            if in_pos[p_i] and t >= exit_bar[p_i]:
                c_exit = close_mat[t, p_i]
                c_entry = entry_pr[p_i]
                dir_i = direction[p_i]
                gross_pnl = (c_exit - c_entry) / c_entry * 50000.0 * dir_i
                comm = sim.profile.commission_per_lot * 0.5
                net_pnl = gross_pnl - comm
                trades.append({"time": pd.to_datetime(df_all.index[t]), "net_pnl": net_pnl, "win": net_pnl > 0})
                in_pos[p_i] = False

        # Trigger: US Session Open Momentum Relay (13:00 UTC)
        # London morning return measured from 07:00 to 13:00 UTC (72 bars)
        if hours[t] == 13 and minutes[t] in [0, 5]:
            for p_i in range(len(PAIRS)):
                c_curr = close_mat[t, p_i]
                c_london_start = close_mat[t-72, p_i]
                if c_curr <= 0 or c_london_start <= 0: continue
                ret_london = (c_curr - c_london_start) / c_london_start
                
                # If London trend >= +0.25% -> BUY US momentum relay
                if ret_london >= 0.0025 and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = 1 # BUY trend relay
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]
                elif ret_london <= -0.0025 and not in_pos[p_i]:
                    in_pos[p_i] = True
                    direction[p_i] = -1 # SELL trend relay
                    exit_bar[p_i] = t + hold_bars
                    entry_pr[p_i] = open_mat[t, p_i]

    return pd.DataFrame(trades)

def main():
    t0 = time.time()
    print("Loading M5 dataset for ORB Breakout Ride (#4) & Session Momentum Relay (#2) Audit...")
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
    print("5-BROKER HONEST BACKTEST AUDIT: ORB BREAKOUT RIDE (#4) & SESSION MOMENTUM RELAY (#2)")
    print("="*85)

    df_orb = run_orb_breakout_ride(df_all, close_mat, open_mat, high_mat, low_mat, hours, minutes)
    df_relay = run_session_momentum_relay(df_all, close_mat, open_mat, hours, minutes)

    for name, df_t in [("ORB Breakout Ride (#4)", df_orb), ("Session Momentum Relay (#2)", df_relay)]:
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
