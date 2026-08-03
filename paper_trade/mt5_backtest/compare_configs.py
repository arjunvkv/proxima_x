"""Systematic comparison of all parameter configs."""
import numpy as np
import pandas as pd
from sim_recon import ReconSim

data = np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy',
               allow_pickle=True)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')

trials = [
    # (name, limit_entry_atr, enable_stability, stab_thresh, z_cum_min)
    ("Baseline (no filters)",           0.0,     False, 0.5, 5.0),
    ("ATR=0.00007 only",                0.00007, False, 0.5, 5.0),
    ("ATR=0.00005 only",                0.00005, False, 0.5, 5.0),
    ("ATR=0.00010 only",                0.00010, False, 0.5, 5.0),
    ("Gate strict (0.5/z5)",            0.0,     True,  0.5, 5.0),
    ("Gate relaxed (0.3/z0)",           0.0,     True,  0.3, 0.0),
    ("ATR07+Gate(0.3/z0)",              0.00007, True,  0.3, 0.0),
    ("ATR07+Gate(0.5/z3)",              0.00007, True,  0.5, 3.0),
    ("ATR05+Gate(0.4/z0)",              0.00005, True,  0.4, 0.0),
    ("Gate(0.3/z3)",                    0.0,     True,  0.3, 3.0),
    ("ATR07+Gate(0.3/z3)",              0.00007, True,  0.3, 3.0),
]

results = []
for name, atr, stab, thresh, zcum in trials:
    sim = ReconSim(
        z_threshold=4.0, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.5, max_spread=5,
        limit_entry_atr=atr,
        enable_stability=stab,
        stab_thresh=thresh,
        z_cum_min=zcum,
        trade_dir=1, start_hour=0, end_hour=7
    )
    trades = sim.run(
        df['time'].astype(np.int64)//10**9, df['open'].values,
        df['high'].values, df['low'].values, df['close'].values,
        df['spread'].values, df['tick_volume'].values
    )
    tp = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    n = len(trades)
    wr = len(wins)/(len(trades)-len(neutral))*100 if (len(trades)-len(neutral)) > 0 else 0
    results.append((name, n, len(wins), len(losses), wr, tp))

print(f"{'Config':35s} | {'Trades':>6s} | {'W/L':>7s} | {'WR':>5s} | {'PnL':>8s}")
print("-"*70)
for name, n, w, l, wr, tp in results:
    print(f"{name:35s} | {n:6d} | {w}/{l:<3d} | {wr:4.1f}% | ${tp:>+6.2f}")
