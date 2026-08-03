"""Sweep parameters to increase trade count while preserving WR on FTMO data."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sim_recon import ReconSim

data = np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy',
               allow_pickle=True)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')

trials = []

# === LEVER 1: z_threshold ===
for z in [4.0, 3.5, 3.0, 2.5]:
    sim = ReconSim(z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
                   trade_dir=1, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z={z} Gate(0.3/z0) LONG 0-7", n, wins, losses, wr, tp))

# === LEVER 2: both directions ===
for z in [4.0, 3.5, 3.0]:
    sim = ReconSim(z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
                   trade_dir=0, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z={z} Gate(0.3/z0) BOTH 0-7", n, wins, losses, wr, tp))

# === LEVER 3: wider hours ===
hours_opts = [(0, 9), (22, 10), (0, 12)]
for label, h0, h1 in [("0-9", 0, 9), ("22-10", 22, 10), ("0-12", 0, 12)]:
    for z in [4.0, 3.5]:
        sim = ReconSim(z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                       max_hold=54, base_lot=0.5, max_spread=5,
                       enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
                       trade_dir=1, start_hour=h0, end_hour=h1)
        trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                         df['high'].values, df['low'].values, df['close'].values,
                         df['spread'].values, df['tick_volume'].values)
        tp = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        n = len(trades)
        wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
        trials.append((f"z={z} Gate(0.3/z0) LONG {label}", n, wins, losses, wr, tp))

# === LEVER 4: lower stability threshold ===
for stab in [0.2, 0.0]:
    sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   enable_stability=True, stab_thresh=stab, z_cum_min=0.0,
                   trade_dir=1, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z=3.5 Gate(stab={stab}/z0) LONG 0-7", n, wins, losses, wr, tp))

# === LEVER 5: stability gate off, ATR filter ===
for z in [3.5, 3.0]:
    sim = ReconSim(z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   limit_entry_atr=0.00007, enable_stability=False,
                   trade_dir=1, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z={z} ATR07 LONG 0-7", n, wins, losses, wr, tp))

# === LEVER 6: adjusted trailing to capture more winners ===
for stop_a in [2.0, 4.0]:
    sim = ReconSim(z_threshold=3.5, stop_a=stop_a, trig_a=1.0, gap_a=0.05,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
                   trade_dir=1, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z=3.5 Gate(0.3/z0) LONG 0-7 stop_a={stop_a}", n, wins, losses, wr, tp))

# === LEVER 7: triggers/gap to capture quicker profits ===
for trig_a, gap_a in [(0.5, 0.03), (2.0, 0.10)]:
    sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=trig_a, gap_a=gap_a,
                   max_hold=54, base_lot=0.5, max_spread=5,
                   enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
                   trade_dir=1, start_hour=0, end_hour=7)
    trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                     df['high'].values, df['low'].values, df['close'].values,
                     df['spread'].values, df['tick_volume'].values)
    tp = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n = len(trades)
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    trials.append((f"z=3.5 Gate(0.3/z0) LONG 0-7 t={trig_a} g={gap_a}", n, wins, losses, wr, tp))

# === LEVER 8: combined best ===
sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=0.5, gap_a=0.03,
               max_hold=54, base_lot=0.5, max_spread=5,
               enable_stability=True, stab_thresh=0.3, z_cum_min=0.0,
               trade_dir=1, start_hour=0, end_hour=9)
trades = sim.run(df['time'].astype(np.int64)//10**9, df['open'].values,
                 df['high'].values, df['low'].values, df['close'].values,
                 df['spread'].values, df['tick_volume'].values)
tp = sum(t.pnl for t in trades)
wins = sum(1 for t in trades if t.pnl > 0)
losses = sum(1 for t in trades if t.pnl < 0)
n = len(trades)
wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
trials.append((f"z=3.5 Gate(0.3/z0) LONG 0-9 t=0.5g=0.03", n, wins, losses, wr, tp))


# Sort by trade count descending, show top combos
trials.sort(key=lambda x: -x[1])

print(f"{'Config':55s} | {'Trades':>6s} | {'W/L':>7s} | {'WR':>5s} | {'PnL':>8s}")
print("="*85)
for name, n, w, l, wr, tp in trials:
    print(f"{name:55s} | {n:6d} | {w}/{l:<3d} | {wr:4.1f}% | ${tp:>+6.2f}")

# Best trade-off: grouped by WR >= 80%
print("\n\n=== BEST TRADE-OFFS (WR >= 80%) ===")
best = [t for t in trials if t[4] >= 80.0]
best.sort(key=lambda x: -x[1])
for name, n, w, l, wr, tp in best:
    print(f"{name:55s} | {n:6d} | {w}/{l:<3d} | {wr:4.1f}% | ${tp:>+6.2f}")

# Per-month estimate (46 trading days = ~2 months)
print(f"\n\nPer-month estimates (~46 trading days = 2 months):")
print(f"{'Config':55s} | {'/mo':>5s} | {'WR':>5s} | {'PnL/mo':>8s}")
for name, n, w, l, wr, tp in best[:10]:
    est_trades = round(n * 30 / 46)
    est_pnl = tp * 30 / 46
    print(f"{name:55s} | {est_trades:5d} | {wr:4.1f}% | ${est_pnl:>+6.2f}")
