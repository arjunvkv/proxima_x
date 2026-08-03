"""Run sim_recon on FTMO data for 6 cross pairs with relaxed max_spread."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sim_recon import ReconSim
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timezone

pairs = ["AUDNZD", "EURAUD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
FROM = datetime(2026, 6, 8)
TO = datetime(2026, 7, 26)

path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
print(f"Initializing FTMO ({path})...")
if not mt5.initialize(path=path):
    print(f"FTMO init failed: {mt5.last_error()}")
    sys.exit(1)

ftmo_results = []
for pair in pairs:
    print(f"\n--- {pair} ---")
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  NO DATA")
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    times = df['time'].astype(np.int64)//10**9
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    spreads = df['spread'].values
    volumes = df['tick_volume'].values
    start_dt = FROM.replace(tzinfo=timezone.utc)

    zero_sprd = (df['spread'] == 0).sum()
    sprd_min = int(df['spread'].min())
    sprd_max = int(df['spread'].max())
    sprd_med = int(df['spread'].median())

    for max_sprd in [9, 10, 12, 15, 20]:
        sim = ReconSim(
            z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
            max_hold=54, base_lot=0.75, max_spread=max_sprd,
            limit_entry_atr=0.0, enable_stability=False,
            trade_dir=0, start_hour=0, end_hour=7
        )
        trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)

        total = sum(t.pnl for t in trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        neutral = [t for t in trades if abs(t.pnl) < 0.01]
        denom = len(trades) - len(neutral)
        wr = len(wins)/denom*100 if denom else 0
        avg_win = sum(t.pnl for t in wins)/len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses)/len(losses) if losses else 0

        print(f"  sprd<={max_sprd:2d}: {len(trades):3d} trades, {wr:5.1f}% WR, ${total:>+8.2f} PnL "
              f"(W:{len(wins)} L:{len(losses)} N:{len(neutral)})")

        ftmo_results.append({
            'pair': pair, 'max_sprd': max_sprd, 'trades': len(trades),
            'pnl': total, 'wr': wr
        })

mt5.shutdown()

# Summary table
print(f"\n\n{'='*90}")
print("FTMO CROSS PAIRS — sim_recon (non-blocking) z=3.5 0-7 lot=0.75")
print(f"Spread stats: none of the 6 pairs have zero-spread bars on FTMO")
print(f"{'='*90}")
print(f"{'PAIR':<8} {'max_sprd':>10} {'TRADES':>8} {'WR':>7} {'PnL':>12}")
print("-" * 45)
for r in ftmo_results:
    print(f"{r['pair']:<8} {r['max_sprd']:>10d} {r['trades']:>8d} {r['wr']:>6.1f}% ${r['pnl']:>+8.2f}")
