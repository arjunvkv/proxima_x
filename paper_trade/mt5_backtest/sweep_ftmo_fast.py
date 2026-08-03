"""Focused sweep: z x sprd x lot on FTMO. Hours fixed to 0-7 Asian session."""
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
print("Initializing FTMO...")
if not mt5.initialize(path=path):
    print(f"FTMO init failed: {mt5.last_error()}")
    sys.exit(1)

pair_data = {}
for pair in pairs:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"{pair}: NO DATA")
        continue
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    pair_data[pair] = df
    print(f"{pair}: {len(df)} bars, sprd min={int(df['spread'].min())} med={int(df['spread'].median())}")

mt5.shutdown()

# Reduced sweep: z x sprd (lots handled by scaling)
zs = [3.0, 3.5, 4.0]
spreads = [9, 10, 12, 15, 20]
base_lot = 0.75
lot_scale = [1.0, 2.0, 3.0, 5.0]

print(f"\n{'='*80}")
print("SWEEP (base_lot=0.75, 0-7 UTC, both dir)")
print(f"{'='*80}")

results = []
for pair, df in pair_data.items():
    times = df['time'].astype(np.int64)//10**9
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    spreads = df['spread'].values
    volumes = df['tick_volume'].values
    start_dt = FROM.replace(tzinfo=timezone.utc)

    for z in zs:
        for ms in spreads:
            sim = ReconSim(
                z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                max_hold=54, base_lot=base_lot, max_spread=ms,
                limit_entry_atr=0.0, enable_stability=False,
                trade_dir=0, start_hour=0, end_hour=7
            )
            trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)
            total_base = sum(t.pnl for t in trades)
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            neutral = [t for t in trades if abs(t.pnl) < 0.01]
            denom = len(trades) - len(neutral)
            wr = len(wins)/denom*100 if denom else 0

            results.append({
                'pair': pair, 'z': z, 'sprd': ms,
                'trades': len(trades), 'wr': wr,
                'pnl_base': total_base
            })

            if len(trades) > 0:
                print(f"{pair:<8} z={z} sprd<={ms:2d}: {len(trades):3d}t {wr:5.1f}% ${total_base:>+8.2f}")

# Best per pair
print(f"\n\n{'='*80}")
print("BEST CONFIG PER PAIR (base_lot=0.75)")
print(f"{'='*80}")
print(f"{'PAIR':<8} {'z':<4} {'sprd≤':<6} {'TRADES':>7} {'WR':>7} {'PnL(base)':>10}")
print("-" * 45)
best_configs = {}
for pair in pairs:
    pr = [r for r in results if r['pair'] == pair and r['trades'] >= 3]
    if not pr:
        print(f"{pair:<8} NO VALID CONFIG")
        continue
    best = max(pr, key=lambda r: r['pnl_base'] if r['wr'] >= 60 else -9999)
    if best['wr'] < 60:
        best = max(pr, key=lambda r: r['pnl_base'])
    best_configs[pair] = best
    scaled = {f"lot={lot}": best['pnl_base'] * (lot / base_lot) for lot in lot_scale}
    print(f"{pair:<8} z={best['z']} sprd<={best['sprd']:2d}: "
          f"{best['trades']:3d}t {best['wr']:5.1f}% ${best['pnl_base']:>+8.2f} "
          f"| scaled: {', '.join(f'{k}=${v:.0f}' for k,v in scaled.items())}")

# Portfolio
print(f"\n\n{'='*80}")
print("PORTFOLIO (best config per pair)")
print(f"{'='*80}")
for lot in lot_scale:
    total_pnl = sum(c['pnl_base'] * (lot / base_lot) for c in best_configs.values())
    total_trades = sum(c['trades'] for c in best_configs.values())
    avg_wr = sum(c['wr'] for c in best_configs.values())/len(best_configs)
    print(f"  lot={lot:.1f}: {total_trades:3d}t total, {avg_wr:.1f}% avg WR, ${total_pnl:>+9.2f} PnL")

# Top single configs across all
print(f"\n\n{'='*80}")
print("TOP 10 CONFIGS by PnL (WR>=60%, trades>=5)")
print(f"{'='*80}")
print(f"{'PAIR':<8} {'z':<4} {'sprd≤':<6} {'TRADES':>7} {'WR':>7} {'PnL(base)':>10}")
print("-" * 45)
top = sorted([r for r in results if r['wr'] >= 60 and r['trades'] >= 5],
             key=lambda r: -r['pnl_base'])[:10]
for r in top:
    print(f"{r['pair']:<8} {r['z']:<4} {r['sprd']:>4d}   {r['trades']:>7d} {r['wr']:>6.1f}% ${r['pnl_base']:>+8.2f}")
