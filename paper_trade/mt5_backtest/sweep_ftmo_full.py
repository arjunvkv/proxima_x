"""Full sweep on FTMO: z x spread x hours x lot to find best PnL + WR combo."""
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
print(f"Initializing FTMO...")
if not mt5.initialize(path=path):
    print(f"FTMO init failed: {mt5.last_error()}")
    sys.exit(1)

# Load all pair data into memory
pair_data = {}
for pair in pairs:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"{pair}: NO DATA")
        continue
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    pair_data[pair] = df
    sprd_med = int(df['spread'].median())
    sprd_min = int(df['spread'].min())
    zero = (df['spread'] == 0).sum()
    print(f"{pair}: {len(df)} bars, spread min={sprd_min} med={sprd_med} zero={zero}")

mt5.shutdown()

# Sweep parameters
zs = [3.0, 3.5, 4.0]
spreads = [9, 10, 12, 15, 20]
hours = [(0,7), (0,12), (0,0)]  # 0-7, 0-12, 0-24
lots = [0.75, 1.5, 3.0, 5.0]

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
            for (sh, eh) in hours:
                for lot in lots:
                    sim = ReconSim(
                        z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                        max_hold=54, base_lot=lot, max_spread=ms,
                        limit_entry_atr=0.0, enable_stability=False,
                        trade_dir=0, start_hour=sh, end_hour=eh
                    )
                    trades = sim.run(times, opens, highs, lows, closes,
                                     spreads, volumes, start_dt)
                    total = sum(t.pnl for t in trades)
                    wins = [t for t in trades if t.pnl > 0]
                    losses = [t for t in trades if t.pnl < 0]
                    neutral = [t for t in trades if abs(t.pnl) < 0.01]
                    denom = len(trades) - len(neutral)
                    wr = len(wins)/denom*100 if denom else 0
                    hours_str = f"{sh}-{eh}" if eh > 0 else "0-24"

                    results.append({
                        'pair': pair, 'z': z, 'sprd': ms,
                        'hours': hours_str, 'lot': lot,
                        'trades': len(trades), 'wr': wr, 'pnl': total
                    })

    # Print per-pair progress
    pair_results = [r for r in results if r['pair'] == pair]
    best = max(pair_results, key=lambda r: r['pnl'])
    print(f"{pair} best PnL: z={best['z']} sprd<={best['sprd']} "
          f"{best['hours']} lot={best['lot']}: {best['trades']}t "
          f"{best['wr']:.1f}% ${best['pnl']:.2f}")

# Print full summary
print(f"\n\n{'='*120}")
print("FULL SWEEP — FTMO CROSS PAIRS (non-blocking sim_recon)")
print(f"{'='*120}")

# Top by PnL with WR >= 60%
candidates = [r for r in results if r['wr'] >= 60 and r['trades'] >= 5]
candidates.sort(key=lambda r: -r['pnl'])

print(f"\nTop 30 by PnL (WR>=60%, trades>=5):")
print(f"{'PAIR':<8} {'z':<4} {'sprd≤':>5} {'hours':<6} {'lot':<5} {'TRADES':>7} {'WR':>7} {'PnL':>10}")
print("-" * 60)
for r in candidates[:30]:
    print(f"{r['pair']:<8} {r['z']:<4} {r['sprd']:>5d} {r['hours']:<6} "
          f"{r['lot']:<5.1f} {r['trades']:>7d} {r['wr']:>6.1f}% ${r['pnl']:>+8.2f}")

# Multi-pair portfolio (best config per pair summed)
print(f"\n\n{'='*80}")
print("PORTFOLIO — Best config per pair (WR>=60%)")
print(f"{'='*80}")
portfolio = {}
for pair in pairs:
    pr = [r for r in candidates if r['pair'] == pair]
    if not pr:
        print(f"{pair}: no valid config")
        continue
    best = max(pr, key=lambda r: r['pnl'])
    portfolio[pair] = best
    print(f"  {pair:<8} z={best['z']} sprd<={best['sprd']} {best['hours']} "
          f"lot={best['lot']:.1f}: {best['trades']:3d}t {best['wr']:5.1f}% ${best['pnl']:>+8.2f}")

if portfolio:
    total_pnl = sum(r['pnl'] for r in portfolio.values())
    total_trades = sum(r['trades'] for r in portfolio.values())
    avg_wr = sum(r['wr'] for r in portfolio.values())/len(portfolio)
    print(f"\n  {'TOTAL':<8} {'':>8} {'':>8} {'':>6} {'':>5} "
          f"{total_trades:3d}t {avg_wr:5.1f}% ${total_pnl:>+8.2f}")

# Also try same pair repeated with lot scaling
print(f"\n\n{'='*80}")
print("TOP SINGLE-PAIR SCALING (best config by PnL)")
print(f"{'='*80}")
for r in candidates[:5]:
    print(f"  {r['pair']:<8} z={r['z']} sprd<={r['sprd']} {r['hours']} "
          f"lot={r['lot']:.1f}: {r['trades']:3d}t {r['wr']:5.1f}% ${r['pnl']:>+8.2f}")
