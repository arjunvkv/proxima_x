"""Fixed FTMO sweep — no variable shadowing. z x sprd_limit x lot."""
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
    sp_med = int(df['spread'].median())
    sp_min = int(df['spread'].min())
    sp_max = int(df['spread'].max())
    zero = int((df['spread'] == 0).sum())
    print(f"{pair}: {len(df)} bars, spread min={sp_min} med={sp_med} max={sp_max} zero={zero}")

mt5.shutdown()

# FIXED: use 'limits' not 'spreads' to avoid shadowing
z_vals = [3.0, 3.5, 4.0, 5.0]
limits = [5, 6, 7, 8, 9, 10, 12, 15, 20, 50]
base_lot = 0.75
lot_scales = [1.0, 2.0, 3.0, 5.0]

all_results = []
for pair, df in pair_data.items():
    times = df['time'].astype(np.int64)//10**9
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    sprds = df['spread'].values
    vols = df['tick_volume'].values
    start_dt = FROM.replace(tzinfo=timezone.utc)

    for z in z_vals:
        for lim in limits:
            sim = ReconSim(
                z_threshold=z, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                max_hold=54, base_lot=base_lot, max_spread=min(lim, 999),
                limit_entry_atr=0.0, enable_stability=False,
                trade_dir=0, start_hour=0, end_hour=7
            )
            trades = sim.run(times, opens, highs, lows, closes, sprds, vols, start_dt)
            total = sum(t.pnl for t in trades)
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            neutral = [t for t in trades if abs(t.pnl) < 0.01]
            denom = len(trades) - len(neutral)
            wr = len(wins)/denom*100 if denom else 0

            all_results.append({
                'pair': pair, 'z': z, 'lim': lim,
                'trades': len(trades), 'wr': wr, 'pnl': total
            })

            if len(trades) > 0:
                print(f"{pair:<8} z={z} sprd<={lim:3d}: {len(trades):3d}t {wr:5.1f}% ${total:>+9.2f}")

    # per-pair summary
    pr = [r for r in all_results if r['pair'] == pair and r['trades'] >= 3]
    if pr:
        best = max(pr, key=lambda r: r['pnl'] if r['wr'] >= 60 else -9999)
        if best['wr'] < 60:
            best = max(pr, key=lambda r: r['pnl'])
        print(f"  >> {pair} BEST: z={best['z']} sprd<={best['lim']} {best['trades']}t {best['wr']:.1f}% ${best['pnl']:+.2f}")

# Top 20 overall
print(f"\n\n{'='*80}")
print("TOP 20 CONFIGS (WR>=60%, trades>=5)")
print(f"{'='*80}")
print(f"{'PAIR':<8} {'z':<4} {'sprd≤':<6} {'TR':>5} {'WR':>6} {'PnL':>10}")
candidates = [r for r in all_results if r['wr'] >= 60 and r['trades'] >= 5]
candidates.sort(key=lambda r: -r['pnl'])
for r in candidates[:20]:
    print(f"{r['pair']:<8} {r['z']:<4} {r['lim']:>4d}   {r['trades']:>5d} {r['wr']:>5.1f}% ${r['pnl']:>+8.2f}")

# Portfolio with best per pair
print(f"\n\n{'='*80}")
print("PORTFOLIO (best config per pair, WR>=60%)")
print(f"{'='*80}")
best_cfgs = {}
for pair in pairs:
    pr = [r for r in candidates if r['pair'] == pair]
    if not pr:
        continue
    best = max(pr, key=lambda r: r['pnl'])
    best_cfgs[pair] = best
    print(f"  {pair:<8}: z={best['z']} sprd<={best['lim']} {best['trades']:3d}t {best['wr']:5.1f}% ${best['pnl']:>+8.2f}")

if best_cfgs:
    base_pnl = sum(c['pnl'] for c in best_cfgs.values())
    base_trd = sum(c['trades'] for c in best_cfgs.values())
    avg_wr = sum(c['wr'] for c in best_cfgs.values())/len(best_cfgs)
    print(f"  {'TOTAL':<8}: {base_trd:3d}t {avg_wr:5.1f}% ${base_pnl:>+8.2f} (lot={base_lot})")
    for lot in lot_scales:
        scaled = base_pnl * (lot / base_lot)
        print(f"  {'':<8}  scaled lot={lot:.1f}: ${scaled:>+9.2f}")
