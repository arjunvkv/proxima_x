"""Run sim_recon on FTMO data for 6 cross pairs and compare vs normal MT5 data."""
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

# Try initializing FTMO terminal (might need explicit path)
print("Initializing FTMO terminal...")
path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if not mt5.initialize(path=path):
    print(f"FTMO init failed: {mt5.last_error()}")
    # fallback to default
    if not mt5.initialize():
        print("Default MT5 also failed!")
        sys.exit(1)
    print("Using default MT5 terminal")

# Check what symbols are available
available = mt5.symbols_get()
avail_names = {s.name for s in available} if available else set()
print(f"Available symbols: {len(avail_names)}")

ftmo_results = []
for pair in pairs:
    if pair not in avail_names:
        print(f"\n{pair}: NOT AVAILABLE on this terminal")
        ftmo_results.append({'pair': pair, 'trades': 0, 'pnl': 0, 'wr': 0, 'bars': 0, 'available': False})
        continue

    print(f"\n--- {pair} ---")
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  NO DATA for {pair}")
        ftmo_results.append({'pair': pair, 'trades': 0, 'pnl': 0, 'wr': 0, 'bars': 0, 'available': True})
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
    sim = ReconSim(
        z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.75, max_spread=5,
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

    zero_spread = (df['spread'] == 0).sum()
    sprd_min = df['spread'].min()
    sprd_median = df['spread'].median()

    print(f"  Bars: {len(df)}  Zero-spread: {zero_spread} ({zero_spread/len(df)*100:.1f}%)")
    print(f"  Spread: min={sprd_min}, median={sprd_median}")
    print(f"  Trades: {len(trades)}  W:{len(wins)} L:{len(losses)} N:{len(neutral)}")
    print(f"  PnL: ${total:.2f}  WR: {wr:.1f}%  AvgW: ${avg_win:.2f}  AvgL: ${avg_loss:.2f}")

    ftmo_results.append({
        'pair': pair, 'trades': len(trades), 'pnl': total, 'wr': wr,
        'bars': len(df), 'zero_spread_pct': zero_spread/len(df)*100,
        'sprd_median': sprd_median, 'available': True
    })

mt5.shutdown()

# Print comparison
print(f"\n\n{'='*100}")
print("FTMO DATA vs NORMAL MT5 DATA — sim_recon (non-blocking) | z=3.5 0-7 lot=0.75")
print(f"{'='*100}")
print(f"{'PAIR':<10} {'FTMO_TRD':>10} {'NORM_TRD':>10} {'FTMO_WR':>9} {'NORM_WR':>9} "
      f"{'FTMO_PnL':>12} {'NORM_PnL':>12} {'FTMO_0sprd':>10} {'NORM_0sprd':>10}")
print("-" * 90)

# Normal MT5 data results from sim_recon (previous run)
norm = {
    'AUDNZD': (98, 68.4, 270.37),
    'EURAUD': (91, 71.4, 190.57),
    'EURNZD': (49, 73.5, 921.27),
    'GBPAUD': (70, 65.7, 15.79),
    'GBPCAD': (76, 77.3, 586.57),
    'GBPNZD': (54, 66.7, 1205.31),
}

total_ftmo = {'trades': 0, 'pnl': 0}
total_norm = {'trades': 0, 'pnl': 0}

for r in ftmo_results:
    p = r['pair']
    n = norm[p]
    available = r.get('available', True)
    if not available:
        print(f"{p:<10} {'NOT AVAILABLE':<21} {n[0]:>10d} {'':>9} {n[1]:>8.1f}% ${n[2]:>+9.2f}")
        continue

    total_ftmo['trades'] += r['trades']
    total_ftmo['pnl'] += r['pnl']
    total_norm['trades'] += n[0]
    total_norm['pnl'] += n[2]

    print(f"{p:<10} {r['trades']:>10d} {n[0]:>10d} {r['wr']:>8.1f}% {n[1]:>8.1f}% "
          f"${r['pnl']:>+9.2f} ${n[2]:>+9.2f} {r.get('zero_spread_pct',0):>9.1f}% {'—':>10}")

print("-" * 90)
print(f"{'TOTAL':<10} {total_ftmo['trades']:>10d} {total_norm['trades']:>10d} "
      f"{'':>8} {'':>8} "
      f"${total_ftmo['pnl']:>+9.2f} ${total_norm['pnl']:>+9.2f}")

print(f"\nZERO-SPREAD ANALYSIS:")
for r in ftmo_results:
    if r.get('available'):
        print(f"  {r['pair']}: FTMO zero-spread = {r.get('zero_spread_pct',0):.1f}% (median spread = {r.get('sprd_median',0):.0f})")
