"""Run sim_recon.py (non-blocking) on all 6 cross pairs with V2+z CPPF config."""
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

if not mt5.initialize():
    print("MT5 init failed!")
    sys.exit(1)

results = []
for pair in pairs:
    print(f"\n--- {pair} ---")
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  NO DATA for {pair}")
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

    # V2+z CPPF exact config: z=3.5, stop=3, trig=1, gap=0.05, lot=0.75
    # Hours: 0-7, both directions, no stability, no ATR filter
    sim = ReconSim(
        z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.75, max_spread=5,
        limit_entry_atr=0.0, enable_stability=False,
        trade_dir=0, start_hour=0, end_hour=7
    )
    trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, FROM.replace(tzinfo=timezone.utc))

    total = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    denom = len(trades) - len(neutral)
    wr = len(wins)/denom*100 if denom else 0
    avg_win = sum(t.pnl for t in wins)/len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses)/len(losses) if losses else 0

    print(f"  Bars: {len(df)}  Trades: {len(trades)}  W:{len(wins)} L:{len(losses)} N:{len(neutral)}")
    print(f"  PnL: ${total:.2f}  WR: {wr:.1f}%  AvgW: ${avg_win:.2f}  AvgL: ${avg_loss:.2f}")

    results.append({
        'pair': pair, 'trades': len(trades), 'wins': len(wins),
        'losses': len(losses), 'pnl': total, 'wr': wr
    })

mt5.shutdown()

# Print comparison table
print(f"\n\n{'='*90}")
print("NON-BLOCKING (sim_recon.py) vs LOOK-AHEAD (sim_backtest.py) vs REAL MT5")
print(f"{'='*90}")
print(f"{'PAIR':<10} {'RECON_TRD':>10} {'SIM_TRD':>9} {'MT5_TRD':>9} "
      f"{'RECON_WR':>9} {'SIM_WR':>8} {'MT5_WR':>8} "
      f"{'RECON_PnL':>10} {'SIM_PnL':>9} {'MT5_PnL':>9}")
print("-" * 90)

# Previous sim_backtest.py results (reproduced)
sim_pnl_map = {"AUDNZD": 1654.94, "EURAUD": 2070.51, "EURNZD": 4062.06,
               "GBPAUD": 2129.52, "GBPCAD": 2397.74, "GBPNZD": 4175.41}
sim_trd_map = {"AUDNZD": 109, "EURAUD": 103, "EURNZD": 109,
               "GBPAUD": 114, "GBPCAD": 126, "GBPNZD": 101}
sim_wr_map = {"AUDNZD": 77.1, "EURAUD": 81.6, "EURNZD": 81.7,
              "GBPAUD": 78.1, "GBPCAD": 84.1, "GBPNZD": 81.2}

mt5_pnl_map = {"AUDNZD": -581, "EURAUD": 313, "EURNZD": -52,
               "GBPAUD": 499, "GBPCAD": -351, "GBPNZD": -792}
mt5_trd_map = {"AUDNZD": 214, "EURAUD": 215, "EURNZD": 119,
               "GBPAUD": 179, "GBPCAD": 180, "GBPNZD": 115}
mt5_wr_map = {"AUDNZD": 58.4, "EURAUD": 66.5, "EURNZD": 66.4,
              "GBPAUD": 70.9, "GBPCAD": 61.7, "GBPNZD": 60.0}

total_recon = {'trades': 0, 'pnl': 0}
total_sim = {'trades': 0, 'pnl': 0}
total_mt5 = {'trades': 0, 'pnl': 0}

for r in results:
    p = r['pair']
    total_recon['trades'] += r['trades']
    total_recon['pnl'] += r['pnl']
    total_sim['trades'] += sim_trd_map[p]
    total_sim['pnl'] += sim_pnl_map[p]
    total_mt5['trades'] += mt5_trd_map[p]
    total_mt5['pnl'] += mt5_pnl_map[p]
    print(f"{p:<10} {r['trades']:>10d} {sim_trd_map[p]:>9d} {mt5_trd_map[p]:>9d} "
          f"{r['wr']:>8.1f}% {sim_wr_map[p]:>7.1f}% {mt5_wr_map[p]:>7.1f}% "
          f"${r['pnl']:>+8.2f} ${sim_pnl_map[p]:>+8.2f} ${mt5_pnl_map[p]:>+8.2f}")

print("-" * 90)
print(f"{'TOTAL':<10} {total_recon['trades']:>10d} {total_sim['trades']:>9d} {total_mt5['trades']:>9d} "
      f"{'':>8} {'':>7} {'':>7} "
      f"${total_recon['pnl']:>+8.2f} ${total_sim['pnl']:>+8.2f} ${total_mt5['pnl']:>+8.2f}")

avg_wr_recon = sum(r['wr'] for r in results)/len(results)
print(f"\nMORE ACCURATE: RECON is non-blocking (no look-ahead)")
print(f"  recon_recon vs reality: ${total_recon['pnl']:.0f} vs ${total_mt5['pnl']:.0f}")
print(f"  sim_backtest vs reality: ${total_sim['pnl']:.0f} vs ${total_mt5['pnl']:.0f}")
