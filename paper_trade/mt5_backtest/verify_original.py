"""Verify original sim_recon works on both FTMO and FundedNext data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sim_recon import ReconSim
from datetime import datetime, timezone

# Load FTMO data
ftmo = np.load(os.path.join(os.path.dirname(__file__), 'ftmo_audusd_m1.npy'), allow_pickle=True)
ftmo_df = pd.DataFrame(ftmo)
print(f"FTMO AUDUSD: {len(ftmo_df)} bars, spread med={ftmo_df['spread'].median()}")

# Run original sim_recon on FTMO data
sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)
trades = sim.run(
    ftmo_df['time'].astype('int64') // 10**9,
    ftmo_df['open'].values, ftmo_df['high'].values,
    ftmo_df['low'].values, ftmo_df['close'].values,
    ftmo_df['spread'].values, ftmo_df['tick_volume'].values
)
if trades:
    pnl = sum(t.pnl for t in trades)
    wins = len([t for t in trades if t.pnl > 0])
    print(f"FTMO: {len(trades)} trades, {wins}W/{len(trades)-wins}L, WR={wins/len(trades)*100:.1f}%, PnL=${pnl:.2f}")
    for t in trades[:3]:
        print(f"  bar={t.entry_time} price={t.entry_price:.5f} dir={t.direction:+d} z={t.z_entry:.2f} -> {t.exit_reason} pnl=${t.pnl:.2f}")
else:
    print("FTMO: 0 trades")

# Load FundedNext data
fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
fn_df = pd.DataFrame(fn)
print(f"\nFundedNext AUDUSD: {len(fn_df)} bars, spread med={fn_df['spread'].median()}")

sim2 = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)
trades2 = sim2.run(
    fn_df['time'].astype('int64') // 10**9,
    fn_df['open'].values, fn_df['high'].values,
    fn_df['low'].values, fn_df['close'].values,
    fn_df['spread'].values, fn_df['tick_volume'].values
)
if trades2:
    pnl2 = sum(t.pnl for t in trades2)
    wins2 = len([t for t in trades2 if t.pnl > 0])
    print(f"FundedNext: {len(trades2)} trades, {wins2}W/{len(trades2)-wins2}L, WR={wins2/len(trades2)*100:.1f}%, PnL=${pnl2:.2f}")
    for t in trades2[:3]:
        print(f"  bar={t.entry_time} price={t.entry_price:.5f} dir={t.direction:+d} z={t.z_entry:.2f} -> {t.exit_reason} pnl=${t.pnl:.2f}")
else:
    print("FundedNext: 0 trades")
    # Check buffer state
    print(f"  close_buf: {len(sim2.close_buf)}")
    print(f"  z at end: {sim2.compute_zscore():.2f}")
    
    # Try with later start
    print(f"\n  Retrying with later start (6 hours in)...")
    sim3 = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                    max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)
    start_dt = datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc)
    trades3 = sim3.run(
        fn_df['time'].astype('int64') // 10**9,
        fn_df['open'].values, fn_df['high'].values,
        fn_df['low'].values, fn_df['close'].values,
        fn_df['spread'].values, fn_df['tick_volume'].values,
        start_dt=start_dt
    )
    if trades3:
        pnl3 = sum(t.pnl for t in trades3)
        wins3 = len([t for t in trades3 if t.pnl > 0])
        print(f"  FundedNext (delayed): {len(trades3)} trades, {wins3}W/{len(trades3)-wins3}L, "
              f"WR={wins3/len(trades3)*100:.1f}%, PnL=${pnl3:.2f}")
    else:
        print(f"  Still 0 trades. close_buf={len(sim3.close_buf)}, "
              f"close_count={sim3.close_count}, z_count={sim3.z_count}")
