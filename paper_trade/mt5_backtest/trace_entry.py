"""Trace entry logic of sim_recon on FundedNext data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Patch ReconSim with debug output
from sim_recon import ReconSim

original_check = ReconSim.check_entry

def debug_check(self, o, h, l, c, sprd, tick_vol, dt, bar_idx):
    if self.trade is not None:
        return
    if not self.check_hour(dt):
        return
    if sprd > self.max_spread:
        return
    
    z = self.compute_zscore()
    av = self.get_atr()
    
    # Print when z is near threshold
    if abs(z) > 3.0:
        print(f"  NEAR ENTRY: bar={bar_idx} dt={dt} z={z:.2f} av={av:.6f} "
              f"sprd={sprd} max_sprd={self.max_spread} "
              f"close_buf={self.close_count} atr_count={self.atr_count}")
        if abs(z) >= self.z_threshold and av > 0:
            print(f"    WOULD ENTER! dir={-1 if z>0 else 1}")
    
    if av <= 0:
        print(f"  ATR=0: bar={bar_idx} atr_count={self.atr_count}")
        return
    
    if abs(z) >= self.z_threshold:
        direction = -1 if z > 0 else 1
        print(f"  ENTERING: bar={bar_idx} dir={direction:+d} z={z:.2f} av={av:.6f} o={o:.5f} sprd={sprd}")

ReconSim.check_entry = debug_check

# Load data
fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
fn_df = pd.DataFrame(fn)

print(f"Data: {len(fn_df)} bars, spread med={fn_df['spread'].median()}")

sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)

# Use a later start so buffers have time to warm up
start_dt = datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc)
trades = sim.run(
    fn_df['time'].astype('int64') // 10**9,
    fn_df['open'].values, fn_df['high'].values,
    fn_df['low'].values, fn_df['close'].values,
    fn_df['spread'].values, fn_df['tick_volume'].values,
    start_dt=start_dt
)

print(f"\n\nRESULT: {len(trades)} trades")
if trades:
    for t in trades[:5]:
        print(f"  entry_bar={t.entry_time} dir={t.direction:+d} z={t.z_entry:.2f} "
              f"atr={t.atr_entry:.6f} -> {t.exit_reason} pnl=${t.pnl:.2f}")
