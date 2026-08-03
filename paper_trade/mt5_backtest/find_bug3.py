"""Direct trace at every bar near the first trigger."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sim_recon import ReconSim
from datetime import datetime, timezone

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
closes = df['close'].values
opens = df['open'].values
spreads = df['spread'].values
times_ns = df['time'].values.astype('int64') // 10**9

sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)

n = len(df)
for i in range(245):  # bars 240-245
    dt = datetime.fromtimestamp(times_ns[i], tz=timezone.utc)
    o, h, l, c, sprd = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i], df['spread'].iloc[i]
    tvol = df['tick_volume'].iloc[i]
    
    if dt < datetime(2026, 6, 8, tzinfo=timezone.utc):
        sim.update_buffers(o, h, l, c, sprd, tvol)
        continue
    
    if sim.trade is not None:
        sim.manage_position(o, h, l, c, sprd)
    if sim.trade is None:
        z_before = sim.compute_zscore()
        av = sim.get_atr()
        
        buf_size = len(sim.close_buf)
        
        sim.check_entry(o, h, l, c, sprd, tvol, dt, i)
        
        if sim.trade is not None:
            print(f"ENTRY at bar {i}: z={z_before:.2f} dir={sim.trade.direction:+d}")
    
    sim.update_buffers(o, h, l, c, sprd, tvol)
    
    z_after = sim.compute_zscore()
    if i >= 240 and i <= 244:
        print(f"bar {i:>3d} dt={dt} close={c:.5f} sprd={sprd} "
              f"z(check_entry)={sim.compute_zscore() if False else '?'} "
              f"z_after={z_after:.2f} buf={len(sim.close_buf)}")

print(f"\nTrades: {len(sim.trades)}")
if sim.trade:
    print(f"Open trade: dir={sim.trade.direction:+d}")
