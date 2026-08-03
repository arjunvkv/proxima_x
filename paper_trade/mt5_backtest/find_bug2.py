"""Trace sim_recon at the exact entry trigger bars."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sim_recon import ReconSim

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
closes = df['close'].values
opens = df['open'].values
spreads = df['spread'].values
times_ns = df['time'].values.astype('int64') // 10**9

# First, find bars where |z|>=3.5 by scanning with close[i] included
z_window = 50
close_buf = np.array([])
trigger_bars = []
for i in range(len(closes)):
    close_buf = np.append(close_buf, closes[i])
    if len(close_buf) < z_window + 2:
        continue
    rets = np.diff(close_buf[-(z_window+2):])
    cur_ret = rets[-1]
    mean = np.mean(rets[:-1])
    var = np.var(rets[:-1], ddof=1)
    if var < 1e-14:
        continue
    z = (cur_ret - mean) / np.sqrt(var)
    if abs(z) >= 3.5:
        trigger_bars.append(i)  # z reached 3.5 at this bar's close

print(f"Trigger bars (|z|>=3.5): {len(trigger_bars)} total")
print(f"First 10 trigger bars: {trigger_bars[:10]}")

# Now manually run sim_recon logic and trace at trigger bars
sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)

n = len(df)
entry_count = 0

for i in range(n):
    dt = datetime.fromtimestamp(times_ns[i], tz=timezone.utc)
    o, h, l, c, sprd = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i], df['spread'].iloc[i]
    
    # Warmup (first bar is at 00:00 UTC, so dt == start_dt → no warmup)
    if dt < datetime(2026, 6, 8, tzinfo=timezone.utc):
        sim.update_buffers(o, h, l, c, sprd, df['tick_volume'].iloc[i])
        continue
    
    # Manage existing trade
    if sim.trade is not None:
        sim.manage_position(o, h, l, c, sprd)
    
    # Check entry
    if sim.trade is None:
        z_check = sim.compute_zscore()
        av = sim.get_atr()
        
        # Trace at trigger bars
        if i in trigger_bars[:20]:
            print(f"TRIGGER BAR {i}: dt={dt} z_check(z before buf)={z_check:.2f} "
                  f"av={av:.6f} sprd={sprd} <=10={sprd<=10}")
        
        sim.check_entry(o, h, l, c, sprd, df['tick_volume'].iloc[i], dt, i)
        
        if sim.trade is not None:
            entry_count += 1
            print(f"  >>> ENTRY #{entry_count} at bar {i}: dir={sim.trade.direction:+d} "
                  f"z={sim.trade.z_entry:.2f} atr={sim.trade.atr_entry:.6f} "
                  f"price={sim.trade.entry_price:.5f} sprd={sim.trade.sprd_entry}")
    
    # Update buffers
    sim.update_buffers(o, h, l, c, sprd, df['tick_volume'].iloc[i])

print(f"\n\nTotal entries: {entry_count}")
print(f"Total trades completed: {len(sim.trades)}")
for t in sim.trades[:5]:
    print(f"  entry_bar={t.entry_time} dir={t.direction:+d} z={t.z_entry:.2f} pnl=${t.pnl:.2f}")
