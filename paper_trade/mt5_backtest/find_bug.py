"""Find why sim_recon produces 0 trades despite z-score extremes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Direct instrumentation: modify check_entry to log when z >= 3.5
from sim_recon import ReconSim, Trade

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
print(f"Data: {len(df)} bars, sprd med={df['spread'].median()}")

# Scan: compute z for every bar and note when abs(z) >= 3.5
closes = df['close'].values
opens = df['open'].values
spreads = df['spread'].values
times = df['time'].values
z_window = 50

close_buf = np.array([])
z_ge_35 = []
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
        z_ge_35.append((i, z, spreads[i], opens[i], closes[i]))

print(f"\nBars with |z|>=3.5: {len(z_ge_35)}")
if z_ge_35:
    print(f"First at bar {z_ge_35[0][0]} ({z_ge_35[0][0]/len(closes)*100:.1f}% into data)")
    print(f"Last at bar {z_ge_35[-1][0]}")
    for i, z, sprd, o, c in z_ge_35[:5]:
        dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
        print(f"  bar={i} dt={dt} z={z:.2f} sprd={sprd} o={o:.5f} c={c:.5f}")

# Now run the sim and check what happens at those bars
print(f"\n\nChecking sim_recon behavior at first extreme bar...")
sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0)

# Process bars up to the trigger
first_extreme = z_ge_35[0][0]
print(f"First extreme at bar {first_extreme}")

# Manually simulate the run up to that point
times_ns = times.astype('int64') // 10**9
n = first_extreme + 5

for i in range(n):
    dt = datetime.fromtimestamp(times_ns[i], tz=timezone.utc)
    
    if dt < datetime(2026, 6, 8, tzinfo=timezone.utc):
        sim.update_buffers(df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i],
                          df['close'].iloc[i], df['spread'].iloc[i], df['tick_volume'].iloc[i])
        continue
    
    if sim.trade is not None:
        sim.manage_position(df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i],
                           df['close'].iloc[i], df['spread'].iloc[i])
    if sim.trade is None:
        sim.check_entry(df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i],
                       df['close'].iloc[i], df['spread'].iloc[i], df['tick_volume'].iloc[i], dt, i)
    sim.update_buffers(df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i],
                      df['close'].iloc[i], df['spread'].iloc[i], df['tick_volume'].iloc[i])

print(f"\nAfter processing {n} bars:")
print(f"  close_buf size: {len(sim.close_buf)}")
print(f"  trade: {sim.trade}")
print(f"  trades: {len(sim.trades)}")

# Now compute what z-score would be inside check_entry at the trigger bar
# At bar first_extreme, before update_buffers, close_buf has closes[0..first_extreme-1]
close_buf_at_entry = np.array([])
for j in range(first_extreme):
    close_buf_at_entry = np.append(close_buf_at_entry, closes[j])

print(f"\nAt bar {first_extreme} entry check:")
print(f"  close_buf size: {len(close_buf_at_entry)}")
if len(close_buf_at_entry) >= z_window + 2:
    rets = np.diff(close_buf_at_entry[-(z_window+2):])
    cur_ret = rets[-1]
    mean = np.mean(rets[:-1])
    var = np.var(rets[:-1], ddof=1)
    z_at_entry = (cur_ret - mean) / np.sqrt(var) if var >= 1e-14 else 0
    print(f"  z at entry check: {z_at_entry:.2f} (expect >=3.5 to trigger)")
    print(f"  cur_ret={cur_ret:.8f} mean={mean:.8f} var={var:.10f}")
    print(f"  sprd={spreads[first_extreme]} <= max_sprd={sim.max_spread}: {spreads[first_extreme] <= sim.max_spread}")
else:
    print(f"  NOT ENOUGH DATA for z-score: {len(close_buf_at_entry)} < {z_window + 2}")
