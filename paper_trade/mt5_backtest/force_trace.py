"""Force-trace the entry check — modify sim_recon at runtime."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Directly test z-score computation on the data
fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
times = df['time'].values.astype('int64') // 10**9
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
spreads = df['spread'].values

n = len(times)
start_dt = datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc)

# Manual tracking
close_buf = np.array([])
atr_buf = np.array([])
z_window = 50
atr_period = 20

first_entry_hit = None

for i in range(n):
    dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
    o, h, l, c, sprd = opens[i], highs[i], lows[i], closes[i], spreads[i]

    if dt < start_dt:
        close_buf = np.append(close_buf, c)
        hl = h - l
        atr_buf = np.append(atr_buf, hl)[-atr_period:]
        continue
    
    # Main phase — replicate check_entry logic
    # Compute z-score
    if len(close_buf) >= z_window + 2:
        rets = np.diff(close_buf[-(z_window+2):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var >= 1e-14:
            z = (cur_ret - mean) / np.sqrt(var)
            if abs(z) >= 3.5:
                print(f"bar[{i}] dt={dt}  z={z:.2f}  sprd={sprd}  o={o:.5f}  c={c:.5f}  "
                      f"close_buf={len(close_buf)}  atr_buf={len(atr_buf)}")
                if first_entry_hit is None:
                    first_entry_hit = i
    
    # Update buffers
    close_buf = np.append(close_buf, c)
    hl = h - l
    atr_buf = np.append(atr_buf, hl)[-atr_period:]
    
    if i > n // 4 and first_entry_hit is not None:
        break

print(f"\nFirst |z|>=3.5 at bar index {first_entry_hit}" if first_entry_hit else "\nNO |z|>=3.5 found!")

# Show buffer state at first entry point
if first_entry_hit:
    i = first_entry_hit
    print(f"\nBuffer state at entry bar {i}:")
    dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
    print(f"  dt={dt}, o={opens[i]:.5f}, c={closes[i]:.5f}")
    # Rebuild buffer to show state at this point
    close_buf2 = np.array([])
    for j in range(i):
        close_buf2 = np.append(close_buf2, closes[j])
    print(f"  close_buf size: {len(close_buf2)}")
    if len(close_buf2) >= z_window + 2:
        rets = np.diff(close_buf2[-(z_window+2):])
        print(f"  Last 52 closes: {close_buf2[-5:]}")
        print(f"  Last 5 returns: {rets[-5:]}")
        print(f"  cur_ret={rets[-1]:.8f}, mean={np.mean(rets[:-1]):.8f}, "
              f"var={np.var(rets[:-1], ddof=1):.10f}")
        z = (rets[-1] - np.mean(rets[:-1])) / np.sqrt(np.var(rets[:-1], ddof=1))
        print(f"  z={z:.2f}")
