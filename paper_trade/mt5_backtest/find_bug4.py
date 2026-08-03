"""Minimal step-through of sim_recon near trigger point."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from datetime import datetime, timezone

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)

# Column indices from MT5: time, open, high, low, close, tick_volume, spread, real_volume
times = fn['time'] if 'time' in fn.dtype.names else fn[:, 0]

# Check dtype
print(f"Data type: {fn.dtype}, names: {fn.dtype.names}")

if fn.dtype.names:
    closes = fn['close']
    opens = fn['open']
    spreads = fn['spread']
    # time is in seconds
else:
    closes = fn[:, 4]
    opens = fn[:, 1]
    spreads = fn[:, 6]

n = len(closes)
print(f"Data: {n} bars, first close={closes[0]:.5f}, last close={closes[-1]:.5f}")

z_window = 50
close_buf = np.array([])

# Check z at bar 243 vs 244
for i in [242, 243, 244, 245, 246]:
    c = closes[i]
    o = opens[i]
    sprd = spreads[i]
    
    # What the entry check sees (before adding close[i])
    print(f"\nBar {i} ENTRY CHECK (buf before adding close[{i}]):")
    print(f"  open={o:.5f} close={c:.5f} sprd={sprd}")
    print(f"  buf size={len(close_buf)}")
    
    if len(close_buf) >= z_window + 2:
        rets = np.diff(close_buf[-(z_window+2):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var >= 1e-14:
            z = (cur_ret - mean) / np.sqrt(var)
            print(f"  cur_ret={cur_ret:.8f} mean={mean:.8f} var={var:.10f} z={z:.2f}")
            print(f"  |z| >= 3.5? {abs(z) >= 3.5}")
        else:
            print(f"  var too small: {var}")
    else:
        print(f"  Not enough data ({len(close_buf)} < {z_window+2})")
    
    # Now add close[i] to buffer (as update_buffers does)
    close_buf = np.append(close_buf, c)
    
    # What the scanner sees (after adding close[i])
    print(f"  AFTER adding close[{i}] (scanner view):")
    if len(close_buf) >= z_window + 2:
        rets = np.diff(close_buf[-(z_window+2):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var >= 1e-14:
            z = (cur_ret - mean) / np.sqrt(var)
            print(f"  cur_ret={cur_ret:.8f} mean={mean:.8f} var={var:.10f} z={z:.2f}")

print(f"\n\nSUMMARY: ")
print(f"  Entry check at bar 243 uses rets[-1]=close[242]-close[241]")
print(f"  Entry check at bar 244 uses rets[-1]=close[243]-close[242]")
print(f"  The trigger at bar 243 found z=-3.50 AFTER adding close[243]")
print(f"  So at bar 244's entry check, rets[-1]=close[243]-close[242] should give z=-3.50")
