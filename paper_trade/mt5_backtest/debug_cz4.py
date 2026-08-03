"""Deep trace of ChallengeZ run at trigger zone."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from challenge_z import ChallengeZ, Trade
from datetime import datetime, timezone

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
closes = df['close'].values
opens = df['open'].values
spreads = df['spread'].values
times_ns = df['time'].values.astype('int64') // 10**9

# Build buffer as ChallengeZ run would
z_window = 50
atr_period = 20
warmup_bars = max(z_window + 3, 60)
start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)

close_buf = np.array([])
atr_buf = np.array([])

def compute_zscore(buf):
    if len(buf) < z_window + 2:
        return 0.0
    rets = np.diff(buf[-(z_window+2):])
    cur_ret = rets[-1]
    mean = np.mean(rets[:-1])
    var = np.var(rets[:-1], ddof=1)
    if var < 1e-14:
        return 0.0
    return (cur_ret - mean) / np.sqrt(var)

def get_atr(buf):
    if len(buf) < atr_period:
        return 0.0
    return float(np.mean(buf))

n = len(df)
for i in range(260):
    dt = datetime.fromtimestamp(times_ns[i], tz=timezone.utc)
    o, h, l, c, sprd = opens[i], df['high'].iloc[i], df['low'].iloc[i], closes[i], spreads[i]
    
    is_warmup = (dt < start_dt or i < warmup_bars)
    if is_warmup:
        close_buf = np.append(close_buf, c)
        hl = h - l
        atr_buf = np.append(atr_buf, hl)[-atr_period:]
        if i < 5 or i >= warmup_bars - 2:
            print(f"WARMUP bar {i}: buf={len(close_buf)} atr={len(atr_buf)}")
        continue
    
    # MAIN PHASE — replicate check_entry logic
    z = compute_zscore(close_buf)
    av = get_atr(atr_buf)
    
    # Check conditions
    hour_ok = 0 <= dt.hour < 24
    sprd_ok = sprd <= 10
    
    if abs(z) >= 3.5 or i in range(240, 250):
        print(f"MAIN bar {i:>3d} dt={dt} o={o:.5f} c={c:.5f} sprd={sprd} "
              f"buf={len(close_buf)} z={z:.2f} av={av:.6f} "
              f"hour_ok={hour_ok} sprd_ok={sprd_ok} "
              f"trigger={abs(z)>=3.5}")
    
    # Update buffers
    close_buf = np.append(close_buf, c)
    hl = h - l
    atr_buf = np.append(atr_buf, hl)[-atr_period:]
