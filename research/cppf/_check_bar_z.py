"""Check if the 18:08 bar triggers a signal in the backtest."""
import pandas as pd
import numpy as np

pair = "gbpnzd"
df = pd.read_parquet(f"research/cppf/_mt5_data/{pair}.parquet")
df.index = pd.to_datetime(df.index, utc=True)

c = df["close"].values
h = df["high"].values
l = df["low"].values

# Find index of 18:08 bar
t = pd.Timestamp(2026, 7, 23, 18, 8, tz="UTC")
mask = df.index <= t
idx_1808 = mask.sum() - 1
print(f"18:08 bar index: {idx_1808}")
print(f"Bar: {df.index[idx_1808]} O={df.iloc[idx_1808]['open']:.5f} H={h[idx_1808]:.5f} L={l[idx_1808]:.5f} C={c[idx_1808]:.5f}")

# Compute z-score at index
ret = np.diff(c)
z_arr = np.full(len(df), np.nan)
for i in range(51, len(df)):
    rw = ret[i - 51:i - 1]
    mu = rw.mean()
    sig = rw.std(ddof=1)
    z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0

atr_arr = np.full(len(df), np.nan)
rng = df["high"] - df["low"]
for i in range(21, len(df)):
    atr_arr[i] = np.mean(rng.iloc[i - 20:i])

print(f"\nZ-score at 18:08 bar (index {idx_1808}): z={z_arr[idx_1808]:.2f}")
print(f"ATR at 18:08: {atr_arr[idx_1808]*10000:.1f} pips")
print(f"Signal: {'SHORT' if abs(z_arr[idx_1808]) >= 2.5 else 'NONE'}")

# Also check position 1 bar (17:58) — the bar used by the live strategy
# copy_rates_from_pos at 18:09:01 returns position 1 = bar ending at 18:09 = bar starting at 17:58
# Wait no, position 1 at time 18:09:00 returns bar for minute 17:58? No...
# Position 1 = last COMPLETED bar. At 18:09:01, the last completed bar is the one ending at 18:09:00.
# The bar ending at 18:09:00 started at 18:08:00.
# So position 1 = bar[18:08:00 - 18:08:59].
# But what time does MT5 report? The bar OPEN time = 18:08:00.
# So copy_rates_from_pos with pos=1 at time 18:09:01 returns bar with time=18:08:00.
# This is correct.

# The LIVE strategy saw close=2.30782 for this bar.
# The ARCHIVED close for this bar is 2.30765.
# The TICK-derived close for this bar is 2.30774.

print(f"\n=== What-if with LIVE close (2.30782) ===")
# If live close = 2.30782, what would ret be?
live_close = 2.30782
prev_close = c[idx_1808 - 1]
live_ret = live_close - prev_close
actual_ret = c[idx_1808] - prev_close
rw = ret[idx_1808 - 50:idx_1808 - 1]
mu = rw.mean()
sig = rw.std(ddof=1)
live_z = (live_ret - mu) / sig if sig > 1e-10 else 0
actual_z = z_arr[idx_1808]
print(f"Archived close: {c[idx_1808]:.5f} -> ret={actual_ret:.6f} -> z={actual_z:.2f}")
print(f"Live close:     {live_close:.5f} -> ret={live_ret:.6f} -> z={live_z:.2f}")
print(f"Live z exceeds 2.5? {abs(live_z) >= 2.5}")

# Check if the 18:09 bar (next bar) would have hit the stop
if abs(live_z) >= 2.5:
    s = 0.15 * atr_arr[idx_1808]
    tg = 0.20 * atr_arr[idx_1808]
    gp = 0.10 * atr_arr[idx_1808]
    direction = -1
    entry = live_close
    
    # Next bar (18:09) high/low
    next_high = h[idx_1808 + 1]
    next_low = l[idx_1808 + 1]
    next_open = df.iloc[idx_1808 + 1]["open"]
    
    best = entry
    if next_low < best:
        best = next_low  # wait, for shorts we track high? No...
        # Direction -1 (short): check next bar LOW, extend stop from best (which tracks LOW)
    # Actually for shorts: if l[bp] < best, best = l[bp]; sl = entry + s; if entry - best > tg, sl = best + gp
    # Check: h[bp] >= sl? 
    
    best_short = entry
    sl = entry + s
    print(f"\nShort trailing stop check:")
    print(f"  Entry={entry:.5f} s={s:.6f} tg={tg:.6f} gp={gp:.6f}")
    print(f"  Initial stop: {sl:.5f}")
    if entry - next_low > tg:
        best_short = next_low
        sl = best_short + gp
        print(f"  Triggered! Best={best_short:.5f} New stop={sl:.5f}")
    if next_high >= sl:
        print(f"  STOP HIT at {sl:.5f}")
        print(f"  PnL: {(sl - entry) * direction * 10000:.0f} pips = ${(sl - entry) * direction * 10000 * 100000:.0f}")
