"""Debug Challenge-Z run method."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from challenge_z import ChallengeZ
from datetime import datetime, timezone

fn = np.load(os.path.join(os.path.dirname(__file__), 'fundednext_audusd_m1.npy'), allow_pickle=True)
df = pd.DataFrame(fn)
closes = df['close'].values

# Direct z-score check (replicating ChallengeZ.compute_zscore)
z_window = 50
close_buf = np.array([])
print("Manual z-score trace around warmup end:")
for i in range(50, 250):
    close_buf = np.append(close_buf, closes[i])
    if len(close_buf) < z_window + 2:
        continue
    rets = np.diff(close_buf[-(z_window+2):])
    cur_ret = rets[-1]
    mean = np.mean(rets[:-1])
    var = np.var(rets[:-1], ddof=1)
    if var < 1e-14:
        z = 0.0
    else:
        z = (cur_ret - mean) / np.sqrt(var)
    if abs(z) >= 3.0 or i < 55:
        print(f"  bar {i}: buf={len(close_buf)} cur_ret={cur_ret:.8f} z={z:.2f}")

print("\n\nNow running ChallengeZ sim...")
sim = ChallengeZ(z_threshold=3.5, hold_bars=10, max_spread=10, lot_size=0.5, trade_dir=0)
trades = sim.run(
    df['time'].astype('int64') // 10**9,
    df['open'].values, df['high'].values, df['low'].values, df['close'].values,
    df['spread'].values, df['tick_volume'].values
)
print(f"Trades: {len(sim.trades)}")
if len(sim.trades) > 0:
    for t in sim.trades[:5]:
        print(f"  bar={t.entry_bar} dir={t.direction:+d} z={t.z_entry:.2f} "
              f"gross=${t.gross_pnl:.2f} net=${t.net_pnl:.2f}")
