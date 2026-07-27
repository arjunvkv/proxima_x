"""Test tick exhaustion on real EURUSD data.
Hypothesis: after N consecutive same-direction ticks, probability of reversal > 65%."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")
df = pd.read_csv(DATA / "EURUSD_mt5.csv", parse_dates=['time'])
print(f"Loaded {len(df):,} EURUSD ticks")

df['mid'] = (df.bid + df.ask) / 2
df['dir'] = np.sign(df.mid.diff()).fillna(0).astype(int)

# Count consecutive same-direction ticks (vectorized)
streaks = []
cur = 0
for d in df.dir:
    if d == 0:
        pass  # unchanged tick: streak unchanged
    elif d == 1:
        cur = cur + 1 if cur >= 0 else 1
    else:
        cur = cur - 1 if cur <= 0 else -1
    streaks.append(cur)
df['streak'] = streaks

max_lookahead = 20
for la in range(1, max_lookahead + 1):
    df[f'fwd_{la}'] = df.mid.shift(-la) - df.mid

print("\n=== Reversal Probability After Consecutive Ticks ===")
results = []
for min_streak in [2, 3, 5, 8, 10, 15]:
    for lookahead in [1, 2, 3, 5, 10, 20]:
        col = f'fwd_{lookahead}'
        mask = (df.streak.abs() >= min_streak) & (df.dir != 0)
        n = mask.sum()
        if n < 10: continue
        up_mask = mask & (df.dir > 0)
        dn_mask = mask & (df.dir < 0)
        up_rev = (df.loc[up_mask, col] < 0).mean() if up_mask.sum() > 0 else 0
        dn_rev = (df.loc[dn_mask, col] > 0).mean() if dn_mask.sum() > 0 else 0
        avg_rev = (up_rev * up_mask.sum() + dn_rev * dn_mask.sum()) / n
        print(f"  streak≥{min_streak:2d} → {lookahead:2d}t: rev={avg_rev*100:.1f}% N={n:,} "
              f"(up={up_rev*100:.1f}% dn={dn_rev*100:.1f}%)")

# Entry simulation: after streak≥3, enter 1-lot in fade direction, exit after N ticks
print("\n=== Simulated Strategy (streak≥3, fade direction, fixed tick exit) ===")
for exit_tick in [1, 3, 5, 10, 20]:
    entry_mask = (df.streak.abs() >= 3) & (df.dir != 0)
    entry_idx = df[entry_mask].index
    if len(entry_idx) == 0: continue
    pips = []
    wins = 0
    for idx in entry_idx:
        exit_idx = min(idx + exit_tick, len(df) - 1)
        entry_mid = df.mid.iloc[idx]
        exit_mid = df.mid.iloc[exit_idx]
        direction = df.dir.iloc[idx]
        # We fade: if direction > 0 (up streak), we short → profit when exit < entry
        if direction > 0:
            ret = (entry_mid - exit_mid) * 10000  # pips
        else:
            ret = (exit_mid - entry_mid) * 10000
        pips.append(ret)
        if ret > 0: wins += 1
    wr = wins / len(pips)
    avg = np.mean(pips)
    print(f"  streak≥3 exit@{exit_tick:2d}t: WR={wr*100:.1f}% avg={avg:+.2f}pip "
          f"trades={len(pips):,}")

# Average pip movement per streak length
print("\n=== Average reversal magnitude by streak length (5-tick exit) ===")
for s in [1, 2, 3, 5, 8, 10, 15]:
    mask = (df.streak.abs() >= s) & (df.streak.abs() < s+3) & (df.dir != 0)
    if mask.sum() < 10: continue
    fwd = df.loc[mask, 'fwd_5'] * 10000
    rev = ((fwd < 0) == (df.loc[mask, 'dir'] > 0)).mean() * 100
    print(f"  streak {s:2d}-{s+2}: N={mask.sum():5,d} rev={rev:.1f}% mean_fwd={fwd.mean():+.2f}pip")

print("\n=== DONE ===")
