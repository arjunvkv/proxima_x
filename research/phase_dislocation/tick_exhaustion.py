"""Tick sequence exhaustion — microstructure edge from queue depletion.
After N consecutive same-direction ticks, the next tick reverses.
"""
import numpy as np
import pandas as pd

files = [
    ('data/cache/ticks_EURJPY_2026-04-01_2026-04-30_50000_42.parquet', 'EURJPY'),
    ('data/cache/ticks_EURJPY_2026-04-21_2026-05-10_50000_42.parquet', 'EURJPY'),
    ('data/cache/ticks_EURJPY_2026-05-01_2026-05-20_50000_42.parquet', 'EURJPY'),
    ('data/cache/ticks_EURJPY_2026-05-21_2026-06-08_50000_42.parquet', 'EURJPY'),
]

all_results = []

for fpath, pair in files:
    df = pd.read_parquet(fpath)
    prices = df['price'].values
    n = len(prices)

    ticks = np.diff(prices)
    directions = np.sign(ticks)
    d_n = len(directions)

    print(f"{pair} — {fpath.split('/')[-1]}: {n} ticks, {np.sum(directions != 0)} directional")

    # run_counts[i] = number of consecutive same-direction ticks up to position i
    run_counts = np.zeros(d_n)
    for i in range(1, d_n):
        if directions[i-1] == 0:
            run_counts[i] = 0
        elif directions[i] == directions[i-1] and directions[i] != 0:
            run_counts[i] = run_counts[i-1] + 1
        else:
            run_counts[i] = 0

    for run_len in range(1, 31):
        # Tick i just completed a run of `run_len` same-direction ticks
        mask = run_counts == run_len
        idx = np.where(mask)[0]
        next_idx = idx + 1
        next_idx = next_idx[next_idx < d_n]
        if len(next_idx) < 5:
            continue

        next_dir = directions[next_idx]
        curr_dir = directions[idx[:len(next_idx)]]
        valid = (next_dir != 0) & (curr_dir != 0)
        if np.sum(valid) < 5:
            continue

        reversal = np.mean(curr_dir[valid] != next_dir[valid])
        n_obs = np.sum(valid)

        all_results.append({
            'pair': pair,
            'run_len': run_len,
            'n_obs': n_obs,
            'reversal_rate': reversal,
            'file': fpath.split('/')[-1],
        })

rd = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("TICK SEQUENCE EXHAUSTION — SUMMARY")
print("=" * 60)
for run_len in sorted(rd['run_len'].unique()):
    sub = rd[rd['run_len'] == run_len]
    avg_rev = sub['reversal_rate'].mean()
    avg_n = sub['n_obs'].mean()
    if avg_n >= 20:
        n_files = len(sub)
        print(f"  Run={run_len:>2d}: rev={avg_rev*100:.1f}% (avg {avg_n:.0f} obs, {n_files} files)")

print("\nBest single results:")
for _, row in rd.sort_values('reversal_rate', ascending=False).head(15).iterrows():
    print(f"  Run={int(row['run_len']):>2d} rev={row['reversal_rate']*100:.1f}% n={int(row['n_obs']):>4d} [{row['pair']} {row['file']}]")

# Edge analysis: does reversal edge = edge AFTER spread?
print("\n" + "=" * 60)
print("SPREAD-ADJUSTED EXPECTED VALUE")
print("=" * 60)

# Load one file for tick size stats
f = 'data/cache/ticks_EURJPY_2026-04-01_2026-04-30_50000_42.parquet'
df = pd.read_parquet(f)
prices = df['price'].values
ticks = np.diff(prices)
abs_ticks = np.abs(ticks[ticks != 0])
avg_tick = np.mean(abs_ticks)
pip = 0.01  # EURJPY quoted in pips

print(f"Avg tick move: {avg_tick:.5f} = {avg_tick/pip:.2f} pips")
print(f"EURJPY spread: ~2.5 pips = {2.5*pip:.5f}")
print(f"Tick-to-spread ratio: {avg_tick/(2.5*pip):.1f}x")

# For this file, compute reversal PnL for each run_len
d = np.sign(np.diff(prices))
d_n = len(d)
rc = np.zeros(d_n)
for i in range(1, d_n):
    if d[i-1] == 0:
        rc[i] = 0
    elif d[i] == d[i-1] and d[i] != 0:
        rc[i] = rc[i-1] + 1
    else:
        rc[i] = 0

for run_len in [1, 3, 5, 10, 15, 20, 25]:
    mask = rc == run_len
    idx = np.where(mask)[0]
    next_idx = idx + 1
    next_idx = next_idx[next_idx < d_n]
    if len(next_idx) < 10:
        continue

    # Trade: after a run of up-ticks, go SHORT. After down-ticks, go LONG.
    curr_d = d[idx[:len(next_idx)]]
    nxt_d = next_dir = d[next_idx]
    valid = (curr_d != 0) & (nxt_d != 0)
    if np.sum(valid) < 5:
        continue

    # PnL from reversal trade
    # If run is up (curr_d=1), we short → profit if next tick is down (nxt_d=-1)
    # If run is down (curr_d=-1), we long → profit if next tick is up (nxt_d=1)
    reversal_pnl = np.where(
        curr_d[valid] == 1,  # up run → short
        (-nxt_d[valid]).astype(float),  # profit if nxt_d == -1
        nxt_d[valid].astype(float)  # down run → long, profit if nxt_d == 1
    )
    # Only count directional ticks (PNL = ±1 for right/wrong direction)
    gross_pnl = np.mean(reversal_pnl)
    # Subtract spread cost
    spread_ticks = 2.5 * pip / avg_tick  # how many ticks of spread
    net_pnl = gross_pnl - spread_ticks * (1 / (np.sum(valid) / len(valid)))  # per trade
    wr = np.mean(reversal_pnl > 0) * 100

    print(f"  Run={run_len:>2d}: WR={wr:.0f}% gross_pnl={gross_pnl:.3f} net_pnl_est={gross_pnl - 0.5*spread_ticks:.3f} [{np.sum(valid)} trades]")
