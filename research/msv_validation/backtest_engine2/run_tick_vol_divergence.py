"""Tick Volume Divergence Backtest.

Tests: When tick_volume spikes on one pair (relative to its baseline)
but a correlated pair's tick_volume stays quiet, does the quiet
pair experience increased volume + directional movement?
"""
import numpy as np
from data import TempCache, PAIRS

PAIR_GROUPS = [
    ('EURUSD', 'GBPUSD', 0, 2),
    ('AUDUSD', 'NZDUSD', 3, 4),
    ('EURJPY', 'GBPJPY', 5, 6),
    ('EURUSD', 'USDJPY', 0, 1),
]

def run():
    print("=" * 70)
    print("ENGINE 2 — Tick Volume Divergence Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    n = data.shape[0]

    all_results = []

    for name_a, name_b, pi_a, pi_b in PAIR_GROUPS:
        print(f"\n--- {name_a} vs {name_b} ---")

        tv_a = data[:, pi_a, 4]  # tick_volume
        tv_b = data[:, pi_b, 4]

        for z_thresh in [1.5, 2.0, 2.5]:
            for lookback in [10, 20, 30]:
                trades = 0
                vol_followed = 0
                dir_matched = 0
                dir_opposite = 0

                for i in range(lookback, n - 5):
                    # Compute z-score of tick volume for pair A
                    recent_a = tv_a[i-lookback:i]
                    if np.std(recent_a) == 0:
                        continue

                    z_a = (tv_a[i] - np.mean(recent_a)) / np.std(recent_a)

                    # Compute z-score for pair B (should be quiet)
                    recent_b = tv_b[i-lookback:i]
                    if np.std(recent_b) == 0:
                        continue

                    z_b = (tv_b[i] - np.mean(recent_b)) / np.std(recent_b)

                    # Divergence: A is active, B is quiet
                    if z_a < z_thresh or z_b > 1.0:
                        continue

                    # Check forward: does B's volume increase?
                    avg_b_fwd = np.mean(tv_b[i+1:i+4])
                    if avg_b_fwd > np.mean(recent_b) * 1.2:
                        vol_followed += 1

                    # Check direction: does B move in same direction as A?
                    dir_a = data[i, pi_a, 3] - data[i, pi_a, 0]  # close-open
                    dir_b = data[i+3, pi_b, 3] - data[i, pi_b, 0]  # close-open 3 bars later

                    if (dir_a > 0) == (dir_b > 0):
                        dir_matched += 1
                    else:
                        dir_opposite += 1

                    trades += 1

                if trades > 5:
                    vol_pct = vol_followed / trades * 100
                    dir_pct = dir_matched / trades * 100
                    print(f"  z>{z_thresh:.1f} lb={lookback:2d} | Trades={trades:4d} | "
                          f"VolFollow={vol_pct:5.1f}% | DirMatch={dir_pct:5.1f}%")

                    all_results.append({
                        'pair_a': name_a, 'pair_b': name_b,
                        'z_thresh': z_thresh, 'lookback': lookback,
                        'trades': trades, 'vol_follow_pct': vol_pct,
                        'dir_match_pct': dir_pct
                    })

    # Summary
    print("\n" + "=" * 70)
    print("TICK VOLUME DIVERGENCE SUMMARY")
    print("=" * 70)

    if not all_results:
        print("No results — data may have zero tick_volume.")
        return

    for name_a, name_b, _, _ in PAIR_GROUPS:
        r = [x for x in all_results if x['pair_a'] == name_a]
        if not r:
            continue
        avg_vol = np.mean([x['vol_follow_pct'] for x in r if x['trades'] > 5])
        avg_dir = np.mean([x['dir_match_pct'] for x in r if x['trades'] > 5])
        print(f"  {name_a}→{name_b}: avg vol_follow={avg_vol:.1f}%, avg dir_match={avg_dir:.1f}%")

    best = max(all_results, key=lambda x: x['dir_match_pct'])
    worst = min(all_results, key=lambda x: x['dir_match_pct'])
    print(f"\n  Best: {best['pair_a']}→{best['pair_b']} z>{best['z_thresh']} lb={best['lookback']} "
          f"dir_match={best['dir_match_pct']:.1f}% ({best['trades']} trades)")
    print(f"  Worst: {worst['pair_a']}→{worst['pair_b']} z>{worst['z_thresh']} lb={worst['lookback']} "
          f"dir_match={worst['dir_match_pct']:.1f}% ({worst['trades']} trades)")

    return all_results

if __name__ == '__main__':
    run()
