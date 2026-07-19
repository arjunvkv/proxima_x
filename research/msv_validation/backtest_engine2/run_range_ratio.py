"""Range Ratio Convergence Backtest.

Tests: When range ratio between correlated pairs diverges (mean + 2σ),
does the lagging pair expand/catch up in the following N bars?

Pairs tested: EURUSD/GBPUSD, AUDUSD/NZDUSD, EURJPY/GBPJPY
"""
import numpy as np
from numba import jit
from data import TempCache, PAIRS, pip

@jit(nopython=True)
def bar_range(bar):
    """Range of a bar in pips."""
    return (bar[1] - bar[2])  # H - L (raw price, convert after)

@jit(nopython=True)
def close_pct(bar):
    """Close position within range: 0.0=bottom, 1.0=top."""
    rng = bar[1] - bar[2]
    if rng == 0:
        return 0.5
    return (bar[3] - bar[2]) / rng

def test_pair_ratio(data, times, pi_a, pi_b, pair_a, pair_b, lookbacks=(10, 20, 30), z_thresh=2.0, fwd_bars=5):
    """Test range ratio convergence between pair A and pair B."""
    results = []
    n = data.shape[0]

    for lb in lookbacks:
        trades = 0
        wins = 0
        ratio_at_entry = []
        ratio_at_exit = []
        dir_matched = 0

        for i in range(lb, n - fwd_bars):
            # Compute rolling ratio of ranges
            ranges_a = np.array([bar_range(data[i-k, pi_a]) for k in range(lb)])
            ranges_b = np.array([bar_range(data[i-k, pi_b]) for k in range(lb)])

            # Avoid div by zero
            valid = (ranges_b > 0)
            if valid.sum() < lb // 2:
                continue

            ratios = ranges_a / (ranges_b + 1e-10)
            mean_r = np.mean(ratios[valid])
            std_r = np.std(ratios[valid])

            if std_r == 0:
                continue

            cur_ratio = ratios[-1]
            z = (cur_ratio - mean_r) / std_r

            # Only trade when pair A expanded relative to pair B
            if z < z_thresh:
                continue

            # At entry: which direction did pair A close?
            entry_bar = data[i, pi_a]
            entry_dir = entry_bar[3] - entry_bar[0]  # close - open
            entry_close_pct = close_pct(entry_bar)

            # Need directional close (close at extreme)
            if abs(entry_close_pct - 0.5) < 0.3:
                continue  # too wicky, not directional

            # Check forward bars: does pair B move in same direction?
            fwd_close_a = data[i + fwd_bars - 1, pi_a, 3]
            fwd_close_b = data[i + fwd_bars - 1, pi_b, 3]
            entry_close_b = data[i, pi_b, 3]

            move_b = fwd_close_b - entry_close_b
            move_a = fwd_close_a - data[i, pi_a, 3]

            # Win: pair B moved in the same direction as pair A's entry direction
            same_dir = (move_b > 0) == (entry_dir > 0)
            # Also check if ratio converged
            end_ratio = bar_range(data[i+fwd_bars-1, pi_a]) / (bar_range(data[i+fwd_bars-1, pi_b]) + 1e-10)
            converged = abs(end_ratio - mean_r) < abs(cur_ratio - mean_r)

            trades += 1
            if same_dir:
                wins += 1
                dir_matched += 1

            ratio_at_entry.append(cur_ratio)
            ratio_at_exit.append(end_ratio)

        if trades > 10:
            wr = wins / trades * 100
            dir_pct = dir_matched / trades * 100
            avg_ratio_entry = np.mean(ratio_at_entry)
            avg_ratio_exit = np.mean(ratio_at_exit)
            print(f"  Lookback={lb:2d} | Trades={trades:4d} | WR={wr:5.1f}% | DirMatch={dir_pct:5.1f}% | "
                  f"Ratio {avg_ratio_entry:.2f}→{avg_ratio_exit:.2f} | Z>{z_thresh}")

            results.append({
                'pair_a': pair_a, 'pair_b': pair_b, 'lookback': lb,
                'trades': trades, 'wr': wr, 'dir_match_pct': dir_pct
            })

    return results

def run():
    print("=" * 70)
    print("ENGINE 2 — Range Ratio Convergence Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    data_pips = data.copy()  # We'll work in raw price, convert at output

    all_results = []

    # 1. EURUSD vs GBPUSD (same block, USD quoted)
    print("\n--- Pair Set 1: EURUSD vs GBPUSD ---")
    r = test_pair_ratio(data_pips, times, 0, 2, 'EURUSD', 'GBPUSD', lookbacks=(10, 20, 30), z_thresh=2.0)
    all_results.extend(r)
    r = test_pair_ratio(data_pips, times, 0, 2, 'EURUSD', 'GBPUSD', lookbacks=(10, 20, 30), z_thresh=1.5)
    all_results.extend(r)

    # 2. AUDUSD vs NZDUSD (commodity block)
    print("\n--- Pair Set 2: AUDUSD vs NZDUSD ---")
    r = test_pair_ratio(data_pips, times, 3, 4, 'AUDUSD', 'NZDUSD', lookbacks=(10, 20, 30), z_thresh=2.0)
    all_results.extend(r)
    r = test_pair_ratio(data_pips, times, 3, 4, 'AUDUSD', 'NZDUSD', lookbacks=(10, 20, 30), z_thresh=1.5)
    all_results.extend(r)

    # 3. EURJPY vs GBPJPY (JPY cross block)
    print("\n--- Pair Set 3: EURJPY vs GBPJPY ---")
    r = test_pair_ratio(data_pips, times, 5, 6, 'EURJPY', 'GBPJPY', lookbacks=(10, 20, 30), z_thresh=2.0)
    all_results.extend(r)
    r = test_pair_ratio(data_pips, times, 5, 6, 'EURJPY', 'GBPJPY', lookbacks=(10, 20, 30), z_thresh=1.5)
    all_results.extend(r)

    # Summary
    print("\n--- RANGE RATIO SUMMARY ---")
    best = max(all_results, key=lambda x: x['wr'])
    worst = min(all_results, key=lambda x: x['wr'])
    print(f"Best: {best['pair_a']}/{best['pair_b']} lb={best['lookback']} WR={best['wr']:.1f}% ({best['trades']} trades)")
    print(f"Worst: {worst['pair_a']}/{worst['pair_b']} lb={worst['lookback']} WR={worst['wr']:.1f}% ({worst['trades']} trades)")

    avg_wr = np.mean([r['wr'] for r in all_results])
    avg_trades = np.mean([r['trades'] for r in all_results])
    print(f"Average WR across all configs: {avg_wr:.1f}%")
    print(f"Average trades per config: {avg_trades:.0f}")

    # Count configs above 55%
    above = sum(1 for r in all_results if r['wr'] > 55)
    print(f"Configs above 55% WR: {above}/{len(all_results)}")

    return all_results

if __name__ == '__main__':
    results = run()
