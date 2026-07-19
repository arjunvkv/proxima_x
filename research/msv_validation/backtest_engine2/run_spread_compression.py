"""Spread Compression Leading Indicator Backtest.

Tests: Does a drop in M1 spread predict wider range / directional movement?
Uses M1 bar spread data (available from MT5).

Reality check: Many compressions are just normal tight liquidity, not signals.
"""
import numpy as np
from data import TempCache, PAIRS

def run():
    print("=" * 70)
    print("ENGINE 2 — Spread Compression Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    n = data.shape[0]
    npairs = len(PAIRS)

    all_results = []

    for pi, pair in enumerate(PAIRS):
        spreads = data[:, pi, 5]
        ranges = data[:, pi, 1] - data[:, pi, 2]
        closes = data[:, pi, 3]
        tick_vols = data[:, pi, 4]

        # Filter bars with valid spread data (spread > 0)
        valid = spreads > 0

        for lookback in [5, 10, 20]:
            for compression_pct in [0.5, 0.3]:  # spread drops to 50% or 30% of average
                trades = 0
                range_increased = 0
                directional = 0
                vol_increased = 0

                for i in range(lookback, n - 2):
                    if not valid[i]:
                        continue

                    # Recent spread history
                    recent_spreads = spreads[i-lookback:i]
                    if np.all(recent_spreads == 0) or np.sum(recent_spreads > 0) < lookback // 2:
                        continue

                    avg_spread = np.mean(recent_spreads[recent_spreads > 0])
                    if avg_spread == 0:
                        continue

                    cur_spread = spreads[i]

                    # Compression: current spread is X% of average
                    if cur_spread > avg_spread * compression_pct:
                        continue

                    # Skip if spread is already very low (normal tightness)
                    if cur_spread <= 1:
                        continue

                    # Now check the NEXT 2 bars
                    next_range = max(ranges[i+1], ranges[i+2])
                    avg_prev_range = np.mean(ranges[max(0, i-10):i])
                    prev_bar_close = closes[i]
                    next_bar_close = closes[i+1]

                    # Does range expand?
                    if next_range > avg_prev_range * 1.2:
                        range_increased += 1

                    # Is there a directional close in next bars?
                    move = closes[i+2] - closes[i]
                    prev_move = closes[i] - closes[max(0, i-1)]
                    if abs(move) > abs(prev_move) * 1.2:
                        directional += 1

                    # Does tick volume increase?
                    avg_vol = np.mean(tick_vols[max(0, i-10):i])
                    if tick_vols[i+1] > avg_vol * 1.2:
                        vol_increased += 1

                    trades += 1

                if trades > 5:
                    range_pct = range_increased / trades * 100
                    dir_pct = directional / trades * 100
                    vol_pct = vol_increased / trades * 100
                    print(f"  {pair:8s} lb={lookback:2d} comp={compression_pct:.0%} | "
                          f"Trades={trades:4d} | RangeExp={range_pct:5.1f}% | "
                          f"Dir={dir_pct:5.1f}% | VolUp={vol_pct:5.1f}%")

                    all_results.append({
                        'pair': pair, 'lookback': lookback, 'compression': compression_pct,
                        'trades': trades, 'range_expand_pct': range_pct,
                        'directional_pct': dir_pct, 'vol_increase_pct': vol_pct
                    })

    # Summary
    print("\n" + "=" * 70)
    print("SPREAD COMPRESSION SUMMARY")
    print("=" * 70)

    if not all_results:
        print("\nNo valid data — likely zero spreads in M1 data.")
        print("MT5 M1 bars show spread=0 for most bars (broker-reported, not calculated).")
        print("This technique requires tick-level bid/ask data to test properly.")
        return

    for pair in PAIRS:
        pair_results = [r for r in all_results if r['pair'] == pair]
        if not pair_results:
            continue
        avg_range = np.mean([r['range_expand_pct'] for r in pair_results])
        avg_dir = np.mean([r['directional_pct'] for r in pair_results])
        print(f"\n  {pair}: avg range_expand={avg_range:.1f}%, avg directional={avg_dir:.1f}%")

    # Reality check
    print("\n--- Reality Check ---")
    print("Spread compression alone is NOT a tradeable signal.")
    print("Direction is ambiguous — tight spreads can precede movement OR dead market.")
    print("Use as a 'get ready' flag only, paired with leading pair direction.")

if __name__ == '__main__':
    run()
