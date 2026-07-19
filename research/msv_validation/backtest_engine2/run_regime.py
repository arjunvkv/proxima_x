"""Cross-Pair Regime Detection Backtest.

Builds regime labels from directional agreement across 8 pairs.
Tests: (1) Regime stability — do regimes persist for N+ bars?
(2) Regime transitions — do they predict directional shifts?
(3) Economic value — can we trade regime persistence?
"""
import numpy as np
from numba import jit
from data import TempCache, PAIRS

def classify_regime(directions):
    """Classify market regime from direction vector of all pairs.
    directions: array of -1/0/+1 per pair
    Returns: regime name, confidence (0-1)
    """
    up = sum(1 for d in directions if d > 0)
    down = sum(1 for d in directions if d < 0)
    total = up + down
    if total == 0:
        return 'UNKNOWN', 0.0

    agreement = max(up, down) / total
    eurusd = directions[0]  # EURUSD
    usdjpy = directions[1]  # USDJPY
    audusd = directions[3]  # AUDUSD
    nzdusd = directions[4]  # NZDUSD

    # Risk-on: all USD-short pairs up, USDJPY flat or up
    if up >= total * 0.75 and eurusd > 0 and audusd > 0 and nzdusd > 0:
        return 'RISK_ON', agreement

    # USD bid: EURUSD down, USDJPY up, AUDUSD down
    if eurusd < 0 and usdjpy > 0 and audusd < 0:
        return 'USD_BID', agreement

    # USD offer: EURUSD up, USDJPY down, AUDUSD up
    if eurusd > 0 and usdjpy < 0 and audusd > 0:
        return 'USD_OFFER', agreement

    # EM stress: commodity pairs down, EURUSD flat/up, USDJPY up
    if audusd < 0 and nzdusd < 0 and usdjpy > 0 and abs(eurusd) == 0:
        return 'EM_STRESS', agreement

    # Carry unwind: JPY crosses all down
    jpy_crosses = [directions[5], directions[6]]  # EURJPY, GBPJPY
    if all(d < 0 for d in jpy_crosses if d != 0) and len([d for d in jpy_crosses if d < 0]) >= 1:
        return 'CARRY_UNWIND', agreement

    # Strong direction
    if agreement > 0.7:
        if up > down:
            return 'BROAD_UP', agreement
        else:
            return 'BROAD_DOWN', agreement

    # Weak / mixed
    if agreement < 0.55:
        return 'MIXED', agreement

    return 'LEANING', agreement


def run():
    print("=" * 70)
    print("ENGINE 2 — Cross-Pair Regime Detection Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    n = data.shape[0]
    npairs = len(PAIRS)

    # Compute per-bar directions for each pair
    directions = np.zeros((n, npairs), dtype='i1')
    for pi in range(npairs):
        for i in range(1, n):
            if data[i, pi, 3] > data[i-1, pi, 3]:
                directions[i, pi] = 1
            elif data[i, pi, 3] < data[i-1, pi, 3]:
                directions[i, pi] = -1
            # else 0 (unchanged)

    # Analyze regime stability
    print("\n--- Regime Stability Analysis ---")
    regimes = []
    confidences = []
    regime_runs = []

    for i in range(n):
        reg, conf = classify_regime(directions[i])
        regimes.append(reg)
        confidences.append(conf)

    # Count consecutive same-regime runs
    from itertools import groupby
    runs = [(k, len(list(g))) for k, g in groupby(regimes)]
    run_lengths = {}
    for reg, length in runs:
        if reg not in run_lengths:
            run_lengths[reg] = []
        run_lengths[reg].append(length)

    print(f"Total bars analyzed: {n}")
    print(f"\nRegime distribution:")
    for reg, lengths in sorted(run_lengths.items(), key=lambda x: sum(x[1]), reverse=True):
        avg_run = np.mean(lengths)
        total_bars = sum(lengths)
        pct = total_bars / n * 100
        print(f"  {reg:15s}: {total_bars:5d} bars ({pct:5.1f}%), "
              f"avg run={avg_run:5.1f} bars, max run={max(lengths):5d}")

    # Test: does regime persistence predict next bar direction?
    print("\n--- Regime Persistence as Predictor ---")
    for reg in ['RISK_ON', 'USD_BID', 'USD_OFFER', 'CARRY_UNWIND', 'BROAD_UP', 'BROAD_DOWN']:
        if reg not in run_lengths:
            continue

        # For each run of >= 3 bars of this regime, check bar 2:
        # does the market continue in the regime direction?
        entry_points = []
        predictions_correct = 0
        total_predictions = 0

        run_count = 1
        current_run = 0
        for i in range(1, n):
            if regimes[i] == regimes[i-1]:
                current_run += 1
            else:
                current_run = 0
                run_count += 1

            if current_run >= 2 and regimes[i] == reg:
                # We've seen 3+ bars of this regime (entry at bar 2 after confirmation)
                # Predict next bar continues same direction
                if i + 1 < n:
                    # What's the majority direction of this regime?
                    reg_up_count = sum(1 for d in directions[i] if d > 0)
                    reg_down_count = sum(1 for d in directions[i] if d < 0)
                    expected_up = reg_up_count > reg_down_count

                    # Check next bar
                    next_up = sum(1 for d in directions[i+1] if d > 0)
                    next_down = sum(1 for d in directions[i+1] if d < 0)

                    if expected_up and next_up > next_down:
                        predictions_correct += 1
                    elif not expected_up and next_down > next_up:
                        predictions_correct += 1
                    elif next_up == next_down:
                        pass  # skip ties
                    else:
                        pass
                    total_predictions += 1

        if total_predictions > 0:
            pct = predictions_correct / total_predictions * 100
            print(f"  {reg:15s}: {predictions_correct:4d}/{total_predictions:4d} correct = {pct:5.1f}%")

    # Test: profitable regime transitions
    print("\n--- Regime Transition Trading ---")
    print("(Entry at regime change, exit after X bars)")

    for hold_bars in [3, 5, 10]:
        trades = 0
        wins = 0
        for i in range(1, n - hold_bars):
            if regimes[i] != regimes[i-1]:
                # Regime changed at bar i
                new_reg = regimes[i]
                old_reg = regimes[i-1]

                # Skip transitions from/to UNKNOWN or MIXED
                if new_reg in ('UNKNOWN', 'MIXED') or old_reg in ('UNKNOWN', 'MIXED'):
                    continue

                # Enter at bar i, hold for hold_bars
                # Direction: majority of pairs at entry
                up_at_entry = sum(1 for d in directions[i] if d > 0)
                down_at_entry = sum(1 for d in directions[i] if d < 0)

                if up_at_entry == down_at_entry:
                    continue

                expected_up = up_at_entry > down_at_entry

                # After hold_bars
                up_at_exit = sum(1 for d in directions[i + hold_bars] if d > 0)
                down_at_exit = sum(1 for d in directions[i + hold_bars] if d < 0)

                if up_at_exit == down_at_exit:
                    continue

                actual_up = up_at_exit > down_at_exit

                trades += 1
                if expected_up == actual_up:
                    wins += 1

        if trades > 0:
            wr = wins / trades * 100
            print(f"  Hold={hold_bars:2d} bars | Trades={trades:4d} | WR={wr:5.1f}%")
        else:
            print(f"  Hold={hold_bars:2d} bars | No trades")

    return regimes, confidences, run_lengths

if __name__ == '__main__':
    run()
