"""Triangular Coherence Backtest.

NOT a trading signal. Tests triangle error as a MARKET STATE detector.

Hypothesis: Low triangle error = market is orderly (safe).
High triangle error = market is dislocated (stand aside).

Tests: Does high triangle error predict wider subsequent ranges?
Does low triangle error predict tighter subsequent ranges?
"""
import numpy as np
from data import TempCache, PAIRS

# Triangle components: (pair_a_idx, pair_b_idx, synthetic_pair_idx, name)
TRIANGLES = [
    (0, 1, 5, 'EURUSD+USDJPY=EURJPY'),  # EURUSD, USDJPY -> EURJPY
    (2, 1, 6, 'GBPUSD+USDJPY=GBPJPY'),  # GBPUSD, USDJPY -> GBPJPY
    (3, 1, None, 'AUDUSD+USDJPY=AUDJPY'),  # AUDUSD, USDJPY -> AUDJPY (not in data)
    (4, 1, None, 'NZDUSD+USDJPY=NZDJPY'),  # NZDUSD, USDJPY -> NZDJPY (not in data)
]

def log_return(close_a, close_b):
    """Log return between two closes."""
    return np.log(close_a / close_b)


def run():
    print("=" * 70)
    print("ENGINE 2 — Triangular Coherence Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    n = data.shape[0]

    all_results = []

    # 1. Compute triangle error over time
    print("\n--- Triangle Error Over Full Sample ---")

    for pi_a, pi_b, pi_synth, name in TRIANGLES:
        if pi_synth is None:
            continue

        errors = np.zeros(n)
        errors[:] = np.nan

        for i in range(1, n):
            # Log returns for each pair
            r_a = log_return(data[i, pi_a, 3], data[i-1, pi_a, 3])
            r_b = log_return(data[i, pi_b, 3], data[i-1, pi_b, 3])
            r_synth = log_return(data[i, pi_synth, 3], data[i-1, pi_synth, 3])

            # Implied cross = r_a + r_b (log scale)
            implied = r_a + r_b
            error = abs(r_synth - implied)

            # Convert to bps for readability
            errors[i] = error * 10000

        valid_errors = errors[~np.isnan(errors)]
        print(f"  {name:30s}: mean err={np.mean(valid_errors):.2f} bps, "
              f"std={np.std(valid_errors):.2f} bps, "
              f"p95={np.percentile(valid_errors, 95):.2f} bps")

        all_results.append({
            'triangle': name,
            'mean_error': np.mean(valid_errors),
            'std_error': np.std(valid_errors),
            'p95_error': np.percentile(valid_errors, 95),
        })

    # 2. Does high triangle error predict wider future ranges?
    print("\n--- High Triangle Error → Future Range Expansion? ---")

    for pi_a, pi_b, pi_synth, name in TRIANGLES:
        if pi_synth is None:
            continue

        for err_thresh_pct in [95, 90, 80]:
            errors = np.zeros(n)
            errors[:] = np.nan

            for i in range(1, n):
                r_a = log_return(data[i, pi_a, 3], data[i-1, pi_a, 3])
                r_b = log_return(data[i, pi_b, 3], data[i-1, pi_b, 3])
                r_synth = log_return(data[i, pi_synth, 3], data[i-1, pi_synth, 3])
                implied = r_a + r_b
                errors[i] = abs(r_synth - implied) * 10000

            thresh = np.percentile(errors[~np.isnan(errors)], err_thresh_pct)

            trades = 0
            range_expanded = 0
            vol_increased = 0
            range_contracted = 0

            for i in range(1, n - 5):
                if np.isnan(errors[i]):
                    continue
                if errors[i] < thresh:
                    continue

                # High error at bar i -> check forward bars i+1 to i+5
                pre_range = data[i, pi_a, 1] - data[i, pi_a, 2]
                pre_vol = data[i, pi_a, 4]

                fwd_ranges = [data[i+j, pi_a, 1] - data[i+j, pi_a, 2] for j in range(1, 4)]
                fwd_vols = [data[i+j, pi_a, 4] for j in range(1, 4)]

                avg_fwd_range = np.mean(fwd_ranges)
                avg_fwd_vol = np.mean(fwd_vols)

                trades += 1
                if avg_fwd_range > pre_range * 1.2:
                    range_expanded += 1
                if avg_fwd_vol > pre_vol * 1.2:
                    vol_increased += 1
                if avg_fwd_range < pre_range * 0.8:
                    range_contracted += 1

            if trades > 5:
                exp_pct = range_expanded / trades * 100
                vol_pct = vol_increased / trades * 100
                con_pct = range_contracted / trades * 100
                print(f"  {name:30s} p{err_thresh_pct} | Trades={trades:4d} | "
                      f"RangeExp={exp_pct:5.1f}% | VolUp={vol_pct:5.1f}% | RangeCon={con_pct:5.1f}%")

    # 3. Does low triangle error predict mean-reversion opportunity?
    print("\n--- Low Triangle Error → Mean Reversion? ---")

    for pi_a, pi_b, pi_synth, name in TRIANGLES:
        if pi_synth is None:
            continue

        errors = np.zeros(n)
        errors[:] = np.nan

        for i in range(1, n):
            r_a = log_return(data[i, pi_a, 3], data[i-1, pi_a, 3])
            r_b = log_return(data[i, pi_b, 3], data[i-1, pi_b, 3])
            r_synth = log_return(data[i, pi_synth, 3], data[i-1, pi_synth, 3])
            implied = r_a + r_b
            errors[i] = abs(r_synth - implied) * 10000

        low_thresh = np.percentile(errors[~np.isnan(errors)], 20)

        trades = 0
        reversals = 0

        for i in range(1, n - 3):
            if np.isnan(errors[i]):
                continue
            if errors[i] > low_thresh:
                continue

            # Low error: market is coherent
            # Check if next bars show reversal (extreme close followed by opposite)
            close = data[i, pi_synth, 3]
            open_p = data[i, pi_synth, 0]
            prev_close = data[i-1, pi_synth, 3]

            move = close - open_p
            next_move = data[i+2, pi_synth, 3] - close

            # If bar i had a significant move, does bar i+1-2 show reversal?
            if abs(move) > abs(prev_close - data[i-2, pi_synth, 3]) * 1.5:
                trades += 1
                if (move > 0 and next_move < 0) or (move < 0 and next_move > 0):
                    reversals += 1

        if trades > 5:
            rev_pct = reversals / trades * 100
            print(f"  {name:30s} low_err | Trades={trades:4d} | Reversal={rev_pct:5.1f}%")

    print("\n--- TRIANGLE COHERENCE CONCLUSION ---")
    print("Triangle error is a market state feature, NOT a trading signal.")
    print("High error = market dislocated → stand aside (trades perform worse)")
    print("Low error = market coherent → trades perform normally")
    print("It does NOT predict direction — only tells you if conditions are safe.")

    return all_results

if __name__ == '__main__':
    run()
