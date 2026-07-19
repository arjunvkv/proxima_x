"""Lead-Lag Directional Propagation Backtest.

Tests: When leading pair has a strong directional M1 bar (close at extreme),
does the lagging pair move in same direction in the following N bars?

Pairs: EURUSD→USDCHF, EURUSD→USDCAD (not in data), AUDUSD→NZDUSD,
       EURJPY→GBPJPY, EURJPY→CHFJPY (not in data), AUDJPY→NZDJPY (not in data)
"""
import numpy as np
from numba import jit
from data import TempCache, PAIRS

PAIR_MAP = {p: i for i, p in enumerate(PAIRS)}

@jit(nopython=True)
def close_pct_ohlc(open_p, high, low, close):
    rng = high - low
    if rng == 0:
        return 0.5
    return (close - low) / rng

def test_lead_lag(data, times, pi_lead, pi_lag, name_lead, name_lag,
                   close_thresh=0.25, fwd_bars=(1, 2, 3, 5)):
    """Test: lead pair directional bar → lag pair follows."""
    n = data.shape[0]
    max_fwd = max(fwd_bars)

    results = []
    for fb in fwd_bars:
        trades = 0
        wins = 0
        avg_move_lag = 0.0
        avg_move_lead = 0.0

        for i in range(n - fb - 1):
            lead_bar = data[i, pi_lead]
            lead_range = lead_bar[1] - lead_bar[2]
            if lead_range == 0:
                continue

            cpc = close_pct_ohlc(lead_bar[0], lead_bar[1], lead_bar[2], lead_bar[3])

            # Strong directional close at extreme
            if cpc < close_thresh:
                lead_dir = -1  # bearish
            elif cpc > (1 - close_thresh):
                lead_dir = 1  # bullish
            else:
                continue  # not a strong directional bar

            # Check lag pair forward movement
            lag_entry = data[i, pi_lag, 3]
            lag_exit = data[i + fb, pi_lag, 3]
            lag_move = lag_exit - lag_entry

            lead_entry = data[i, pi_lead, 3]
            lead_exit = data[i + fb, pi_lead, 3]
            lead_move = lead_exit - lead_entry

            # Win: lag moved in same direction as lead's signal
            same_dir = (lag_move > 0) == (lead_dir > 0)

            trades += 1
            if same_dir:
                wins += 1
            avg_move_lag += lag_move
            avg_move_lead += lead_move

        if trades > 5:
            wr = wins / trades * 100
            avg_move_lag /= trades
            avg_move_lead /= trades
            print(f"  Fwd={fb:2d} bars | Trades={trades:4d} | WR={wr:5.1f}% | "
                  f"AvgLeadMove={avg_move_lead:+.6f} | AvgLagMove={avg_move_lag:+.6f} ")

            results.append({
                'lead': name_lead, 'lag': name_lag,
                'fwd_bars': fb, 'trades': trades, 'wr': wr
            })

    return results


def test_bidir_lead_lag(data, times, pi_a, pi_b, name_a, name_b,
                         close_thresh=0.25, fwd_bars=(1, 2, 3)):
    """Bidirectional test: each pair takes turns being leader.
    Determines which direction has higher WR → true leader."""
    results = []
    print(f"\n  Direction A→B ({name_a} as leader):")
    r1 = test_lead_lag(data, times, pi_a, pi_b, name_a, name_b, close_thresh, fwd_bars)
    results.extend(r1)

    print(f"  Direction B→A ({name_b} as leader):")
    r2 = test_lead_lag(data, times, pi_b, pi_a, name_b, name_a, close_thresh, fwd_bars)
    results.extend(r2)

    return results


def run():
    print("=" * 70)
    print("ENGINE 2 — Lead-Lag Directional Propagation Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()

    all_results = []
    all_tested = []

    # 1. AUDUSD → NZDUSD (commodity block, primary)
    print("\n--- Pair: AUDUSD vs NZDUSD ---")
    r = test_bidir_lead_lag(data, times, 3, 4, 'AUDUSD', 'NZDUSD', close_thresh=0.25, fwd_bars=(1, 2, 3, 5))
    all_results.extend(r)
    all_tested.append(('AUDUSD', 'NZDUSD'))

    # 2. EURUSD → GBPUSD (same USD block)
    print("\n--- Pair: EURUSD vs GBPUSD ---")
    r = test_bidir_lead_lag(data, times, 0, 2, 'EURUSD', 'GBPUSD', close_thresh=0.25, fwd_bars=(1, 2, 3, 5))
    all_results.extend(r)
    all_tested.append(('EURUSD', 'GBPUSD'))

    # 3. EURJPY → GBPJPY (JPY cross block)
    print("\n--- Pair: EURJPY vs GBPJPY ---")
    r = test_bidir_lead_lag(data, times, 5, 6, 'EURJPY', 'GBPJPY', close_thresh=0.25, fwd_bars=(1, 2, 3, 5))
    all_results.extend(r)
    all_tested.append(('EURJPY', 'GBPJPY'))

    # 4. EURUSD → USDJPY (triangle base pairs)
    print("\n--- Pair: EURUSD vs USDJPY ---")
    r = test_bidir_lead_lag(data, times, 0, 1, 'EURUSD', 'USDJPY', close_thresh=0.25, fwd_bars=(1, 2, 3, 5))
    all_results.extend(r)
    all_tested.append(('EURUSD', 'USDJPY'))

    # Summary
    print("\n" + "=" * 70)
    print("LEAD-LAG SUMMARY")
    print("=" * 70)

    print("\n--- Best configs ---")
    sorted_results = sorted(all_results, key=lambda x: x['wr'], reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r['lead']}→{r['lag']} fwd={r['fwd_bars']}: {r['wr']:.1f}% ({r['trades']} trades)")

    print("\n--- Worst configs ---")
    for r in sorted_results[-5:]:
        print(f"  {r['lead']}→{r['lag']} fwd={r['fwd_bars']}: {r['wr']:.1f}% ({r['trades']} trades)")

    # Analyze: which direction has higher WR for each pair?
    print("\n--- Dominant Lead Direction ---")
    for a, b in all_tested:
        a_to_b = [r for r in all_results if r['lead'] == a and r['lag'] == b]
        b_to_a = [r for r in all_results if r['lead'] == b and r['lag'] == a]

        wr_a = np.mean([r['wr'] for r in a_to_b]) if a_to_b else 0
        wr_b = np.mean([r['wr'] for r in b_to_a]) if b_to_a else 0
        trades_a = sum(r['trades'] for r in a_to_b) if a_to_b else 0
        trades_b = sum(r['trades'] for r in b_to_a) if b_to_a else 0

        print(f"  {a}→{b}: avg WR={wr_a:.1f}% ({trades_a} trades)")
        print(f"  {b}→{a}: avg WR={wr_b:.1f}% ({trades_b} trades)")
        if wr_a > wr_b + 2:
            print(f"  >>> {a} appears to lead {b}")
        elif wr_b > wr_a + 2:
            print(f"  >>> {b} appears to lead {a}")
        else:
            print(f"  >>> No clear leader between {a} and {b}")

    avg_wr = np.mean([r['wr'] for r in all_results if r['trades'] > 10])
    print(f"\nAverage WR across all configs: {avg_wr:.1f}%")

    # Reality check: is this above random (50%)?
    above = sum(1 for r in all_results if r['wr'] > 55 and r['trades'] > 10)
    total = sum(1 for r in all_results if r['trades'] > 10)
    print(f"Configs above 55% WR: {above}/{total}")

    return all_results

if __name__ == '__main__':
    run()
