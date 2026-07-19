"""Combined 3-Condition Entry Gate Backtest.

Tests the joint condition:
  1) Direction confirmed (lead-lag or tick_volume divergence)
  2) Spread acceptable (within session baseline)
  3) Range ratio active (imbalance present)

A 3-condition gate should have HIGHER WR than any single condition.
"""
import numpy as np
from data import TempCache, PAIRS

def close_pct_ohlc(bar):
    rng = bar[1] - bar[2]
    if rng == 0:
        return 0.5
    return (bar[3] - bar[2]) / rng

def run():
    print("=" * 70)
    print("ENGINE 2 — 3-Condition Entry Gate Backtest")
    print("=" * 70)
    print()

    cache = TempCache(7)
    data, times, pairs = cache.get()
    n = data.shape[0]

    all_results = []

    for pi_a, pi_b, name_a, name_b in [
        (3, 4, 'AUDUSD', 'NZDUSD'),
        (0, 2, 'EURUSD', 'GBPUSD'),
        (5, 6, 'EURJPY', 'GBPJPY'),
    ]:
        print(f"\n--- {name_a} → {name_b} ---")

        for lb in [10, 20]:
            trades_single = 0
            wins_single = 0
            trades_gate = 0
            wins_gate = 0

            for i in range(lb, n - 5):
                # ===== CONDITION 1: Direction from lead-lag =====
                bar_a = data[i, pi_a]
                cpc_a = close_pct_ohlc(bar_a)

                if cpc_a < 0.25:
                    direction = -1
                elif cpc_a > 0.75:
                    direction = 1
                else:
                    continue  # no clear direction

                # ===== CONDITION 2: Range ratio =====
                ranges_a = np.array([data[i-k, pi_a, 1] - data[i-k, pi_a, 2] for k in range(lb)])
                ranges_b = np.array([data[i-k, pi_b, 1] - data[i-k, pi_b, 2] for k in range(lb)])

                valid = ranges_b > 0
                if valid.sum() < lb // 2:
                    continue

                ratios = ranges_a / (ranges_b + 1e-10)
                mean_r = np.mean(ratios[valid])
                std_r = np.std(ratios[valid])
                if std_r == 0:
                    continue

                cur_ratio = ratios[-1]
                z = (cur_ratio - mean_r) / std_r
                ratio_active = z > 1.5 or z < -1.5

                # ===== CONDITION 3: Spread acceptable =====
                spread = data[i, pi_b, 5]
                recent_spreads = data[max(0, i-20):i, pi_b, 5]
                recent_spreads = recent_spreads[recent_spreads > 0]
                if len(recent_spreads) == 0:
                    spread_ok = True
                else:
                    baseline = np.median(recent_spreads)
                    spread_ok = spread <= baseline * 1.5 if baseline > 0 else True

                # ===== CHECK FORWARD BARS =====
                move_b = data[i+3, pi_b, 3] - data[i, pi_b, 3]
                correct = (move_b > 0) == (direction > 0)

                # SINGLE CONDITION (direction only)
                trades_single += 1
                if correct:
                    wins_single += 1

                # 3-CONDITION GATE
                if ratio_active and spread_ok:
                    trades_gate += 1
                    if correct:
                        wins_gate += 1

            if trades_single > 0:
                wr_single = wins_single / trades_single * 100
                wr_gate = wins_gate / trades_gate * 100 if trades_gate > 0 else 0
                print(f"  lb={lb:2d} | Single: {wins_single:4d}/{trades_single:4d} = {wr_single:5.1f}% | "
                      f"Gate: {wins_gate:4d}/{trades_gate:4d} = {wr_gate:5.1f}% | "
                      f"Diff: {wr_gate - wr_single:+5.1f}pp")

                all_results.append({
                    'lead': name_a, 'lag': name_b, 'lookback': lb,
                    'wr_single': wr_single, 'trades_single': trades_single,
                    'wr_gate': wr_gate, 'trades_gate': trades_gate,
                })

    # Summary
    print("\n" + "=" * 70)
    print("ENTRY GATE SUMMARY")
    print("=" * 70)

    for r in all_results:
        if r['trades_gate'] > 0:
            delta = r['wr_gate'] - r['wr_single']
            print(f"  {r['lead']}→{r['lag']} lb={r['lookback']}: "
                  f"Single={r['wr_single']:.1f}%({r['trades_single']}) → "
                  f"Gate={r['wr_gate']:.1f}%({r['trades_gate']}) "
                  f"Δ={delta:+.1f}pp")

    # Cross-pair regime + direction (even higher-level gate)
    print("\n--- Layer 2 Filter: Regime Stability ---")
    # Test: trade only when regime is stable (not UNKNOWN/MIXED)

    # Build regimes
    directions = np.zeros((n, len(PAIRS)), dtype='i1')
    for pi in range(len(PAIRS)):
        for i in range(1, n):
            if data[i, pi, 3] > data[i-1, pi, 3]:
                directions[i, pi] = 1
            elif data[i, pi, 3] < data[i-1, pi, 3]:
                directions[i, pi] = -1

    def has_stable_regime(idx):
        """Check if market has >60% directional agreement."""
        if idx < 5:
            return False
        agreements = []
        for j in range(5):
            d = directions[idx - j]
            up = sum(1 for x in d if x > 0)
            down = sum(1 for x in d if x < 0)
            total = up + down
            if total == 0:
                agreements.append(0.5)
            else:
                agreements.append(max(up, down) / total)
        return np.mean(agreements) >= 0.6

    for pi_a, pi_b, name_a, name_b in [
        (3, 4, 'AUDUSD', 'NZDUSD'),
        (0, 2, 'EURUSD', 'GBPUSD'),
    ]:
        trades_raw = 0
        wins_raw = 0
        trades_filtered = 0
        wins_filtered = 0

        for i in range(10, n - 5):
            bar_a = data[i, pi_a]
            cpc_a = close_pct_ohlc(bar_a)
            if cpc_a < 0.25:
                direction = -1
            elif cpc_a > 0.75:
                direction = 1
            else:
                continue

            move_b = data[i+3, pi_b, 3] - data[i, pi_b, 3]
            correct = (move_b > 0) == (direction > 0)

            trades_raw += 1
            if correct:
                wins_raw += 1

            if has_stable_regime(i):
                trades_filtered += 1
                if correct:
                    wins_filtered += 1

        if trades_raw > 0:
            wr_raw = wins_raw / trades_raw * 100
            wr_filtered = wins_filtered / trades_filtered * 100 if trades_filtered > 0 else 0
            print(f"  {name_a}→{name_b}: "
                  f"Raw={wr_raw:.1f}%({trades_raw}) → "
                  f"RegimeFiltered={wr_filtered:.1f}%({trades_filtered}) "
                  f"Δ={wr_filtered - wr_raw:+.1f}pp")

    # Final verdict
    print("\n--- ENTRY GATE VERDICT ---")
    print("The 3-condition gate should IMPROVE WR vs single signals.")
    print("If WR drops, the gate is overblocking (trading fewer but worse).")
    print("If WR improves, the gate filters noise effectively.")

    return all_results

if __name__ == '__main__':
    run()
