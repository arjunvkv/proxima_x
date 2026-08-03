"""Fast debug: z-score mean reversion analysis on FundedNext data."""
import numpy as np
import pandas as pd
import time

z_window = 50
pairs = ['audusd', 'euraud', 'gbpaud']

for pair in pairs:
    t0 = time.time()
    data = np.load(f'paper_trade/mt5_backtest/fundednext_{pair}_m1.npy', allow_pickle=True)
    df = pd.DataFrame(data)
    closes = df['close'].values
    opens = df['open'].values
    spreads = df['spread'].values

    n = len(closes)

    # Compute z-scores in one pass
    z_scores = np.zeros(n)
    close_buf = np.array([])
    for i in range(n):
        close_buf = np.append(close_buf, closes[i])
        if len(close_buf) < z_window + 2:
            continue
        rets = np.diff(close_buf[-(z_window+2):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var >= 1e-14:
            z_scores[i] = (cur_ret - mean) / np.sqrt(var)

    timestamps = df['time'].values
    days = (timestamps[-1] - timestamps[0]) / 86400.0

    print(f"\n{'='*60}")
    print(f"{pair.upper()} ({n} bars)")

    # Trigger counts
    for thresh in [2.0, 2.5, 3.0, 3.5, 4.0]:
        triggered = np.where(np.abs(z_scores) >= thresh)[0]
        print(f"  |z|>={thresh:.1f}: {len(triggered):>4d} triggers ({len(triggered)/days:.1f}/day)")

    # Test strategy: enter at open[i], exit at close[i+hold]
    # Also test with trailing stop
    for hold in [5, 10, 15]:
        triggered = np.where(np.abs(z_scores) >= 3.5)[0]
        total_pnl_fixed = 0.0
        long_signal_pnl = 0.0
        short_signal_pnl = 0.0
        for idx in triggered:
            exit_idx = min(idx + hold, n - 1)
            entry = opens[idx]
            exit_p = closes[exit_idx]
            if z_scores[idx] < 0:  # LONG
                pnl = (exit_p - entry) * 100000 * 0.5
                long_signal_pnl += pnl
            else:  # SHORT
                pnl = (entry - exit_p) * 100000 * 0.5
                short_signal_pnl += pnl
            total_pnl_fixed += pnl
        lt = len(np.where(z_scores[triggered] < 0)[0])
        st = len(np.where(z_scores[triggered] > 0)[0])
        print(f"  Fixed hold {hold:>2d}: LONG=${long_signal_pnl:>+8.2f} ({lt}t)  "
              f"SHORT=${short_signal_pnl:>+8.2f} ({st}t)  "
              f"TOTAL=${total_pnl_fixed:>+8.2f}")

    # Test with TRAILING STOP (like original sim_recon):
    # stop_a=3.0, trig_a=1.0, gap_a=0.05
    print(f"  TRAILING STOP (3.0/1.0/0.05, lot=0.5):")
    triggered = np.where(np.abs(z_scores) >= 3.5)[0]
    total_pnl = 0.0
    wins = 0
    losses = 0
    for idx in triggered:
        direction = -1 if z_scores[idx] > 0 else 1  # z>0→SHORT, z<0→LONG
        entry = opens[idx]
        atr_val = np.mean(np.abs(np.diff(closes[max(0, idx-20):idx+1]))) if idx >= 20 else 0.0
        if atr_val <= 0:
            continue
        sl_dist = 3.0 * atr_val
        trig_dist = 1.0 * atr_val
        gap_dist = 0.05 * atr_val
        if direction > 0:
            sl = entry - sl_dist
            best = entry
        else:
            sl = entry + sl_dist
            best = entry

        pnl = None
        for j in range(idx, min(idx + hold + 5, n)):
            o, h, l, c = opens[j], df['high'].iloc[j], df['low'].iloc[j], closes[j]
            if direction > 0:
                if h > best:
                    best = h
                    if best - entry > trig_dist:
                        ns = best - gap_dist
                        if ns > sl:
                            sl = ns
                if l <= sl:
                    pnl = (sl - entry) * 100000 * 0.5
                    break
            else:
                if l < best:
                    best = l
                    if entry - best > trig_dist:
                        ns = best + gap_dist
                        if ns < sl:
                            sl = ns
                if h >= sl:
                    pnl = (entry - sl) * 100000 * 0.5
                    break
        if pnl is None:
            last_c = closes[min(idx + hold + 5, n - 1)]
            if direction > 0:
                pnl = (last_c - entry) * 100000 * 0.5
            else:
                pnl = (entry - last_c) * 100000 * 0.5

        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"    Total=${total_pnl:>+8.2f}  W/L={wins}/{losses}  WR={wr:.1f}%")

    # Test with spread filter (only enter when spread <= 10)
    print(f"  + SPREAD <= 10 filter:")
    triggered2 = [idx for idx in triggered if spreads[idx] <= 10]
    total_pnl2 = 0.0
    wins2 = 0
    losses2 = 0
    for idx in triggered2:
        direction = -1 if z_scores[idx] > 0 else 1
        entry = opens[idx]
        atr_val = np.mean(np.abs(np.diff(closes[max(0, idx-20):idx+1]))) if idx >= 20 else 0.0
        if atr_val <= 0:
            continue
        sl_dist = 3.0 * atr_val
        trig_dist = 1.0 * atr_val
        gap_dist = 0.05 * atr_val
        if direction > 0:
            sl = entry - sl_dist
            best = entry
        else:
            sl = entry + sl_dist
            best = entry

        pnl = None
        for j in range(idx, min(idx + 15, n)):
            o, h, l, c = opens[j], df['high'].iloc[j], df['low'].iloc[j], closes[j]
            if direction > 0:
                if h > best:
                    best = h
                    if best - entry > trig_dist:
                        ns = best - gap_dist
                        if ns > sl:
                            sl = ns
                if l <= sl:
                    pnl = (sl - entry) * 100000 * 0.5
                    break
            else:
                if l < best:
                    best = l
                    if entry - best > trig_dist:
                        ns = best + gap_dist
                        if ns < sl:
                            sl = ns
                if h >= sl:
                    pnl = (entry - sl) * 100000 * 0.5
                    break
        if pnl is None:
            last_c = closes[min(idx + 15, n - 1)]
            if direction > 0:
                pnl = (last_c - entry) * 100000 * 0.5
            else:
                pnl = (entry - last_c) * 100000 * 0.5
        total_pnl2 += pnl
        if pnl > 0:
            wins2 += 1
        else:
            losses2 += 1
    wr2 = wins2 / (wins2 + losses2) * 100 if (wins2 + losses2) > 0 else 0
    print(f"    Total=${total_pnl2:>+8.2f}  W/L={wins2}/{losses2}  WR={wr2:.1f}%")

    t1 = time.time()
    print(f"  [took {t1-t0:.1f}s]")
