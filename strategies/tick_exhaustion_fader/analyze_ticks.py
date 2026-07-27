"""Analyze real GBPNZD tick data for exhaustion patterns.
Downloads ticks from MT5 broker feed and runs the EA detection logic."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def download_ticks(symbol="GBPNZD", days=7, max_ticks=2_000_000):
    import MetaTrader5 as mt5
    mt5.initialize()
    now = datetime.now()
    from_dt = now - timedelta(days=days)
    ticks = mt5.copy_ticks_range(symbol, from_dt, now, mt5.COPY_TICKS_ALL)
    mt5.shutdown()
    if ticks is None or len(ticks) == 0:
        print(f"No tick data for {symbol}")
        return None
    n = min(len(ticks), max_ticks)
    ticks = ticks[-n:]
    print(f"Downloaded {len(ticks)} ticks from {symbol}")
    print(f"  Range: {datetime.utcfromtimestamp(ticks[0][0])} to {datetime.utcfromtimestamp(ticks[-1][0])}")
    print(f"  Rate: {len(ticks)/(ticks[-1][0]-ticks[0][0]):.1f}/s")
    return ticks

def detect_signals(mid_prices, consecutive=4, min_move_pips=0.4, exhaust_ratio=0.7, pip_price=0.0001):
    n = len(mid_prices)
    if n < consecutive + 2:
        return np.zeros(n, dtype=int)

    diffs = np.diff(mid_prices)
    signs = np.sign(diffs)
    abs_diffs = np.abs(diffs)
    min_move_abs = min_move_pips * pip_price

    sigs = np.zeros(n, dtype=int)
    for i in range(consecutive + 1, n):
        segment = diffs[i-consecutive-1:i]
        seg_signs = signs[i-consecutive-1:i]
        seg_abs = abs_diffs[i-consecutive-1:i]

        up_count = int(np.sum(seg_signs > 0))
        dn_count = int(np.sum(seg_signs < 0))

        if up_count < consecutive and dn_count < consecutive:
            continue

        rising = up_count >= dn_count
        cons = up_count if rising else dn_count
        if cons < consecutive:
            continue

        total_move = abs(mid_prices[i] - mid_prices[i - cons])
        if total_move < min_move_abs:
            continue

        last_move = abs_diffs[i-1]
        prev_move = abs_diffs[i-2]
        if prev_move > 0 and last_move > prev_move * exhaust_ratio:
            continue

        p2_move = abs_diffs[i-3] if i >= 3 else 0
        if p2_move > 0 and prev_move > p2_move:
            continue

        sigs[i] = -1 if rising else 1

    return sigs

def analyze():
    ticks = download_ticks(symbol="GBPNZD", days=7, max_ticks=2_000_000)
    if ticks is None:
        return

    mid = (ticks["bid"] + ticks["ask"]) / 2.0
    spread = ticks["ask"] - ticks["bid"]
    time_ms = ticks["time_msc"] if "time_msc" in ticks.dtype.names else ticks["time"] * 1000

    ts = [datetime.utcfromtimestamp(t) for t in ticks["time"]]
    hours = np.array([t.hour + t.minute/60 for t in ts])

    pip_price = 0.0001  # GBPNZD is 5-digit

    configs = [
        ("default", 4, 0.4, 0.7),
        ("tight", 3, 0.3, 0.6),
        ("loose", 5, 0.5, 0.8),
        ("aggressive", 3, 0.2, 0.5),
    ]

    for label, cons, min_move, exhaust in configs:
        sigs = detect_signals(mid, consecutive=cons, min_move_pips=min_move,
                              exhaust_ratio=exhaust, pip_price=pip_price)
        n_long = int(np.sum(sigs == 1))
        n_short = int(np.sum(sigs == -1))
        total = n_long + n_short

        duration_h = (ticks[-1][0] - ticks[0][0]) / 3600
        per_day = total / max(duration_h / 24, 0.01)

        spread_at_sig = spread[sigs != 0]
        avg_spread = np.mean(spread_at_sig) if len(spread_at_sig) > 0 else 0

        print(f"\n{label}: cons={cons} min={min_move} exhaust={exhaust}")
        print(f"  Signals: {total} ({n_long}L/{n_short}S) in {duration_h:.1f}h = {per_day:.0f}/day")
        if total > 0:
            print(f"  Avg spread at signal: {avg_spread:.1f} pts")

    # Per-session breakdown
    sigs = detect_signals(mid, consecutive=4, min_move_pips=0.4,
                          exhaust_ratio=0.7, pip_price=pip_price)
    sig_mask = sigs != 0
    sig_hours = hours[sig_mask]
    if len(sig_hours) > 0:
        asia = int(np.sum((sig_hours >= 0) & (sig_hours < 8)))
        london = int(np.sum((sig_hours >= 8) & (sig_hours < 16)))
        ny = int(np.sum(sig_hours >= 16))
        print(f"\nSession breakdown (default params):")
        print(f"  Asia (0-8 UTC):  {asia}")
        print(f"  London (8-16):   {london}")
        print(f"  NY (16-24):      {ny}")

    # Tick-level profile around signals
    sig_indices = np.where(sig_mask)[0]
    if len(sig_indices) > 0:
        lookback = 20
        lookahead = 10
        profiles = []
        for idx in sig_indices:
            if idx >= lookback and idx + lookahead < len(mid):
                seg = mid[idx-lookback:idx+lookahead+1]
                seg = (seg - seg[lookback]) / pip_price  # normalize to entry
                profiles.append(seg)
        if profiles:
            arr = np.array(profiles)
            avg_profile = np.mean(arr, axis=0)
            print(f"\nAvg tick profile around signal ({len(profiles)} events):")
            print(f"  Pre-signal ({lookback} ticks): {avg_profile[0]:.2f} -> {avg_profile[lookback]:.2f} pips")
            print(f"  Post-signal ({lookahead} ticks): {avg_profile[lookback]:.2f} -> {avg_profile[-1]:.2f} pips")

if __name__ == "__main__":
    analyze()
