"""
Live tick collector for Engine 2 validation.

Poll symbol_info_tick across all 7 pairs at ~1ms intervals.
Write ticks to per-pair memory-mapped files for later backtesting.

Usage:
  python tick_collector.py              # Collect until Ctrl+C
  python tick_collector.py --duration 3600  # Collect for 1 hour
  python tick_collector.py --analyze        # Analyze collected files

File format:
  Each pair gets a .npy file with structured array:
    time_ms, bid, ask, spread, volume
"""
import argparse
import time
import os
import signal
import sys
from datetime import datetime
import numpy as np

import MetaTrader5 as mt5

PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY']
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tick_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICK_DTYPE = np.dtype([
    ('time_ms', 'u8'),
    ('bid', 'f8'),
    ('ask', 'f8'),
    ('spread', 'f4'),
    ('volume', 'u4'),
])

running = True

def signal_handler(sig, frame):
    global running
    print(f"\nCaught signal {sig}. Flushing and exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def collect(duration=None):
    """Collect ticks from all pairs into per-pair arrays."""
    for attempt in range(3):
        init = mt5.initialize()
        if init:
            break
        print(f"MT5 init attempt {attempt+1}/3: {mt5.last_error()}")
        time.sleep(1)
    else:
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    print(f"Connected to MT5. Collecting ticks for {len(PAIRS)} pairs...")
    if duration:
        print(f"Duration: {duration}s")
    print("Press Ctrl+C to stop.")
    print()

    # Pre-allocate large arrays (expand as needed)
    cap = 10_000_000  # 10M ticks per pair (should be plenty)
    buffers = {}
    counts = {}
    warns = {p: 0 for p in PAIRS}

    for p in PAIRS:
        buffers[p] = np.zeros(cap, dtype=TICK_DTYPE)
        counts[p] = 0

    last_ts = {p: None for p in PAIRS}
    last_bid = {p: None for p in PAIRS}
    last_ask = {p: None for p in PAIRS}
    start_time = time.time()
    report_interval = 30  # seconds
    last_report = start_time

    global running
    total_polls = 0

    while running:
        now = time.time()

        if duration and (now - start_time) > duration:
            print(f"\nDuration {duration}s reached.")
            break

        for p in PAIRS:
            tick = mt5.symbol_info_tick(p)
            if tick is None:
                continue

            total_polls += 1
            c = counts[p]

            if c >= cap:
                warns[p] += 1
                if warns[p] == 1:
                    print(f"WARN: {p} buffer full at {cap}")
                continue

            ts_ms = int(tick.time * 1000 + (tick.time_msc % 1000))

            # Only store if bid or ask changed (skip redundant polls)
            if last_bid[p] == tick.bid and last_ask[p] == tick.ask:
                continue

            buffers[p][c] = (ts_ms, tick.bid, tick.ask,
                             tick.ask - tick.bid, tick.volume or 0)
            counts[p] += 1
            last_bid[p] = tick.bid
            last_ask[p] = tick.ask

        # Report progress
        if now - last_report >= report_interval:
            elapsed = now - start_time
            print(f"[{elapsed:6.0f}s]", end="")
            for p in PAIRS:
                if counts[p] > 0:
                    rate = counts[p] / elapsed
                    print(f" {p}={counts[p]}({rate:.0f}/s)", end="")
            print()
            last_report = now

    # Flush to disk
    elapsed = time.time() - start_time
    print(f"\nFlushing {elapsed:.0f}s of tick data...")

    total = 0
    for p in PAIRS:
        c = counts[p]
        if c == 0:
            print(f"  {p}: 0 ticks — SKIPPING")
            continue
        fname = os.path.join(OUTPUT_DIR, f"{p}_ticks.npy")
        np.save(fname, buffers[p][:c])
        total += c
        rate = c / elapsed if elapsed > 0 else 0
        print(f"  {p}: {c} ticks saved ({rate:.0f}/s avg)")

    print(f"\nTotal: {total} ticks across {len(PAIRS)} pairs")
    print(f"Output: {OUTPUT_DIR}")
    mt5.shutdown()


def analyze():
    """Analyze collected tick files."""
    total = 0
    print(f"\nAnalyzing tick data in {OUTPUT_DIR}:")
    print()

    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith("_ticks.npy"):
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        data = np.load(fpath)
        n = len(data)
        pair = fname.replace("_ticks.npy", "")
        total += n

        if n == 0:
            print(f"  {pair}: 0 ticks")
            continue

        duration_s = (data[-1]['time_ms'] - data[0]['time_ms']) / 1000
        rate = n / duration_s if duration_s > 0 else 0

        bid_changes = np.sum(np.abs(np.diff(data['bid'])) > 0)
        spread = data['spread']
        avg_spread = np.mean(spread[spread > 0]) if np.any(spread > 0) else 0

        print(f"  {pair}: {n} ticks over {duration_s:.0f}s ({rate:.0f}/s)")
        print(f"         bid changes: {bid_changes}, avg spread: {avg_spread:.1f}")

    print(f"\n  TOTAL: {total} ticks")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live tick collector for Engine 2")
    parser.add_argument('--duration', type=int, default=None,
                       help="Collection duration in seconds (default: continuous)")
    parser.add_argument('--analyze', action='store_true',
                       help="Analyze existing collected files instead of collecting")

    args = parser.parse_args()

    if args.analyze:
        analyze()
    else:
        collect(args.duration)
