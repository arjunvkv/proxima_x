"""Fast V2+z on Dukascopy cross pair M1 bid — vectorized scan, no per-bar sim overhead."""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

DUKA_DIR = Path("research/dark_research/dukascopy_data")
PAIRS = ["EURAUD", "AUDNZD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
CONTRACT = 100000
LOT = 0.75
COMM = 3.0

def load_dukascopy(pair):
    files = sorted(DUKA_DIR.glob(f"{pair.lower()}-m1-bid-*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=["timestamp","open","high","low","close"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fast_backtest(df, z_thresh, sprd, start_hour=0, end_hour=7, stop_a=3.0, trig_a=1.0, gap_a=0.05, max_hold=54):
    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    dt = df['datetime'].values
    n = len(df)

    # precompute z-score and ATR vectorized
    ret = np.diff(c, prepend=c[0])
    ret[:51] = np.nan
    z = np.full(n, np.nan)
    for i in range(51, n):
        seg = ret[i-50:i+1]
        r = seg[-1]
        m = np.mean(seg[:-1])
        v = np.var(seg[:-1], ddof=1)
        if v < 1e-14:
            z[i] = 0.0
        else:
            z[i] = (r - m) / np.sqrt(v)

    # ATR
    hl = h - l
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(hl[i-19:i+1])

    # hour check
    hours = np.array([pd.Timestamp(t).hour for t in dt])
    in_hours = (hours >= start_hour) & (hours < end_hour) if end_hour > start_hour else (hours >= start_hour) | (hours < end_hour)

    trades = []
    i = 1  # start from bar 1, check z of previous completed bar (i-1)
    while i < n:
        if not in_hours[i] or np.isnan(z[i-1]) or np.isnan(atr[i-1]) or atr[i-1] <= 0:
            i += 1
            continue
        if abs(z[i-1]) >= z_thresh:
            direction = 1 if z[i-1] < 0 else -1
            entry = o[i]
            atr_v = atr[i-1]
            sl = entry - stop_a * atr_v if direction > 0 else entry + stop_a * atr_v
            best = entry
            exited = False

            max_j = min(max_hold + 1, n - i)
            for j in range(1, max_j):
                idx = i + j
                if direction > 0:
                    if h[idx] > best:
                        best = h[idx]
                    if best - entry > trig_a * atr_v:
                        ns = best - gap_a * atr_v
                        if ns > sl:
                            sl = ns
                    if l[idx] <= sl:
                        exit_px = sl
                        pnl = (exit_px - entry) * LOT * CONTRACT
                        exit_reason = "stop"
                        exited = True
                        break
                else:
                    if l[idx] < best:
                        best = l[idx]
                    if entry - best > trig_a * atr_v:
                        ns = best + gap_a * atr_v
                        if ns < sl:
                            sl = ns
                    if h[idx] >= sl:
                        exit_px = sl
                        pnl = (entry - exit_px) * LOT * CONTRACT
                        exit_reason = "stop"
                        exited = True
                        break

            if not exited:
                exit_px = c[min(i + max_hold, n - 1)]
                if direction > 0:
                    pnl = (exit_px - entry) * LOT * CONTRACT
                else:
                    pnl = (entry - exit_px) * LOT * CONTRACT
                exit_reason = "expiry"

            pnl -= LOT * COMM
            exit_idx = i + j if exited else min(i + max_hold, n - 1)
            trades.append({
                'entry_idx': i, 'exit_idx': exit_idx,
                'direction': direction, 'entry': entry, 'exit': exit_px,
                'pnl': pnl, 'z': z[i-1], 'atr': atr_v, 'exit_reason': exit_reason,
                'bars_held': (exit_idx - i), 'dt': dt[i]
            })
            i = exit_idx + 1
        else:
            i += 1

    return trades

def print_result(pair, trades, z_thresh, sprd, label=""):
    n = len(trades)
    if n == 0:
        print(f"{pair:<8} z>={z_thresh:.1f}  sprd={sprd}  {'0-24h' if '24' in label else '0-7h'}: NO TRADES")
        return
    gross = sum(t['pnl'] for t in trades) + n * LOT * COMM
    net = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    neutral = [t for t in trades if abs(t['pnl']) < 0.01]
    denom = n - len(neutral)
    wr = len(wins) / denom * 100 if denom else 0
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    label_txt = label if label else f"z>={z_thresh:.1f}"
    survive = "SURVIVES" if net > 0 else "DIES"
    print(f"{pair:<8} {label_txt:<12} sprd={sprd}  {n:>4d} trades  {wr:>5.1f}%  "
          f"gross ${gross:>+7.2f}  net ${net:>+7.2f}  "
          f"avgW ${avg_win:>+6.2f}  avgL ${avg_loss:>+6.2f}  ← {survive}")

print("=" * 120)
print("V2+z ON DUKASCOPY M1 BID — 6 CROSS PAIRS (fast vectorized)")
print(f"Fixed spread=2 pips , $3/round-turn , 0.75 lot , Asian session 0-7 UTC")
print("=" * 120)

for pair in PAIRS:
    t0 = time.time()
    df = load_dukascopy(pair)
    if len(df) == 0:
        print(f"\n{pair}: NO DATA"); continue
    print(f"\n--- {pair}: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')} ({len(df)} bars) ---")

    for z in [2.0, 2.5, 3.0, 3.5, 4.0]:
        trades = fast_backtest(df, z_thresh=z, sprd=2)
        print_result(pair, trades, z, 2)

    trades_24h = fast_backtest(df, z_thresh=3.5, sprd=2, start_hour=0, end_hour=24)
    print_result(pair, trades_24h, 3.5, 2, label="0-24h z>=3.5")

    print(f"  Runtime: {time.time()-t0:.1f}s")
