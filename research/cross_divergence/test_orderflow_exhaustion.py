"""Test order flow imbalance and bid-ask spread dynamics during tick exhaustion.
Hypothesis: during extreme order flow imbalance (mostly aggressive buys/sells),
the spread widens. When spread contracts back, exhaustion is confirmed."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

for pair_name, fname in [("EURUSD","EURUSD_mt5.csv"), ("GBPUSD","GBPUSD_mt5.csv"),
                          ("EURJPY","EURJPY_mt5_v2.csv"), ("USDJPY","USDJPY_mt5_v2.csv")]:
    path = DATA / fname
    if not path.exists(): continue
    df = pd.read_csv(path, parse_dates=['time'])
    pip_mul = 0.01 if 'JPY' in pair_name else 0.0001
    df['spread'] = (df.ask - df.bid) / pip_mul  # spread in pips
    df['mid'] = (df.bid + df.ask) / 2

    # Order flow: last == ask means aggressive buy, last == bid means aggressive sell
    # Using flags column if available (1=bid, 2=ask, 4=last)
    if 'flags' in df.columns:
        df['side'] = df['flags'].apply(lambda x: 'ask' if x & 4 else 'bid' if x & 2 else 'unknown')
    else:
        # Approximate: if last >= ask, aggressive buy; if last <= bid, aggressive sell
        df['side'] = np.where(df.last >= df.ask, 'ask',
                     np.where(df.last <= df.bid, 'bid', 'mid'))

    # 1. Spread dynamics during tick streaks
    print(f"\n=== {pair_name}: Spread During Tick Streaks ===")
    dirs = np.sign(df.mid.diff().fillna(0)).astype(np.int8)
    streaks = np.zeros(len(df), dtype=np.int32)
    cur = 0
    for i in range(1, len(df)):
        d = int(dirs[i])
        if d == 1: cur = cur + 1 if cur >= 0 else 1
        elif d == -1: cur = cur - 1 if cur <= 0 else -1
        streaks[i] = cur
    df['streak'] = streaks
    df['abs_streak'] = abs(streaks)

    # Spread at streak start vs streak end
    for min_strk in [3, 5, 8]:
        mask = (df.abs_streak >= min_strk) & (df.streak != 0)
        entry_idx = df[mask].index.values
        # Also get before-streak spread
        before_idx = np.maximum(0, entry_idx - 1)
        if len(entry_idx) < 10: continue

        entry_spread = df.spread.values[entry_idx]
        before_spread = df.spread.values[before_idx]
        # Shift-based: spread at entry vs spread at entry+5 ticks
        exit_n = min(60, len(df) - 1)
        after_idx = np.minimum(entry_idx + 5, len(df)-1)
        after_spread = df.spread.values[after_idx]
        after_10_idx = np.minimum(entry_idx + 10, len(df)-1)
        after_10_spread = df.spread.values[after_10_idx]

        print(f"  streak≥{min_strk}: before={before_spread.mean():.2f} entry={entry_spread.mean():.2f} "
              f"+5={after_spread.mean():.2f} +10={after_10_spread.mean():.2f} N={len(entry_idx):,}")

    # 2. Order flow imbalance during tick streaks
    print(f"\n=== {pair_name}: Order Flow During Tick Streaks ===")
    df['aggressive_buy'] = (df.side == 'ask').astype(int)
    df['aggressive_sell'] = (df.side == 'bid').astype(int)

    for window in [5, 10, 20]:
        df['ofi'] = (df.aggressive_buy.rolling(window).sum() -
                     df.aggressive_sell.rolling(window).sum()) / window
        df['ofi_abs'] = df.ofi.abs()

        for ofi_thresh in [0.6, 0.8, 0.9]:
            mask = (df.ofi_abs >= ofi_thresh) & (df.streak != 0)
            if mask.sum() < 10: continue
            # After extreme OFI, how does price move?
            entry_idx = df[mask].index.values
            exit_idx = np.minimum(entry_idx + 5, len(df)-1)
            entry_mid = df.mid.values[entry_idx]
            exit_mid = df.mid.values[exit_idx]

            # If extreme buying (ofi > 0), expect short-term mean reversion (sell)
            entry_dir = np.sign(df.ofi.values[entry_idx])
            gains = np.where(entry_dir > 0, (entry_mid - exit_mid) * (1/pip_mul),
                             (exit_mid - entry_mid) * (1/pip_mul))
            wr = (gains > 0).mean() * 100
            avg_pip = gains.mean()
            print(f"  OFI≥{ofi_thresh} w={window}: WR={wr:.1f}% avg={avg_pip:+.2f} N={mask.sum():,}")

    # 3. Combined: extreme OFI + streak exhaustion
    print(f"\n=== {pair_name}: OFI + Streak Combined ===")
    for min_strk in [3, 5]:
        for ofi_thresh in [0.6, 0.8]:
            mask = (df.abs_streak >= min_strk) & (df.ofi_abs >= ofi_thresh) & (df.streak != 0)
            if mask.sum() < 10: continue
            entry_idx = df[mask].index.values
            exit_idx = np.minimum(entry_idx + 5, len(df)-1)
            entry_mid = df.mid.values[entry_idx]
            exit_mid = df.mid.values[exit_idx]
            entry_dir = np.sign(df.streak.values[entry_idx]) * -1  # fade the streak
            gains = np.where(entry_dir > 0, (exit_mid - entry_mid) * (1/pip_mul),
                             (entry_mid - exit_mid) * (1/pip_mul))
            wr = (gains > 0).mean() * 100
            avg_pip = gains.mean()
            print(f"  streak≥{min_strk} OFI≥{ofi_thresh}: WR={wr:.1f}% avg={avg_pip:+.2f} N={mask.sum():,}")

print("\n=== DONE ===")
