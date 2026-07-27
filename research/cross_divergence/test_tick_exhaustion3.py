"""Fast tick exhaustion test across pairs. Uses index-based exit (n ticks forward)."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

PAIRS = {
    "EURUSD": "EURUSD_mt5.csv",
    "GBPUSD": "GBPUSD_mt5.csv",
    "EURJPY": "EURJPY_mt5_v2.csv",
    "USDJPY": "USDJPY_mt5_v2.csv",
    "GBPJPY": "GBPJPY_mt5_v2.csv",
    "EURGBP": "EURGBP_mt5.csv",
    "EURCHF": "EURCHF_mt5.csv",
}

def pip_mult(name):
    return 1.0 / (0.01 if ('JPY' in name or name == 'USDJPY') else 0.0001)

print("=== Tick Exhaustion Across All Pairs ===")
print(f"{'Pair':<8} {'Streak':<6} {'ExitTicks':<9} {'WR%':<6} {'AvgPip':<8} {'Trades':<8} {'Freq/hr':<8}")
print("-" * 60)

for pair, fname in PAIRS.items():
    path = DATA / fname
    if not path.exists():
        print(f"{pair:<8} FILE NOT FOUND"); continue
    df = pd.read_csv(path, parse_dates=['time'])
    df['mid'] = (df.bid + df.ask) / 2

    # Compute streaks (vectorized loop)
    dirs = np.sign(df.mid.diff().fillna(0)).astype(np.int8)
    streaks = np.zeros(len(df), dtype=np.int32)
    cur = 0
    for i in range(1, len(df)):
        d = int(dirs[i])
        if d == 1:
            cur = cur + 1 if cur >= 0 else 1
        elif d == -1:
            cur = cur - 1 if cur <= 0 else -1
        streaks[i] = cur
    df['streak'] = abs(streaks)
    df['dir'] = dirs
    mult = pip_mult(pair)

    for min_strk in [3, 5]:
        mask = (df.streak >= min_strk) & (df.dir != 0)
        entry_idx = df[mask].index.values
        if len(entry_idx) < 10: continue

        for exit_n in [5, 10, 30, 60, 120, 300]:
            exit_idx = entry_idx + exit_n
            valid = exit_idx < len(df)
            ei, xi = entry_idx[valid], exit_idx[valid]
            if len(ei) < 5: continue

            entry_mid = df.mid.values[ei]
            exit_mid = df.mid.values[xi]
            direction = df.dir.values[ei]

            gains = np.where(direction > 0, (entry_mid - exit_mid) * mult,
                             (exit_mid - entry_mid) * mult)
            wins = (gains > 0).sum()
            n = len(gains)
            wr = wins / n * 100
            avg = gains.mean()
            duration_h = (df.time.max() - df.time.min()).total_seconds() / 3600
            freq = n / duration_h

            print(f"{pair:<8} ≥{min_strk:<4} {exit_n:<9} {wr:5.1f}% {avg:+7.2f} {n:<8,} {freq:6.0f}")

print("\n=== DONE ===")
