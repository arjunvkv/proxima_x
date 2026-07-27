"""Test tick exhaustion across multiple pairs with time-based exits."""
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

def load(path):
    df = pd.read_csv(path, parse_dates=['time'])
    df['mid'] = (df.bid + df.ask) / 2
    print(f"  {path.name}: {len(df):,} ticks, {df.time.min()} → {df.time.max()}")
    return df

def compute_streak(df):
    dirs = np.sign(df.mid.diff().fillna(0)).astype(int)
    streaks = [0]
    cur = 0
    for d in dirs[1:]:
        if d == 0:
            pass
        elif d == 1:
            cur = cur + 1 if cur >= 0 else 1
        else:
            cur = cur - 1 if cur <= 0 else -1
        streaks.append(cur)
    df['dir'] = dirs
    df['streak'] = streaks
    return df

# Per-pair pip value
def pip_value(name):
    if 'JPY' in name or name == 'USDJPY':
        return 0.01
    else:
        return 0.0001

print("=== Load & Compute Streaks ===")
for pair, fname in PAIRS.items():
    path = DATA / fname
    if not path.exists(): continue
    df = load(path)
    df = compute_streak(df)

    pip = pip_value(pair)
    mult = 1.0 / pip

    # Streak ≥ 3 → exit after N seconds
    print(f"\n  --- {pair} (pip={pip}, streak≥3 entry, time-based exit) ---")
    for hold_s in [5, 10, 30, 60, 120, 300]:
        entry_mask = (df.streak.abs() >= 3) & (df.dir != 0)
        entry_idx = df[entry_mask].index
        if len(entry_idx) < 10: continue
        pips = []
        wins = 0
        for idx in entry_idx:
            entry_time = df.time.iloc[idx]
            exit_time = entry_time + pd.Timedelta(seconds=hold_s)
            exit_df = df[df.time >= exit_time]
            if len(exit_df) == 0: continue
            exit_idx = exit_df.index[0]
            entry_mid = df.mid.iloc[idx]
            exit_mid = df.mid.iloc[exit_idx]
            direction = df.dir.iloc[idx]
            if direction > 0:
                ret = (entry_mid - exit_mid) * mult
            else:
                ret = (exit_mid - entry_mid) * mult
            pips.append(ret)
            if ret > 0: wins += 1
        if len(pips) < 5: continue
        wr = wins / len(pips)
        avg = np.mean(pips)
        n = len(pips)
        freq_h = n / (df.time.max() - df.time.min()).total_seconds() * 3600
        print(f"    hold={hold_s:3d}s: WR={wr*100:5.1f}% avg={avg:+.2f}pip "
              f"trades={n:,} freq={freq_h:.0f}/hr")

    # Also test: streak≥5 entry
    print(f"\n  --- {pair} (streak≥5 entry, time-based exit) ---")
    for hold_s in [10, 30, 60, 120]:
        entry_mask = (df.streak.abs() >= 5) & (df.dir != 0)
        entry_idx = df[entry_mask].index
        if len(entry_idx) < 10: continue
        pips = []
        wins = 0
        for idx in entry_idx:
            entry_time = df.time.iloc[idx]
            exit_time = entry_time + pd.Timedelta(seconds=hold_s)
            exit_df = df[df.time >= exit_time]
            if len(exit_df) == 0: continue
            exit_idx = exit_df.index[0]
            entry_mid = df.mid.iloc[idx]
            exit_mid = df.mid.iloc[exit_idx]
            direction = df.dir.iloc[idx]
            if direction > 0:
                ret = (entry_mid - exit_mid) * mult
            else:
                ret = (exit_mid - entry_mid) * mult
            pips.append(ret)
            if ret > 0: wins += 1
        if len(pips) < 5: continue
        wr = wins / len(pips)
        avg = np.mean(pips)
        n = len(pips)
        freq_h = n / (df.time.max() - df.time.min()).total_seconds() * 3600
        print(f"    hold={hold_s:3d}s: WR={wr*100:5.1f}% avg={avg:+.2f}pip "
              f"trades={n:,} freq={freq_h:.0f}/hr")

print("\n=== DONE ===")
