"""
Spread Recovery Scanner — Generalized Recovery Time Signal.

Tests if slow spread recovery after ANY large move predicts mean reversion.
Session-independent. Multiple thresholds. Both GBPJPY + EURJPY.
"""
import sys, os, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}

np.random.seed(42)


def load_ticks(pairs):
    tick_data = {}
    for pair in pairs:
        t0 = time.time()
        dfs = []
        for year, month in MONTHS:
            fn = TICK_DIR / f'{pair}_Raw_Spread_{year}_{month:02d}.zip'
            if not fn.exists():
                continue
            df = pd.read_csv(fn, compression='zip',
                names=['Exness', 'Symbol', 'Timestamp', 'Bid', 'Ask'],
                skiprows=1, header=None, on_bad_lines='skip',
                dtype={'Exness': str, 'Symbol': str, 'Timestamp': str,
                       'Bid': np.float64, 'Ask': np.float64})
            df['Timestamp'] = df['Timestamp'].str.replace('Z', '', regex=False)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            df = df.dropna(subset=['Timestamp'])
            dfs.append(df)
        tick_data[pair] = pd.concat(dfs, ignore_index=True)
        tick_data[pair] = tick_data[pair].sort_values('Timestamp').reset_index(drop=True)
        print(f"  {pair}: {len(tick_data[pair]):>8,d} ticks  ({time.time()-t0:.1f}s)")
    return tick_data


def build_m1(tick_data, pair):
    df = tick_data[pair].copy()
    s = SCALE[pair]

    df['Mid'] = (df['Bid'] + df['Ask']) / 2
    df['Spread_pips'] = (df['Ask'] - df['Bid']) * s
    df = df.set_index('Timestamp')

    ohlc = df['Mid'].resample('1min').ohlc()
    tick_count = df['Mid'].resample('1min').count()
    med_spread = df['Spread_pips'].resample('1min').median()
    max_spread = df['Spread_pips'].resample('1min').max()

    bars = pd.DataFrame({
        'open': ohlc['open'], 'high': ohlc['high'],
        'low': ohlc['low'], 'close': ohlc['close'],
        'tick_count': tick_count,
        'med_spread': med_spread,
        'max_spread': max_spread,
    }).dropna(subset=['open', 'close'])

    bars['ret_pips'] = (bars['close'] - bars['open']) * s
    bars['hl_range'] = ((bars['high'] - bars['low']) * s).clip(lower=0)

    roll = 20
    bars['roll_spread_med'] = bars['med_spread'].rolling(roll).median()
    bars['roll_spread_std'] = bars['med_spread'].rolling(roll).std()
    bars['roll_vol_5'] = bars['ret_pips'].rolling(5).std()

    bars['z_ret'] = bars['ret_pips'] / bars['roll_vol_5'].clip(lower=1e-8)
    bars['spread_ratio'] = bars['max_spread'] / bars['roll_spread_med'].clip(lower=1e-8)

    bars['fwd5_ret'] = bars['ret_pips'].rolling(5).sum().shift(-5)
    bars['fwd15_ret'] = bars['ret_pips'].rolling(15).sum().shift(-15)

    bars['dir_adj_fwd5'] = -np.sign(bars['z_ret'].fillna(0)) * bars['fwd5_ret'].fillna(0)
    bars['dir_adj_fwd15'] = -np.sign(bars['z_ret'].fillna(0)) * bars['fwd15_ret'].fillna(0)

    bars['hour_utc'] = bars.index.hour
    bars['session'] = np.select(
        [bars['hour_utc'] <= 6,
         (bars['hour_utc'] >= 7) & (bars['hour_utc'] <= 15),
         (bars['hour_utc'] >= 16) & (bars['hour_utc'] <= 23)],
        ['TOKYO', 'LONDON', 'NY'], 'UNKNOWN'
    )

    return bars, df


def measure_recovery(event_idx, bars, tick_raw, pair):
    s = SCALE[pair]
    bar_time = bars.index[event_idx]
    roll_med = bars['roll_spread_med'].iloc[event_idx]

    tick_slice = tick_raw.loc[bar_time:bar_time + timedelta(minutes=2)]
    if len(tick_slice) < 3:
        tick_slice = tick_raw.loc[bar_time:bar_time + timedelta(minutes=5)]
    if len(tick_slice) < 3:
        return None

    spreads = (tick_slice['Ask'] - tick_slice['Bid']) * s
    peak_pos = spreads.idxmax()
    peak_spread = spreads.max()

    after = spreads.loc[peak_pos:]
    if len(after) < 2:
        return None

    threshold = max(1.5 * roll_med, roll_med + 0.03)
    below = after < threshold
    first_below = below.idxmax() if below.any() else None

    if first_below is None or (first_below == after.index[0] and not below.iloc[0]):
        wider = tick_raw.loc[bar_time:bar_time + timedelta(minutes=10)]
        if len(wider) > len(tick_slice):
            wider_spreads = (wider['Ask'] - wider['Bid']) * s
            after2 = wider_spreads.loc[peak_pos:]
            below2 = after2 < threshold
            first_below = below2.idxmax() if below2.any() else None

    if first_below is None:
        return None

    recovery_sec = (first_below - peak_pos).total_seconds()
    ticks_window = tick_raw.loc[peak_pos:first_below]
    recovery_ticks = len(ticks_window)
    if recovery_ticks < 1:
        return None

    return recovery_ticks, recovery_sec, peak_spread


def run_scan(pairs, z_thresholds, recover_quantiles):
    tick_data = load_ticks(pairs)
    spread_costs = {'EURJPY': 0.5, 'GBPJPY': 0.6}

    print(f"\n{'='*110}")
    print(f"{'Pair':<8s} {'z_thr':>6s} {'rec_q':>6s} {'n':>5s} {'n_day':>6s} {'adjWR5':>7s} {'adjWR15':>7s} {'adj_p5':>8s} {'adj_p15':>8s} {'net5':>8s} {'net15':>8s} {'e/c5':>6s} {'e/c15':>6s}")
    print(f"{'='*110}")

    all_results = {pair: [] for pair in pairs}

    for pair in pairs:
        if pair == 'EURUSD':
            continue
        bars, tick_raw = build_m1(tick_data, pair)
        cost = spread_costs[pair]
        n_total = len(bars)
        print(f"  {pair}: {n_total:,d} M1 bars, {bars.index[0].date()} to {bars.index[-1].date()}")

        for z_thr in z_thresholds:
            big_moves = bars['z_ret'].abs() > z_thr
            n_big = big_moves.sum()
            if n_big < 30:
                continue

            for rec_q in recover_quantiles:
                event_indices = np.where(big_moves.values)[0]
                results = []
                for idx in event_indices:
                    rec = measure_recovery(idx, bars, tick_raw, pair)
                    if rec is None:
                        continue
                    recovery_ticks, _, _ = rec
                    fwd5 = bars['dir_adj_fwd5'].iloc[idx]
                    fwd15 = bars['dir_adj_fwd15'].iloc[idx]
                    if np.isnan(fwd5) or np.isnan(fwd15):
                        continue
                    results.append({'rec_ticks': recovery_ticks, 'fwd5': fwd5, 'fwd15': fwd15})

                if len(results) < 10:
                    continue

                rdf = pd.DataFrame(results)
                threshold_ticks = rdf['rec_ticks'].quantile(1 - rec_q)
                fast = rdf['rec_ticks'] <= threshold_ticks
                slow = rdf['rec_ticks'] > threshold_ticks

                for mask, label in [(fast, 'FAST'), (slow, 'SLOW'), (rdf.index, ' ALL')]:
                    is_all = isinstance(mask, pd.Index)
                    n = len(mask) if is_all else mask.sum()
                    if n < 5:
                        continue
                    subset = rdf if is_all else rdf.loc[mask]
                    adj5 = (subset['fwd5'] > 0).mean()
                    adj15 = (subset['fwd15'] > 0).mean()
                    avg5 = subset['fwd5'].mean()
                    avg15 = subset['fwd15'].mean()
                    n_day = n / 66
                    net5 = avg5 - cost
                    net15 = avg15 - cost
                    ec5 = avg5 / cost if cost > 0 else 0
                    ec15 = avg15 / cost if cost > 0 else 0

                    if label == 'SLOW' and adj5 >= 0.65:
                        print(f"  {pair:<8s} {z_thr:>5.1f}  {rec_q:>5.2f} {label:>4s} {n:>5d} {n_day:>5.2f}  {adj5:>6.1%} {adj15:>6.1%} {avg5:>+8.3f}p {avg15:>+8.3f}p {net5:>+8.3f}p {net15:>+8.3f}p {ec5:>5.1f}x {ec15:>5.1f}x")

        del bars, tick_raw
        gc.collect()

    return tick_data


print("=" * 110)
print("SPREAD RECOVERY SCANNER")
print("Grid search: z_thresholds [1.0, 1.25, 1.5, 1.75, 2.0], recovery quantiles [0.3, 0.4, 0.5]")
print("=" * 110)

run_scan(
    pairs=['GBPJPY', 'EURJPY'],
    z_thresholds=[1.0, 1.25, 1.5, 1.75, 2.0],
    recover_quantiles=[0.3, 0.4, 0.5]
)
