"""
Spread Recovery Time Test — Dealer Capitulation Deep Dive.

Hypothesis: Dealer capitulation events (large move + spread widening) where spread
recovers FAST (<5 ticks) have HIGHER forward WR than slow-recovery events.

The dealer's spread recovery time reveals whether they absorbed the flow (fast = done)
or are still under pressure (slow = trend continues).

Session-independent. Tests on GBPJPY + EURJPY (3 months Exness tick data).
"""
import sys, os, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
PAIRS = ['GBPJPY', 'EURJPY', 'EURUSD']
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}

np.random.seed(42)


def load_ticks():
    print("=" * 70)
    print("LOADING EXNESS TICK DATA (Oct-Dec 2025)")
    print("=" * 70)
    t0_total = time.time()
    tick_data = {}
    for pair in PAIRS:
        t0 = time.time()
        dfs = []
        for year, month in MONTHS:
            fn = TICK_DIR / f'{pair}_Raw_Spread_{year}_{month:02d}.zip'
            if not fn.exists():
                print(f"  MISSING: {fn.name}")
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
        elapsed = time.time() - t0
        print(f"  {pair}: {len(tick_data[pair]):>8,d} ticks  ({elapsed:.1f}s)")
    print(f"  Total load time: {time.time() - t0_total:.1f}s")
    return tick_data


def build_m1(tick_data, pair):
    df = tick_data[pair].copy()
    s = SCALE[pair]

    df['Mid'] = (df['Bid'] + df['Ask']) / 2
    df['Spread_pips'] = (df['Ask'] - df['Bid']) * s
    df['Spread_pips_raw'] = df['Ask'] - df['Bid']
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
    })
    bars = bars.dropna(subset=['open', 'close'])

    bars['ret_pips'] = (bars['close'] - bars['open']) * s
    bars['hl_range'] = ((bars['high'] - bars['low']) * s).clip(lower=0)

    roll = 20
    bars['roll_spread_med'] = bars['med_spread'].rolling(roll).median()
    bars['roll_spread_std'] = bars['med_spread'].rolling(roll).std()
    bars['roll_vol_5'] = bars['ret_pips'].rolling(5).std()
    bars['roll_tick_count'] = bars['tick_count'].rolling(roll).median()

    bars['z_ret'] = bars['ret_pips'] / bars['roll_vol_5'].clip(lower=1e-8)
    bars['spread_ratio'] = bars['max_spread'] / bars['roll_spread_med'].clip(lower=1e-8)
    bars['spread_widen'] = bars['max_spread'] > 2 * bars['roll_spread_med'].clip(lower=1e-8)

    bars['fwd1_ret'] = bars['ret_pips'].shift(-1)
    bars['fwd5_ret'] = bars['ret_pips'].rolling(5).sum().shift(-5)
    bars['fwd15_ret'] = bars['ret_pips'].rolling(15).sum().shift(-15)

    bars['hour_utc'] = bars.index.hour
    bars['session'] = np.select(
        [bars['hour_utc'] <= 6,
         (bars['hour_utc'] >= 7) & (bars['hour_utc'] <= 15),
         (bars['hour_utc'] >= 16) & (bars['hour_utc'] <= 23)],
        ['TOKYO', 'LONDON', 'NY'], 'UNKNOWN'
    )

    return bars, df


def compute_spread_recovery_tick_level(event_idx, bars, tick_raw, pair):
    bar_time = bars.index[event_idx]
    roll_med = bars['roll_spread_med'].iloc[event_idx]
    s = SCALE[pair]
    next_time = bar_time + timedelta(minutes=1)

    tick_slice = tick_raw.loc[bar_time:bar_time + timedelta(minutes=2)]
    if len(tick_slice) < 3:
        tick_slice = tick_raw.loc[bar_time:bar_time + timedelta(minutes=5)]
    if len(tick_slice) < 3:
        return None

    tick_slice['Spread_pips'] = (tick_slice['Ask'] - tick_slice['Bid']) * s

    peak_pos = tick_slice['Spread_pips'].idxmax()
    peak_spread = tick_slice['Spread_pips'].max()

    after = tick_slice.loc[peak_pos:]
    if len(after) < 2:
        return None

    recovery_threshold = max(1.5 * roll_med, roll_med + 0.03)
    below = after['Spread_pips'] < recovery_threshold
    first_below = below.idxmax() if below.any() else None

    if first_below is None or first_below == after.index[0] and not below.iloc[0]:
        tick_slice2 = tick_raw.loc[bar_time:bar_time + timedelta(minutes=10)]
        if len(tick_slice2) > len(tick_slice):
            tick_slice2['Spread_pips'] = (tick_slice2['Ask'] - tick_slice2['Bid']) * s
            after2 = tick_slice2.loc[peak_pos:]
            below2 = after2['Spread_pips'] < recovery_threshold
            first_below = below2.idxmax() if below2.any() else None

    if first_below is None:
        return None

    recovery_seconds = (first_below - peak_pos).total_seconds()
    bar_end = bar_time + timedelta(minutes=1)
    ticks_in_bar = tick_raw.loc[bar_time:bar_end]
    if len(ticks_in_bar) < 2:
        return None
    ticks_in_bar['Spread_pips'] = (ticks_in_bar['Ask'] - ticks_in_bar['Bid']) * s
    recovery_ticks = ((ticks_in_bar.index >= peak_pos) & (ticks_in_bar.index <= first_below)).sum()

    return recovery_ticks, recovery_seconds, peak_spread


def compute_cost_model(wr, avg_ret_pips, spread_cost_pip, pair):
    gross_per_trade = avg_ret_pips
    net_per_trade = gross_per_trade - spread_cost_pip
    return {'gross_pip': gross_per_trade, 'spread_pip': spread_cost_pip,
            'net_pip': net_per_trade}


def main():
    t_main = time.time()

    tick_data = load_ticks()

    spread_costs = {'EURJPY': 0.5, 'GBPJPY': 0.6, 'EURUSD': 0.2}

    for pair in PAIRS:
        print(f"\n{'='*70}")
        print(f"PAIR: {pair}")
        print(f"{'='*70}")

        bars, tick_raw = build_m1(tick_data, pair)
        print(f"  M1 bars: {len(bars):,d}")
        print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")

        bars['dir_adj_fwd5'] = -np.sign(bars['z_ret'].fillna(0)) * bars['fwd5_ret'].fillna(0)
        bars['dir_adj_fwd15'] = -np.sign(bars['z_ret'].fillna(0)) * bars['fwd15_ret'].fillna(0)

        big_move = bars['z_ret'].abs() > 2.0
        widen = bars['spread_widen'].astype(bool)
        events = big_move & widen
        n_events = events.sum()
        print(f"\n  Dealer capitulation events: {n_events}")
        if n_events < 5:
            print(f"  SKIP — insufficient events")
            continue

        event_indices = np.where(events.values)[0]
        results = []

        for idx in event_indices:
            rec = compute_spread_recovery_tick_level(
                idx, bars, tick_raw, pair)
            if rec is None:
                continue
            recovery_ticks, recovery_sec, peak_spread = rec

            fwd5 = bars['fwd5_ret'].iloc[idx]
            fwd15 = bars['fwd15_ret'].iloc[idx]
            if np.isnan(fwd5) or np.isnan(fwd15):
                continue

            results.append({
                'time': bars.index[idx],
                'recovery_ticks': recovery_ticks,
                'recovery_sec': recovery_sec,
                'peak_spread': peak_spread,
                'z_ret': bars['z_ret'].iloc[idx],
                'spread_ratio': bars['spread_ratio'].iloc[idx],
                'fwd5_ret': fwd5,
                'fwd15_ret': fwd15,
                'dir_adj_fwd5': bars['dir_adj_fwd5'].iloc[idx],
                'dir_adj_fwd15': bars['dir_adj_fwd15'].iloc[idx],
                'session': bars['session'].iloc[idx],
                'hour': bars['hour_utc'].iloc[idx],
            })

        if len(results) < 5:
            print(f"  SKIP — only {len(results)} valid events after tick recovery measurement")
            continue

        rdf = pd.DataFrame(results)
        print(f"  Events with tick recovery data: {len(rdf)}")

        recovery_med = rdf['recovery_ticks'].median()
        print(f"  Median recovery ticks: {recovery_med:.1f}")
        print(f"  Median recovery seconds: {rdf['recovery_sec'].median():.1f}")
        print(f"  Median peak spread: {rdf['peak_spread'].median():.2f}p")

        fast = rdf['recovery_ticks'] <= recovery_med
        slow = rdf['recovery_ticks'] > recovery_med

        n_fast = fast.sum()
        n_slow = slow.sum()

        print(f"\n  {'Metric':<30s} {'FAST recovery':>16s} {'SLOW recovery':>16s} {'ALL events':>12s}")
        print(f"  {'-'*74}")

        for label, col in [('n_events', None), ('WR (fwd15)', 'fwd15_ret'),
                           ('WR (fwd5)', 'fwd5_ret'), ('adjWR (fwd15)', 'dir_adj_fwd15'),
                           ('adjWR (fwd5)', 'dir_adj_fwd5'), ('avg_fwd15', 'fwd15_ret'),
                           ('avg_fwd5', 'fwd5_ret'), ('avg_adj_fwd15', 'dir_adj_fwd15'),
                           ('avg_adj_fwd5', 'dir_adj_fwd5'), ('peak_spread', 'peak_spread'),
                           ('recovery_ticks', 'recovery_ticks')]:
            if col == 'recovery_ticks':
                for name, mask in [('FAST', fast), ('SLOW', slow), ('ALL', slice(None))]:
                    pass
                f_val = rdf.loc[fast, col].mean()
                s_val = rdf.loc[slow, col].mean()
                a_val = rdf[col].mean()
                print(f"  {'avg_'+col:<30s} {f_val:>16.1f} {s_val:>16.1f} {a_val:>12.1f}")
                continue

            if col is None:
                print(f"  {'n_events':<30s} {n_fast:>16d} {n_slow:>16d} {len(rdf):>12d}")
                continue

            if 'WR' in label:
                f_val = (rdf.loc[fast, col] > 0).mean() * 100
                s_val = (rdf.loc[slow, col] > 0).mean() * 100
                a_val = (rdf[col] > 0).mean() * 100
                print(f"  {label:<30s} {f_val:>15.1f}% {s_val:>15.1f}% {a_val:>11.1f}%")
            else:
                f_val = rdf.loc[fast, col].mean()
                s_val = rdf.loc[slow, col].mean()
                a_val = rdf[col].mean()
                print(f"  {label:<30s} {f_val:>+16.3f}p {s_val:>+16.3f}p {a_val:>+12.3f}p")

        spread_cost = spread_costs[pair]
        print(f"\n  --- COST MODEL ({spread_cost}p spread round-trip) ---")
        print(f"  {'Bucket':<12s} {'n':>4s} {'raw_WR':>7s} {'adj_WR':>7s} {'gross':>8s} {'net':>8s} {'adj_gross':>10s} {'adj_net':>8s} {'e/c':>5s}")
        print(f"  {'-'*70}")

        for mask, label in [(fast, 'FAST'), (slow, 'SLOW'), (rdf.index, 'ALL')]:
            is_all = isinstance(mask, pd.Index)
            n = len(mask) if is_all else mask.sum()
            if n == 0:
                continue
            subset = rdf if is_all else rdf.loc[mask]
            wr = (subset['fwd15_ret'] > 0).mean()
            adj_wr = (subset['dir_adj_fwd15'] > 0).mean()
            avg = subset['fwd15_ret'].mean()
            adj_avg = subset['dir_adj_fwd15'].mean()
            net = avg - spread_cost
            adj_net = adj_avg - spread_cost
            ec = avg / spread_cost if spread_cost > 0 else float('inf')
            print(f"  [{label:>4s}] {n:>4d} {wr:>6.1%} {adj_wr:>6.1%} {avg:>+8.3f}p {net:>+8.3f}p {adj_avg:>+10.3f}p {adj_net:>+8.3f}p {ec:>4.1f}x")

        print(f"\n  --- SESSION BREAKDOWN (SLOW recovery — the real signal) ---")
        slow_df = rdf[slow]
        if len(slow_df) >= 5:
            for session in ['TOKYO', 'LONDON', 'NY']:
                smask = slow_df['session'] == session
                sn = smask.sum()
                if sn < 3:
                    continue
                swr = (slow_df.loc[smask, 'dir_adj_fwd15'] > 0).mean()
                savg = slow_df.loc[smask, 'dir_adj_fwd15'].mean()
                print(f"    {session:<8s} n={sn:>4d}  adjWR={swr:.1%}  adj_avg={savg:+.3f}p")

        print(f"\n  --- SPREAD RATIO BREAKDOWN ---")
        for lo, hi, label in [(2, 3, '2-3x'), (3, 5, '3-5x'), (5, 99, '5x+')]:
            smask = (rdf['spread_ratio'] >= lo) & (rdf['spread_ratio'] < hi)
            sn = smask.sum()
            if sn < 3:
                continue
            swr = (rdf.loc[smask, 'fwd15_ret'] > 0).mean()
            savg = rdf.loc[smask, 'fwd15_ret'].mean()
            net = savg - spread_cost
            print(f"    {label:<6s} spread n={sn:>4d}  WR={swr:.1%}  gross={savg:+.3f}p  net={net:+.3f}p")

        del bars, tick_raw, rdf
        gc.collect()

    elapsed = time.time() - t_main
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
