"""
Multi-Timeframe State Compression Test.
Tests whether state alignment across M30/M15/M5/M1 predicts forward returns.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
PAIRS = ['EURJPY', 'GBPJPY']
MONTHS = [(2025, 12), (2025, 11), (2025, 10)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100}
TIMEFRAMES = {'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min'}
LOOKBACK = 20

np.random.seed(42)

def load_ticks():
    print("=" * 70)
    print("LOADING EXNESS TICK DATA")
    print("=" * 70)
    t0 = time.time()
    tick_data = {}
    for pair in PAIRS:
        t1 = time.time()
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
        print(f"  {pair}: {len(tick_data[pair]):>8,d} ticks ({time.time()-t1:.1f}s)")
    print(f"  Total: {time.time()-t0:.1f}s")
    return tick_data


def build_bars(tick_data, rule, scale):
    df = tick_data.copy()
    df['Mid'] = (df['Bid'] + df['Ask']) / 2
    df['Spread_pips'] = (df['Ask'] - df['Bid']) * scale
    df = df.set_index('Timestamp')
    ohlc = df['Mid'].resample(rule).ohlc()
    tick_count = df['Mid'].resample(rule).count()
    med_spread = df['Spread_pips'].resample(rule).median()
    bars = pd.DataFrame({
        'open': ohlc['open'], 'high': ohlc['high'],
        'low': ohlc['low'], 'close': ohlc['close'],
        'tick_count': tick_count, 'med_spread': med_spread,
    })
    bars = bars.dropna(subset=['open', 'close'])
    bars['ret'] = (bars['close'] - bars['open']) * scale
    bars['ret_close'] = bars['close'].diff() * scale
    return bars


def compute_regime_features(bars):
    result = bars.copy()
    n = LOOKBACK
    result['vol'] = result['ret'].rolling(n).std()
    result['vol_rank'] = result['vol'].rank(pct=True, na_option='keep')
    result['vol_regime'] = pd.cut(result['vol_rank'],
        bins=[-0.01, 0.33, 0.67, 1.01], labels=['LOW', 'MED', 'HIGH'])
    result['ma'] = result['close'].rolling(n).mean()
    result['ma_dist'] = (result['close'] - result['ma']) / result['ma'].clip(lower=1e-8)
    result['trend_regime'] = pd.cut(result['ma_dist'],
        bins=[-np.inf, -0.001, 0.001, np.inf], labels=['DOWN', 'FLAT', 'UP'])
    result['spread_rank'] = result['med_spread'].rank(pct=True, na_option='keep')
    result['spread_regime'] = pd.cut(result['spread_rank'],
        bins=[-0.01, 0.33, 0.67, 1.01], labels=['TIGHT', 'NORMAL', 'WIDENED'])
    result['tick_rank'] = result['tick_count'].rank(pct=True, na_option='keep')
    result['tick_regime'] = pd.cut(result['tick_rank'],
        bins=[-0.01, 0.33, 0.67, 1.01], labels=['QUIET', 'NORMAL', 'ACTIVE'])
    return result


def align_to_m1(m1_index, hf_bars_dict):
    aligned = pd.DataFrame(index=m1_index)
    for tf_name, hf_bars in hf_bars_dict.items():
        rule = TIMEFRAMES[tf_name]
        duration = pd.Timedelta(rule)
        hf = hf_bars[['vol_regime', 'trend_regime', 'spread_regime', 'tick_regime',
                       'vol', 'ma_dist', 'med_spread', 'tick_count',
                       'ret_close']].copy()
        hf.index = hf.index + duration
        hf = hf.add_suffix(f'_{tf_name}')
        aligned = pd.merge_asof(aligned, hf, left_index=True, right_index=True,
                                 direction='backward')
    return aligned


def compute_compression(aligned):
    c = aligned.copy()
    tf_labels = ['M1', 'M5', 'M15', 'M30']
    for metric in ['vol', 'trend', 'spread', 'tick']:
        col = f'{metric}_regime'
        regime_cols = [f'{col}_{tf}' for tf in tf_labels if f'{col}_{tf}' in c.columns]
        if len(regime_cols) < 2:
            continue
        regimes = c[regime_cols].values
        c[f'{metric}_all_same'] = np.array([
            len(set(r[~pd.isna(r)])) == 1 if sum(~pd.isna(r)) >= 2 else np.nan
            for r in regimes
        ])
    vol_cols = [f'vol_regime_{tf}' for tf in tf_labels if f'vol_regime_{tf}' in c.columns]
    trend_cols = [f'trend_regime_{tf}' for tf in tf_labels if f'trend_regime_{tf}' in c.columns]
    c['all_aligned'] = np.array([
        (v == 1 and t == 1) if not (np.isnan(v) or np.isnan(t)) else False
        for v, t in zip(c['vol_all_same'].fillna(0).values,
                        c['trend_all_same'].fillna(0).values)
    ])
    c['vol_expansion'] = np.array([
        (r_M30 == 'LOW' and r_M1 == 'HIGH') if (not pd.isna(r_M30) and not pd.isna(r_M1)) else False
        for r_M30, r_M1 in zip(c['vol_regime_M30'].values, c['vol_regime_M1'].values)
    ])
    c['vol_contraction'] = np.array([
        (r_M30 == 'HIGH' and r_M1 == 'LOW') if (not pd.isna(r_M30) and not pd.isna(r_M1)) else False
        for r_M30, r_M1 in zip(c['vol_regime_M30'].values, c['vol_regime_M1'].values)
    ])
    trend_cols_avail = [f'trend_regime_{tf}' for tf in tf_labels if f'trend_regime_{tf}' in c.columns]
    if len(trend_cols_avail) >= 2:
        trend_matrix = c[trend_cols_avail].values
        trend_all_up = np.full(len(c), False)
        trend_all_down = np.full(len(c), False)
        for i in range(len(c)):
            row = trend_matrix[i]
            valid = ~pd.isna(row)
            if valid.sum() >= 2:
                if all(r == 'UP' for r in row[valid]):
                    trend_all_up[i] = True
                if all(r == 'DOWN' for r in row[valid]):
                    trend_all_down[i] = True
        c['trend_all_up'] = trend_all_up
        c['trend_all_down'] = trend_all_down
    return c


def test_events(aligned, pair):
    print(f"\n{'='*70}")
    print(f"MULTI-TIMEFRAME COMPRESSION — {pair}")
    print(f'='*70)
    results = {}
    tf_ret_col = f'ret_close_M1'
    m1_ret = aligned[tf_ret_col].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    def to_bool(arr):
        arr = np.asarray(arr)
        if arr.dtype == object:
            out = np.full(len(arr), False)
            for i in range(len(arr)):
                try:
                    out[i] = bool(arr[i]) and not (isinstance(arr[i], float) and np.isnan(arr[i]))
                except Exception:
                    out[i] = False
            return out
        if np.issubdtype(arr.dtype, np.bool_):
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            return (~np.isnan(arr)) & (arr > 0.5)
        return arr.astype(bool)
    events = {
        'ALL_ALIGNED': to_bool(aligned['all_aligned'].values),
        'VOL_ALL_SAME': to_bool(aligned['vol_all_same'].values),
        'TREND_ALL_SAME': to_bool(aligned['trend_all_same'].values),
        'SPREAD_ALL_SAME': to_bool(aligned['spread_all_same'].values),
        'VOL_EXPANSION': to_bool(aligned['vol_expansion'].values),
        'VOL_CONTRACTION': to_bool(aligned['vol_contraction'].values),
        'TREND_ALL_UP': to_bool(aligned['trend_all_up'].values),
        'TREND_ALL_DOWN': to_bool(aligned['trend_all_down'].values),
    }
    print(f"\n{'Event Type':<20s} {'n':>6s} {'Fwd15':>8s} {'WR':>6s} {'z':>7s}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")
    baseline = None
    for label, mask in events.items():
        n = int(mask.sum())
        if n < 10:
            print(f"{label:<20s} {n:>6d} {'--':>8s}")
            continue
        fwd = fwd15.values.copy()
        valid = ~np.isnan(fwd)
        ev_fwd = fwd[valid & mask]
        if len(ev_fwd) < 10:
            print(f"{label:<20s} {n:>6d} {'--':>8s} (no valid fwd)")
            continue
        mean_ret = np.nanmean(ev_fwd)
        wr = np.nanmean(ev_fwd > 0)
        se = np.nanstd(ev_fwd) / np.sqrt(len(ev_fwd))
        z = mean_ret / se if se > 0 else 0
        print(f"{label:<20s} {len(ev_fwd):>6d} {mean_ret:>+8.3f}p {wr:>6.1%} {z:>+7.2f}")
        results[label] = {'n': len(ev_fwd), 'fwd15': float(mean_ret), 'wr': float(wr), 'z': float(z)}
        if label == 'ALL_ALIGNED':
            baseline = results[label]
    all_mask = np.ones(len(aligned), dtype=bool)
    all_valid = ~np.isnan(fwd15.values)
    all_fwd = fwd15.values[all_valid]
    print(f"{'BASELINE (all)':<20s} {len(all_fwd):>6d} {np.nanmean(all_fwd):>+8.3f}p {np.nanmean(all_fwd>0):>6.1%}")
    results['BASELINE'] = {'n': len(all_fwd), 'fwd15': float(np.nanmean(all_fwd)), 'wr': float(np.nanmean(all_fwd>0))}
    if baseline and baseline['n'] >= 10:
        delta = baseline['fwd15'] - results['BASELINE']['fwd15']
        wr_delta = baseline['wr'] - results['BASELINE']['wr']
        print(f"\n  Delta (ALL_ALIGNED - BASELINE): {delta:+.3f}p  WR: {wr_delta:+.1%}")
        results['DELTA'] = {'fwd15': float(delta), 'wr_delta': float(wr_delta)}
    return results


def test_pair_breakdown(aligned, pair):
    print(f"\n{'='*70}")
    print(f"PAIR-LEVEL DECOMPOSITION — {pair}")
    print(f"{'='*70}")
    results = {}
    m1_ret = aligned['ret_close_M1'].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    vol_regimes = ['LOW', 'MED', 'HIGH']
    trend_regimes = ['DOWN', 'FLAT', 'UP']
    print(f"\nCross-TF Vol Regime Combinations (M30 vs M1):")
    hdr_sep = '/'
    print(f"{'M30' + hdr_sep + 'M1':<10s} {'LOW':>8s} {'MED':>8s} {'HIGH':>8s}")
    for r30 in vol_regimes:
        row_vals = []
        for r1 in vol_regimes:
            mask = (aligned['vol_regime_M30'].values == r30) & (aligned['vol_regime_M1'].values == r1)
            n = int(np.nansum(mask))
            if n >= 10:
                fwd = fwd15.values.copy()
                valid = ~np.isnan(fwd) & ~np.isnan(mask.astype(float))
                ev_fwd = fwd[valid & mask]
                mean_r = np.nanmean(ev_fwd) if len(ev_fwd) > 0 else 0
                wr_r = np.nanmean(ev_fwd > 0) if len(ev_fwd) > 0 else 0
                row_vals.append(f"{mean_r:+.2f}p/{wr_r:.0%}")
            else:
                row_vals.append(f"n={n}")
        print(f"{r30:<10s} {row_vals[0]:>8s} {row_vals[1]:>8s} {row_vals[2]:>8s}")
        sep = '/'
        print(f"\nCross-TF Trend Alignment (M30 vs M1):")
        print(f"{'M30' + sep + 'M1':<10s} {'DOWN':>8s} {'FLAT':>8s} {'UP':>8s}")
    for t30 in trend_regimes:
        row_vals = []
        for t1 in trend_regimes:
            mask = (aligned['trend_regime_M30'].values == t30) & (aligned['trend_regime_M1'].values == t1)
            n = int(np.nansum(mask))
            if n >= 10:
                fwd = fwd15.values.copy()
                valid = ~np.isnan(fwd) & ~np.isnan(mask.astype(float))
                ev_fwd = fwd[valid & mask]
                mean_r = np.nanmean(ev_fwd) if len(ev_fwd) > 0 else 0
                wr_r = np.nanmean(ev_fwd > 0) if len(ev_fwd) > 0 else 0
                row_vals.append(f"{mean_r:+.2f}p/{wr_r:.0%}")
            else:
                row_vals.append(f"n={n}")
        print(f"{t30:<10s} {row_vals[0]:>8s} {row_vals[1]:>8s} {row_vals[2]:>8s}")
    return results


def test_session_alignment(aligned, pair):
    print(f"\n{'='*70}")
    print(f"SESSION BREAKDOWN — {pair}")
    print(f"{'='*70}")
    m1_ret = aligned['ret_close_M1'].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    aligned['hour'] = aligned.index.hour
    for session, hour_mask in [('TOKYO', (0, 6)), ('LONDON', (7, 15)), ('NY', (16, 23))]:
        h_start, h_end = hour_mask
        mask = (aligned['hour'] >= h_start) & (aligned['hour'] <= h_end)
        sub = aligned[mask].copy()
        n_total = len(sub)
        if n_total < 100:
            continue
        for regime in ['LOW', 'HIGH']:
            vol_mask = sub['vol_regime_M30'].values == regime
            n_vol = int(np.nansum(vol_mask))
            if n_vol < 10:
                continue
            fwd = fwd15.loc[sub.index].values.copy()
            vmask = vol_mask & ~np.isnan(fwd)
            ev_fwd = fwd[vmask]
            if len(ev_fwd) < 10:
                continue
            print(f"  {session} M30-{regime}: n={len(ev_fwd):>5d}  fwd15={np.nanmean(ev_fwd):+>.3f}p  "
                  f"WR={np.nanmean(ev_fwd>0):.1%}")
    return {}


def test_dealer_cap_multiframe(aligned, pair):
    print(f"\n{'='*70}")
    print(f"DEALER CAPITULATION × MULTI-TIMEFRAME — {pair}")
    print(f"{'='*70}")
    m1_ret = aligned['ret_close_M1'].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    z_ret = m1_ret / np.nanstd(m1_ret.clip(1e-8))
    big_move = np.abs(z_ret) > 2.0
    spread_widen = aligned['spread_regime_M1'].values == 'WIDENED'
    dealer_cap = big_move & spread_widen
    results = {}
    for label, extra_mask, desc in [
        ('DC_ONLY', np.ones(len(aligned), dtype=bool), 'No multi-TF filter'),
        ('DC+TREND_ALIGN', aligned['trend_all_same'].fillna(0).values.astype(bool), 'All TFs same trend'),
        ('DC+VOL_EXPAND', aligned['vol_expansion'].values.astype(bool), 'M30 LOW → M1 HIGH'),
        ('DC+LOW_VOL', (aligned['vol_regime_M30'].values == 'LOW') & (aligned['vol_regime_M15'].values == 'LOW'), 'M30+M15 low vol'),
        ('DC+HIGH_VOL', (aligned['vol_regime_M30'].values == 'HIGH') & (aligned['vol_regime_M15'].values == 'HIGH'), 'M30+M15 high vol'),
    ]:
        mask = dealer_cap & extra_mask
        n = int(np.nansum(mask))
        if n < 5:
            print(f"  {desc:<35s} n={n:>4d}  --")
            continue
        fwd = fwd15.values.copy()
        valid = ~np.isnan(fwd) & ~np.isnan(mask.astype(float))
        ev_fwd = fwd[valid & mask]
        if len(ev_fwd) < 5:
            continue
        mean_r = np.nanmean(ev_fwd)
        wr = np.nanmean(ev_fwd > 0)
        se = np.nanstd(ev_fwd) / np.sqrt(len(ev_fwd))
        z = mean_r / se if se > 0 else 0
        print(f"  {desc:<35s} n={len(ev_fwd):>4d}  fwd15={mean_r:+.3f}p  WR={wr:.1%}  z={z:+.2f}")
        results[label] = {'n': len(ev_fwd), 'fwd15': float(mean_r), 'wr': float(wr), 'z': float(z)}
    return results


def test_ma_distribution(aligned, pair):
    print(f"\n{'='*70}")
    print(f"MA DISTANCE DISTRIBUTION — {pair}")
    print(f"{'='*70}")
    m1_ret = aligned['ret_close_M1'].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    for tf in ['M1', 'M5', 'M15', 'M30']:
        col = f'ma_dist_{tf}'
        if col not in aligned.columns:
            continue
        vals = aligned[col].values
        for pctile, label in [(10, 'Extreme low'), (25, 'Low'), (75, 'High'), (90, 'Extreme high')]:
            thresh = np.nanpercentile(vals, pctile)
            if pctile < 50:
                mask = vals < thresh
                side = 'below'
            else:
                mask = vals > thresh
                side = 'above'
            n = int(np.nansum(mask))
            if n < 10:
                continue
            fwd = fwd15.values.copy()
            valid = ~np.isnan(fwd) & ~np.isnan(mask.astype(float))
            ev_fwd = fwd[valid & mask]
            if len(ev_fwd) < 10:
                continue
            mean_r = np.nanmean(ev_fwd)
            wr = np.nanmean(ev_fwd > 0)
            print(f"  {tf} {label:<15s} ({side} {pctile}%ile): n={len(ev_fwd):>5d}  "
                  f"fwd15={mean_r:+.3f}p  WR={wr:.1%}")
    return {}


def test_spread_vol_combined(aligned, pair):
    print(f"\n{'='*70}")
    print(f"SPREAD × VOL COMBINED — {pair}")
    print(f"{'='*70}")
    m1_ret = aligned['ret_close_M1'].values.copy()
    m1_ret_aligned = np.roll(m1_ret, -15)
    fwd15 = pd.Series(m1_ret_aligned, index=aligned.index)
    for spread_tf in ['M1', 'M5', 'M15', 'M30']:
        spread_col = f'spread_regime_{spread_tf}'
        vol_col = f'vol_regime_{spread_tf}'
        if spread_col not in aligned.columns or vol_col not in aligned.columns:
            continue
        for sr in ['TIGHT', 'WIDENED']:
            for vr in ['LOW', 'HIGH']:
                mask = ((aligned[spread_col].values == sr) &
                        (aligned[vol_col].values == vr))
                n = int(np.nansum(mask))
                if n < 10:
                    continue
                fwd = fwd15.values.copy()
                valid = ~np.isnan(fwd) & ~np.isnan(mask.astype(float))
                ev_fwd = fwd[valid & mask]
                if len(ev_fwd) < 10:
                    continue
                mean_r = np.nanmean(ev_fwd)
                wr = np.nanmean(ev_fwd > 0)
                print(f"  {spread_tf} sr={sr:<7s} vr={vr:<4s}: n={len(ev_fwd):>5d}  "
                      f"fwd15={mean_r:+.3f}p  WR={wr:.1%}")
    return {}


def main():
    t_main = time.time()
    tick_data = load_ticks()
    all_results = {}
    for pair in PAIRS:
        pair_results = {}
        s = SCALE[pair]
        print(f"\n{'#'*70}")
        print(f"BUILDING MULTI-TIMEFRAME BARS — {pair}")
        print(f"{'#'*70}")
        bars = {}
        for tf_name, rule in TIMEFRAMES.items():
            t0 = time.time()
            b = build_bars(tick_data[pair], rule, s)
            b = compute_regime_features(b)
            bars[tf_name] = b
            n_bars = len(b)
            print(f"  {tf_name}: {n_bars:>5d} bars ({time.time()-t0:.1f}s)")
        print(f"\n  Aligning all timeframes to M1...")
        m1_bars = bars['M1']
        hf_bars = {k: v for k, v in bars.items() if k != 'M1'}
        aligned = align_to_m1(m1_bars.index, hf_bars)
        m1_features = m1_bars[['ret_close', 'vol_regime', 'trend_regime',
                                'spread_regime', 'tick_regime']].copy()
        m1_features.columns = [f'{c}_M1' for c in m1_features.columns]
        aligned = pd.concat([aligned, m1_features], axis=1)
        aligned = compute_compression(aligned)
        aligned = aligned.dropna(subset=['vol_regime_M1', 'trend_regime_M1'])
        print(f"  Aligned bars: {len(aligned):,d}")
        pair_results['compression'] = test_events(aligned, pair)
        pair_results['pair_breakdown'] = test_pair_breakdown(aligned, pair)
        pair_results['session'] = test_session_alignment(aligned, pair)
        pair_results['dealer_cap'] = test_dealer_cap_multiframe(aligned, pair)
        pair_results['ma_dist'] = test_ma_distribution(aligned, pair)
        pair_results['spread_vol'] = test_spread_vol_combined(aligned, pair)
        all_results[pair] = pair_results
    elapsed = time.time() - t_main
    print(f"\n{'='*70}")
    print(f"MULTI-TIMEFRAME COMPRESSION TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Pairs: {PAIRS}")
    print(f"  Data: Oct-Dec 2025 Exness ticks")
    print(f"  Timeframes: {list(TIMEFRAMES.keys())}")
    return all_results


if __name__ == '__main__':
    main()
