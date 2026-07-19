"""
Tests 16-17: Dealer Capitulation Validation & Mechanism.
Self-contained — copies data loading, M1 building, and feature computation
from run_market_state_auth.py with forced pandas boolean parentheses safety.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
PAIRS = ['EURJPY', 'GBPJPY', 'EURUSD']
MONTHS = [(2025, 12), (2025, 11), (2025, 10)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}
N_FWD_BARS = 15

np.random.seed(42)

# ====================================================================
# DATA LOADING (exact copy from run_market_state_auth.py)
# ====================================================================
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


# ====================================================================
# M1 BAR BUILDING (exact copy + fwd30 for test 17)
# ====================================================================
def build_m1(tick_data):
    print("\n" + "=" * 70)
    print("BUILDING M1 BARS WITH TICK FEATURES")
    print("=" * 70)
    m1_dict = {}
    for pair in PAIRS:
        t0 = time.time()
        df = tick_data[pair].copy()
        s = SCALE[pair]

        df['Mid'] = (df['Bid'] + df['Ask']) / 2
        df['Spread_pips'] = (df['Ask'] - df['Bid']) * s
        df = df.set_index('Timestamp')

        dt_us = df.index.to_series().diff().dt.total_seconds() * 1e6
        df['dt_us'] = dt_us.fillna(0)

        df['dbid'] = df['Bid'].diff().fillna(0)
        df['dask'] = df['Ask'].diff().fillna(0)
        df['dmid'] = df['Mid'].diff().fillna(0)
        dmid_abs = df['dmid'].abs().clip(lower=1e-8)
        df['QAI'] = (df['dbid'] - df['dask']).abs() / dmid_abs

        ohlc = df['Mid'].resample('1min').ohlc()
        tick_count = df['Mid'].resample('1min').count()

        med_spread = df['Spread_pips'].resample('1min').median()
        mean_spread = df['Spread_pips'].resample('1min').mean()
        max_spread = df['Spread_pips'].resample('1min').max()
        std_spread = df['Spread_pips'].resample('1min').std()

        med_qai = df['QAI'].resample('1min').median()
        qai_stress_pct = df['QAI'].resample('1min').apply(
            lambda x: np.mean(x > 2.0) if len(x) > 0 else 0)

        mean_dt_us = df['dt_us'].resample('1min').mean()
        med_dt_us = df['dt_us'].resample('1min').median()

        bid_close = df['Bid'].resample('1min').last()
        bid_open = df['Bid'].resample('1min').first()
        ask_close = df['Ask'].resample('1min').last()
        ask_open = df['Ask'].resample('1min').first()
        bid_drift = (bid_close - bid_open).fillna(0) * s
        ask_drift = (ask_close - ask_open).fillna(0) * s
        spread_open = ((ask_open - bid_open) * s).fillna(0)
        spread_close = ((ask_close - bid_close) * s).fillna(0)

        bars = pd.DataFrame({
            'open': ohlc['open'], 'high': ohlc['high'],
            'low': ohlc['low'], 'close': ohlc['close'],
            'tick_count': tick_count, 'med_spread': med_spread,
            'mean_spread': mean_spread, 'max_spread': max_spread,
            'std_spread': std_spread,
            'med_qai': med_qai, 'qai_stress_pct': qai_stress_pct,
            'mean_dt_us': mean_dt_us, 'med_dt_us': med_dt_us,
            'bid_drift': bid_drift, 'ask_drift': ask_drift,
            'spread_open': spread_open, 'spread_close': spread_close,
        })
        bars = bars.dropna(subset=['open', 'close'])

        bars['ret_pips'] = (bars['close'] - bars['open']) * s
        bars['ret_high_low'] = ((bars['high'] - bars['low']) * s).clip(lower=0)
        bars['body_pct'] = ((bars['close'] - bars['open']).abs() /
                            (bars['high'] - bars['low']).clip(lower=1e-8))

        roll = 20
        bars['roll_spread_med'] = bars['med_spread'].rolling(roll).median()
        bars['roll_spread_std'] = bars['med_spread'].rolling(roll).std()
        bars['roll_vol_5'] = bars['ret_pips'].rolling(5).std()
        bars['roll_vol_20'] = bars['ret_pips'].rolling(roll).std()
        bars['roll_mean_dt'] = bars['mean_dt_us'].rolling(roll).mean()
        bars['roll_tick_count'] = bars['tick_count'].rolling(roll).median()

        bars['spread_shock'] = bars['med_spread'] > 3 * bars['roll_spread_med'].clip(lower=1e-8)
        bars['tick_accel'] = bars['mean_dt_us'] < 0.3 * bars['roll_mean_dt'].clip(lower=1e-8)
        bars['tick_drop'] = bars['tick_count'] < 0.25 * bars['roll_tick_count'].clip(lower=1e-8)
        bars['z_ret'] = bars['ret_pips'] / bars['roll_vol_5'].clip(lower=1e-8)
        bars['spread_z'] = ((bars['med_spread'] - bars['roll_spread_med'].clip(lower=1e-8))
                            / bars['roll_spread_std'].clip(lower=1e-8))
        bars['spread_widen'] = bars['med_spread'] > 2 * bars['roll_spread_med'].clip(lower=1e-8)

        bars['fwd1_ret'] = bars['ret_pips'].shift(-1)
        bars['fwd5_ret'] = bars['ret_pips'].rolling(5).sum().shift(-5)
        bars['fwd15_ret'] = bars['ret_pips'].rolling(15).sum().shift(-15)
        bars['fwd30_ret'] = bars['ret_pips'].rolling(30).sum().shift(-30)
        bars['fwd5_vol'] = bars['ret_pips'].rolling(5).std().shift(-1).rolling(5).mean().shift(-5)

        for col in bars.columns:
            bars = bars.rename(columns={col: f'{pair}_{col}'})

        m1_dict[pair] = bars
        elapsed = time.time() - t0
        print(f"  {pair}: {len(bars):,d} M1 bars ({elapsed:.1f}s)")

    return m1_dict


# ====================================================================
# ALIGNMENT & CROSS-PAIR FEATURES (exact copy)
# ====================================================================
def align_and_enrich(m1_dict):
    print("\n" + "=" * 70)
    print("ALIGNING & COMPUTING CROSS-PAIR FEATURES")
    print("=" * 70)

    common_idx = m1_dict[PAIRS[0]].index
    for pair in PAIRS[1:]:
        common_idx = common_idx.intersection(m1_dict[pair].index)
    print(f"  Common index: {len(common_idx):,d} M1 bars")

    pieces = [m1_dict[p].loc[common_idx] for p in PAIRS]
    df = pd.concat(pieces, axis=1)
    print(f"  Combined shape: {df.shape}")

    ret_cols = [f'{p}_ret_pips' for p in PAIRS]
    sign_matrix = np.sign(df[ret_cols].values)
    df['agreement'] = np.array([
        max(np.mean(s == 1), np.mean(s == -1), np.mean(s == 0))
        for s in sign_matrix
    ])
    df['dispersion'] = df[ret_cols].std(axis=1)
    df['direction'] = np.sign(df[ret_cols].mean(axis=1))

    y = df['EURJPY_ret_pips'].values
    X = df[['GBPJPY_ret_pips', 'EURUSD_ret_pips']].values
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    df['msv_residual'] = np.nan
    if valid.sum() > 10:
        model = LinearRegression().fit(X[valid], y[valid])
        df.loc[valid, 'msv_residual'] = y[valid] - model.predict(X[valid])
        print(f"  MSV residual model: EURJPY = {model.coef_[0]:.3f}*GBPJPY + {model.coef_[1]:.3f}*EURUSD + {model.intercept_:.3f}")
        print(f"  R\u00b2 = {model.score(X[valid], y[valid]):.4f}")

    df['hour_utc'] = df.index.hour
    df['is_tokyo_h0'] = (df['hour_utc'] == 0).astype(int)
    df['is_london_h0'] = (df['hour_utc'] == 8).astype(int)
    df['is_ny_h0'] = (df['hour_utc'] == 13).astype(int)
    df['is_sunday_open'] = (df['hour_utc'] == 22).astype(int)
    conds = [df['hour_utc'] <= 6,
             (df['hour_utc'] >= 7) & (df['hour_utc'] <= 15),
             (df['hour_utc'] >= 16) & (df['hour_utc'] <= 23)]
    df['session'] = np.select(conds, ['TOKYO', 'LONDON', 'NY'], 'UNKNOWN')

    df['is_declining_basket'] = (df[ret_cols] < 0).all(axis=1).astype(int)
    df['is_rising_basket'] = (df[ret_cols] > 0).all(axis=1).astype(int)

    df['normal_spread'] = ((~df['EURJPY_spread_widen']) &
                           (~df['GBPJPY_spread_widen']) &
                           (~df['EURUSD_spread_widen'])).astype(int)
    df['any_spread_stress'] = (df['EURJPY_spread_widen'] |
                               df['GBPJPY_spread_widen'] |
                               df['EURUSD_spread_widen']).astype(int)

    df['stable_quotes'] = ((df['EURJPY_med_qai'] < df['EURJPY_med_qai'].median()) &
                           (df['GBPJPY_med_qai'] < df['GBPJPY_med_qai'].median()) &
                           (df['EURUSD_med_qai'] < df['EURUSD_med_qai'].median())).astype(int)

    df['high_participation'] = ((df['EURJPY_tick_count'] > df['EURJPY_tick_count'].median()) &
                                (df['GBPJPY_tick_count'] > df['GBPJPY_tick_count'].median()) &
                                (df['EURUSD_tick_count'] > df['EURUSD_tick_count'].median())).astype(int)

    df['authenticated'] = ((df['agreement'] > 0.66) &
                           df['normal_spread'].astype(bool) &
                           df['high_participation'].astype(bool)).astype(int)

    df['fake_breakout'] = ((df['agreement'] < 0.66) &
                           df['any_spread_stress'].astype(bool) &
                           (df['EURJPY_z_ret'].abs() > 2.0)).astype(int)

    df['low_vol'] = (df['EURJPY_roll_vol_5'] < df['EURJPY_roll_vol_5'].quantile(0.25)).astype(int)

    print(f"  Authentication rate: {df['authenticated'].mean():.1%}")
    print(f"  Fake breakout rate: {df['fake_breakout'].mean():.1%}")
    print(f"  Normal spread rate: {df['normal_spread'].mean():.1%}")
    print(f"  Any spread stress:  {df['any_spread_stress'].mean():.1%}")

    return df


# ====================================================================
# TEST 16: DEALER CAPITULATION VALIDATION
# ====================================================================
def test_dealer_capitulation_validation(df):
    print("\n" + "=" * 70)
    print("TEST 16: DEALER CAPITULATION VALIDATION")
    print("=" * 70)

    n_total = len(df)
    n_trading_days = 66

    # ---- event definitions ----
    big_move_ej = (df['EURJPY_z_ret'].abs() > 2.0)
    widen_ej = df['EURJPY_spread_widen'].astype(bool)
    event_ej = big_move_ej & widen_ej

    big_move_gj = (df['GBPJPY_z_ret'].abs() > 2.0)
    widen_gj = df['GBPJPY_spread_widen'].astype(bool)
    event_gj = big_move_gj & widen_gj

    # EURUSD: adjusted z_ret threshold because scale differs (0.0001 units)
    big_move_eur = (df['EURUSD_z_ret'].abs() > 1.5)
    widen_eur = df['EURUSD_spread_widen'].astype(bool)
    event_eur = big_move_eur & widen_eur

    overall_med_spread = df['EURJPY_med_spread'].median()

    # ------------------------------------------------------------------
    # A) BY PAIR
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("A) BY PAIR")
    print("-" * 70)
    print(f"  {'Pair':<10s} {'n_events':>9s} {'fwd5_mean':>10s} {'fwd5_pos%':>9s} {'fwd15_mean':>11s} {'fwd15_pos%':>10s} {'post_vol':>9s}")
    print(f"  {'-'*68}")

    for label, mask, pair in [
        ("EURJPY", event_ej, 'EURJPY'),
        ("GBPJPY", event_gj, 'GBPJPY'),
        ("EURUSD", event_eur, 'EURUSD'),
    ]:
        n = mask.sum()
        f5 = df.loc[mask, f'{pair}_fwd5_ret'].dropna()
        f15 = df.loc[mask, f'{pair}_fwd15_ret'].dropna()
        pv = df.loc[mask, f'{pair}_fwd5_vol'].dropna()
        print(f"  {label:<10s} {n:>9,d} {f5.mean():>+10.3f} {(f5 > 0).mean():>9.1%} {f15.mean():>+11.3f} {(f15 > 0).mean():>10.1%} {pv.mean():>9.3f}")

    # ------------------------------------------------------------------
    # B) BY SESSION (EURJPY)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("B) BY SESSION (EURJPY events)")
    print("-" * 70)
    print(f"  {'Session':<15s} {'n_events':>9s} {'fwd5_mean':>10s} {'fwd5_pos%':>9s} {'fwd15_mean':>11s} {'fwd15_pos%':>10s} {'post_vol':>9s}")
    print(f"  {'-'*68}")

    for slabel, smask in [
        ("Asia (0-6)", (df['hour_utc'] >= 0) & (df['hour_utc'] <= 6)),
        ("London (7-15)", (df['hour_utc'] >= 7) & (df['hour_utc'] <= 15)),
        ("NY (16-23)", (df['hour_utc'] >= 16) & (df['hour_utc'] <= 23)),
    ]:
        mask = event_ej & smask
        n = mask.sum()
        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()
        pv = df.loc[mask, 'EURJPY_fwd5_vol'].dropna()
        pad = "" if n >= 3 else " (low n)"
        print(f"  {slabel:<15s} {n:>9,d}{pad} {f5.mean():>+10.3f} {(f5 > 0).mean():>9.1%} {f15.mean():>+11.3f} {(f15 > 0).mean():>10.1%} {pv.mean():>9.3f}")

    # ------------------------------------------------------------------
    # C) BY SPREAD MAGNITUDE (EURJPY)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("C) BY SPREAD MAGNITUDE (EURJPY events)")
    print("-" * 70)
    print(f"  {'Magnitude':<18s} {'n_events':>9s} {'fwd5_mean':>10s} {'fwd5_pos%':>9s} {'fwd15_mean':>11s} {'fwd15_pos%':>10s} {'post_vol':>9s}")
    print(f"  {'-'*68}")

    spread_ratio = df['EURJPY_med_spread'] / overall_med_spread
    for slabel, smask in [
        ("3-5x median", (spread_ratio >= 3) & (spread_ratio < 5)),
        ("5-10x median", (spread_ratio >= 5) & (spread_ratio < 10)),
        ("10x+ median", spread_ratio >= 10),
    ]:
        mask = event_ej & smask
        n = mask.sum()
        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()
        pv = df.loc[mask, 'EURJPY_fwd5_vol'].dropna()
        pad = "" if n >= 3 else " (low n)"
        print(f"  {slabel:<18s} {n:>9,d}{pad} {f5.mean():>+10.3f} {(f5 > 0).mean():>9.1%} {f15.mean():>+11.3f} {(f15 > 0).mean():>10.1%} {pv.mean():>9.3f}")

    # ------------------------------------------------------------------
    # D) BY PRICE MAGNITUDE (EURJPY)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("D) BY PRICE MAGNITUDE (EURJPY events)")
    print("-" * 70)
    print(f"  {'|z_ret| range':<18s} {'n_events':>9s} {'fwd5_mean':>10s} {'fwd5_pos%':>9s} {'fwd15_mean':>11s} {'fwd15_pos%':>10s} {'post_vol':>9s}")
    print(f"  {'-'*68}")

    z_abs = df['EURJPY_z_ret'].abs()
    for plabel, pmask in [
        ("2-3 sigma", (z_abs >= 2) & (z_abs < 3)),
        ("3-5 sigma", (z_abs >= 3) & (z_abs < 5)),
        ("5+ sigma", z_abs >= 5),
    ]:
        mask = event_ej & pmask
        n = mask.sum()
        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()
        pv = df.loc[mask, 'EURJPY_fwd5_vol'].dropna()
        pad = "" if n >= 3 else " (low n)"
        print(f"  {plabel:<18s} {n:>9,d}{pad} {f5.mean():>+10.3f} {(f5 > 0).mean():>9.1%} {f15.mean():>+11.3f} {(f15 > 0).mean():>10.1%} {pv.mean():>9.3f}")

    # ------------------------------------------------------------------
    # E) CONTROL
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("E) CONTROL (EURJPY) — large move +/- spread stress")
    print("-" * 70)
    print(f"  {'Condition':<25s} {'n_events':>9s} {'fwd5_mean':>10s} {'fwd5_pos%':>9s} {'fwd15_mean':>11s} {'fwd15_pos%':>10s} {'post_vol':>9s}")
    print(f"  {'-'*68}")

    big_move_only = big_move_ej & (~widen_ej)
    for clabel, cmask in [("With spread stress", event_ej), ("Without spread stress", big_move_only)]:
        n = cmask.sum()
        f5 = df.loc[cmask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[cmask, 'EURJPY_fwd15_ret'].dropna()
        pv = df.loc[cmask, 'EURJPY_fwd5_vol'].dropna()
        print(f"  {clabel:<25s} {n:>9,d} {f5.mean():>+10.3f} {(f5 > 0).mean():>9.1%} {f15.mean():>+11.3f} {(f15 > 0).mean():>10.1%} {pv.mean():>9.3f}")

    print("\n" + "=" * 70)
    print("TEST 16 COMPLETE")
    print("=" * 70)


# ====================================================================
# TEST 17: DEALER CAPITULATION MECHANISM
# ====================================================================
def test_dealer_capitulation_mechanism(df):
    print("\n" + "=" * 70)
    print("TEST 17: DEALER CAPITULATION MECHANISM")
    print("=" * 70)

    n_trading_days = 66
    pair = 'EURJPY'

    big_move = (df[f'{pair}_z_ret'].abs() > 2.0)
    widen = df[f'{pair}_spread_widen'].astype(bool)
    event = big_move & widen

    qai_p90 = df[f'{pair}_med_qai'].quantile(0.9)
    tick_q25 = df[f'{pair}_tick_count'].quantile(0.25)

    qai_spike = (df[f'{pair}_med_qai'] > qai_p90)
    tick_collapse = (df[f'{pair}_tick_count'] < tick_q25)

    buckets = [
        ("A) Baseline (capitulation only)", event & (~qai_spike) & (~tick_collapse)),
        ("B) + QAI spike (not collapsed)", event & qai_spike & (~tick_collapse)),
        ("C) + tick collapse (not QAI)", event & (~qai_spike) & tick_collapse),
        ("D) + QAI spike + tick collapse (extreme)", event & qai_spike & tick_collapse),
    ]

    # ------------------------------------------------------------------
    # Bucket analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("BUCKET ANALYSIS")
    print("-" * 70)

    header = (f"  {'Bucket':<40s} {'n':>6s} {'fwd5_m':>8s} {'fwd5_pos':>8s} "
              f"{'fwd15_m':>8s} {'fwd15_pos':>8s} {'fwd30_m':>8s} {'fwd30_pos':>8s} "
              f"{'adv_exc':>8s} {'fav_exc':>8s} {'exc_r':>6s} {'tpd':>5s}")
    print(header)
    print("  " + "-" * 120)

    results = {}

    for blabel, bmask in buckets:
        n = bmask.sum()
        if n < 5:
            print(f"  {blabel:<40s} {n:>6,d}  (< 5, skipping)")
            continue

        f5 = df.loc[bmask, f'{pair}_fwd5_ret'].dropna()
        f15 = df.loc[bmask, f'{pair}_fwd15_ret'].dropna()
        f30 = df.loc[bmask, f'{pair}_fwd30_ret'].dropna()

        adv, fav = compute_excursions(df, bmask, pair, n_bars=15)
        exc_r = fav / max(abs(adv), 0.001)

        print(f"  {blabel:<40s} {n:>6,d} {f5.mean():>+8.3f} {(f5>0).mean():>8.1%} "
              f"{f15.mean():>+8.3f} {(f15>0).mean():>8.1%} {f30.mean():>+8.3f} {(f30>0).mean():>8.1%} "
              f"{adv:>+8.3f} {fav:>+8.3f} {exc_r:>6.2f} {n/n_trading_days:>5.2f}")

        results[blabel] = {
            'n': n,
            'fwd5_mean': float(f5.mean()), 'fwd5_pos': float((f5 > 0).mean()),
            'fwd15_mean': float(f15.mean()), 'fwd15_pos': float((f15 > 0).mean()),
            'fwd30_mean': float(f30.mean()), 'fwd30_pos': float((f30 > 0).mean()),
            'adverse': float(adv), 'favorable': float(fav), 'exc_ratio': float(exc_r),
            'trades_per_day': float(n / n_trading_days),
        }

    # ------------------------------------------------------------------
    # QAI adaptive exit on baseline (bucket A)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("QAI ADAPTIVE EXIT ON DEALER CAPITULATION")
    print("(Baseline bucket A: with spread stress, no qai spike, no tick collapse)")
    print("-" * 70)

    baseline_mask = event & (~qai_spike) & (~tick_collapse)
    n_base = baseline_mask.sum()
    print(f"\n  Baseline events: {n_base}")

    if n_base >= 5:
        qai_series = df[f'{pair}_med_qai']
        qai_diff = qai_series.diff()
        qai_med_delta = qai_diff.median()
        print(f"  QAI bar-to-bar delta median (aggregate): {qai_med_delta:.4f}")

        ret_col = f'{pair}_ret_pips'
        event_indices = np.where(baseline_mask.values)[0]

        fixed_rets = []
        adaptive_rets = []
        fixed_durs = []
        adaptive_durs = []
        fixed_adv_list = []
        adaptive_adv_list = []

        for idx in event_indices:
            entry_dir = np.sign(df[ret_col].iloc[idx])
            if np.isnan(entry_dir) or entry_dir == 0:
                continue

            # --- Fixed 15-bar ---
            end_fixed = min(len(df), idx + 15)
            fwd15 = df[f'{pair}_fwd15_ret'].iloc[idx]
            if not np.isnan(fwd15):
                pnl = entry_dir * fwd15
                fixed_rets.append(pnl)
                fixed_durs.append(15)
                # adverse excursion for fixed hold
                seg = df[ret_col].iloc[idx+1:end_fixed].dropna().values
                if len(seg) >= 2:
                    cum = np.cumsum(seg)
                    adv_seg = min(cum) if entry_dir > 0 else -max(cum)
                else:
                    adv_seg = 0.0
                fixed_adv_list.append(adv_seg)

            # --- QAI adaptive exit ---
            exit_offset = None
            for offset in range(1, 16):
                bp = idx + offset
                if bp >= len(df):
                    break
                dq = qai_diff.iloc[bp]
                if np.isnan(dq):
                    continue
                if dq > qai_med_delta:
                    exit_offset = offset
                    break

            if exit_offset is not None:
                hold = exit_offset
                end_a = min(len(df), idx + hold + 1)
                cum_ret = df[ret_col].iloc[idx+1:end_a].sum()
            else:
                hold = 15
                end_a = min(len(df), idx + 16)
                cum_ret = df[ret_col].iloc[idx+1:end_a].sum()

            adaptive_rets.append(entry_dir * cum_ret)
            adaptive_durs.append(hold)
            # adverse excursion for adaptive hold
            seg_a = df[ret_col].iloc[idx+1:min(len(df), idx+hold+1)].dropna().values
            if len(seg_a) >= 2:
                cum_a = np.cumsum(seg_a)
                adv_seg_a = min(cum_a) if entry_dir > 0 else -max(cum_a)
            else:
                adv_seg_a = 0.0
            adaptive_adv_list.append(adv_seg_a)

        fixed_rets = np.array(fixed_rets)
        adaptive_rets = np.array(adaptive_rets)
        fixed_adv = np.array(fixed_adv_list)
        adaptive_adv = np.array(adaptive_adv_list)

        print(f"\n  {'Exit strategy':<25s} {'n_trades':>9s} {'WR':>8s} {'avg_ret':>10s} {'avg_dur':>8s} {'avg_adv_exc':>12s}")
        print(f"  {'-'*72}")
        print(f"  {'Fixed 15-bar':<25s} {len(fixed_rets):>9,d} {(fixed_rets>0).mean():>8.1%} {fixed_rets.mean():>+10.3f} {np.mean(fixed_durs):>8.1f} {np.mean(fixed_adv):>+12.3f}")
        print(f"  {'QAI adaptive':<25s} {len(adaptive_rets):>9,d} {(adaptive_rets>0).mean():>8.1%} {adaptive_rets.mean():>+10.3f} {np.mean(adaptive_durs):>8.1f} {np.mean(adaptive_adv):>+12.3f}")

        if len(fixed_rets) >= 5 and len(adaptive_rets) >= 5:
            min_len = min(len(fixed_rets), len(adaptive_rets))
            try:
                t_stat, p_val = scipy_stats.ttest_rel(fixed_rets[:min_len], adaptive_rets[:min_len])
                print(f"\n  Paired t-test (fixed vs adaptive): t={t_stat:.3f}  p={p_val:.4f}")
            except Exception as e:
                print(f"\n  t-test skipped: {e}")

    print("\n" + "=" * 70)
    print("TEST 17 COMPLETE")
    print("=" * 70)
    return results


def compute_excursions(df, mask, pair, n_bars=15):
    """Compute mean adverse and favorable excursion within n_bars after each event."""
    adv_list = []
    fav_list = []
    ret_col = f'{pair}_ret_pips'
    idx_positions = np.where(mask.values)[0]
    for idx in idx_positions:
        end = min(len(df), idx + n_bars)
        seg = df[ret_col].iloc[idx+1:end].dropna().values
        if len(seg) < 2:
            continue
        cum = np.cumsum(seg)
        entry_dir = np.sign(df[ret_col].iloc[idx])
        if np.isnan(entry_dir) or entry_dir == 0:
            continue
        if entry_dir > 0:
            adv_list.append(min(cum))
            fav_list.append(max(cum))
        else:
            adv_list.append(-max(cum))
            fav_list.append(-min(cum))
    mean_adv = np.mean(adv_list) if adv_list else 0.0
    mean_fav = np.mean(fav_list) if fav_list else 0.0
    return mean_adv, mean_fav


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_main = time.time()

    tick_data = load_ticks()
    m1_dict = build_m1(tick_data)
    df = align_and_enrich(m1_dict)
    del tick_data, m1_dict

    ej_event = ((df['EURJPY_z_ret'].abs() > 2.0) & df['EURJPY_spread_widen'].astype(bool))

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"  Total bars: {len(df):,d}")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')}  to  {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  EURJPY large move + spread stress events: {ej_event.sum():,d}")
    print(f"  EURJPY median spread: {df['EURJPY_med_spread'].median():.2f}p")
    print(f"  EURJPY QAI p90:       {df['EURJPY_med_qai'].quantile(0.9):.3f}")
    print(f"  EURJPY tick count q25: {df['EURJPY_tick_count'].quantile(0.25):.0f}")
    print(f"  Trading days (approx): ~66")

    test_dealer_capitulation_validation(df)
    test_dealer_capitulation_mechanism(df)

    elapsed = time.time() - t_main
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
