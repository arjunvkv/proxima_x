"""
Market State Authentication Validation — Tests 9-14.
Reuses exact data loading and feature computation from run_market_state_auth.py
so results are comparable.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
# DATA LOADING — (copied verbatim from run_market_state_auth.py)
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
# M1 BAR BUILDING — (copied verbatim from run_market_state_auth.py)
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
        bars['fwd5_vol'] = bars['ret_pips'].rolling(5).std().shift(-1).rolling(5).mean().shift(-5)

        for col in bars.columns:
            bars = bars.rename(columns={col: f'{pair}_{col}'})

        m1_dict[pair] = bars
        elapsed = time.time() - t0
        print(f"  {pair}: {len(bars):,d} M1 bars ({elapsed:.1f}s)")

    return m1_dict


# ====================================================================
# ALIGNMENT & CROSS-PAIR FEATURES — (copied verbatim from run_market_state_auth.py)
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

    df['normal_spread'] = (~df['EURJPY_spread_widen'] &
                           ~df['GBPJPY_spread_widen'] &
                           ~df['EURUSD_spread_widen']).astype(int)
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
# TEST 9: INDEPENDENCE TEST
# ====================================================================
def test_independence(df):
    print("\n" + "=" * 70)
    print("TEST 9: INDEPENDENCE TEST")
    print("=" * 70)

    features = ['agreement', 'normal_spread', 'high_participation',
                'EURJPY_med_qai', 'EURJPY_tick_accel', 'dispersion', 'EURJPY_z_ret']
    labels  = ['agreement', 'normal_spread', 'high_particip',
               'EURJPY_med_qai', 'EURJPY_tick_accel', 'dispersion', 'EURJPY_z_ret']
    feat_df = df[features].dropna()

    corr = feat_df.corr(method='pearson')
    print("\n  Pearson Correlation Matrix:")
    header = ''.join(f'{lab:>13s}' for lab in labels)
    print(f"  {'':>20s}{header}")
    for i, f1 in enumerate(features):
        row = f'{labels[i]:>20s}'
        for j, f2 in enumerate(features):
            row += f'{corr.loc[f1, f2]:>13.3f}'
        print(row)

    triu = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
    max_corr = np.max(np.abs(corr.values[triu]))
    print(f"\n  Max absolute off-diagonal correlation: {max_corr:.4f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df)
    pca = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cum_var, 0.90) + 1)

    print(f"\n  PCA Explained Variance:")
    for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cum_var)):
        print(f"    PC{i+1}: {ev:.3f}  (cumulative: {cv:.3f})")
    print(f"  Components to explain >90% variance: {n_components_90}")

    if n_components_90 == 1:
        status = "FAIL"
        msg = f"FAIL — single 'market activity' factor (PC1 = {pca.explained_variance_ratio_[0]:.3f})"
    elif max_corr > 0.5:
        status = "FAIL"
        msg = f"FAIL — layers not independent (max corr = {max_corr:.2f} > 0.5)"
    elif max_corr > 0.3:
        status = "INCONCLUSIVE"
        msg = f"INCONCLUSIVE — moderate correlations (max corr = {max_corr:.2f})"
    else:
        status = "PASS"
        msg = f"PASS — all |correlations| < 0.3 (max = {max_corr:.2f})"

    print(f"\n  \u2192 {msg}")

    # List correlations > 0.3 for detail
    high_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr.values[i, j]) > 0.3:
                high_pairs.append(f"{labels[i]} vs {labels[j]} = {corr.values[i, j]:.3f}")
    if high_pairs:
        print(f"  Correlations > 0.3:")
        for hp in high_pairs:
            print(f"    {hp}")

    return {'status': status, 'max_corr': float(max_corr), 'n_components_90': n_components_90}


# ====================================================================
# TEST 10: DEALER CAPITULATION CONTROL
# ====================================================================
def test_dealer_capitulation_control(df):
    print("\n" + "=" * 70)
    print("TEST 10: DEALER CAPITULATION CONTROL")
    print("=" * 70)

    big_move = df['EURJPY_z_ret'].abs() > 2.0
    bucket_a = big_move & df['EURJPY_spread_widen'].astype(bool)
    bucket_b = big_move & ~df['EURJPY_spread_widen'].astype(bool)

    results = {}
    for label, mask in [("Bucket A: Large move WITH spread stress", bucket_a),
                         ("Bucket B: Large move WITHOUT spread stress", bucket_b)]:
        n = mask.sum()
        if n < 5:
            print(f"\n  {label}: only {n} events, skipping")
            results[label] = {'n': n}
            continue

        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()
        post_vol = df.loc[mask, 'EURJPY_roll_vol_5'].dropna()

        print(f"\n  {label} (n={n}):")
        print(f"    Fwd5  mean return:   {f5.mean():+.3f}p   %pos = {(f5>0).mean():.1%}")
        print(f"    Fwd15 mean return:   {f15.mean():+.3f}p   %pos = {(f15>0).mean():.1%}")
        print(f"    Post-event vol:      {post_vol.mean():.3f}p")

        results[label] = {
            'n': n,
            'fwd5_mean': float(f5.mean()), 'fwd5_pos': float((f5>0).mean()),
            'fwd15_mean': float(f15.mean()), 'fwd15_pos': float((f15>0).mean()),
            'post_vol': float(post_vol.mean()),
        }

    if 'Bucket A: Large move WITH spread stress' in results and 'Bucket B: Large move WITHOUT spread stress' in results:
        r_a = results['Bucket A: Large move WITH spread stress']
        r_b = results['Bucket B: Large move WITHOUT spread stress']
        if r_a.get('n', 0) >= 5 and r_b.get('n', 0) >= 5:
            delta_fwd15 = r_a['fwd15_mean'] - r_b['fwd15_mean']
            delta_pos = r_a['fwd15_pos'] - r_b['fwd15_pos']
            print(f"\n  Delta (A - B) fwd15 mean: {delta_fwd15:+.3f}p")
            print(f"  Delta (A - B) fwd15 pos%: {delta_pos:+.1%}")
            results['delta_fwd15'] = float(delta_fwd15)
            results['delta_pos'] = float(delta_pos)

    # Determine PASS/FAIL — Bucket A and B should be DIFFERENT
    r_a = results.get('Bucket A: Large move WITH spread stress', {})
    r_b = results.get('Bucket B: Large move WITHOUT spread stress', {})
    if r_a.get('n', 0) >= 5 and r_b.get('n', 0) >= 5:
        fwd15_diff = abs(r_a.get('fwd15_mean', 0) - r_b.get('fwd15_mean', 0))
        pos_diff = abs(r_a.get('fwd15_pos', 0) - r_b.get('fwd15_pos', 0))
        if fwd15_diff > 0.3 or pos_diff > 0.10:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
    else:
        status = "INCONCLUSIVE"

    results['status'] = status
    print(f"\n  \u2192 {status}")
    return results


# ====================================================================
# TEST 11: EDGE AUTHENTICATION — Engine 2 (Response Deficit)
# ====================================================================
def test_edge_auth_engine2(df):
    print("\n" + "=" * 70)
    print("TEST 11: EDGE AUTHENTICATION \u2014 Engine 2 (Response Deficit)")
    print("=" * 70)

    # Response deficit: EURJPY moves > 0.5 sigma, GBPJPY hasn't moved (|z_ret| < 0.5)
    ej_moved = df['EURJPY_z_ret'].abs() > 0.5
    gj_quiet = df['GBPJPY_z_ret'].abs() < 0.5
    deficit = ej_moved & gj_quiet

    # Compute forward returns for GBPJPY
    gj_fwd5 = df['GBPJPY_fwd5_ret']
    gj_fwd15 = df['GBPJPY_fwd15_ret']

    # Baseline: all deficit events
    baseline = deficit

    # Authenticated: deficit + normal_spread + high_participation + agreement > 0.5
    authenticated = deficit & df['normal_spread'].astype(bool) & df['high_participation'].astype(bool) & (df['agreement'] > 0.5)

    # Rejected: deficit + any_spread_stress
    rejected = deficit & df['any_spread_stress'].astype(bool)

    groups = [
        ("Baseline (unfiltered)", baseline),
        ("Authenticated (deficit + 3 layers)", authenticated),
        ("Rejected (deficit + spread stress)", rejected),
    ]

    results = {}
    for label, mask in groups:
        n = mask.sum()
        if n < 3:
            print(f"\n  {label}: only {n} events, skipping")
            results[label] = {'n': n}
            continue

        f5 = gj_fwd5[mask].dropna()
        f15 = gj_fwd15[mask].dropna()

        # Catch-up rate: % where GBPJPY moves same direction as EURJPY within 15 bars
        ej_dir = np.sign(df.loc[mask, 'EURJPY_ret_pips'])
        gj_dir_f15 = np.sign(gj_fwd15[mask])
        valid_catch = ~(ej_dir.isna() | gj_dir_f15.isna())
        catch_up = (ej_dir[valid_catch] == gj_dir_f15[valid_catch]).mean() if valid_catch.sum() > 0 else 0

        # Information transmission efficiency: actual GBPJPY adjustment / EURJPY adjustment magnitude
        ej_abs = df.loc[mask, 'EURJPY_ret_pips'].abs()
        gj_abs = gj_fwd15[mask].abs()
        valid_eff = ~(ej_abs.isna() | gj_abs.isna()) & (ej_abs > 0)
        if valid_eff.sum() > 0:
            efficiency = (gj_abs[valid_eff] / ej_abs[valid_eff]).mean()
        else:
            efficiency = 0

        print(f"\n  {label} (n={n}):")
        print(f"    GBPJPY fwd5  mean return:  {f5.mean():+.3f}p   %pos = {(f5>0).mean():.1%}")
        print(f"    GBPJPY fwd15 mean return:  {f15.mean():+.3f}p   %pos = {(f15>0).mean():.1%}")
        print(f"    Catch-up rate:              {catch_up:.1%}")
        print(f"    Info transmission eff:      {efficiency:.3f}")

        results[label] = {
            'n': n,
            'fwd5_mean': float(f5.mean()), 'fwd5_pos': float((f5>0).mean()),
            'fwd15_mean': float(f15.mean()), 'fwd15_pos': float((f15>0).mean()),
            'catch_up': float(catch_up), 'efficiency': float(efficiency),
        }

    # Compute gain: Authenticated efficiency / Baseline efficiency
    r_auth = results.get('Authenticated (deficit + 3 layers)', {})
    r_base = results.get('Baseline (unfiltered)', {})
    r_rej = results.get('Rejected (deficit + spread stress)', {})

    if r_auth.get('n', 0) >= 3 and r_base.get('n', 0) >= 3 and r_rej.get('n', 0) >= 3:
        eff_gain = r_auth['efficiency'] / r_base['efficiency'] - 1 if r_base['efficiency'] > 0 else 0
        print(f"\n  Efficiency gain (Auth / Baseline): {eff_gain:+.1%}")
        print(f"  Rejected efficiency:              {r_rej['efficiency']:.3f}")
        print(f"  Authenticated catch-up:           {r_auth['catch_up']:.1%}")
        print(f"  Rejected catch-up:                {r_rej['catch_up']:.1%}")
        results['efficiency_gain'] = float(eff_gain)

        if r_auth['catch_up'] > r_base['catch_up'] and r_auth['catch_up'] > r_rej['catch_up'] and r_rej['catch_up'] < r_base['catch_up']:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
    else:
        status = "INCONCLUSIVE"

    results['status'] = status
    print(f"\n  \u2192 {status}")
    return results


# ====================================================================
# TEST 12: CAUSAL ORDERING
# ====================================================================
def test_causal_ordering(df):
    print("\n" + "=" * 70)
    print("TEST 12: CAUSAL ORDERING")
    print("=" * 70)

    # Precompute conditions for each bar
    N = len(df)
    info_cond = df['msv_residual'].abs() > 1.0
    liq_cond = df['EURJPY_spread_widen'].astype(bool)
    part_cond = df['high_participation'].astype(bool)
    price_cond = df['EURJPY_z_ret'].abs() > 1.0

    # Track spread_widen activation (transition 0→1)
    liq_activate = liq_cond.values.copy().astype(int)
    liq_activate[1:] = (liq_cond.values[1:].astype(int) - liq_cond.values[:-1].astype(int) == 1).astype(int)

    # Track high_participation activation (transition 0→1)
    part_activate = part_cond.values.copy().astype(int)
    part_activate[1:] = (part_cond.values[1:].astype(int) - part_cond.values[:-1].astype(int) == 1).astype(int)

    window = 5
    orderings = []
    fwd15_returns = []

    layer_names = ['info', 'liquidity', 'participation', 'price']
    correct_order = [0, 1, 2, 3]  # info → liquidity → participation → price

    for i in range(N - window - 15):
        triggers = {0: [], 1: [], 2: [], 3: []}  # layer -> list of bar offsets where triggered
        for j in range(window):
            idx = i + j
            if info_cond.iloc[idx]:
                triggers[0].append(j)
            if liq_activate[idx]:
                triggers[1].append(j)
            if part_activate[idx]:
                triggers[2].append(j)
            if price_cond.iloc[idx]:
                triggers[3].append(j)

        # All 4 must trigger within window
        triggered_layers = [k for k, v in triggers.items() if len(v) > 0]
        if len(triggered_layers) < 4:
            continue

        # Get first trigger bar offset for each layer
        first = {k: min(v) for k, v in triggers.items()}
        # Sort by first trigger time
        order = [k for k in sorted(first, key=first.__getitem__)]

        orderings.append(order)

        # Measure fwd15 return from bar i
        fwd15 = df['EURJPY_fwd15_ret'].iloc[i]
        if not np.isnan(fwd15):
            fwd15_returns.append(fwd15)
        else:
            fwd15_returns.append(0)

    orderings = np.array(orderings)
    fwd15_returns = np.array(fwd15_returns)
    n_total = len(orderings)

    print(f"\n  Total events with all 4 triggers in {window}-bar window: {n_total}")

    if n_total < 10:
        print("  Too few events, inconclusive")
        return {'status': 'INCONCLUSIVE', 'n_events': int(n_total)}

    # Count orderings
    correct_mask = np.all(orderings == correct_order, axis=1)
    correct_pct = correct_mask.mean()

    # Other orderings
    other_orderings = {}
    for ord_vec in orderings:
        key = tuple(ord_vec)
        other_orderings[key] = other_orderings.get(key, 0) + 1

    print(f"\n  Ordering distribution:")
    sorted_orders = sorted(other_orderings.items(), key=lambda x: -x[1])
    for ord_vec, count in sorted_orders[:10]:
        names = [layer_names[o] for o in ord_vec]
        pct = count / n_total * 100
        avg_fwd = fwd15_returns[np.all(orderings == ord_vec, axis=1)].mean()
        arrow = ' -> '.join(names)
        print(f"    {arrow:>55s}  {pct:5.1f}%  (fwd15={avg_fwd:+.3f}p)")

    # Measure forward returns per ordering type
    correct_fwd15 = fwd15_returns[correct_mask].mean() if correct_mask.sum() > 0 else 0
    incorrect_fwd15 = fwd15_returns[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

    arrow_sep = ' \u2192 '
    print(f"\n  Correct ordering ({arrow_sep.join(layer_names)}):  {correct_pct:.1%}")
    print(f"    Fwd15 mean return: {correct_fwd15:+.3f}p")
    print(f"  Other orderings combined:")
    print(f"    Fwd15 mean return: {incorrect_fwd15:+.3f}p")

    if correct_fwd15 > incorrect_fwd15 and correct_pct > 0.1:
        status = "PASS"
    elif correct_pct > 0.05:
        status = "INCONCLUSIVE"
    else:
        status = "FAIL"

    results = {
        'status': status,
        'n_events': int(n_total),
        'correct_pct': float(correct_pct),
        'correct_fwd15': float(correct_fwd15),
        'incorrect_fwd15': float(incorrect_fwd15),
    }
    print(f"\n  \u2192 {status}")
    return results


# ====================================================================
# TEST 13: ADVERSARIAL FAILURE TEST
# ====================================================================
def test_adversarial_failure(df):
    print("\n" + "=" * 70)
    print("TEST 13: ADVERSARIAL FAILURE TEST")
    print("=" * 70)

    # False positives: authenticated == 1 but fwd5 reverses (opposite sign from current bar's return)
    auth = df['authenticated'].astype(bool)
    ej_ret = df['EURJPY_ret_pips']
    ej_fwd5 = df['EURJPY_fwd5_ret']
    same_dir = (np.sign(ej_ret) == np.sign(ej_fwd5))
    false_pos = auth & ~same_dir
    false_pos = false_pos & ej_ret.notna() & ej_fwd5.notna()

    fp_mask = false_pos.fillna(False).astype(bool)
    fp_indices = np.where(fp_mask.values)[0]
    n_fp = len(fp_indices)
    n_auth = auth.sum()
    fp_rate = n_fp / n_auth if n_auth > 0 else 0

    print(f"\n  Total authenticated events:    {n_auth}")
    print(f"  Total false positives:          {n_fp}")
    print(f"  False positive rate:            {fp_rate:.2%}")

    if n_fp < 5:
        print("  Too few false positives for detailed analysis")
        results = {'status': 'INCONCLUSIVE', 'n_fp': int(n_fp), 'fp_rate': float(fp_rate),
                   'n_auth': int(n_auth)}
        print(f"\n  \u2192 INCONCLUSIVE")
        return results

    N = len(df)

    # For each false positive, check various conditions
    near_session = 0
    tick_collapse = 0
    spread_after = 0
    qai_spike = 0

    tick_median = df['EURJPY_tick_count'].median()
    qai_median = df['EURJPY_med_qai'].median()

    for idx in fp_indices:
        # Near session boundary (within 5 bars of Tokyo/London/NY open)
        window_start = max(0, idx - 5)
        window_end = min(N, idx + 5)
        near = (df['is_tokyo_h0'].iloc[window_start:window_end+1].any() |
                df['is_london_h0'].iloc[window_start:window_end+1].any() |
                df['is_ny_h0'].iloc[window_start:window_end+1].any())
        if near:
            near_session += 1

        # Tick activity collapses after auth (bars idx+1 to idx+5)
        post_end = min(N, idx + 6)
        post_tick = df['EURJPY_tick_count'].iloc[idx+1:post_end]
        if len(post_tick) > 0 and (post_tick < tick_median).any():
            tick_collapse += 1

        # Spread stress increases after auth
        post_spread = df['EURJPY_spread_widen'].iloc[idx+1:post_end]
        if len(post_spread) > 0 and post_spread.any():
            spread_after += 1

        # Quote asymmetry spikes after auth
        post_qai = df['EURJPY_med_qai'].iloc[idx+1:post_end]
        if len(post_qai) > 0 and (post_qai > qai_median).any():
            qai_spike += 1

    pct_near = near_session / n_fp
    pct_tick = tick_collapse / n_fp
    pct_spread = spread_after / n_fp
    pct_qai = qai_spike / n_fp

    print(f"\n  False Positive Decomposition:")
    print(f"    % near session boundary:      {pct_near:.1%}  ({near_session}/{n_fp})")
    print(f"    % tick count < median after:  {pct_tick:.1%}  ({tick_collapse}/{n_fp})")
    print(f"    % spread_widen after auth:    {pct_spread:.1%}  ({spread_after}/{n_fp})")
    print(f"    % qai spike after auth:       {pct_qai:.1%}  ({qai_spike}/{n_fp})")

    predicted_by_session_or_collapse = 0
    for idx in fp_indices:
        window_start = max(0, idx - 5)
        window_end = min(N, idx + 5)
        near = (df['is_tokyo_h0'].iloc[window_start:window_end+1].any() |
                df['is_london_h0'].iloc[window_start:window_end+1].any() |
                df['is_ny_h0'].iloc[window_start:window_end+1].any())
        post_end = min(N, idx + 6)
        post_tick = df['EURJPY_tick_count'].iloc[idx+1:post_end]
        tick_down = len(post_tick) > 0 and (post_tick < tick_median).any()
        if near or tick_down:
            predicted_by_session_or_collapse += 1
    pct_predicted = predicted_by_session_or_collapse / n_fp if n_fp > 0 else 0

    print(f"\n    % explained (session OR tick collapse): {pct_predicted:.1%}")

    if pct_predicted > 0.6:
        status = "PASS"
    elif pct_predicted > 0.3:
        status = "INCONCLUSIVE"
    else:
        status = "FAIL"

    results = {
        'status': status,
        'n_fp': int(n_fp), 'fp_rate': float(fp_rate),
        'n_auth': int(n_auth),
        'pct_near_session': float(pct_near),
        'pct_tick_collapse': float(pct_tick),
        'pct_spread_after': float(pct_spread),
        'pct_qai_spike': float(pct_qai),
        'pct_predicted': float(pct_predicted),
    }
    print(f"\n  \u2192 {status}")
    return results


# ====================================================================
# TEST 14: ABLATION TEST
# ====================================================================
def test_ablation(df):
    print("\n" + "=" * 70)
    print("TEST 14: ABLATION TEST")
    print("=" * 70)

    big_move = df['EURJPY_z_ret'].abs() > 2.0

    models = {
        "Model A: high_participation only": big_move & df['high_participation'].astype(bool),
        "Model B: + normal_spread": big_move & df['high_participation'].astype(bool) & df['normal_spread'].astype(bool),
        "Model C: + agreement > 0.5 (full)": big_move & df['high_participation'].astype(bool) & df['normal_spread'].astype(bool) & (df['agreement'] > 0.5),
    }

    results = {}
    for label, mask in models.items():
        n = mask.sum()
        if n < 5:
            print(f"\n  {label}: only {n} events, skipping")
            results[label] = {'n': n}
            continue

        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()
        ej_ret = df.loc[mask, 'EURJPY_ret_pips'].dropna()
        f15_aligned = df.loc[mask, 'EURJPY_fwd15_ret']
        common = ej_ret.index.intersection(f15_aligned.dropna().index)
        persistence = (np.sign(ej_ret.loc[common]) == np.sign(f15_aligned.loc[common])).mean() if len(common) > 0 else 0

        print(f"\n  {label} (n={n}):")
        print(f"    Fwd15 mean return:  {f15.mean():+.3f}p")
        print(f"    Fwd15 %% positive:   {(f15>0).mean():.1%}")
        print(f"    Persistence rate:   {persistence:.1%}")

        results[label] = {
            'n': n,
            'fwd15_mean': float(f15.mean()),
            'fwd15_pos': float((f15>0).mean()),
            'persistence': float(persistence),
        }

    r_c = results.get("Model C: + agreement > 0.5 (full)", {})
    r_b = results.get("Model B: + normal_spread", {})
    r_a = results.get("Model A: high_participation only", {})

    if r_c.get('n', 0) >= 5 and r_b.get('n', 0) >= 5 and r_a.get('n', 0) >= 5:
        print(f"\n  Persistence comparison:")
        print(f"    Model A (ticks only):          {r_a['persistence']:.1%}")
        print(f"    Model B (+ spread):            {r_b['persistence']:.1%}")
        print(f"    Model C (+ agreement, full):   {r_c['persistence']:.1%}")

        if r_c['persistence'] >= r_b['persistence'] >= r_a['persistence']:
            status = "PASS"
            print(f"  C >= B >= A \u2714 \u2014 each layer adds value")
        elif r_c['persistence'] >= r_a['persistence'] and r_b['persistence'] >= r_a['persistence']:
            if r_c['persistence'] > r_b['persistence']:
                status = "PASS"
                print(f"  C > B, B > A \u2714")
            elif r_c['persistence'] == r_b['persistence']:
                status = "INCONCLUSIVE"
                print(f"  C == B \u2014 MSV agreement adds nothing beyond activity + liquidity")
            else:
                status = "INCONCLUSIVE"
                print(f"  C < B \u2014 agreement layer reduces persistence")
        else:
            status = "INCONCLUSIVE"
            print(f"  Unexpected ordering")
    else:
        status = "INCONCLUSIVE"
        print(f"\n  Insufficient events for comparison")

    results['status'] = status
    print(f"\n  \u2192 {status}")
    return results


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_main = time.time()

    tick_data = load_ticks()
    m1_dict = build_m1(tick_data)
    df = align_and_enrich(m1_dict)
    del tick_data, m1_dict

    N = len(df)
    stats = df[['EURJPY_ret_pips', 'EURJPY_med_spread', 'EURJPY_tick_count',
                'agreement', 'dispersion', 'authenticated']].describe()
    print("\n  Key stats:")
    print(f"    EURJPY ret_pips: mean={stats.loc['mean','EURJPY_ret_pips']:+.3f} "
          f"std={stats.loc['std','EURJPY_ret_pips']:.3f}")
    print(f"    EURJPY med_spread: mean={stats.loc['mean','EURJPY_med_spread']:.2f}p "
          f"median={df['EURJPY_med_spread'].median():.2f}p")
    print(f"    EURJPY tick_count: mean={stats.loc['mean','EURJPY_tick_count']:.0f} "
          f"median={df['EURJPY_tick_count'].median():.0f}")
    print(f"    Agreement: mean={stats.loc['mean','agreement']:.3f}")
    print(f"    Dispersion: mean={stats.loc['mean','dispersion']:.3f}")
    print(f"    Authenticated: {df['authenticated'].mean():.1%}")
    print(f"    Data range: {df.index[0]} to {df.index[-1]} ({N:,d} bars)")

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION TESTS 9-14")
    print("=" * 70)

    all_results = {}

    print("\n" + "\u2500" * 70)
    all_results['test_9'] = test_independence(df)

    print("\n" + "\u2500" * 70)
    all_results['test_10'] = test_dealer_capitulation_control(df)

    print("\n" + "\u2500" * 70)
    all_results['test_11'] = test_edge_auth_engine2(df)

    print("\n" + "\u2500" * 70)
    all_results['test_12'] = test_causal_ordering(df)

    print("\n" + "\u2500" * 70)
    all_results['test_13'] = test_adversarial_failure(df)

    print("\n" + "\u2500" * 70)
    all_results['test_14'] = test_ablation(df)

    elapsed = time.time() - t_main

    # Summary table
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (Tests 9-14)")
    print("=" * 70)
    print(f"{'TEST':<25s}  {'RESULT':<15s}  KEY METRIC")
    print("\u2500" * 70)

    summary_rows = [
        ("9. Independence", all_results['test_9']),
        ("10. Dealer Control", all_results['test_10']),
        ("11. Edge Auth", all_results['test_11']),
        ("12. Causal Ordering", all_results['test_12']),
        ("13. Adversarial", all_results['test_13']),
        ("14. Ablation", all_results['test_14']),
    ]

    for name, res in summary_rows:
        status = res.get('status', 'ERROR')
        metric = ""
        if 'max_corr' in res:
            metric = f"max corr = {res['max_corr']:.2f}"
        elif 'delta_fwd15' in res:
            metric = f"delta = {res['delta_fwd15']:+.2f}p"
        elif 'efficiency_gain' in res:
            metric = f"eff gain = {res['efficiency_gain']:+.1%}"
        elif 'correct_pct' in res:
            metric = f"correct = {res['correct_pct']:.1%}"
        elif 'pct_predicted' in res:
            metric = f"predicted = {res['pct_predicted']:.1%}"
        elif res.get('n', 0) > 0:
            status_val = res.get('persistence', 0)
            if status_val:
                metric = f"C > B > A persistence"
        print(f"{name:<25s}  {status:<15s}  {metric}")

    print("\n" + "\u2500" * 70)
    passes = sum(1 for v in all_results.values() if v.get('status') == 'PASS')
    fails = sum(1 for v in all_results.values() if v.get('status') == 'FAIL')
    inconclusive = sum(1 for v in all_results.values() if v.get('status') == 'INCONCLUSIVE')
    print(f"  PASS: {passes}  |  FAIL: {fails}  |  INCONCLUSIVE: {inconclusive}")
    print(f"  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == '__main__':
    main()
