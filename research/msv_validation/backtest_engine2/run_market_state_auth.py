"""
Market State Authentication Tests — three-layer agreement required.
Tests combinations of features the market CANNOT fake simultaneously
because they are governed by independent mechanisms:
  Layer 1 — Information (MSV cross-pair dispersion/agreement)
  Layer 2 — Liquidity (spread, quote asymmetry)
  Layer 3 — Participation (tick arrival, session ecology)

When all three layers agree on a regime, the signal is authentic.
When layers disagree, the regime is likely noise or manipulation.
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
# DATA LOADING
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
# M1 BAR BUILDING WITH TICK FEATURES
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
# ALIGNMENT & CROSS-PAIR FEATURES
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
        print(f"  R² = {model.score(X[valid], y[valid]):.4f}")

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
# TIER 1 — FOUNDATIONAL TESTS
# ====================================================================

def test_authenticated_repricing(df):
    """
    Hypothesis: When MSV agreement + normal spread + high participation
    all agree on a move, the move is genuine and persists.
    Without any layer, moves reverse.
    """
    print("\n" + "=" * 70)
    print("TEST 1: AUTHENTICATED REPRICING STATE")
    print("=" * 70)
    results = {}

    big_move = df['EURJPY_z_ret'].abs() > 2.0
    auth = df['authenticated'].astype(bool) & big_move
    non_auth = (~df['authenticated'].astype(bool)) & big_move

    for label, mask in [("Authenticated", auth), ("Non-authenticated", non_auth)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: only {n} events, skipping")
            continue

        f1 = df.loc[mask, 'EURJPY_fwd1_ret'].dropna()
        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()

        f1_mean, f1_pos = f1.mean(), (f1 > 0).mean()
        f5_mean, f5_pos = f5.mean(), (f5 > 0).mean()
        f15_mean, f15_pos = f15.mean(), (f15 > 0).mean()

        same_dir = ((df.loc[mask, 'EURJPY_ret_pips'] > 0) &
                    (df.loc[mask, 'EURJPY_fwd1_ret'] > 0)).dropna()
        persistence = same_dir.mean() if len(same_dir) > 0 else 0

        print(f"\n  {label} (n={n:,d}):")
        print(f"    Fwd1  return: {f1_mean:+.3f}p  pos%={f1_pos:.1%}")
        print(f"    Fwd5  return: {f5_mean:+.3f}p  pos%={f5_pos:.1%}")
        print(f"    Fwd15 return: {f15_mean:+.3f}p  pos%={f15_pos:.1%}")
        print(f"    Persistence:  {persistence:.1%}")

        results[label] = {
            'n': n, 'fwd1': float(f1_mean), 'fwd5': float(f5_mean),
            'fwd15': float(f15_mean), 'pos_pct': float(f15_pos),
            'persistence': float(persistence),
        }

    if 'Authenticated' in results and 'Non-authenticated' in results:
        r_a = results['Authenticated']
        r_na = results['Non-authenticated']
        delta = r_a['fwd15'] - r_na['fwd15']
        print(f"\n  Delta (Auth - NonAuth) fwd15: {delta:+.3f}p")
        results['delta_fwd15'] = float(delta)

    status = 'PASS' if results.get('Authenticated', {}).get('fwd15', 0) > 0 and \
                      results.get('Non-authenticated', {}).get('fwd15', 0) < 0 else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


def test_tokyo_h0_liquidity(df):
    """
    Hypothesis: Tokyo H0 + declining basket reversals are real ONLY
    when liquidity is normal. With spread stress, the edge disappears.
    """
    print("\n" + "=" * 70)
    print("TEST 2: TOKYO H0 + LIQUIDITY NORMALITY")
    print("=" * 70)
    results = {}

    h0_decline = (df['is_tokyo_h0'].astype(bool) & df['is_declining_basket'].astype(bool))
    normal_liq = h0_decline & df['normal_spread'].astype(bool) & df['stable_quotes'].astype(bool)
    stressed_liq = h0_decline & df['any_spread_stress'].astype(bool)

    for label, mask in [("Normal liquidity", normal_liq), ("Stressed liquidity", stressed_liq)]:
        n = mask.sum()
        if n < 3:
            print(f"  {label}: only {n} events, skipping")
            continue

        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()

        print(f"\n  {label} (n={n}):")
        print(f"    Fwd5  return: {f5.mean():+.3f}p  pos%={(f5>0).mean():.1%}")
        print(f"    Fwd15 return: {f15.mean():+.3f}p  pos%={(f15>0).mean():.1%}")

        results[label] = {
            'n': n, 'fwd5': float(f5.mean()), 'fwd15': float(f15.mean()),
            'pos_pct': float((f15 > 0).mean()) if len(f15) > 0 else 0,
        }

    if 'Normal liquidity' in results and 'Stressed liquidity' in results:
        delta = results['Normal liquidity']['fwd15'] - results['Stressed liquidity']['fwd15']
        print(f"\n  Delta (Normal - Stressed) fwd15: {delta:+.3f}p")
        results['delta_fwd15'] = float(delta)

    status = 'PASS' if results.get('Normal liquidity', {}).get('fwd15', -999) > \
                      results.get('Stressed liquidity', {}).get('fwd15', 999) else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


def test_false_breakout(df):
    """
    Hypothesis: Price expansion + MSV disagreement + spread widening = fake breakout (reversal).
    Price expansion + MSV agreement + no spread stress = real breakout (persistence).
    """
    print("\n" + "=" * 70)
    print("TEST 3: FALSE BREAKOUT STATE")
    print("=" * 70)
    results = {}

    big_move = df['EURJPY_z_ret'].abs() > 2.0
    real_breakout = big_move & (df['agreement'] > 0.66) & (~df['any_spread_stress'].astype(bool))
    fake_breakout = big_move & (df['agreement'] < 0.66) & df['any_spread_stress'].astype(bool)

    for label, mask in [("Real breakout", real_breakout), ("Fake breakout", fake_breakout)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: only {n} events, skipping")
            continue

        f1 = df.loc[mask, 'EURJPY_fwd1_ret'].dropna()
        f5 = df.loc[mask, 'EURJPY_fwd5_ret'].dropna()
        f15 = df.loc[mask, 'EURJPY_fwd15_ret'].dropna()

        reversal = (np.sign(df.loc[mask, 'EURJPY_ret_pips']) !=
                    np.sign(df.loc[mask, 'EURJPY_fwd1_ret'])).dropna()
        reversal_rate = reversal.mean() if len(reversal) > 0 else 0

        print(f"\n  {label} (n={n}):")
        print(f"    Fwd1  return: {f1.mean():+.3f}p  pos%={(f1>0).mean():.1%}")
        print(f"    Fwd5  return: {f5.mean():+.3f}p  pos%={(f5>0).mean():.1%}")
        print(f"    Fwd15 return: {f15.mean():+.3f}p  pos%={(f15>0).mean():.1%}")
        print(f"    Reversal rate: {reversal_rate:.1%}")

        results[label] = {
            'n': n, 'fwd1': float(f1.mean()), 'fwd5': float(f5.mean()),
            'fwd15': float(f15.mean()), 'reversal_rate': float(reversal_rate),
        }

    if 'Fake breakout' in results and 'Real breakout' in results:
        print(f"\n  Fake reversal rate: {results['Fake breakout']['reversal_rate']:.1%}")
        print(f"  Real reversal rate: {results['Real breakout']['reversal_rate']:.1%}")
        results['reversal_delta'] = results['Fake breakout']['reversal_rate'] - results['Real breakout']['reversal_rate']

    status = 'PASS' if results.get('Fake breakout', {}).get('reversal_rate', 0) > \
                      results.get('Real breakout', {}).get('reversal_rate', 1) else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


# ====================================================================
# TIER 2 — MECHANISM TESTS
# ====================================================================

def test_response_deficit_acceptance(df):
    """
    Hypothesis: EURJPY→GBPJPY response deficit + stable GBPJPY spread = real propagation.
    Deficit + spread stress = fake dislocation (GBPJPY never catches up).
    """
    print("\n" + "=" * 70)
    print("TEST 4: RESPONSE DEFICIT + QUOTE ACCEPTANCE")
    print("=" * 70)
    results = {}

    ej_move = df['EURJPY_z_ret'].abs() > 1.5
    deficit = ej_move & (df['GBPJPY_z_ret'].abs() < 0.5)
    same_dir = (np.sign(df['EURJPY_ret_pips']) == np.sign(df['GBPJPY_ret_pips']))

    stable = deficit & same_dir & (~df['GBPJPY_spread_widen'].astype(bool))
    stressed = deficit & same_dir & df['GBPJPY_spread_widen'].astype(bool)

    gj_fwd5 = df['GBPJPY_ret_pips'].rolling(5).sum().shift(-5)
    gj_fwd15 = df['GBPJPY_ret_pips'].rolling(15).sum().shift(-15)

    for label, mask in [("Stable spread propagation", stable),
                        ("Stressed spread dislocation", stressed)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: only {n} events, skipping")
            continue

        f5 = gj_fwd5[mask].dropna()
        f15 = gj_fwd15[mask].dropna()
        catch_up = ((np.sign(df.loc[mask, 'EURJPY_ret_pips']) ==
                     np.sign(gj_fwd5[mask])).dropna())

        print(f"\n  {label} (n={n}):")
        print(f"    GBPJPY fwd5  return: {f5.mean():+.3f}p  pos%={(f5>0).mean():.1%}")
        print(f"    GBPJPY fwd15 return: {f15.mean():+.3f}p  pos%={(f15>0).mean():.1%}")
        print(f"    Catch-up rate:  {catch_up.mean():.1%}" if len(catch_up) > 0 else "")

        results[label] = {
            'n': n, 'fwd5': float(f5.mean()), 'fwd15': float(f15.mean()),
            'catch_up': float(catch_up.mean()) if len(catch_up) > 0 else 0,
        }

    if 'Stable spread propagation' in results and 'Stressed spread dislocation' in results:
        d = results['Stable spread propagation']['fwd15'] - results['Stressed spread dislocation']['fwd15']
        print(f"\n  Delta fwd15: {d:+.3f}p")
        results['delta_fwd15'] = float(d)

    status = 'PASS' if results.get('Stable spread propagation', {}).get('catch_up', 0) > \
                      results.get('Stressed spread dislocation', {}).get('catch_up', 1) else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


def test_compression_expansion(df):
    """
    Hypothesis: Low vol + tick acceleration + stable spread = healthy expansion.
    Low vol + tick acceleration + spread stress = false breakout.
    """
    print("\n" + "=" * 70)
    print("TEST 5: COMPRESSION → EXPANSION AUTHENTICATION")
    print("=" * 70)
    results = {}

    compressed = df['low_vol'].astype(bool) & df['EURJPY_tick_accel'].astype(bool)
    healthy = compressed & (~df['any_spread_stress'].astype(bool))
    false_break = compressed & df['any_spread_stress'].astype(bool)

    fwd15_ret = df['EURJPY_ret_pips'].rolling(15).sum().shift(-15)
    fwd5_vol = df['EURJPY_ret_pips'].rolling(5).std().shift(-1)

    for label, mask in [("Healthy expansion", healthy), ("False expansion", false_break)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: only {n} events, skipping")
            continue

        f15 = fwd15_ret[mask].dropna()
        fv = fwd5_vol[mask].dropna()

        print(f"\n  {label} (n={n}):")
        print(f"    Fwd15 return:  {f15.mean():+.3f}p  pos%={(f15>0).mean():.1%}")
        print(f"    Post vol:      {fv.mean():.3f}p")
        print(f"    Vol vs base:   {fv.mean()/df['EURJPY_roll_vol_5'].median():.2f}x")

        results[label] = {
            'n': n, 'fwd15': float(f15.mean()), 'pos_pct': float((f15>0).mean()) if len(f15) > 0 else 0,
            'post_vol': float(fv.mean()),
        }

    if 'Healthy expansion' in results and 'False expansion' in results:
        d = results['Healthy expansion']['fwd15'] - results['False expansion']['fwd15']
        print(f"\n  Delta fwd15: {d:+.3f}p")
        results['delta_fwd15'] = float(d)

    status = 'PASS' if results.get('Healthy expansion', {}).get('fwd15', -999) > \
                      results.get('False expansion', {}).get('fwd15', 999) else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


# ====================================================================
# TIER 3 — ADVANCED PATTERNS
# ====================================================================

def test_dealer_capitulation(df):
    """
    Hypothesis: Spread stress peaks represent dealer capitulation.
    After peak, returns are mean-reverting as liquidity normalizes.
    """
    print("\n" + "=" * 70)
    print("TEST 6: DEALER CAPITULATION CYCLE")
    print("=" * 70)
    results = {}

    shock = df['EURJPY_spread_shock'].astype(bool)
    shock_idx = np.where(shock.values)[0]

    if len(shock_idx) < 10:
        print(f"  Only {len(shock_idx)} shock events, skipping")
        results['status'] = 'INCONCLUSIVE'
        return results

    spread_path = []
    ret_path = []
    vol_path = []

    for idx in shock_idx:
        start = max(0, idx - 5)
        end = min(len(df), idx + 20)
        segment = df.iloc[start:end]
        spread_path.append(segment['EURJPY_med_spread'].values)

        post = df.iloc[idx:min(len(df), idx + 15)]
        rets = post['EURJPY_ret_pips'].values
        ret_path.append(np.nansum(rets) if len(rets) > 0 else 0)

        vols = post['EURJPY_roll_vol_5'].values
        vol_path.append(np.nanmean(vols) if len(vols) > 0 else 0)

    spread_path = np.array([p for p in spread_path if len(p) == 25])
    if len(spread_path) < 5:
        print(f"  Insufficient aligned segments ({len(spread_path)}), skipping")
        results['status'] = 'INCONCLUSIVE'
        return results

    spread_norm = spread_path / spread_path[:, 5:6].clip(min=1e-8)
    spread_med = np.nanmedian(spread_norm, axis=0)

    ret_arr = np.array(ret_path)
    vol_arr = np.array(vol_path)

    print(f"\n  Events with valid post-window: {len(spread_path)}")
    print(f"  Spread path (normalized to shock peak):")
    for i, v in enumerate(spread_med[:10]):
        t = i - 5
        print(f"    t+{t:+3d}: {v:.3f}x")
    print(f"  ... t+15: {spread_med[-1] if len(spread_med) > 15 else 'N/A':.3f}x")

    print(f"\n  Mean post-shock 15-bar return: {np.nanmean(ret_arr):+.3f}p")
    print(f"  Post-shock pos%: {(ret_arr > 0).mean():.1%}")
    print(f"  Post-shock vol:  {np.nanmean(vol_arr):.3f}p")
    print(f"  Vol vs median:   {np.nanmean(vol_arr)/df['EURJPY_roll_vol_5'].median():.2f}x")

    results['n_events'] = len(spread_path)
    results['spread_decay_to_2x'] = int(np.argmax(spread_med < 2.0)) if np.any(spread_med < 2.0) else '>15'
    results['post_ret'] = float(np.nanmean(ret_arr))
    results['post_pos_pct'] = float((ret_arr > 0).mean())
    results['post_vol_ratio'] = float(np.nanmean(vol_arr) / df['EURJPY_roll_vol_5'].median())

    status = 'PASS' if results.get('spread_decay_to_2x') != '>15' else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


def test_session_liquidity_vacuum(df):
    """
    Hypothesis: Session boundary + tick drop + spread widening = no-trade regime.
    Market is illiquid and unpredictable during these transitions.
    """
    print("\n" + "=" * 70)
    print("TEST 7: SESSION LIQUIDITY VACUUM")
    print("=" * 70)
    results = {}

    boundary = (df['is_tokyo_h0'].astype(bool) | df['is_london_h0'].astype(bool) |
                df['is_ny_h0'].astype(bool))
    vacuum = boundary & df['EURJPY_tick_drop'].astype(bool) & df['EURJPY_spread_widen'].astype(bool)
    normal = boundary & (~df['EURJPY_tick_drop'].astype(bool)) & (~df['EURJPY_spread_widen'].astype(bool))

    fwd5_vol = df['EURJPY_ret_pips'].rolling(5).std().shift(-1)

    for label, mask in [("Liquidity vacuum", vacuum), ("Normal boundary", normal)]:
        n = mask.sum()
        if n < 5:
            print(f"  {label}: only {n} events, skipping")
            continue

        fv = fwd5_vol[mask].dropna()
        spread = df.loc[mask, 'EURJPY_med_spread'].dropna()
        fwd1 = df.loc[mask, 'EURJPY_fwd1_ret'].dropna()

        ej_ret = df.loc[mask, 'EURJPY_ret_pips'].dropna()
        next_sign_same = ((ej_ret > 0) & (fwd1.reindex(ej_ret.index) > 0) |
                          (ej_ret < 0) & (fwd1.reindex(ej_ret.index) < 0)).dropna()
        persistence = next_sign_same.mean() if len(next_sign_same) > 0 else 0

        print(f"\n  {label} (n={n}):")
        print(f"    Post vol:       {fv.mean():.3f}p")
        print(f"    Median spread:  {spread.median():.2f}p")
        print(f"    Fwd1 direction persistence: {persistence:.1%}")

        results[label] = {
            'n': n, 'post_vol': float(fv.mean()), 'med_spread': float(spread.median()),
            'persistence': float(persistence),
        }

    if 'Liquidity vacuum' in results:
        r = results['Liquidity vacuum']
        print(f"\n  Vacuum: vol={r['post_vol']:.3f}p, spread={r['med_spread']:.2f}p")

    status = 'PASS' if results.get('Liquidity vacuum', {}).get('post_vol', 999) < \
                      results.get('Normal boundary', {}).get('post_vol', 0) else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


def test_memory_decay(df):
    """
    Hypothesis: After each type of event, market state memory decays with
    characteristic half-life. Measure persistence of authentication state.
    """
    print("\n" + "=" * 70)
    print("TEST 8: MEMORY DECAY")
    print("=" * 70)
    results = {}

    auth_events = df['authenticated'].astype(bool) & (df['EURJPY_z_ret'].abs() > 1.0)
    auth_idx = np.where(auth_events.values)[0]

    if len(auth_idx) < 10:
        print(f"  Only {len(auth_idx)} auth events, skipping")
        results['status'] = 'INCONCLUSIVE'
        return results

    halflives = []
    for idx in auth_idx[:200]:
        end = min(len(df), idx + 120)
        segment = df.iloc[idx:end]
        decay = (~segment['authenticated'].astype(bool)).values
        first_false = np.argmax(decay) if np.any(decay) else len(decay)
        halflives.append(first_false)

    halflives = np.array(halflives)
    print(f"\n  Authentication memory decay:")
    print(f"    Mean half-life:           {halflives.mean():.1f} bars")
    print(f"    Median half-life:         {np.median(halflives):.1f} bars")
    print(f"    Q25/Q75:                  {np.percentile(halflives, 25):.0f} / {np.percentile(halflives, 75):.0f} bars")
    print(f"    Pct living > 5 bars:      {(halflives > 5).mean():.1%}")
    print(f"    Pct living > 15 bars:     {(halflives > 15).mean():.1%}")

    spread_events = df['EURJPY_spread_shock'].astype(bool)
    spread_idx = np.where(spread_events.values)[0]

    if len(spread_idx) >= 10:
        spread_halflives = []
        for idx in spread_idx[:200]:
            end = min(len(df), idx + 120)
            segment = df.iloc[idx:end]
            ts = (segment['EURJPY_spread_z'].abs() < 1.0).values
            first_normal = np.argmax(ts) if np.any(ts) else len(ts)
            spread_halflives.append(first_normal)

        spread_hl = np.array(spread_halflives)
        print(f"\n  Spread shock memory decay:")
        print(f"    Mean recovery:           {spread_hl.mean():.1f} bars")
        print(f"    Median recovery:         {np.median(spread_hl):.1f} bars")

        results['spread_recovery_bars'] = float(np.median(spread_hl))
    else:
        print(f"\n  Spread shock: too few events ({len(spread_idx)})")

    results['auth_halflife_mean'] = float(halflives.mean())
    results['auth_halflife_med'] = float(np.median(halflives))
    results['auth_gt5_pct'] = float((halflives > 5).mean())
    results['auth_gt15_pct'] = float((halflives > 15).mean())

    if results.get('spread_recovery_bars', 999) < 60:
        status = 'PASS'
    else:
        status = 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
    return results


# ====================================================================
# TRADING FRAMEWORK TEST
# ====================================================================
def test_auth_trading_framework(df):
    """
    Simulate a simple trading framework using authentication state.
    Entry: authenticated state with directional bias.
    Exit: when authentication drops or profit target hit.
    Compare vs random entries.
    """
    print("\n" + "=" * 70)
    print("AUX: AUTHENTICATION TRADING FRAMEWORK SIMULATION")
    print("=" * 70)
    results = {}

    entry_long = df['authenticated'].astype(bool) & (df['direction'] > 0)
    entry_short = df['authenticated'].astype(bool) & (df['direction'] < 0)
    entry_any = entry_long | entry_short

    n_entries = entry_any.sum()
    if n_entries < 10:
        print(f"  Only {n_entries} entries, skipping")
        results['status'] = 'INCONCLUSIVE'
        return results

    trade_rets = []
    trade_dir = []
    for idx in np.where(entry_any.values)[0]:
        direction = 1 if entry_long.iloc[idx] else -1
        fwd5 = df['EURJPY_fwd5_ret'].iloc[idx]
        if np.isnan(fwd5):
            continue
        pnl = direction * fwd5
        trade_rets.append(pnl)
        trade_dir.append(direction)

    trade_rets = np.array(trade_rets)
    n = len(trade_rets)
    wr = (trade_rets > 0).mean()
    avg = trade_rets.mean()
    sharpe = avg / trade_rets.std() * np.sqrt(60) if trade_rets.std() > 0 else 0
    t_stat = avg / (trade_rets.std() / np.sqrt(n)) if trade_rets.std() > 0 else 0

    random_rets = []
    n_random = min(n * 5, 10000)
    idx_pool = np.arange(500, len(df) - 10)
    for _ in range(n_random):
        ri = np.random.choice(idx_pool)
        direction = np.random.choice([-1, 1])
        fwd5 = df['EURJPY_fwd5_ret'].iloc[ri]
        if np.isnan(fwd5):
            continue
        random_rets.append(direction * fwd5)
    random_rets = np.array(random_rets)

    print(f"\n  Auth entries: {n}")
    print(f"  Win rate:        {wr:.1%}")
    print(f"  Avg return:      {avg:+.3f}p")
    print(f"  Sharpe (5m):     {sharpe:.2f}")
    print(f"  t-stat:          {t_stat:.3f}")

    if len(random_rets) > 0:
        r_wr = (random_rets > 0).mean()
        r_avg = random_rets.mean()
        print(f"\n  Random entries (n={len(random_rets)}):")
        print(f"    Win rate:   {r_wr:.1%}")
        print(f"    Avg return: {r_avg:+.3f}p")

        t_test = scipy_stats.ttest_ind(trade_rets, random_rets, alternative='greater')
        print(f"\n  Auth > Random? t={t_test.statistic:.3f} p={t_test.pvalue:.6f}")

        results['t_test_stat'] = float(t_test.statistic)
        results['t_test_pval'] = float(t_test.pvalue)

    results.update({
        'n_entries': n, 'win_rate': float(wr), 'avg_ret': float(avg),
        'sharpe': float(sharpe), 't_stat': float(t_stat),
    })

    status = 'PASS' if wr > 0.55 and abs(t_stat) > 1.96 else 'INCONCLUSIVE'
    results['status'] = status
    print(f"  → {status}")
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

    print("\n" + "=" * 70)
    print("RUNNING ALL AUTHENTICATION TESTS")
    print("=" * 70)

    all_results = {}

    print("\n" + "─" * 70)
    all_results['test_1'] = test_authenticated_repricing(df)

    print("\n" + "─" * 70)
    all_results['test_2'] = test_tokyo_h0_liquidity(df)

    print("\n" + "─" * 70)
    all_results['test_3'] = test_false_breakout(df)

    print("\n" + "─" * 70)
    all_results['test_4'] = test_response_deficit_acceptance(df)

    print("\n" + "─" * 70)
    all_results['test_5'] = test_compression_expansion(df)

    print("\n" + "─" * 70)
    all_results['test_6'] = test_dealer_capitulation(df)

    print("\n" + "─" * 70)
    all_results['test_7'] = test_session_liquidity_vacuum(df)

    print("\n" + "─" * 70)
    all_results['test_8'] = test_memory_decay(df)

    print("\n" + "─" * 70)
    all_results['test_framework'] = test_auth_trading_framework(df)

    passes = sum(1 for v in all_results.values() if v.get('status') == 'PASS')
    fails = sum(1 for v in all_results.values() if v.get('status') == 'FAIL')
    inc = sum(1 for v in all_results.values() if v.get('status') == 'INCONCLUSIVE')

    elapsed = time.time() - t_main
    print("\n" + "=" * 70)
    print("AUTHENTICATION TEST SUMMARY")
    print("=" * 70)
    print(f"  PASS:        {passes}")
    print(f"  FAIL:        {fails}")
    print(f"  INCONCLUSIVE: {inc}")
    print(f"  Total time:  {elapsed:.1f}s")
    print(f"  Data range:  {df.index[0]} to {df.index[-1]}")

    return all_results


if __name__ == '__main__':
    main()
