"""
TEST 15: Dynamic Conviction Ladder with QAI Adaptive Exit
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
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
# TEST 15: DYNAMIC CONVICTION LADDER WITH QAI ADAPTIVE EXIT
# ====================================================================
def test_conviction_ladder(df):
    print("\n" + "=" * 70)
    print("TEST 15: DYNAMIC CONVICTION LADDER WITH QAI ADAPTIVE EXIT")
    print("=" * 70)

    s = SCALE['GBPJPY']
    n = len(df)

    # --- Pre-compute exit signal columns ---
    df['GBPJPY_dQAI'] = df['GBPJPY_med_qai'].diff().fillna(0)
    df['GBPJPY_roll_dQAI_med'] = df['GBPJPY_dQAI'].rolling(20).median()
    df['GBPJPY_qai_signal'] = df['GBPJPY_dQAI'] > df['GBPJPY_roll_dQAI_med']
    df['GBPJPY_qai_signal'] = df['GBPJPY_qai_signal'].fillna(False)

    df['GBPJPY_tick_collapse'] = df['GBPJPY_tick_count'] < df['GBPJPY_roll_tick_count'] * 0.7
    df['GBPJPY_tick_collapse'] = df['GBPJPY_tick_collapse'].fillna(False)

    # --- Response deficit entry condition ---
    ej_big = df['EURJPY_z_ret'].abs() > 0.5
    gj_small = df['GBPJPY_z_ret'].abs() < 0.5
    deficit = ej_big & gj_small

    print(f"\n  Response deficit events: {deficit.sum():,d} ({deficit.mean():.1%} of bars)")

    # --- Conviction level masks ---
    norm_bool = df['normal_spread'].astype(bool)
    highp_bool = df['high_participation'].astype(bool)
    stress_bool = df['any_spread_stress'].astype(bool)

    level_masks = {
        0: deficit,
        1: deficit & norm_bool,
        2: deficit & norm_bool & highp_bool,
        3: deficit & norm_bool & highp_bool & (df['agreement'] > 0.5),
        4: deficit & stress_bool,
    }

    for level, mask in level_masks.items():
        print(f"  Level {level}: {mask.sum():,d} entries ({mask.mean():.1%})")

    # Pre-compute direction for every bar
    ej_dir = np.sign(df['EURJPY_ret_pips'].values)

    # --- Exit strategies ---
    strategies = {
        'A_fixed_5':  {'max_hold': 5,  'type': 'fixed'},
        'B_fixed_15': {'max_hold': 15, 'type': 'fixed'},
        'C_qai':      {'max_hold': 15, 'type': 'qai'},
        'D_qai_tick': {'max_hold': 15, 'type': 'qai_tick'},
    }

    # --- Simulation ---
    trades = []
    entry_indices = np.where(deficit.values)[0]

    t_start = time.time()
    print(f"\n  Simulating trades over {len(entry_indices)} deficit events...")

    for idx_pos, t in enumerate(entry_indices):
        if idx_pos > 0 and idx_pos % 5000 == 0:
            print(f"    ... {idx_pos}/{len(entry_indices)} ({time.time()-t_start:.1f}s)")

        direction = ej_dir[t]
        if direction == 0:
            continue

        entry_price = df.iloc[t]['GBPJPY_close']

        # Determine which levels this bar qualifies for
        row = df.iloc[t]
        q_levels = [0]
        if row['normal_spread']:
            q_levels.append(1)
            if row['high_participation']:
                q_levels.append(2)
                if row['agreement'] > 0.5:
                    q_levels.append(3)
        if row['any_spread_stress']:
            q_levels.append(4)

        for level in q_levels:
            for strat_name, strat in strategies.items():
                max_hold = strat['max_hold']
                end = min(n, t + max_hold + 1)

                if end <= t + 1:
                    continue

                t_exit = None

                if strat['type'] == 'fixed':
                    t_exit = t + max_hold
                    if t_exit >= n:
                        continue

                elif strat['type'] == 'qai':
                    for k in range(t + 1, end):
                        if df.iloc[k]['GBPJPY_qai_signal']:
                            t_exit = k
                            break
                    if t_exit is None:
                        if t + max_hold < n:
                            t_exit = t + max_hold
                        else:
                            continue

                elif strat['type'] == 'qai_tick':
                    for k in range(t + 1, end):
                        if df.iloc[k]['GBPJPY_qai_signal'] or df.iloc[k]['GBPJPY_tick_collapse']:
                            t_exit = k
                            break
                    if t_exit is None:
                        if t + max_hold < n:
                            t_exit = t + max_hold
                        else:
                            continue

                if t_exit is None or t_exit >= n:
                    continue

                # --- Compute trade metrics ---
                exit_price = df.iloc[t_exit]['GBPJPY_close']
                return_pips = direction * (exit_price - entry_price) * s
                duration = t_exit - t

                # Excursions using bar high/low
                seg = df.iloc[t + 1 : t_exit + 1]
                if len(seg) > 0:
                    if direction > 0:
                        fav_series = direction * (seg['GBPJPY_high'] - entry_price) * s
                        adv_series = direction * (seg['GBPJPY_low'] - entry_price) * s
                    else:
                        fav_series = direction * (seg['GBPJPY_low'] - entry_price) * s
                        adv_series = direction * (seg['GBPJPY_high'] - entry_price) * s
                    favorable_excursion = float(fav_series.max())
                    adverse_excursion = float(-adv_series.min())
                    adverse_excursion = max(0.0, adverse_excursion)
                else:
                    favorable_excursion = 0.0
                    adverse_excursion = 0.0

                trades.append({
                    'entry_idx': t,
                    'level': level,
                    'strategy': strat_name,
                    'direction': direction,
                    'return_pips': return_pips,
                    'hit': 1 if return_pips > 0 else 0,
                    'duration_bars': duration,
                    'favorable_excursion': favorable_excursion,
                    'adverse_excursion': adverse_excursion,
                })

    dt_sim = time.time() - t_start
    print(f"  Simulation complete: {len(trades):,d} trades in {dt_sim:.1f}s")

    if len(trades) == 0:
        print("  No trades generated!")
        return

    # --- Aggregate results ---
    trade_df = pd.DataFrame(trades)

    print(f"\n  Total simulated trades: {len(trade_df):,d}")
    print(f"  Trade date range: {df.index[0]} to {df.index[-1]}")

    # Trading days (Mon-Fri)
    weekday_dates = df.index[df.index.dayofweek < 5].normalize().unique()
    trading_days = len(weekday_dates)

    print(f"  Trading days: {trading_days}")

    print("\n" + "=" * 90)
    print("CONVICTION LADDER RESULTS BY (LEVEL, EXIT STRATEGY)")
    print("=" * 90)

    results_rows = []
    group_cols = ['level', 'strategy']

    for (level, strategy), grp in trade_df.groupby(group_cols):
        n_trades = len(grp)
        win_rate = grp['hit'].mean()
        avg_return = grp['return_pips'].mean()
        total_return = grp['return_pips'].sum()
        avg_dur = grp['duration_bars'].mean()
        avg_adv = grp['adverse_excursion'].mean()
        avg_fav = grp['favorable_excursion'].mean()
        exc_ratio = avg_fav / avg_adv if avg_adv > 0 else np.inf
        avg_ret_pb = avg_return / avg_dur if avg_dur > 0 else 0
        wr_sqrt_n = win_rate * np.sqrt(n_trades)

        results_rows.append({
            'level': level,
            'strategy': strategy,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'avg_dur_bars': avg_dur,
            'avg_adv_exc': avg_adv,
            'avg_fav_exc': avg_fav,
            'exc_ratio': exc_ratio,
            'avg_ret_per_bar': avg_ret_pb,
            'wr_sqrt_n': wr_sqrt_n,
        })

    res = pd.DataFrame(results_rows)

    # Pretty print
    fmt = {
        'n_trades': '{:>8,d}'.format,
        'win_rate': '{:>7.1%}'.format,
        'avg_return': '{:>+9.3f}'.format,
        'total_return': '{:>+10.3f}'.format,
        'avg_dur_bars': '{:>8.1f}'.format,
        'avg_adv_exc': '{:>9.3f}'.format,
        'avg_fav_exc': '{:>9.3f}'.format,
        'exc_ratio': '{:>8.2f}'.format,
        'avg_ret_per_bar': '{:>+10.4f}'.format,
        'wr_sqrt_n': '{:>9.4f}'.format,
    }

    print(f"\n{'Lvl':<4} {'Strat':<15} {'Trades':>8} {'WinRate':>7} {'AvgRet':>9} "
          f"{'TotRet':>10} {'DurBars':>8} {'AdvExc':>9} {'FavExc':>9} "
          f"{'ExcR':>8} {'Ret/Bar':>10} {'WR*sqrtN':>9}")
    print("-" * 116)

    for _, r in res.sort_values(['level', 'strategy']).iterrows():
        print(f"{int(r['level']):<4} {r['strategy']:<15} "
              f"{fmt['n_trades'](r['n_trades'])} "
              f"{fmt['win_rate'](r['win_rate'])} "
              f"{fmt['avg_return'](r['avg_return'])} "
              f"{fmt['total_return'](r['total_return'])} "
              f"{fmt['avg_dur_bars'](r['avg_dur_bars'])} "
              f"{fmt['avg_adv_exc'](r['avg_adv_exc'])} "
              f"{fmt['avg_fav_exc'](r['avg_fav_exc'])} "
              f"{fmt['exc_ratio'](r['exc_ratio'])} "
              f"{fmt['avg_ret_per_bar'](r['avg_ret_per_bar'])} "
              f"{fmt['wr_sqrt_n'](r['wr_sqrt_n'])}")

    # ===================================================================
    # ANSWERS TO SPECIFIC QUESTIONS
    # ===================================================================
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    # Q1: Does adding more layers increase WR without killing trade count?
    print("\n--- Q1: Conviction Layer Impact on Win Rate ---")
    for strat_name in sorted(trade_df['strategy'].unique()):
        sub = res[res['strategy'] == strat_name].sort_values('level')
        print(f"\n  Strategy: {strat_name}")
        for _, r in sub.iterrows():
            print(f"    Level {int(r['level'])}: WR={r['win_rate']:.1%}  "
                  f"n={r['n_trades']:,d}  "
                  f"Ret/Bar={r['avg_ret_per_bar']:+.4f}")

    # Q2: Does QAI adaptive exit improve WR vs fixed hold?
    print("\n--- Q2: QAI Adaptive Exit vs Fixed Hold (WR comparison) ---")
    for level_val in sorted(trade_df['level'].unique()):
        sub = res[res['level'] == level_val]
        print(f"\n  Level {int(level_val)}:")
        for _, r in sub.iterrows():
            print(f"    {r['strategy']:<15} WR={r['win_rate']:.1%}  "
                  f"AvgRet={r['avg_return']:+.3f}  "
                  f"AvgAdv={r['avg_adv_exc']:.3f}  "
                  f"AvgFav={r['avg_fav_exc']:.3f}")

    # Q3: Best (level, strategy) for WR * sqrt(n_trades)
    print("\n--- Q3: Best (Level, Strategy) by WR * sqrt(n_trades) ---")
    best = res.loc[res['wr_sqrt_n'].idxmax()]
    print(f"  Level {int(best['level'])} + {best['strategy']}: "
          f"WR={best['win_rate']:.1%}, n={best['n_trades']:,d}, "
          f"WR*sqrtN={best['wr_sqrt_n']:.4f}")

    # Also show top 5
    top5 = res.nlargest(5, 'wr_sqrt_n')
    print(f"\n  Top 5 combinations:")
    for _, r in top5.iterrows():
        print(f"    Level {int(r['level'])} + {r['strategy']:<15}: "
              f"WR={r['win_rate']:.1%}  n={r['n_trades']:,d}  "
              f"WR*sqrtN={r['wr_sqrt_n']:.4f}")

    # Q4: Trades per day at each level
    print("\n--- Q4: Trades per Day at Each Level ---")
    for level_val in sorted(trade_df['level'].unique()):
        sub = trade_df[trade_df['level'] == level_val]
        n_uniq = sub['entry_idx'].nunique()
        tpd = n_uniq / trading_days if trading_days > 0 else 0
        print(f"  Level {int(level_val)}: {n_uniq:,d} unique entry bars, "
              f"{tpd:.1f} trades/day")

    print("\n" + "=" * 90)
    print("TEST 15 COMPLETE")
    print("=" * 90)


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_main = time.time()

    tick_data = load_ticks()
    m1_dict = build_m1(tick_data)
    df = align_and_enrich(m1_dict)
    del tick_data, m1_dict

    # Key stats (same as auth test)
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

    # Run Test 15
    test_conviction_ladder(df)

    elapsed = time.time() - t_main
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Data range: {df.index[0]} to {df.index[-1]}")


if __name__ == '__main__':
    main()
