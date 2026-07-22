"""
Combined Signal: Response Deficit + Spread Recovery.
EURJPY moves, GBPJPY lags, GBPJPY spread stays wide = high-conviction catch-up.
Fast test on Oct-Dec 2025 Exness data.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path
from datetime import timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100}

t0 = time.time()

# Load both pairs
all_ticks = {}
for pair in ['EURJPY', 'GBPJPY']:
    dfs = []
    for y, m in MONTHS:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts': str, 'B': np.float64, 'A': np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['Mid'] = (df['B'] + df['A']) / 2
    df['Sprd'] = (df['A'] - df['B']) * SCALE[pair]
    df = df.set_index('Ts')
    all_ticks[pair] = df
    print(f"  {pair}: {len(df):,d} ticks")

# Build aligned M1 bars
bars = {}
for pair in ['EURJPY', 'GBPJPY']:
    s = SCALE[pair]
    t = all_ticks[pair]
    ohlc = t['Mid'].resample('1min').ohlc()
    tc = t['Mid'].resample('1min').count().to_frame('tc')
    tc['ms'] = t['Sprd'].resample('1min').median()
    tc['xs'] = t['Sprd'].resample('1min').max()
    b = pd.concat([ohlc, tc], axis=1).dropna(subset=['open','close'])
    b['ret'] = (b['close'] - b['open']) * s
    b['v5'] = b['ret'].rolling(20).std()
    b['z'] = b['ret'] / b['v5'].clip(lower=1e-8)
    b['sr'] = b['xs'] / b['ms'].rolling(20).median().clip(lower=1e-8)
    b['sw'] = b['sr'] > 2.0
    b['f5'] = b['ret'].rolling(5).sum().shift(-5)
    b['f15'] = b['ret'].rolling(15).sum().shift(-15)
    bars[pair] = b

# Align on common index
common = bars['EURJPY'].index.intersection(bars['GBPJPY'].index)
ej = bars['EURJPY'].loc[common]
gj = bars['GBPJPY'].loc[common]

# Compute response deficit: EURJPY moves but GBPJPY doesn't follow proportionally
# deficit = beta * EURJPY_ret - GBPJPY_ret
ej_ret = ej['ret'].values
gj_ret = gj['ret'].values
valid = ~(np.isnan(ej_ret) | np.isnan(gj_ret))

# Rolling beta: cov(ej_ret, gj_ret) / var(ej_ret) over 20 bars
roll_cov = pd.Series(ej_ret * gj_ret, index=common).rolling(20).mean()
roll_var = pd.Series(ej_ret**2, index=common).rolling(20).mean()
roll_mean_ej = pd.Series(ej_ret, index=common).rolling(20).mean()
roll_mean_gj = pd.Series(gj_ret, index=common).rolling(20).mean()
roll_cov = roll_cov - roll_mean_ej * roll_mean_gj
roll_var = roll_var - roll_mean_ej**2
beta = roll_cov / roll_var.clip(lower=1e-8)

expected_gj = beta * ej_ret
deficit = gj_ret - expected_gj  # positive = GBPJPY overshot, negative = undershot
deficit_z = deficit / pd.Series(deficit, index=common).rolling(20).std().clip(lower=1e-8).values

# Entry rules:
# 1. EURJPY moves > 1.5 sigma
# 2. Deficit > 2.0 sigma (GBPJPY under-responded)
# 3. Direction: if EURJPY up, go LONG GBPJPY (catch up); if down, go SHORT
ej_moved = ej['z'].abs() > 1.5
big_deficit = deficit_z.abs() > 2.0
signal = ej_moved & big_deficit

print(f"\nResponse deficit signals: {signal.sum()}")
print(f"  EURJPY z>1.5: {ej_moved.sum()}")
print(f"  Deficit z>2.0: {big_deficit.sum()}")

# Direction-adjusted returns
signal_idx = np.where(signal.values)[0]
results = []

for pos in signal_idx:
    dt = common[pos]
    entry_dir = np.sign(ej_ret[pos])  # go same direction as EURJPY
    
    f5 = gj['f5'].iloc[pos]
    f15 = gj['f15'].iloc[pos]
    if np.isnan(f5) or np.isnan(f15):
        continue
    
    trade_ret_5 = entry_dir * f5
    trade_ret_15 = entry_dir * f15
    
    # Add spread recovery measurement
    rm = gj['ms'].iloc[pos]
    gj_ticks = all_ticks['GBPJPY']
    sl = gj_ticks.loc[dt:dt + timedelta(minutes=2)]
    if len(sl) < 3:
        sl = gj_ticks.loc[dt:dt + timedelta(minutes=5)]
    if len(sl) >= 3:
        sp = (sl['A'] - sl['B']) * SCALE['GBPJPY']
        pp = sp.idxmax()
        ps = sp.max()
        af = sp.loc[pp:]
        thr = max(1.5 * rm, rm + 0.03)
        fb = af.index[(af < thr).values][0] if (af < thr).any() else None
        if fb is None:
            sl2 = gj_ticks.loc[dt:dt + timedelta(minutes=10)]
            if len(sl2) > len(sl):
                sp2 = (sl2['A'] - sl2['B']) * SCALE['GBPJPY']
                af2 = sp2.loc[pp:]
                fb = af2.index[(af2 < thr).values][0] if (af2 < thr).any() else None
        rec_ticks = len(gj_ticks.loc[pp:fb]) if fb is not None else None
        rec_sec = (fb - pp).total_seconds() if fb is not None else None
    else:
        rec_ticks = None
        rec_sec = None
    
    results.append({'ret5': trade_ret_5, 'ret15': trade_ret_15,
                    'rec_t': rec_ticks, 'rec_s': rec_sec,
                    'ej_z': ej['z'].iloc[pos], 'dz': deficit_z[pos],
                    'sr': gj['sr'].iloc[pos], 'sw': gj['sw'].iloc[pos]})

rdf = pd.DataFrame(results)
print(f"Measurable trades: {len(rdf)}")
print(f"Overall adjWR5: {(rdf['ret5'] > 0).mean():.1%}")
print(f"Overall adjWR15: {(rdf['ret15'] > 0).mean():.1%}")
print(f"Overall avg5: {rdf['ret5'].mean():+.3f}p")
print(f"Overall avg15: {rdf['ret15'].mean():+.3f}p")

# Split by spread recovery time
has_rec = rdf['rec_t'].notna()
if has_rec.sum() > 20:
    med_rt = rdf.loc[has_rec, 'rec_t'].median()
    fast = has_rec & (rdf['rec_t'] <= med_rt)
    slow = has_rec & (rdf['rec_t'] > med_rt)
    
    print(f"\n--- Split by spread recovery (median={med_rt:.0f} ticks) ---")
    print(f"  {'Group':<8s} {'n':>5s} {'adjWR5':>7s} {'adjWR15':>7s} {'avg5':>9s} {'avg15':>9s} {'net5':>9s} {'net15':>9s}")
    print(f"  {'-'*63}")
    
    for label, mask in [('FAST', fast), ('SLOW', slow), ('ALL', has_rec)]:
        is_all = isinstance(mask, pd.Index)
        n = len(mask) if is_all else mask.sum()
        if n < 5: continue
        sub = rdf if is_all else rdf.loc[mask]
        w5 = (sub['ret5'] > 0).mean()
        w15 = (sub['ret15'] > 0).mean()
        a5 = sub['ret5'].mean()
        a15 = sub['ret15'].mean()
        print(f"  {label:<8s} {n:>5d} {w5:>6.1%} {w15:>6.1%} {a5:>+9.3f}p {a15:>+9.3f}p {a5-0.6:>+9.3f}p {a15-0.6:>+9.3f}p")
    
    # Also split by whether spread widened
    sw_mask = rdf['sw'].astype(bool)
    print(f"\n--- Split by spread_widen ---")
    for label, mask in [('WIDEN', sw_mask), ('NO_WIDEN', ~sw_mask)]:
        n = mask.sum()
        if n < 5: continue
        sub = rdf.loc[mask]
        w5 = (sub['ret5'] > 0).mean()
        a5 = sub['ret5'].mean()
        print(f"  {label:<8s} {n:>5d} adjWR5={w5:.1%} avg5={a5:+.3f}p net5={a5-0.6:+.3f}p")
    
    # Combined: deficit + spread_widen + slow recovery
    combo = slow & sw_mask
    n_combo = combo.sum()
    if n_combo >= 5:
        sub = rdf.loc[combo]
        w5 = (sub['ret5'] > 0).mean()
        w15 = (sub['ret15'] > 0).mean()
        a5 = sub['ret5'].mean()
        a15 = sub['ret15'].mean()
        print(f"\n=== COMBINED: Deficit + SpreadWiden + SlowRec ===")
        print(f"  n={n_combo}, adjWR5={w5:.1%} adjWR15={w15:.1%} avg5={a5:+.3f}p avg15={a15:+.3f}p net5={a5-0.6:+.3f}p n/day={n_combo/66:.2f}")

print(f"\nTotal: {time.time()-t0:.1f}s")
