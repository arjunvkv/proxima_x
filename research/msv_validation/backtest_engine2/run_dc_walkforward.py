"""
Walk-forward validation of best DC configurations.
Train: Oct+Nov 2025  |  Test: Dec 2025
"""
import numpy as np, pandas as pd, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SCALE = {'EURJPY': 100, 'GBPJPY': 100}

def load(pair):
    s = []
    for y, m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str, 'B':np.float64, 'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        s.append(d.dropna(subset=['Ts']))
    df = pd.concat(s, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['Mid'] = (df['B'] + df['A']) / 2
    df['Sprd'] = (df['A'] - df['B']) * SCALE[pair]
    return df.set_index('Ts')

def get_bars(t, pair):
    s = SCALE[pair]
    ohlc = t['Mid'].resample('1min').ohlc()
    tc = t['Mid'].resample('1min').count().to_frame('tc')
    tc['ms'] = t['Sprd'].resample('1min').median()
    tc['xs'] = t['Sprd'].resample('1min').max()
    b = pd.concat([ohlc, tc], axis=1).dropna(subset=['open','close'])
    b['ret'] = (b['close'] - b['open']) * s
    b['v5'] = b['ret'].rolling(20).std()
    b['z'] = b['ret'] / b['v5'].clip(lower=1e-8)
    b['rm'] = b['ms'].rolling(20).median()
    b['sr'] = b['xs'] / b['rm'].clip(lower=1e-8)
    b['sw'] = b['sr'] > 2.0
    for h in [5, 10, 15]:
        f = b['ret'].rolling(h).sum().shift(-h)
        b[f'f{h}'] = f
        b[f'af{h}'] = -np.sign(b['z'].fillna(0)) * f.fillna(0)
    return b

def measure_recovery(t, dt, rm, pair, offset_min=5):
    s = SCALE[pair]
    sl = t.loc[dt:dt + pd.Timedelta(minutes=offset_min)]
    if len(sl) < 3:
        sl = t.loc[dt:dt + pd.Timedelta(minutes=10)]
    if len(sl) < 3: return None, None
    sp = (sl['A'] - sl['B']) * s
    pp = sp.idxmax()
    pk = sp.max()
    af = sp.loc[pp:]
    if len(af) < 2: return None, None
    results = {}
    for r_mult in [1.3, 1.5]:
        thr = max(r_mult * rm, rm + 0.03)
        fb = af.index[(af < thr).values][0] if (af < thr).any() else None
        if fb is None:
            sl2 = t.loc[dt:dt + pd.Timedelta(minutes=15)]
            if len(sl2) > len(sl) and len(sl2) >= 3:
                sp2 = (sl2['A'] - sl2['B']) * s
                af2 = sp2.loc[pp:]
                fb = af2.index[(af2 < thr).values][0] if (af2 < thr).any() else None
        rec_t = len(t.loc[pp:fb]) if fb is not None else 999
        results[r_mult] = (rec_t, fb, pp)
    return results, pk

def test_config(b, t, pair, z_thr, r_mult, rec_fn):
    """Apply a config with recovery function returning True if trade qualifies."""
    events_mask = (b['z'].abs() > z_thr) & b['sw']
    event_idx = np.where(events_mask.values)[0]
    
    trades = []
    for idx in event_idx:
        dt = b.index[idx]
        rm = b['rm'].iloc[idx]
        res, pk = measure_recovery(t, dt, rm, pair)
        if res is None: continue
        if r_mult not in res: continue
        rec_t, fb, pp = res[r_mult]
        if not rec_fn(rec_t, b, idx): continue
        
        for hold, h in [(5,'5'),(10,'10'),(15,'15')]:
            col = f'af{hold}'
            if col not in b.columns or idx >= len(b) - hold: continue
            trades.append({
                'dt': dt, 'rec_t': rec_t,
                'result': b[col].iloc[idx],
                'win': b[col].iloc[idx] > 0,
                'hold': hold,
            })
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

for pair, z_thr, r_mult, label in [('EURJPY',1.75,1.5,'Q4_SLOW'),
                                      ('EURJPY',1.75,1.3,'Q4_SLOW'),
                                      ('GBPJPY',1.75,1.3,'MED_SPLIT')]:
    print(f"\n{'='*70}")
    print(f"{pair} z>{z_thr} r×{r_mult} {label}")
    print(f"{'='*70}")
    
    t = load(pair)
    
    # Walk forward
    train_mask = (t.index.year == 2025) & (t.index.month.isin([10, 11]))
    test_mask = (t.index.year == 2025) & (t.index.month == 12)
    
    for phase_name, mask in [('IS (Oct+Nov)', train_mask), ('OOS (Dec)', test_mask)]:
        sub_t = t.loc[mask]
        nticks = len(sub_t)
        if nticks < 1000:
            print(f"  {phase_name}: SKIP ({nticks} ticks < 1000)")
            continue
        b = get_bars(sub_t, pair)
        nbars = len(b)
        
        events_mask = (b['z'].abs() > z_thr) & b['sw']
        event_idx = np.where(events_mask.values)[0]
        print(f"  {phase_name}: {nticks:,d}t {nbars}b raw_events={len(event_idx)}", end='')
        
        rec_times = []
        for idx in event_idx:
            dt = b.index[idx]
            rm = b['rm'].iloc[idx]
            res, pk = measure_recovery(sub_t, dt, rm, pair)
            if res and r_mult in res:
                rec_times.append(res[r_mult][0])
        if not rec_times:
            print(" → no measurable recovery")
            continue
        
        # Match optimization logic: exclude never-recovered (999), compute threshold on rest
        rec_arr = np.array(rec_times)
        rec_clean = rec_arr[rec_arr < 999]
        n_never = (rec_arr == 999).sum()
        med_r = np.median(rec_clean) if len(rec_clean) > 0 else 999
        q3_r = np.quantile(rec_clean, 0.75) if len(rec_clean) > 0 else 999
        print(f" recovered={len(rec_clean)} never={n_never} med={med_r:.0f} Q3={q3_r:.0f}")
        
        # Determine threshold from IS data (carries to OOS)
        if phase_name == 'IS (Oct+Nov)':
            if label == 'Q4_SLOW':
                thr_val = q3_r
            elif label == 'MED_SPLIT':
                thr_val = med_r
        
        # Apply recovery filter (only for events that DO recover)
        def make_rec_fn(tv):
            return lambda r, b_, i_: r > tv and r < 999
        
        trades = test_config(b, sub_t, pair, z_thr, r_mult, make_rec_fn(thr_val))
        if len(trades) == 0:
            print(f"  {phase_name} trades=0 (thr_val={thr_val:.0f})")
            continue
        
        for hold in [5, 10, 15]:
            sub = trades[trades['hold'] == hold]
            if len(sub) < 5: continue
            wr = sub['win'].mean()
            avg = sub['result'].mean()
            nd = len(sub) / 22
            print(f"  {phase_name:<14s} h={hold:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                  f"WR={wr:.1%} avg={avg:+.3f}p")

print(f"\nNOTE: OOS uses IS recovery threshold to avoid lookahead bias.")
