"""
Optimize Dealer Capitulation signal across recovery definitions.
Tests: threshold, hold time, z threshold — to find best WR.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path
from datetime import timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
SCALE = {'EURJPY': 100, 'GBPJPY': 100}

t0 = time.time()

# Load
all_t = {}
for pair in ['GBPJPY', 'EURJPY']:
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
    all_t[pair] = df.set_index('Ts')
    print(f"  {pair}: {len(df):,d} ticks  ({time.time()-t0:.1f}s)")

for pair in ['GBPJPY', 'EURJPY']:
    s = SCALE[pair]
    t = all_t[pair]
    cost = 0.6 if pair == 'GBPJPY' else 0.5
    
    print(f"\n{'='*70}")
    print(f"{pair}")
    print(f"{'='*70}")
    
    # Build M1 bars
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
    
    b['f5'] = b['ret'].rolling(5).sum().shift(-5)
    b['f10'] = b['ret'].rolling(10).sum().shift(-10)
    b['f15'] = b['ret'].rolling(15).sum().shift(-15)
    b['af5'] = -np.sign(b['z'].fillna(0)) * b['f5'].fillna(0)
    b['af10'] = -np.sign(b['z'].fillna(0)) * b['f10'].fillna(0)
    b['af15'] = -np.sign(b['z'].fillna(0)) * b['f15'].fillna(0)
    
    # Test multiple z thresholds
    for z_thr in [2.0, 1.75]:
        events_mask = (b['z'].abs() > z_thr) & b['sw']
        event_idx = np.where(events_mask.values)[0]
        
        # Measure recovery at tick level for each event
        results = []
        for idx in event_idx:
            dt = b.index[idx]
            rm = b['rm'].iloc[idx]
            
            sl = t.loc[dt:dt + timedelta(minutes=2)]
            if len(sl) < 3: sl = t.loc[dt:dt + timedelta(minutes=5)]
            if len(sl) < 3: continue
            
            sp = (sl['A'] - sl['B']) * s
            pp = sp.idxmax()
            pk = sp.max()
            af = sp.loc[pp:]
            if len(af) < 2: continue
            
            # Try multiple recovery thresholds
            for r_mult in [1.3, 1.5, 2.0]:
                thr = max(r_mult * rm, rm + 0.03)
                fb = af.index[(af < thr).values][0] if (af < thr).any() else None
                if fb is None:
                    sl2 = t.loc[dt:dt + timedelta(minutes=10)]
                    if len(sl2) > len(sl):
                        sp2 = (sl2['A'] - sl2['B']) * s
                        af2 = sp2.loc[pp:]
                        fb = af2.index[(af2 < thr).values][0] if (af2 < thr).any() else None
                
                rec_t = len(t.loc[pp:fb]) if fb is not None else 999
                rec_s = (fb - pp).total_seconds() if fb is not None else 999
                
                results.append({
                    'dt': dt, 'z_thr': z_thr, 'r_mult': r_mult,
                    'rec_t': rec_t, 'rec_s': rec_s, 'pk_sprd': pk,
                    'sr': b['sr'].iloc[idx],
                    'af5': b['af5'].iloc[idx],
                    'af10': b['af10'].iloc[idx],
                    'af15': b['af15'].iloc[idx],
                    'idx': idx
                })
        
        if not results:
            continue
        
        rdf = pd.DataFrame(results)
        
        # For each (z_thr, r_mult) combo, split by median recovery ticks
        for r_mult in [1.3, 1.5, 2.0]:
            sub = rdf[(rdf['r_mult'] == r_mult) & (rdf['rec_t'] < 999)]
            if len(sub) < 10: continue
            
            med = sub['rec_t'].median()
            fast = sub['rec_t'] <= med
            slow = sub['rec_t'] > med
            
            for label, mask in [('Q4_SLOW', slow & (sub['rec_t'] > sub['rec_t'].quantile(0.75))),
                                ('SLOW', slow), ('FAST', fast), ('ALL', sub.index)]:
                is_special = isinstance(mask, pd.Index)
                n = len(mask) if is_special else mask.sum()
                if n < 5: continue
                ssub = sub if is_special else sub.loc[mask]
                
                for hold, col in [('5min', 'af5'), ('10min', 'af10'), ('15min', 'af15')]:
                    wr = (ssub[col] > 0).mean()
                    avg = ssub[col].mean()
                    net = avg - cost
                    nd = n / 66
                    
                    if label == 'Q4_SLOW' and wr >= 0.65:
                        print(f"  z>{z_thr} r×{r_mult} {label:<8s} h={hold:<4s} n={n:>4d} {nd:.2f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                    elif label == 'SLOW' and wr >= 0.60:
                        print(f"  z>{z_thr} r×{r_mult} {label:<8s} h={hold:<4s} n={n:>4d} {nd:.2f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")

print(f"\nTotal: {time.time()-t0:.1f}s")
