"""
Quick spread recovery scan — GBPJPY only, z_thr: 1.5/1.75/2.0, fast execution.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path
from datetime import timedelta

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
SCALE = {'GBPJPY': 100}

def load_pair(pair):
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
    return df

pair = 'GBPJPY'
s = SCALE[pair]
cost = 0.6

t0 = time.time()
ticks = load_pair(pair)
print(f"Loaded {len(ticks):,d} ticks ({time.time()-t0:.1f}s)")

# Build M1 bars
t0 = time.time()
ticks['Mid'] = (ticks['B'] + ticks['A']) / 2
ticks['Sprd'] = (ticks['A'] - ticks['B']) * s
ticks = ticks.set_index('Ts')

ohlc = ticks['Mid'].resample('1min').ohlc()
ts = ticks['Mid'].resample('1min').count().to_frame('tc')
ts['ms'] = ticks['Sprd'].resample('1min').median()
ts['xs'] = ticks['Sprd'].resample('1min').max()

bars = pd.concat([ohlc, ts], axis=1).dropna(subset=['open','close'])
bars['ret'] = (bars['close'] - bars['open']) * s
bars['v5'] = bars['ret'].rolling(20).std()

bars['z'] = bars['ret'] / bars['v5'].clip(lower=1e-8)
bars['sr'] = bars['xs'] / bars['ms'].rolling(20).median().clip(lower=1e-8)
bars['f5'] = bars['ret'].rolling(5).sum().shift(-5)
bars['f15'] = bars['ret'].rolling(15).sum().shift(-15)
bars['af5'] = -np.sign(bars['z'].fillna(0)) * bars['f5'].fillna(0)
bars['af15'] = -np.sign(bars['z'].fillna(0)) * bars['f15'].fillna(0)
print(f"Built {len(bars):,d} M1 bars ({time.time()-t0:.1f}s)")

# Grid scan: z thresholds x recovery splits
bars['sw'] = bars['sr'] > 2.0  # spread_widen flag

for z_thr in [2.0, 1.75, 1.5]:
    t0 = time.time()
    events = bars.index[(bars['z'].abs() > z_thr) & (bars['sw'])]
    n_events = len(events)
    
    results = []
    for i, bt in enumerate(events):
        if i > 0 and i % 500 == 0:
            print(f"  z>{z_thr}: {i}/{n_events} events...")
        
        rm = bars.loc[bt, 'ms']
        if pd.isna(rm) or rm <= 0:
            continue
        
        sl = ticks.loc[bt:bt + timedelta(minutes=2)]
        if len(sl) < 3:
            sl = ticks.loc[bt:bt + timedelta(minutes=5)]
        if len(sl) < 3:
            continue
        
        sp = (sl['A'] - sl['B']) * s
        pp = sp.idxmax()
        ps = sp.max()
        af = sp.loc[pp:]
        if len(af) < 2:
            continue
        
        thr = max(1.5 * rm, rm + 0.03)
        fb = af.index[(af < thr).values][0] if (af < thr).any() else None
        if fb is None:
            sl2 = ticks.loc[bt:bt + timedelta(minutes=10)]
            if len(sl2) <= len(sl):
                continue
            sp2 = (sl2['A'] - sl2['B']) * s
            af2 = sp2.loc[pp:]
            fb = af2.index[(af2 < thr).values][0] if (af2 < thr).any() else None
            if fb is None:
                continue
        
        rec_ticks = len(ticks.loc[pp:fb])
        if rec_ticks < 1:
            continue
        
        f5 = bars.loc[bt, 'af5']
        f15 = bars.loc[bt, 'af15']
        if pd.isna(f5) or pd.isna(f15):
            continue
        
        results.append({'rt': rec_ticks, 'f5': f5, 'f15': f15, 'ps': ps, 'sr': bars.loc[bt, 'sr']})
    
    elapsed = time.time() - t0
    
    if len(results) < 20:
        print(f"\n  z>{z_thr}: {n_events} events, {len(results)} measurable — too few")
        continue
    
    rdf = pd.DataFrame(results)
    
    quartiles = np.percentile(rdf['rt'], [25, 50, 75])
    
    def label_quartile(rt):
        if rt <= quartiles[0]: return 'Q1_fast'
        elif rt <= quartiles[1]: return 'Q2'
        elif rt <= quartiles[2]: return 'Q3'
        else: return 'Q4_slow'
    rdf['bin'] = rdf['rt'].apply(label_quartile)
    
    print(f"\n  === z>{z_thr}: {len(rdf)} trades ({n_events} events, {elapsed:.1f}s) ===")
    print(f"  Recovery quartiles: 25%<={quartiles[0]:.0f} 50%<={quartiles[1]:.0f} 75%<={quartiles[2]:.0f} ticks")
    print(f"  {'Group':<12s} {'n':>5s} {'n/day':>6s} {'adjWR5':>7s} {'adjWR15':>7s} {'avg5':>9s} {'avg15':>9s} {'net5':>9s} {'net15':>9s}")
    print(f"  {'-'*73}")
    
    for label in ['Q1_fast', 'Q2', 'Q3', 'Q4_slow']:
        grp = rdf[rdf['bin'] == label]
        n = len(grp)
        if n < 3: continue
        wr5 = (grp['f5'] > 0).mean()
        wr15 = (grp['f15'] > 0).mean()
        a5 = grp['f5'].mean()
        a15 = grp['f15'].mean()
        nd = n / 66
        print(f"  {label:<12s} {n:>5d} {nd:>5.2f}  {wr5:>6.1%} {wr15:>6.1%} {a5:>+9.3f}p {a15:>+9.3f}p {a5-cost:>+9.3f}p {a15-cost:>+9.3f}p")
    
    # Also show threshold-based split (median)
    med_rt = rdf['rt'].median()
    fast = rdf['rt'] <= med_rt
    slow = rdf['rt'] > med_rt
    for label, mask in [('FAST', fast), ('SLOW', slow), ('ALL', rdf.index)]:
        is_all = isinstance(mask, pd.Index)
        n = len(mask) if is_all else mask.sum()
        if n < 5: continue
        sub = rdf if is_all else rdf.loc[mask]
        wr5 = (sub['f5'] > 0).mean()
        wr15 = (sub['f15'] > 0).mean()
        a5 = sub['f5'].mean()
        a15 = sub['f15'].mean()
        nd = n / 66
        print(f"  {label:<12s} {n:>5d} {nd:>5.2f}  {wr5:>6.1%} {wr15:>6.1%} {a5:>+9.3f}p {a15:>+9.3f}p {a5-cost:>+9.3f}p {a15-cost:>+9.3f}p")
    
    gc.collect()

print(f"\nTotal: {time.time()-t0:.1f}s")
