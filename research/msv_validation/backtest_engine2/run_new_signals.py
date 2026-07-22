"""
Scan new microstructure signals for high trade count + high WR.
All use tick-level data, walk-forward (IS: Oct+Nov, OOS: Dec).
"""
import numpy as np, pandas as pd, gc
from pathlib import Path
from collections import deque

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}

def load(pair):
    s = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn=TICK_DIR/f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d=pd.read_csv(fn, compression='zip', names=['E','S','Ts','B','A'],
            skiprows=1, header=None, dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts']=pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        s.append(d.dropna(subset=['Ts']))
    df=pd.concat(s,ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['Mid']=(df['B']+df['A'])/2
    df['Sprd']=(df['A']-df['B'])*SCALE[pair]
    sc=SCALE[pair]
    df['MP']=df['Mid']*sc
    return df.set_index('Ts')

def load_all():
    return {p:load(p) for p in ['EURJPY','GBPJPY','EURUSD']}

# ─── Strategy 1: Order Flow Imbalance (OFI) via 1min bars ───
def ofi_signal(t, pair):
    """Compute 1min-bar OFI: ratio of up-ticks to total ticks."""
    sc = SCALE[pair]
    # Tick direction
    t['up'] = t['MP'].diff() > 0
    t['dn'] = t['MP'].diff() < 0
    t['fl'] = (t['up'] | t['dn']).astype(int)
    
    # Aggregate per minute
    of = t['up'].resample('1min').sum().to_frame('up')
    of['dn'] = t['dn'].resample('1min').sum()
    of['fl'] = t['fl'].resample('1min').sum()
    of['ofi'] = (of['up'] - of['dn']) / of['fl'].clip(lower=1)
    
    # Price returns
    mp = t['MP'].resample('1min').last()
    # Use pct_change for cross-pair comparability
    of['ret1'] = mp.pct_change(1)  # 1min return
    of['ret5'] = mp.pct_change(5).shift(-5)  # forward 5min
    of['ret10'] = mp.pct_change(10).shift(-10)  # forward 10min
    
    of = of.dropna(subset=['ret1','ret5','ret10'])
    n = len(of)
    if n < 50: return of
    
    of['ofi_z'] = (of['ofi'] - of['ofi'].rolling(50).mean()) / of['ofi'].rolling(50).std().clip(lower=1e-8)
    
    for hold, col in [(1,'ret1'),(5,'ret5'),(10,'ret10')]:
        of[f'af{hold}'] = np.sign(of['ofi_z'].fillna(0)) * of[col].fillna(0) * sc
    return of

def test_ofi(o, phase, z_thr, hold, ndays):
    col = f'af{hold}'
    events = o['ofi_z'].abs() > z_thr
    n = events.sum()
    if n < 5: return None
    wr = (o.loc[events, col] > 0).mean()
    avg = o.loc[events, col].mean()
    nd = n / ndays
    return {'n':n, 'nday':nd, 'wr':wr, 'avg':avg}

# ─── Strategy 2: Tick Streak Continuation ───
def streak_signal(t, pair):
    """For each tick, track consecutive same-direction ticks. 
    Predict next tick direction based on streak length."""
    sc = SCALE[pair]
    mp = t['MP'].values
    idx = t.index.values
    
    up = np.diff(mp) > 0
    dn = np.diff(mp) < 0
    
    streaks = []
    cur_streak = 0
    cur_dir = 0  # 1=up, -1=down
    
    for i in range(1, len(mp)):
        if up[i-1]:
            if cur_dir == 1: cur_streak += 1
            else: cur_streak = 1; cur_dir = 1
        elif dn[i-1]:
            if cur_dir == -1: cur_streak += 1
            else: cur_streak = 1; cur_dir = -1
        else:
            cur_streak = 0; cur_dir = 0
            continue
        
        if cur_streak >= 2:
            streaks.append({
                'dt': idx[i],
                'streak': cur_streak,
                'dir': cur_dir,
                'next_up': up[i] if i < len(up) else None,
                'next_dn': dn[i] if i < len(dn) else None,
                'mp': mp[i],
            })
    
    sdf = pd.DataFrame(streaks)
    if len(sdf) < 10: return sdf
    
    # Forward price change over 1,3,5 ticks
    mp_arr = mp
    for fd in [1, 3, 5]:
        sdf[f'fwd{fd}'] = 0.0
        for i in range(len(sdf)):
            pos = np.where(idx == sdf['dt'].iloc[i])[0]
            if len(pos) == 0: continue
            pi = pos[0]
            if pi + fd < len(mp_arr):
                sdf.loc[sdf.index[i], f'fwd{fd}'] = (mp_arr[pi+fd] - mp_arr[pi]) * sc
        sdf[f'af{fd}'] = sdf['dir'] * sdf[f'fwd{fd}']
        sdf[f'win{fd}'] = sdf[f'af{fd}'] > 0
    
    return sdf

def test_streak(sdf, phase, min_streak, hold, ndays):
    col = f'win{hold}'
    events = sdf['streak'] >= min_streak
    n = events.sum()
    if n < 10: return None
    wr = sdf.loc[events, col].mean()
    avg = sdf.loc[events, f'af{hold}'].mean()
    nd = n / ndays
    return {'n':n, 'nday':nd, 'wr':wr, 'avg':avg}

# ─── Strategy 3: Cross-pair flow (EURUSD · EURJPY coherence) ───
def cross_flow_signal(teurjpy, teurusd):
    """When EURUSD and EURJPY tick simultaneously, does the EUR
    direction predict JPY cross moves?"""
    # Align both to 1min bars
    pass  # Will implement if OFI/streaks are promising

# ─── RUN ───
all_t = load_all()

# Strategy 1: OFI
print("=" * 70)
print("STRATEGY 1: ORDER FLOW IMBALANCE (1min bars)")
print("=" * 70)
for pair in ['EURJPY', 'GBPJPY']:
    t = all_t[pair]
    print(f"\n--- {pair} ---")
    for phase_name, mask in [('IS', (t.index.year==2025)&(t.index.month.isin([10,11]))),
                              ('OOS', (t.index.year==2025)&(t.index.month==12))]:
        sub_t = t.loc[mask]
        if len(sub_t) < 1000: continue
        o = ofi_signal(sub_t, pair)
        if len(o) < 50: continue
        unique_dates = sub_t.index.normalize().nunique()
        ndays = int(unique_dates * 5 / 7)
        
        for z_thr in [1.0, 1.5, 2.0, 2.5]:
            for hold in [5, 10]:
                r = test_ofi(o, phase_name, z_thr, hold, ndays)
                if r is None: continue
                wr_s = f"{r['wr']:.1%}" if phase_name == 'OOS' else f"{r['wr']:.1%}"
                print(f"  {phase_name:<5s} z>{z_thr:.1f} h={hold:<2}min "
                      f"n={r['n']:>4d} {r['nday']:>5.2f}/d WR={wr_s} avg={r['avg']:+.3f}p")

# Strategy 2: Tick streaks
print("\n" + "=" * 70)
print("STRATEGY 2: TICK STREAK CONTINUATION")
print("=" * 70)
for pair in ['EURJPY', 'GBPJPY']:
    t = all_t[pair]
    print(f"\n--- {pair} ---")
    for phase_name, mask in [('IS', (t.index.year==2025)&(t.index.month.isin([10,11]))),
                              ('OOS', (t.index.year==2025)&(t.index.month==12))]:
        sub_t = t.loc[mask]
        if len(sub_t) < 1000: continue
        sdf = streak_signal(sub_t, pair)
        if len(sdf) < 10: continue
        total_ticks = len(sub_t)
        total_events = len(sdf)
        trades_per_day = total_events / 66  # ~66 trading days
        
        unique_dates = sub_t.index.normalize().nunique()
        ndays = int(unique_dates * 5 / 7)
        print(f"  {phase_name:<5s} total_ticks={total_ticks:,d} streak_events={total_events:,d} "
              f"({total_events/ndays:.0f}/d) ndays={ndays}")
        
        for min_s in [2, 3, 4, 5]:
            for hold in [1, 3, 5]:
                r = test_streak(sdf, phase_name, min_s, hold, ndays)
                if r is None: continue
                print(f"  {phase_name:<5s} streak>={min_s} h={hold}t "
                      f"n={r['n']:>6d} {r['nday']:>6.0f}/d WR={r['wr']:.1%} avg={r['avg']:+.3f}p")

print("\nDone")
