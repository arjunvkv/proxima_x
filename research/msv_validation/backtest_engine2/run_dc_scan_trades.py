"""
Scan DC configs for max trades/day while keeping WR>=70% OOS.
Tests: z threshold, recovery filter, hold time, recovered vs never-recovered.
Walk-forward: IS=Oct+Nov, OOS=Dec.
"""
import numpy as np, pandas as pd
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

def get_trades(b, t, pair, z_thr, r_mult):
    """Get all events with their recovery times, no filtering."""
    events_mask = (b['z'].abs() > z_thr) & b['sw']
    event_idx = np.where(events_mask.values)[0]
    trades = []
    for idx in event_idx:
        dt = b.index[idx]
        rm = b['rm'].iloc[idx]
        res, pk = measure_recovery(t, dt, rm, pair)
        if res is None or r_mult not in res: continue
        rec_t, fb, pp = res[r_mult]
        for hold in [5, 10, 15]:
            col = f'af{hold}'
            if col not in b.columns or idx >= len(b) - hold: continue
            trades.append({'dt': dt, 'rec_t': rec_t, 'result': b[col].iloc[idx],
                           'win': b[col].iloc[idx] > 0, 'hold': hold})
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def evaluate(trades, phase, thr_val, filter_type):
    """Print stats for a group of trades."""
    if len(trades) < 3: return
    for hold in [5, 10, 15]:
        sub = trades[trades['hold'] == hold]
        if len(sub) < 3: continue
        wr = sub['win'].mean()
        avg = sub['result'].mean()
        nd = len(sub) / 22
        print(f"  {phase:<14s} {filter_type:<12s} h={hold:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
              f"WR={wr:.1%} avg={avg:+.3f}p thr={thr_val:.0f}")

for pair in ['EURJPY', 'GBPJPY']:
    print(f"\n{'='*70}")
    print(f"{pair}")
    print(f"{'='*70}")
    t = load(pair)
    
    for z_thr in [2.0, 1.75, 1.5]:
        for r_mult in [1.3, 1.5]:
            print(f"\n--- z>{z_thr} r×{r_mult} ---")
            
            for phase_name, mask in [('IS', (t.index.year==2025)&(t.index.month.isin([10,11]))),
                                      ('OOS', (t.index.year==2025)&(t.index.month==12))]:
                sub_t = t.loc[mask]
                if len(sub_t) < 1000: continue
                b = get_bars(sub_t, pair)
                trades = get_trades(b, sub_t, pair, z_thr, r_mult)
                if len(trades) == 0: continue
                
                # Split trades
                all_trades = trades.copy()
                recovered = trades[trades['rec_t'] < 999]
                never = trades[trades['rec_t'] == 999]
                
                if len(recovered) < 3 and len(never) < 3: continue
                
                # Compute IS thresholds
                if phase_name == 'IS':
                    if len(recovered) >= 10:
                        rec_times = recovered['rec_t'].values
                        is_med = np.median(rec_times)
                        is_q3 = np.quantile(rec_times, 0.75)
                        is_q1 = np.quantile(rec_times, 0.25)
                        is_never_wr = never['win'].mean() if len(never) >= 5 else None
                        is_never_avg = never['result'].mean() if len(never) >= 5 else None
                        is_never_n = len(never)
                        is_all_wr = all_trades['win'].mean()
                        is_all_avg = all_trades['result'].mean()
                        is_all_n = len(all_trades)
                        print(f"  IS recovered: n={len(recovered)} Q1={is_q1:.0f} med={is_med:.0f} Q3={is_q3:.0f}")
                        print(f"  IS never:     n={is_never_n} WR={is_never_wr:.1%} avg={is_never_avg:+.3f}p" if is_never_wr is not None else f"  IS never:     n={is_never_n} (too few)")
                        print(f"  IS all:       n={is_all_n} WR={is_all_wr:.1%} avg={is_all_avg:+.3f}p")
                    else:
                        print(f"  IS: only {len(recovered)} recovered, skipping")
                        continue
                
                # OOS: apply IS thresholds
                else:
                    if len(recovered) < 3: continue
                    
                    # Q4_SLOW: r > Q3 from IS
                    q4 = recovered[recovered['rec_t'] > is_q3]
                    # MED_SPLIT: r > med from IS
                    med_split = recovered[recovered['rec_t'] > is_med]
                    # ALL recovered
                    all_rec = recovered
                    # ALL trades
                    all_all = all_trades
                    # NEVER
                    never_grp = never
                    
                    eval_order = [
                        ('Q4_SLOW', q4, is_q3),
                        ('MED_SPLIT', med_split, is_med),
                        ('ALL_REC', all_rec, is_med),
                        ('ALL+NEVER', all_all, is_med),
                    ]
                    if len(never) >= 3:
                        eval_order.append(('NEVER', never, 999))
                    
                    for fname, grp, thr in eval_order:
                        if len(grp) >= 3:
                            evaluate(grp, 'OOS', thr, fname)

print(f"\n{'='*70}")
print("DONE")
