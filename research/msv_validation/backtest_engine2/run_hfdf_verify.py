"""
HF-DF Code Verification — strict no-lookahead, per-month cross-validation.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'

def load_pair(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    df['Sprd'] = (df['A']-df['B']) * 10000
    return df.set_index('Ts')

def test_pair(pair, cost, stop_a, trig_a, gap_a, sr_thr, label):
    t = load_pair(pair)
    print(f"\n{'='*70}")
    print(f"{pair}  stop={stop_a} trig={trig_a} gap={gap_a} sr>{sr_thr}  ({label})")
    print(f"{'='*70}")

    b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    b['ret'] = b['close'].diff()
    # STRICT no-lookahead: z uses mean/std from previous 50 bars only
    b['z'] = (b['ret'] - b['ret'].shift(1).rolling(50).mean()) / b['ret'].shift(1).rolling(50).std().clip(1e-8)
    sp_max = t['Sprd'].resample('10s').max()
    sp_med = t['Sprd'].resample('10s').median().fillna(method='ffill')
    # STRICT no lookahead: rm from previous 30 bars only
    b['rm'] = sp_med.shift(1).rolling(30, min_periods=1).median().fillna(method='ffill')
    b['sr'] = sp_max / b['rm'].clip(1e-8)
    # STRICT no lookahead: atr from previous 20 bars only
    b['atr_v'] = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

    # Test each month as OOS against the other two as IS
    months = [(2025,10,'Oct'), (2025,11,'Nov'), (2025,12,'Dec')]
    
    for y, m, mname in months:
        mask = (b.index.year == y) & (b.index.month == m)
        other_mask = ~mask
        train = b[other_mask].dropna(subset=['z','sr','atr_v'])
        test = b[mask].dropna(subset=['z','sr','atr_v'])

        for phase_name, phase in [('IS', train), ('OOS', test)]:
            if len(phase) < 50: continue
            events = phase[phase['sr'] > sr_thr]
            n_raw = len(events)
            if n_raw < 5: continue
            
            ndays = len(set(idx.date() for idx in events.index)) if len(events) > 0 else 1
            if ndays < 2: ndays = 22 if y == 2025 and (m == 12) else 44

            pnls = []
            max_bars = 54
            for idx in events.index:
                pos = b.index.get_loc(idx)
                if pos + 2 >= len(b): continue
                row = b.loc[idx]
                z = row['z']
                direction = -1 if z > 0 else 1
                entry = row['close']
                atr = row['atr_v']

                if np.isnan(atr) or atr < 1e-10: continue
                s = stop_a * atr; tg = trig_a * atr; gp = gap_a * atr
                best = entry; exited = False

                for j in range(1, max_bars + 1):
                    if pos + j >= len(b): break
                    bar = b.iloc[pos + j]
                    if direction == 1:
                        best = max(best, bar['high'])
                        sl = entry - s
                        if best - entry > tg: sl = best - gp
                        if bar['low'] <= sl:
                            pnls.append((sl - entry) * direction)
                            exited = True; break
                    else:
                        best = min(best, bar['low'])
                        sl = entry + s
                        if entry - best > tg: sl = best + gp
                        if bar['high'] >= sl:
                            pnls.append((sl - entry) * direction)
                            exited = True; break
                if not exited:
                    exit_px = b['close'].iloc[min(pos + max_bars, len(b)-1)]
                    pnls.append((exit_px - entry) * direction)

            if len(pnls) < 5: continue
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            avg = np.mean(pnls)
            net = avg - cost
            tpd_adj = max(len(pnls) / max(ndays, 1), 0)
            print(f"  {mname} {phase_name:<4s} sr>{sr_thr:.2f} n={len(pnls):>5d} {tpd_adj:>5.1f}/d "
                  f"WR={wr:.1%} avg={avg:+.2f} net={net:+.2f}")

    # Full 3-month result with STRICT no lookahead
    b_clean = b.dropna(subset=['z','sr','atr_v'])
    events = b_clean[b_clean['sr'] > sr_thr]
    if len(events) >= 5:
        pnls = []
        for idx in events.index:
            pos = b.index.get_loc(idx)
            if pos + 2 >= len(b): continue
            row = b.loc[idx]
            z = row['z']; direction = -1 if z > 0 else 1
            entry = row['close']; atr = row['atr_v']
            if np.isnan(atr) or atr < 1e-10: continue
            s = stop_a * atr; tg = trig_a * atr; gp = gap_a * atr
            best = entry; exited = False
            for j in range(1, 55):
                if pos + j >= len(b): break
                bar = b.iloc[pos + j]
                if direction == 1:
                    best = max(best, bar['high'])
                    sl = entry - s
                    if best - entry > tg: sl = best - gp
                    if bar['low'] <= sl: pnls.append((sl - entry) * direction); exited = True; break
                else:
                    best = min(best, bar['low'])
                    sl = entry + s
                    if entry - best > tg: sl = best + gp
                    if bar['high'] >= sl: pnls.append((sl - entry) * direction); exited = True; break
            if not exited:
                pnls.append((b['close'].iloc[min(pos+54, len(b)-1)] - entry) * direction)
        if len(pnls) >= 5:
            wr = sum(1 for p in pnls if p>0)/len(pnls)
            avg = np.mean(pnls)
            net = avg - cost
            tpd = len(pnls)/66
            print(f"  ALL <<< n={len(pnls):>5d} {tpd:>5.1f}/d WR={wr:.1%} avg={avg:+.2f} net={net:+.2f} >>>")

t0 = time.time()

# Test both pair + best configs
test_pair('EURUSD', 0.15, 0.15, 0.20, 0.10, 1.05, 'tight')
test_pair('GBPJPY', 60, 0.15, 0.20, 0.10, 1.05, 'tight')
test_pair('EURJPY', 50, 0.15, 0.20, 0.10, 1.05, 'tight')

# Also try higher sr threshold on EURJPY
test_pair('EURJPY', 50, 0.15, 0.20, 0.10, 1.50, 'tight sr>1.5')

print(f"\nTotal: {time.time()-t0:.1f}s")
print("\nNOTE: All z/atr/rm use shift(1) to eliminate lookahead.")
