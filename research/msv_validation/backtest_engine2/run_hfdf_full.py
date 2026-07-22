"""
HF-DF Full Validation — all 3 pairs × 3 months (Oct/Nov/Dec).
Best params from sweep: stop=0.5, trig=0.2, gap=0.1
Also test tighter stops.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025,10),(2025,11),(2025,12)]
COST = {'EURUSD':0.15, 'EURJPY':50, 'GBPJPY':60}

def load_pair(pair, scale):
    dfs = []
    for y,m in MONTHS:
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

t0 = time.time()
pairs = [('EURUSD',1), ('EURJPY',100), ('GBPJPY',100)]

for pair, scale in pairs:
    t = load_pair(pair, scale)
    cost = COST[pair]
    print(f"\n{'='*70}\n{pair}  ({len(t):,d} ticks  {time.time()-t0:.1f}s)\n{'='*70}")

    b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    b['ret'] = b['close'].diff()
    b['z'] = (b['ret'] - b['ret'].rolling(50).mean()) / b['ret'].rolling(50).std().clip(1e-8)
    sp_max = t['Sprd'].resample('10s').max()
    sp_med = t['Sprd'].resample('10s').median().fillna(method='ffill')
    b['rm'] = sp_med.rolling(30, min_periods=1).median().fillna(method='ffill')
    b['sr'] = sp_max / b['rm'].clip(1e-8)
    b['atr'] = (b['high'] - b['low']).rolling(20).mean().clip(1e-8)

    ndays_all = 66
    ndays_oos = 22

    # Test multiple configs
    configs = [
        (0.15, 0.20, 0.10, 'tight'),
        (0.20, 0.20, 0.10, 'med'),
        (0.30, 0.20, 0.10, 'loose'),
        (0.50, 0.20, 0.10, 'wide'),
    ]

    for sr_thr in [1.05, 1.10, 1.15, 1.20]:
        mask = (b['sr'].fillna(0) > sr_thr) & b['z'].notna()
        events = b[mask]
        n_raw = len(events)
        if n_raw < 10: continue

        for stop_a, trig_a, gap_a, label in configs:
            pnls = []
            max_bars = 54
            for idx in events.index:
                pos = b.index.get_loc(idx)
                if pos + 2 >= len(b): continue
                row = b.iloc[pos]
                z = row['z']; direction = -1 if z > 0 else 1
                entry = row['close']; atr = row['atr']
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
            tpd = len(pnls) / ndays_all

            # OOS (Dec only)
            oos_pnls = []
            for idx in events.index:
                if idx < pd.Timestamp('2025-12-01'): continue
                pos = b.index.get_loc(idx)
                if pos + 2 >= len(b): continue
                row = b.iloc[pos]
                z = row['z']; direction = -1 if z > 0 else 1
                entry = row['close']; atr = row['atr']
                s = stop_a * atr; tg = trig_a * atr; gp = gap_a * atr
                best = entry; exited = False
                for j in range(1, max_bars + 1):
                    if pos + j >= len(b): break
                    bar = b.iloc[pos + j]
                    if direction == 1:
                        best = max(best, bar['high'])
                        sl = entry - s
                        if best - entry > tg: sl = best - gp
                        if bar['low'] <= sl: oos_pnls.append((sl-entry)*direction); exited=True; break
                    else:
                        best = min(best, bar['low'])
                        sl = entry + s
                        if entry - best > tg: sl = best + gp
                        if bar['high'] >= sl: oos_pnls.append((sl-entry)*direction); exited=True; break
                if not exited:
                    exit_px = b['close'].iloc[min(pos + max_bars, len(b)-1)]
                    oos_pnls.append((exit_px - entry) * direction)

            if len(oos_pnls) >= 5:
                oos_wr = sum(1 for p in oos_pnls if p>0) / len(oos_pnls)
                oos_avg = np.mean(oos_pnls)
                oos_net = oos_avg - cost
                oos_tpd = len(oos_pnls) / ndays_oos
                if tpd >= 10:
                    print(f"  sr>{sr_thr:.2f} {label:<5s} n={len(pnls):>5d} {tpd:>5.1f}/d "
                          f"WR={wr:.1%} avg={avg:+.2f} net={net:+.2f}"
                          f" | OOS n={len(oos_pnls):>4d} {oos_tpd:.1f}/d WR={oos_wr:.1%} avg={oos_avg:+.2f} net={oos_net:+.2f}")

print(f"\nTotal: {time.time()-t0:.1f}s")
