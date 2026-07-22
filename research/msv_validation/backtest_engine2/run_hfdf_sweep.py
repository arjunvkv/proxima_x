"""
HF-DF 10s bar — sweep trailing stop parameters for max WR.
EURUSD Oct 2025. sr>1.05 (all events).
"""
import numpy as np, pandas as pd, time
from pathlib import Path
from itertools import product

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
t0 = time.time()

fn = TICK_DIR / 'EURUSD_Raw_Spread_2025_10.zip'
d = pd.read_csv(fn, compression='zip',
    names=['E','S','Ts','B','A'], skiprows=1, header=None,
    dtype={'Ts':str,'B':np.float64,'A':np.float64})
d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
    format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
t = d.dropna(subset=['Ts']).set_index('Ts')
t['MP'] = ((t['B']+t['A'])/2) * 10000
t['Sprd'] = (t['A']-t['B']) * 10000

b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
b['ret'] = b['close'].diff()
b['z'] = (b['ret'] - b['ret'].rolling(50).mean()) / b['ret'].rolling(50).std().clip(1e-8)
sp_max = t['Sprd'].resample('10s').max()
sp_med = t['Sprd'].resample('10s').median().fillna(method='ffill')
b['rm'] = sp_med.rolling(30, min_periods=1).median().fillna(method='ffill')
b['sr'] = sp_max / b['rm'].clip(1e-8)
b['atr'] = (b['high'] - b['low']).rolling(20).mean().clip(1e-8)

mask = (b['sr'].fillna(0) > 1.05) & b['z'].notna()
events = b[mask]
ndays = 22
print(f"Events: {len(events)} ({len(events)/ndays:.1f}/d)")
print(f"{'stop':<6s} {'trig':<6s} {'gap':<6s} {'WR':>6s} {'avg':>7s} {'net':>7s} {'tpd':>6s}")

best_wr = 0
results = []

for stop, trig, gap in product(
    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
    [0.1, 0.15, 0.2, 0.3],
):
    if gap >= stop: continue
    if trig <= gap: continue

    pnls = []
    max_bars = 54
    for idx in events.index:
        pos = b.index.get_loc(idx)
        if pos + 2 >= len(b): continue
        row = b.iloc[pos]
        z = row['z']
        direction = -1 if z > 0 else 1
        entry = row['close']
        atr = row['atr']

        s = stop * atr
        tg = trig * atr
        gp = gap * atr
        best = entry
        exited = False

        for j in range(1, max_bars + 1):
            if pos + j >= len(b): break
            bar = b.iloc[pos + j]

            if direction == 1:
                best = max(best, bar['high'])
                sl = entry - s
                if best - entry > tg:
                    sl = best - gp
                if bar['low'] <= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
            else:
                best = min(best, bar['low'])
                sl = entry + s
                if entry - best > tg:
                    sl = best + gp
                if bar['high'] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break

        if not exited:
            exit_px = b['close'].iloc[min(pos + max_bars, len(b) - 1)]
            pnls.append((exit_px - entry) * direction)

    wr = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
    avg = np.mean(pnls) if pnls else 0
    net = avg - 0.15
    tpd = len(pnls) / ndays

    if wr >= 0.60:
        print(f"  {stop:<5.2f} {trig:<5.2f} {gap:<5.2f} {wr:>5.1%} {avg:>+6.2f} {net:>+6.2f} {tpd:>5.1f}")

    if net > 0 and (wr > best_wr or (wr == best_wr and net > best_net)):
        best_wr = wr
        best_stop = stop
        best_trig = trig
        best_gap = gap
        best_avg = avg
        best_net = net
        best_tpd = tpd
    results.append((wr, stop, trig, gap, avg, net, tpd))

print(f"\nBest WR: {best_wr:.1%} stop={best_stop} trig={best_trig} gap={best_gap} "
      f"avg={best_avg:+.2f} net={best_net:+.2f} tpd={best_tpd:.1f}")

# Also show top 5
results.sort(key=lambda r: -r[0])
print(f"\nTop 5 by WR (net>0):")
for wr, stop, trig, gap, avg, net, tpd in results[:5]:
    if net > 0:
        print(f"  WR={wr:.1%} stop={stop} trig={trig} gap={gap} avg={avg:+.2f} net={net:+.2f} tpd={tpd:.1f}")

print(f"\nTime: {time.time()-t0:.1f}s")
