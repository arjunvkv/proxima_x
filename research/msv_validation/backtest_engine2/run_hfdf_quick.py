"""
HF-DF Quick Test — High-Frequency Dealer Flow with trailing stops.
Tests sr thresholds 1.15-2.0: fixed-hold vs trailing-stop exit.
Small sample: EURUSD Oct 2025 (~22 trading days).
"""
import numpy as np, pandas as pd, time
from pathlib import Path

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
print(f"EURUSD Oct: {len(t):,d} ticks ({time.time()-t0:.1f}s)")

b1 = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
b1['ret'] = b1['close'] - b1['open']
b1['v'] = b1['ret'].rolling(20).std()
b1['z'] = b1['ret'] / b1['v'].clip(1e-8)

sp_max = t['Sprd'].resample('1min').max()
sp_med = t['Sprd'].resample('1min').median().fillna(method='ffill')
b1['rm'] = sp_med.rolling(20, min_periods=1).median().fillna(method='ffill')
b1['sr'] = sp_max / b1['rm'].clip(1e-8)

# Fixed-hold baseline
b1['fwd10'] = b1['ret'].rolling(10).sum().shift(-10)
b1['af10'] = -np.sign(b1['z'].fillna(0)) * b1['fwd10'].fillna(0)

# ATR from 10-bar high-low range
b1['atr'] = (b1['high'] - b1['low']).rolling(10).mean().clip(1e-8)

ndays = 22
print(f"{'SR':<6s} {'n':>4s} {'t/d':>5s} {'WR_f10':>8s} {'avg_f10':>9s} "
      f"{'trailWR':>8s} {'trailAvg':>9s} {'t_tpd':>6s}")

best_net = -999

for sr_thr in [1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0]:
    mask = (b1['sr'] > sr_thr)
    mask &= b1['z'].notna() & b1['af10'].notna()
    events = b1[mask]
    n_raw = len(events)
    if n_raw < 5: continue

    tpd = n_raw / ndays
    wr10 = events['af10'].mean()  # NOPE: this is avg
    wr10 = (events['af10'] > 0).mean()
    avg10 = events['af10'].mean()

    # Trailing stop
    pnls = []
    for idx in events.index:
        pos = b1.index.get_loc(idx)
        if pos + 10 >= len(b1): continue
        row = b1.iloc[pos]
        z = row['z']
        direction = -1 if z > 0 else 1
        entry = row['close']
        atr = row['atr']

        stop_dist = 0.3 * atr
        trail_trig = 0.5 * atr
        trail_gap = 0.2 * atr
        max_bars = 10
        best = entry
        exited = False

        for j in range(1, max_bars + 1):
            bar = b1.iloc[pos + j]

            if direction == 1:  # long
                best = max(best, bar['high'])
                sl = entry - stop_dist
                if best - entry > trail_trig:
                    sl = best - trail_gap
                if bar['low'] <= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
            else:  # short
                best = min(best, bar['low'])
                sl = entry + stop_dist
                if entry - best > trail_trig:
                    sl = best + trail_gap
                if bar['high'] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break

        if not exited:
            exit_px = b1['close'].iloc[min(pos + max_bars, len(b1) - 1)]
            pnls.append((exit_px - entry) * direction)

    trail_wr = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
    trail_avg = np.mean(pnls) if pnls else 0
    trail_net = trail_avg - 0.15  # spread cost
    ttpd = len(pnls) / ndays

    print(f"  {sr_thr:<6.2f} {n_raw:>4d} {tpd:>5.1f} {wr10:>7.1%} {avg10:>+8.2f} "
          f"{trail_wr:>7.1%} {trail_avg:>+8.2f} {ttpd:>5.1f}")

    if trail_net > best_net:
        best_net = trail_net
        best_sr = sr_thr
        best_wr = trail_wr
        best_avg = trail_avg
        best_ttpd = ttpd

print(f"\nBest: sr>{best_sr} trail_WR={best_wr:.1%} avg={best_avg:+.2f} net={best_net:+.2f} tpd={best_ttpd:.1f}")
print(f"Time: {time.time()-t0:.1f}s")
