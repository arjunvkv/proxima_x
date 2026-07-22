"""
HF-DF on 10-second bars. sr as primary filter, z for direction.
Trailing stop exit. EURUSD Oct 2025 only.
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

# 10s bars
b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
b['ret'] = b['close'] - b['open']
b['ret_prev'] = b['close'].diff()  # close-to-close return for z
b['z'] = (b['ret_prev'] - b['ret_prev'].rolling(50).mean()) / b['ret_prev'].rolling(50).std().clip(1e-8)

sp_max = t['Sprd'].resample('10s').max()
sp_med = t['Sprd'].resample('10s').median().fillna(method='ffill')
b['rm'] = sp_med.rolling(30, min_periods=1).median().fillna(method='ffill')
b['sr'] = sp_max / b['rm'].clip(1e-8)

b['atr'] = (b['high'] - b['low']).rolling(20).mean().clip(1e-8)

ndays = 22
print(f"10s bars: {len(b):,d}")
print(f"{'sr_thr':<7s} {'n':>5s} {'t/d':>6s} {'WR_fix':>7s} {'avg_fix':>8s} "
      f"{'trailWR':>7s} {'trailAvg':>8s} {'t_tpd':>6s} {'net':>8s}")

best_net = -999

for sr_thr in [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0]:
    mask = (b['sr'].fillna(0) > sr_thr) & b['z'].notna()
    events = b[mask]
    n_raw = len(events)
    if n_raw < 10: continue
    tpd = n_raw / ndays

    # Fixed hold 18 bars (3 min)
    pnls_fix = []
    for idx in events.index:
        pos = b.index.get_loc(idx)
        if pos + 18 >= len(b): continue
        z = b['z'].iloc[pos]
        direction = -1 if z > 0 else 1
        entry = b['close'].iloc[pos]
        fwd = b['ret'].iloc[pos+1:pos+19].sum()
        pnls_fix.append(fwd * direction)
    wr_fix = sum(1 for p in pnls_fix if p > 0) / max(len(pnls_fix), 1)
    avg_fix = np.mean(pnls_fix) if pnls_fix else 0

    # Trailing stop on 10s bars
    pnls = []
    max_bars = 54  # 9 minutes max hold (3× original)
    for idx in events.index:
        pos = b.index.get_loc(idx)
        if pos + 3 >= len(b): continue
        row = b.iloc[pos]
        z = row['z']
        direction = -1 if z > 0 else 1
        entry = row['close']
        atr = row['atr']

        stop = 0.3 * atr
        trail_trig = 0.5 * atr
        trail_gap = 0.15 * atr
        best = entry
        exited = False

        for j in range(1, max_bars + 1):
            if pos + j >= len(b): break
            bar = b.iloc[pos + j]

            if direction == 1:
                best = max(best, bar['high'])
                sl = entry - stop
                if best - entry > trail_trig:
                    sl = best - trail_gap
                if bar['low'] <= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
            else:
                best = min(best, bar['low'])
                sl = entry + stop
                if entry - best > trail_trig:
                    sl = best + trail_gap
                if bar['high'] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break

        if not exited:
            exit_px = b['close'].iloc[min(pos + max_bars, len(b) - 1)]
            pnls.append((exit_px - entry) * direction)

    trail_wr = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
    trail_avg = np.mean(pnls) if pnls else 0
    trail_net = trail_avg - 0.15
    ttpd = len(pnls) / ndays

    print(f"  {sr_thr:<6.2f} {n_raw:>5d} {tpd:>5.1f} {wr_fix:>6.1%} {avg_fix:>+7.2f} "
          f"{trail_wr:>6.1%} {trail_avg:>+7.2f} {ttpd:>5.1f} {trail_net:>+7.2f}")

    if trail_net > best_net and trail_wr > 0.55:
        best_net = trail_net
        best_sr = sr_thr
        best_wr = trail_wr
        best_avg = trail_avg
        best_ttpd = ttpd

print(f"\nBest: sr>{best_sr:.2f} trail_WR={best_wr:.1%} avg={best_avg:+.2f} net={best_net:+.2f} tpd={best_ttpd:.1f}")
print(f"Time: {time.time()-t0:.1f}s")
