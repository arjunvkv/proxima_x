"""
HF-DF on Dukascopy M1 — completely independent sample (Oct2024–Jun2026).
No spread data → skip sr filter. Tests pure trailing stop on bid-close.
Trades per day will be higher since every bar fires.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

DUKA_DIR = Path(__file__).resolve().parents[3] / 'research' / 'dark_research' / 'dukascopy_data'
COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

def load_dukascopy(pair):
    files = sorted(DUKA_DIR.glob(f'{pair.lower()}-m1-bid-*.csv'))
    dfs = [pd.read_csv(f, usecols=['timestamp','open','high','low','close'],
                       parse_dates=['timestamp'])
           for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    df = df.set_index('timestamp').astype(float)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit='ms')
    # Convert to MP units (multiply by 10000) to match HF-DF convention
    df = df * 10000
    return df[['open','high','low','close']]

def hfdf_m1(b, cost):
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

    valid = z.notna() & atr.notna()
    idxs = np.where(valid)[0]
    if len(idxs) < 5:
        return np.array([])

    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
    max_bars = 54
    pnls = []
    closes = b['close'].values
    highs = b['high'].values
    lows = b['low'].values
    z_vals = z.values
    atr_vals = atr.values

    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -1 if z_vals[pos] > 0 else 1
        entry = closes[pos]
        atr_v = atr_vals[pos]
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry
        exited = False
        for j in range(1, max_bars + 1):
            bar_pos = pos + j
            if bar_pos >= len(b): break
            if direction == 1:
                if highs[bar_pos] > best: best = highs[bar_pos]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bar_pos] <= sl:
                    pnls.append((sl - entry))
                    exited = True; break
            else:
                if lows[bar_pos] < best: best = lows[bar_pos]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bar_pos] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True; break
        if not exited:
            exit_bar = min(pos + max_bars, len(b) - 1)
            pnls.append((closes[exit_bar] - entry) * direction)
    return np.array(pnls)

def run_test(pair):
    print(f"\n{'='*65}")
    print(f"{pair}  Dukascopy M1 bid  Oct2024–Jun2026")
    print(f"{'='*65}")
    b = load_dukascopy(pair)
    print(f"  Bars={len(b):,d}")

    # Split by date windows
    splits = [
        ('2024-08-01','2025-01-01','Q4 2024'),
        ('2025-01-01','2026-01-01','2025 (gap)'),
        ('2026-01-01','2026-05-01','Q1 2026'),
        ('2026-05-01','2026-07-01','Q2 2026'),
    ]
    all_p = []
    for sd, ed, label in splits:
        mask = (b.index >= sd) & (b.index < ed)
        if mask.sum() < 100: continue
        pnls = hfdf_m1(b[mask].copy(), COST[pair])
        all_p.append(pnls)
        if len(pnls) >= 5:
            ndays = b[mask].index.date.__len__()  # not perfect but fast
            ndays = len(set(b[mask].index.date))
            wr = np.mean(pnls > 0)
            avg = np.mean(pnls)
            net = avg - COST[pair]
            tpd = len(pnls)/max(ndays, 1)
            print(f"  {label}: n={len(pnls):>5d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")

    combined = np.concatenate(all_p)
    if len(combined) >= 5:
        wr = np.mean(combined > 0)
        avg = np.mean(combined)
        net = avg - COST[pair]
        ndays = len(set(b.index.date))
        tpd = len(combined)/max(ndays, 1)
        print(f"  TOTAL: n={len(combined):>5d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")

t0 = time.time()
for pair in ['EURUSD','EURJPY','GBPJPY']:
    run_test(pair)
print(f"\nTotal: {time.time()-t0:.1f}s")
