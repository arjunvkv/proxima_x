"""
Overfit resistance tests for Idea 2 (z>2.0 + ATR>25%).
Tests: direction reversal, parameter sweep, time-split CV, source cross-validation.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

EXNESS_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
DUKA_CSV_DIR = Path(__file__).resolve().parents[3] / 'research' / 'dark_research' / 'dukascopy_data'
DUKA_PAR_DIR = Path(__file__).resolve().parents[3] / 'research' / 'phase_dislocation' / 'dukascopy_data'

COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

def load_exness(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = EXNESS_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    return df.set_index('Ts')

def load_duka_csv(pair):
    files = sorted(DUKA_CSV_DIR.glob(f'{pair.lower()}-m1-bid-*.csv'))
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms')
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    df = df.set_index('timestamp').astype(float) * 10000
    return df[['open','high','low','close']]

def load_duka_par(pair):
    df = pd.read_parquet(DUKA_PAR_DIR / f'{pair.lower()}.parquet')
    df = df.set_index('timestamp').astype(float) * 10000
    return df[['open','high','low','close']]

def run_with_params(b, cost, z_thresh, atr_pctl, stop_a, trig_a, gap_a, direction_sign):
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    atr_gate = atr.rolling(100, min_periods=10).quantile(atr_pctl).bfill()
    atr_pass = atr > atr_gate

    valid = z.notna() & atr.notna() & (z.abs() > z_thresh) & atr_pass
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values; index_dates = b.index

    pnls = []; ndays = set()
    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -direction_sign if z_vals[pos] > 0 else direction_sign
        entry = closes[pos]; atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                best = max(best, highs[bp]); sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl: pnls.append(sl - entry); exited = True; break
            else:
                best = min(best, lows[bp]); sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
        if not exited:
            pnls.append((closes[min(pos+54, len(b)-1)] - entry) * direction)
        ndays.add(index_dates[pos].date())

    pnls = np.array(pnls)
    if len(pnls) < 10: return None
    wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost; tpd = len(pnls)/max(len(ndays),1)
    win_avg = np.mean(pnls[pnls>0]) if np.any(pnls>0) else 0
    loss_avg = np.mean(pnls[pnls<=0]) if np.any(pnls<=0) else 0
    payoff = win_avg/abs(loss_avg) if loss_avg!=0 else 0
    return {'n':len(pnls),'tpd':tpd,'wr':wr,'avg':avg,'net':net,'payoff':payoff,'days':len(ndays)}

t0 = time.time()

# ═══════════════════════════════════════════════════════════
# TEST 1: Direction reversal — trade WITH z instead of AGAINST
# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("TEST 1: Direction Reversal  (trade WITH sign(z) instead of AGAINST)")
print("=" * 65)
for pair in ['EURUSD','EURJPY','GBPJPY']:
    t = load_exness(pair)
    b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    r_against = run_with_params(b, COST[pair], 2.0, 0.25, 0.15, 0.20, 0.10, -1)  # -sign(z) = against
    r_with = run_with_params(b, COST[pair], 2.0, 0.25, 0.15, 0.20, 0.10, 1)      # +sign(z) = with
    if r_against:
        print(f"  {pair:>7s}  AGAINST z: tpd={r_against['tpd']:5.0f}  WR={r_against['wr']:5.1%}  net={r_against['net']:+.2f}  payoff={r_against['payoff']:.1f}")
    if r_with:
        print(f"  {pair:>7s}  WITH z:    tpd={r_with['tpd']:5.0f}  WR={r_with['wr']:5.1%}  net={r_with['net']:+.2f}  payoff={r_with['payoff']:.1f}")

# ═══════════════════════════════════════════════════════════
# TEST 2: Parameter sensitivity — sweep stop/trigger/gap
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TEST 2: Parameter Sensitivity  (GBPJPY, Exness)")
print(f"{'='*65}")
print(f"{'stop':>5s} {'trig':>5s} {'gap':>5s}  {'tpd':>5s}  {'WR':>5s}  {'net':>8s}  {'payoff':>5s}")
for stop in [0.10, 0.15, 0.20, 0.30]:
    for trig in [1.0, 1.33, 1.5]:
        for gap in [0.50, 0.67, 0.75]:
            t = load_exness('GBPJPY')
            b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
            r = run_with_params(b, COST['GBPJPY'], 2.0, 0.25, stop, stop*trig, stop*gap, -1)
            if r:
                print(f"{stop:>5.2f} {stop*trig:>5.2f} {stop*gap:>5.2f}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+8.2f}  {r['payoff']:>5.1f}")

# ═══════════════════════════════════════════════════════════
# TEST 3: Source cross-validation (train Exness, test Duka CSV)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TEST 3: Source Cross-Validation  (train=Exness, test=DukaCSV)")
print(f"{'='*65}")
print(f"{'Pair':>7s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}  {'n':>6s}")
for pair in ['EURUSD','EURJPY','GBPJPY']:
    b = load_duka_csv(pair)
    r = run_with_params(b, COST[pair], 2.0, 0.25, 0.15, 0.20, 0.10, -1)
    if r:
        print(f"{pair:>7s}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}  {r['n']:>6d}")

# Also test Duka Parquet
print(f"\n{'='*65}")
print("TEST 3b: Source Cross-Validation  (train=Exness, test=DukaPar)")
print(f"{'='*65}")
for pair in ['EURUSD','EURJPY','GBPJPY']:
    b = load_duka_par(pair)
    r = run_with_params(b, COST[pair], 2.0, 0.25, 0.15, 0.20, 0.10, -1)
    if r:
        print(f"{pair:>7s}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}  {r['n']:>6d}")

# ═══════════════════════════════════════════════════════════
# TEST 4: Temporal CV (each month as OOS, trained on all others)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TEST 4: Temporal Cross-Validation  (GBPJPY, Exness, per-month)")
print(f"{'='*65}")
t = load_exness('GBPJPY')
b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
for y,m,mname in [(2025,10,'Oct'),(2025,11,'Nov'),(2025,12,'Dec')]:
    mask = (b.index.year == y) & (b.index.month == m)
    r = run_with_params(b[mask], COST['GBPJPY'], 2.0, 0.25, 0.15, 0.20, 0.10, -1)
    if r:
        print(f"  {mname}: tpd={r['tpd']:5.0f}  WR={r['wr']:5.1%}  net={r['net']:+.2f}  payoff={r['payoff']:.1f}  n={r['n']:5d}")

# ═══════════════════════════════════════════════════════════
# TEST 5: The null hypothesis test — random direction
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TEST 5: Random Direction Baseline  (EURUSD, 10 runs avg)")
print(f"{'='*65}")
np.random.seed(42)
t = load_exness('EURUSD')
b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
ret = b['close'].diff()
z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
atr_gate = atr.rolling(100, min_periods=10).quantile(0.25).bfill()
valid = z.notna() & atr.notna() & (z.abs() > 2.0) & (atr > atr_gate)
base_idxs = np.where(valid)[0]

rand_wrs = []
for run_i in range(10):
    dirs = np.random.choice([-1, 1], size=len(base_idxs))
    pnls = []; ndays = set()
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values; atr_vals = atr.values
    index_dates = b.index
    for ri, pos in enumerate(base_idxs):
        if pos + 2 >= len(b): continue
        direction = dirs[ri]
        entry = closes[pos]; atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue
        s = 0.15*atr_v; tg = 0.20*atr_v; gp = 0.10*atr_v
        best = entry; exited = False
        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                best = max(best, highs[bp]); sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl: pnls.append(sl - entry); exited = True; break
            else:
                best = min(best, lows[bp]); sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
        if not exited:
            pnls.append((closes[min(pos+54, len(b)-1)] - entry) * direction)
        ndays.add(index_dates[pos].date())
    pnls = np.array(pnls)
    wr_r = np.mean(pnls > 0); avg_r = np.mean(pnls); net_r = avg_r - COST['EURUSD']
    rand_wrs.append(wr_r)
    if run_i < 3:
        print(f"  Run {run_i+1}: WR={wr_r:.1%}  avg={avg_r:+.2f}  net={net_r:+.2f}")

print(f"  Random baseline: avg WR = {np.mean(rand_wrs):.1%} (should be ~50%)")

print(f"\nTotal: {time.time()-t0:.0f}s")
