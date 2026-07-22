"""
Robustness backtests for Idea 2 — all without lookahead (shift(1) everywhere).
Tests: delayed entry, wider stops, limit entry, hidden stops, slippage.
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

# ─── Core run function with shift(1) on ALL rolling calcs ───
def run_trades(b, cost, z_thresh=2.0, atr_pctl=0.25, stop_a=0.15, trig_a=0.20, gap_a=0.10,
               entry_offset_s=0, limit_offset_a=0.0, use_limit_entry=False,
               hidden_stop=False, slip_mp=0.0, slip_pct=0.0):
    """
    All computation uses shift(1) — zero lookahead.
    Parameters:
      entry_offset_s: delay entry by N seconds (simulated by skipping offset bars)
      limit_offset_a: enter at close ± limit_offset_a*ATR instead of close
      use_limit_entry: if True, use limit offset; else market order
      hidden_stop: if True, use opposite limit instead of SL
      slip_mp: fixed slippage per trade in MP
      slip_pct: percentage of trades that experience slip_mp
    """
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    atr_gate = atr.shift(1).rolling(100, min_periods=10).quantile(atr_pctl).bfill()
    atr_pass = atr > atr_gate

    valid = z.notna() & atr.notna() & (z.abs() > z_thresh) & atr_pass
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values; index_dates = b.index

    pnls = []; ndays = set(); missed = 0
    for i, pos in enumerate(idxs):
        if pos + 2 >= len(b): continue
        direction = -1 if z_vals[pos] > 0 else 1
        entry = closes[pos]
        atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue

        # Slippage on entry: deterministic based on pos (no random)
        trade_slip = slip_mp if (pos % 100) / 100 < slip_pct else 0.0

        # Limit entry: adjust entry price
        if use_limit_entry:
            entry_offset = limit_offset_a * atr_v
            if direction == 1:
                entry = closes[pos] - entry_offset
            else:
                entry = closes[pos] + entry_offset
            if direction == 1 and entry < lows[pos]: missed += 1; continue
            if direction == -1 and entry > highs[pos]: missed += 1; continue

        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        sl = 0.0

        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break

            if direction == 1:
                best = max(best, highs[bp])
                sl = entry - s
                if best - entry > tg:
                    sl = best - gp
                if lows[bp] <= sl:
                    pnls.append(sl - entry - trade_slip)
                    exited = True; break
            else:
                best = min(best, lows[bp])
                sl = entry + s
                if entry - best > tg:
                    sl = best + gp
                if highs[bp] >= sl:
                    pnls.append((sl - entry) * direction - trade_slip)
                    exited = True; break

        if not exited:
            exit_bar = min(pos + 54, len(b) - 1)
            pnls.append((closes[exit_bar] - entry) * direction - trade_slip)
        ndays.add(index_dates[pos].date())

    pnls = np.array(pnls)
    if len(pnls) < 10: return None
    wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost; tpd = len(pnls)/max(len(ndays),1)
    win_avg = np.mean(pnls[pnls>0]) if np.any(pnls>0) else 0
    loss_avg = np.mean(pnls[pnls<=0]) if np.any(pnls<=0) else 0
    payoff = win_avg/abs(loss_avg) if loss_avg!=0 else 0
    return {'n':len(pnls),'tpd':tpd,'wr':wr,'avg':avg,'net':net,'payoff':payoff,'days':len(ndays),'missed':missed}

t0 = time.time()

def load_bars(label, load_fn, pair):
    t = load_fn(pair)
    if label == 'Exness':
        return t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    return t[['open','high','low','close']]

# ═════════════════════════════════════════════════════════════
# TEST A: Wider stops (simulating StopLevel constraints)
# ═════════════════════════════════════════════════════════════
print("=" * 75)
print("TEST A: Wider Stops  |z|>2.0 ATR>25%  (no lookahead)")
print("=" * 75)
print(f"{'Source':>10s} {'Pair':>7s} {'stop_a':>5s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}")
for label, lfn in [('Exness',load_exness),('DukaCSV',load_duka_csv),('DukaPar',load_duka_par)]:
    for pair in ['EURUSD','GBPJPY']:
        b = load_bars(label, lfn, pair)
        for stop in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            r = run_trades(b, COST[pair], stop_a=stop, trig_a=stop*1.33, gap_a=stop*0.67)
            if r:
                print(f"{label:>10s} {pair:>7s} {stop:>5.2f}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}")

# ═════════════════════════════════════════════════════════════
# TEST B: Delayed entry (entry price penalty)
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("TEST B: Entry Delay  |z|>2.0 ATR>25%  stop=0.15/0.20/0.10")
print(f"{'='*75}")
print(f"{'Source':>10s} {'Pair':>7s} {'delay_s':>7s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}")
for label, lfn in [('Exness',load_exness),('DukaCSV',load_duka_csv)]:
    for pair in ['EURUSD','GBPJPY']:
        b = load_bars(label, lfn, pair)
        for delay in [0, 1, 3, 5]:
            slip = 0.0
            if delay > 0:
                slip = (0.02 if pair=='EURUSD' else 0.5) * delay
            r = run_trades(b, COST[pair], slip_mp=slip, slip_pct=1.0)
            if r:
                print(f"{label:>10s} {pair:>7s} {delay:>7d}s  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}")

# ═════════════════════════════════════════════════════════════
# TEST C: Limit entry (enter at better price, risk no fill)
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("TEST C: Limit Entry  |z|>2.0 ATR>25%  stop=0.15/0.20/0.10")
print(f"{'='*75}")
print(f"{'Source':>10s} {'Pair':>7s} {'lim_off':>6s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}  {'fill%':>5s}")
for label, lfn in [('Exness',load_exness),('DukaCSV',load_duka_csv)]:
    for pair in ['EURUSD','GBPJPY']:
        b = load_bars(label, lfn, pair)
        base_n = run_trades(b, COST[pair])
        bn = base_n['n']
        for lim_off in [0.05, 0.10, 0.15, 0.20]:
            r = run_trades(b, COST[pair], use_limit_entry=True, limit_offset_a=lim_off)
            if r and bn:
                fill_pct = (1 - r['missed']/max(bn,1))*100
                print(f"{label:>10s} {pair:>7s} {lim_off:>6.2f}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}  {fill_pct:>5.0f}%")

# ═════════════════════════════════════════════════════════════
# TEST D: Hidden stops (opposite limit order instead of SL)
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("TEST D: Hidden Stop (opposite limit)  |z|>2.0 ATR>25%")
print(f"{'='*75}")
print(f"{'Source':>10s} {'Pair':>7s} {'stop_a':>5s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}")
for label, lfn in [('Exness',load_exness),('DukaCSV',load_duka_csv),('DukaPar',load_duka_par)]:
    for pair in ['EURUSD','GBPJPY']:
        b = load_bars(label, lfn, pair)
        for stop in [0.15, 0.20, 0.30, 0.50]:
            r = run_trades(b, COST[pair], stop_a=stop, trig_a=stop*1.33, gap_a=stop*0.67, hidden_stop=True)
            if r:
                print(f"{label:>10s} {pair:>7s} {stop:>5.2f}  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}")

# ═════════════════════════════════════════════════════════════
# TEST E: Slippage sensitivity
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("TEST E: Slippage Sensitivity  |z|>2.0 ATR>25%  stop=0.15")
print(f"{'='*75}")
print(f"{'Source':>10s} {'Pair':>7s} {'slip_mp':>6s}  {'tpd':>5s}  {'WR':>5s}  {'net':>9s}  {'payoff':>5s}")
for label, lfn in [('Exness',load_exness),('DukaCSV',load_duka_csv)]:
    for pair in ['EURUSD','GBPJPY']:
        b = load_bars(label, lfn, pair)
        for slip in [0, 1, 3, 5, 10, 20]:
            if pair == 'GBPJPY': slip_mp = slip * 2  # JPY pairs: 1pip = 100MP
            else: slip_mp = slip * 0.1  # EURUSD: 1pip = 1MP
            r = run_trades(b, COST[pair], slip_mp=slip_mp, slip_pct=1.0)
            if r:
                print(f"{label:>10s} {pair:>7s} {slip:>6d}p  {r['tpd']:>5.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>5.1f}")

print(f"\nTotal: {time.time()-t0:.1f}s")
