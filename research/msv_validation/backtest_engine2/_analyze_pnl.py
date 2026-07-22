"""Analyze PnL distribution to explain high WR / low per-trade."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

def load(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    return df.set_index('Ts')

def run(pair, cost):
    t = load(pair)
    b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    valid = z.notna() & atr.notna()
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values
    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10

    pnls = []; exits = []; durations = []
    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -1 if z_vals[pos] > 0 else 1
        entry = closes[pos]; atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                best = max(best, highs[bp])
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl:
                    pnls.append(sl - entry); exits.append('stop'); durations.append(j); exited=True; break
            else:
                best = min(best, lows[bp])
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl:
                    pnls.append((sl - entry) * direction); exits.append('stop'); durations.append(j); exited=True; break
        if not exited:
            exit_px = closes[min(pos+54, len(b)-1)]
            pnls.append((exit_px - entry) * direction)
            exits.append('timeout'); durations.append(54)

    pnls = np.array(pnls); durations = np.array(durations)
    wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost
    avg_win = np.mean(pnls[pnls > 0]) if np.any(pnls > 0) else 0
    avg_loss = np.mean(pnls[pnls <= 0]) if np.any(pnls <= 0) else 0
    avg_dur = np.mean(durations)
    avg_win_dur = np.mean(durations[pnls > 0]) if np.any(pnls > 0) else 0
    avg_loss_dur = np.mean(durations[pnls <= 0]) if np.any(pnls <= 0) else 0

    # ATR stats
    valid_atr = atr.dropna()
    atr_mean = valid_atr.mean()
    stop_px = 0.15 * atr_mean
    trig_px = 0.20 * atr_mean
    gap_px = 0.10 * atr_mean

    print(f"\n{'='*65}")
    print(f"{pair}  M1  (cost={cost})")
    print(f"{'='*65}")
    print(f"  ATR mean: {atr_mean:.2f}")
    print(f"  stop={stop_px:.2f}  trig={trig_px:.2f}  gap={gap_px:.2f}")
    print(f"  n={len(pnls):,d}  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")
    print(f"  avg win={avg_win:+.2f}  avg loss={avg_loss:+.2f}  payoff ratio={avg_win/abs(avg_loss):.2f}")
    print(f"  avg dur={avg_dur:.1f}m  win dur={avg_win_dur:.1f}m  loss dur={avg_loss_dur:.1f}m")
    print(f"  spread cost as % of gross: {cost/abs(avg)*100 if avg else 0:.1f}%")
    print(f"  Required edge per trade (gross): avg_win*WR + avg_loss*(1-WR) = {avg:.4f}")

    # Distribution of PnLs
    pcts = np.percentile(pnls, [1,5,10,25,50,75,90,95,99])
    print(f"  PnL pctiles: 1%={pcts[0]:.2f} 5%={pcts[1]:.2f} 10%={pcts[2]:.2f} "
          f"25%={pcts[3]:.2f} 50%={pcts[4]:.2f} 75%={pcts[5]:.2f} 95%={pcts[6]:.2f}")

    # How many exit methods
    stop_exits = np.sum(np.array(exits) == 'stop')
    timeout_exits = np.sum(np.array(exits) == 'timeout')
    print(f"  Exit: stop={stop_exits} ({100*stop_exits/len(exits):.0f}%)  "
          f"timeout={timeout_exits} ({100*timeout_exits/len(exits):.0f}%)")

t0 = time.time()
for pair in ['EURUSD','EURJPY','GBPJPY']:
    run(pair, COST[pair])
print(f"\nTotal: {time.time()-t0:.0f}s")
