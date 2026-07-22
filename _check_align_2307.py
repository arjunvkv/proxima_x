"""Check if EURJPY signals near z=-2.307 matched between backtest and live paths."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper_trade'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper_trade' / 'strategies' / 'm1_z_reversal'))
import paper_trade.core.config as cfg_mod
cfg_mod.register = lambda n, c: None
from strategy import CONFIG, PairState

TICK_DIR = Path(__file__).resolve().parent / 'data' / 'exness_ticks'
pair = 'EURJPY'

dfs = []
for y,m in [(2025,10),(2025,11),(2025,12)]:
    d = pd.read_csv(TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip',
        names=['E','S','Ts','B','A'], skiprows=1, header=None,
        dtype={'Ts':str,'B':np.float64,'A':np.float64})
    d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    dfs.append(d.dropna(subset=['Ts']))
t = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
t['MP'] = ((t['B']+t['A'])/2)*10000
t = t.set_index('Ts')

b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()

ret = b['close'].diff()
rz = ret.shift(1)
z = (ret - rz.rolling(50).mean()) / rz.rolling(50).std().clip(1e-8)
atr = (b['high']-b['low']).shift(1).rolling(20).mean().clip(1e-8)
atr_gate = atr.shift(1).rolling(100, min_periods=10).quantile(0.25).bfill()
atr_pass = atr > atr_gate
valid = z.notna() & atr.notna() & (z.abs() > 2.0) & atr_pass

bt_by_time = {}
bt_signals = []
for idx in np.where(valid.values)[0]:
    d = {'bar_time': b.index[idx], 'z_score': z.values[idx], 'direction': -1 if z.values[idx] > 0 else 1, 'atr': atr.values[idx]}
    bt_by_time[b.index[idx]] = d
    bt_signals.append(d)

ps = PairState(pair, {**CONFIG, 'z_thresh': 2.0, 'atr_pctl': 0.25})
seed_count = min(60, len(b)-10)
for i in range(seed_count):
    bar = b.iloc[i]
    ps.seed_bar({'open':bar['open'],'high':bar['high'],'low':bar['low'],'close':bar['close'],'time':int(bar.name.timestamp())})

seed_end = int(b.index[seed_count-1].timestamp())
raw_bids = t['B'].values; raw_asks = t['A'].values; raw_times = t.index.astype(np.int64)//10**9
prev_bar_min = -1
live_by_time = {}
live_signals = []
for i in range(len(raw_bids)):
    bid, ask, ts = raw_bids[i], raw_asks[i], int(raw_times[i])
    if ts <= seed_end: continue
    mid = (bid+ask)/2 if bid>0 and ask>0 else bid
    sig = ps.update(mid, ts)
    if sig:
        bar_min = ts // 60
        if bar_min == prev_bar_min: continue
        prev_bar_min = bar_min
        bt = pd.Timestamp(sig['bar_time'], unit='s') if isinstance(sig['bar_time'], (int,np.integer)) else sig['bar_time']
        live_by_time[bt] = sig
        live_signals.append(sig)

print(f"EURJPY: {len(bt_signals)} BT signals, {len(live_signals)} live signals")
print()

# Find matches for z around -2.307
print("Signals where z ≈ -2.307 (both paths):")
target_z = -2.307
matched = 0
for t_s, bt_s in bt_by_time.items():
    if t_s in live_by_time:
        lv_s = live_by_time[t_s]
        z_diff = abs(bt_s['z_score'] - lv_s['z_score'])
        dir_ok = bt_s['direction'] == lv_s['direction']
        if abs(bt_s['z_score'] - target_z) < 0.2:
            matched += 1
            print(f"  {t_s}:")
            print(f"    BT: z={bt_s['z_score']:.3f} dir={bt_s['direction']:+d} atr={bt_s['atr']:.2f}")
            print(f"    LV: z={lv_s['z_score']:.3f} dir={lv_s['direction']:+d} atr={lv_s['atr']:.4f}")
            print(f"    z_diff={z_diff:.4f} dir_match={dir_ok}")

if matched == 0:
    print("  (none found)")
    # Show closest
    closest = min(bt_by_time.keys(), key=lambda x: abs(bt_by_time[x]['z_score'] - target_z))
    bt_s = bt_by_time[closest]
    in_live = closest in live_by_time
    print(f"\nClosest BT signal at {closest}: z={bt_s['z_score']:.3f} dir={bt_s['direction']:+d}")
    print(f"  In live path: {in_live}")
    if in_live:
        lv_s = live_by_time[closest]
        print(f"  Live: z={lv_s['z_score']:.3f} dir={lv_s['direction']:+d}")
