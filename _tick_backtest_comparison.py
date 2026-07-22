"""
Full tick-level backtest comparison: Backtest (M1 bar) vs Live (BarBuilder+Trail)
on the SAME Exness ticks. Compares trades at entry-bar granularity.
"""
import numpy as np, pandas as pd, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper_trade'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper_trade' / 'strategies' / 'm1_z_reversal'))
import paper_trade.core.config as cfg_mod
cfg_mod.register = lambda n, c: None
from strategy import CONFIG, PairState, TrailingStopManager

TICK_DIR = Path(__file__).resolve().parent / 'data' / 'exness_ticks'
pair = 'EURJPY'
PIP = 0.01
COST_MP = 50
MIN_STOP_MP = 1.5 * PIP * 10000
COST_RAW = COST_MP / 10000
MIN_STOP_RAW = 1.5 * PIP

# ─── Load EXACTLY like _align_live.py ───
dfs = []
for y,m in [(2025,10),(2025,11),(2025,12)]:
    d = pd.read_csv(TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip',
        names=['E','S','Ts','B','A'], skiprows=1, header=None,
        dtype={'Ts':str,'B':np.float64,'A':np.float64})
    d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    dfs.append(d.dropna(subset=['Ts']))
t = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
t['mid'] = (t['B']+t['A'])/2
t['mp'] = t['mid'] * 10000
t = t.set_index('Ts')
print(f"Loaded {len(t):,} ticks")

# ══════════════════════════════════════════════
# PATH A: Backtest (M1 bars, MP units)
# ══════════════════════════════════════════════
b = t['mp'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
ret = b['close'].diff()
z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
atr = (b['high']-b['low']).shift(1).rolling(20).mean().clip(1e-8)
atr_gate = atr.shift(1).rolling(100, min_periods=10).quantile(0.25).bfill()
atr_pass = atr > atr_gate
valid = z.notna() & atr.notna() & (z.abs() > 2.0) & atr_pass
idxs = np.where(valid)[0]
closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
z_vals = z.values; atr_vals = atr.values

bt_trades = {}
for pos in idxs:
    if pos + 2 >= len(b): continue
    direction = -1 if z_vals[pos] > 0 else 1
    entry = closes[pos]
    atr_v = atr_vals[pos]
    if np.isnan(atr_v) or atr_v < 1e-10: continue
    s = max(0.15 * atr_v, MIN_STOP_MP); tg = 0.20 * atr_v; gp = 0.10 * atr_v
    best = entry; exited = False
    for j in range(1, 55):
        bp = pos + j
        if bp >= len(b): break
        if direction == 1:
            best = max(best, highs[bp]); sl = entry - s
            if best - entry > tg: sl = best - gp
            if lows[bp] <= sl:
                pnl = sl - entry - COST_MP
                bt_trades[b.index[pos]] = {'dir': direction, 'entry': entry, 'exit': sl,
                    'pnl': pnl, 'exit_bar': j, 'z': z_vals[pos], 'atr': atr_v}
                exited = True; break
        else:
            best = min(best, lows[bp]); sl = entry + s
            if entry - best > tg: sl = best + gp
            if highs[bp] >= sl:
                pnl = (sl - entry) * direction - COST_MP
                bt_trades[b.index[pos]] = {'dir': direction, 'entry': entry, 'exit': sl,
                    'pnl': pnl, 'exit_bar': j, 'z': z_vals[pos], 'atr': atr_v}
                exited = True; break
    if not exited:
        eb = min(pos + 54, len(b) - 1)
        pnl = (closes[eb] - entry) * direction - COST_MP
        bt_trades[b.index[pos]] = {'dir': direction, 'entry': entry, 'exit': closes[eb],
            'pnl': pnl, 'exit_bar': 54, 'z': z_vals[pos], 'atr': atr_v}
print(f"Path A (BT): {len(bt_trades)} trades")

# ══════════════════════════════════════════════
# PATH B: Live (BarBuilder+PairState, RAW units)
# ══════════════════════════════════════════════
lv_config = {**CONFIG, 'z_thresh': 2.0, 'atr_pctl': 0.25, 'min_stop_pips': 1.5,
             'stop_a': 0.15, 'trig_a': 0.20, 'gap_a': 0.10, 'max_hold_min': 54}
ps = PairState(pair, lv_config)
tsm = TrailingStopManager(lv_config)

seed_count = min(60, len(b)-10)
for i in range(seed_count):
    bar = b.iloc[i]
    ps.seed_bar({'open': bar['open']/10000, 'high': bar['high']/10000,
                 'low': bar['low']/10000, 'close': bar['close']/10000,
                 'time': int(bar.name.timestamp())})

seed_end = int(b.index[seed_count-1].timestamp())
raw_times = t.index.astype(np.int64)//10**9
raw_bids = t['B'].values; raw_asks = t['A'].values

live_trades = {}
prev_bar_min = -1
t0 = time.time()

for i in range(len(raw_times)):
    bid = raw_bids[i]; ask = raw_asks[i]; ts = int(raw_times[i])
    if ts <= seed_end: continue
    mid = (bid + ask) / 2.0

    # Trailing stop checks (uses bid/ask, handles grace period + expiry)
    closed = tsm.update(bid, ask, ts)
    for cp in closed:
        tr = live_trades.get(cp['ticket'])
        # trade already recorded at signal time below, just update exit info
        if tr:
            tr['exit'] = bid if tr['dir'] == 1 else ask
            tr['exit_time'] = ts
            tr['pnl_raw'] = (tr['exit'] - tr['entry']) * tr['dir'] - COST_RAW
            tr['pnl'] = tr['pnl_raw'] * 10000

    # New signals
    sig = ps.update(mid, ts)
    if sig:
        bar_min = ts // 60
        if bar_min == prev_bar_min: continue
        prev_bar_min = bar_min
        entry = ask if sig['direction'] == 1 else bid
        atr_v = sig['atr']
        ticket = tsm.add(pair, sig['direction'], entry, atr_v, timestamp=ts)
        bt = pd.Timestamp(sig['bar_time'], unit='s') if isinstance(sig['bar_time'], (int,np.integer)) else sig['bar_time']
        live_trades[ticket] = {'bar_time': bt, 'dir': sig['direction'], 'entry': entry,
            'entry_time': ts, 'z': sig['z_score'], 'atr_raw': atr_v,
            'exit': None, 'exit_time': None, 'pnl_raw': None, 'pnl': None}

# Close any remaining via expiry
for ticket, tr in list(live_trades.items()):
    if tr['exit'] is None:
        tr['exit'] = (raw_bids[-1] + raw_asks[-1]) / 2.0
        tr['exit_time'] = int(raw_times[-1])
        tr['pnl_raw'] = (tr['exit'] - tr['entry']) * tr['dir'] - COST_RAW
        tr['pnl'] = tr['pnl_raw'] * 10000

print(f"Path B (LV): {len(live_trades)} trades")

# Build lv_by_time index from ticket-based dict
lv_by_time = {}
for ticket, tr in live_trades.items():
    lv_by_time[tr['bar_time']] = tr

# ══════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════
matched = 0
entry_diff = []; exit_diff = []; pnl_diff = []; dur_diff = []
z_match = []; dir_match = []

for bt_time in sorted(bt_trades.keys()):
    bt = bt_trades[bt_time]
    if bt_time in lv_by_time:
        lv = lv_by_time[bt_time]
        matched += 1
        entry_diff.append(abs(bt['entry']/10000 - lv['entry']))
        exit_diff.append(abs(bt['exit']/10000 - lv['exit']))
        pnl_diff.append(abs(bt['pnl'] - lv['pnl']))  # both in MP units
        dir_match.append(bt['dir'] == lv['dir'])
        z_match.append(abs(bt['z'] - lv['z']) < 0.01)
        lv_dur = (lv.get('exit_time',0) - lv.get('entry_time',0)) / 60
        dur_diff.append(abs(bt['exit_bar'] - lv_dur))

bt_only = [t for t in bt_trades if t not in lv_by_time]
lv_only = [t for t in lv_by_time if t not in bt_trades]

print(f"\n{'='*85}")
print(f"COMPARISON: BT vs LV on {pair} (Exness Oct-Dec 2025)")
print(f"{'='*85}")
print(f"  Total BT trades: {len(bt_trades)}")
print(f"  Total LV trades: {len(live_trades)}")
print(f"  Matched by bar_time: {matched}")
print(f"  BT-only (no LV match): {len(bt_only)}")
print(f"  LV-only (no BT match): {len(lv_only)}")

if matched > 0:
    ed = np.array(entry_diff); ed_mp = ed * 10000
    xd = np.array(exit_diff); xd_mp = xd * 10000
    pd_arr = np.array(pnl_diff)  # already in MP units
    dd = np.array(dur_diff)
    print(f"\n  Entry diff (raw): mean={ed.mean():.5f} median={np.median(ed):.5f} 95th={np.percentile(ed,95):.5f}")
    print(f"  Entry diff (pips): mean={ed_mp.mean()/100:.2f} median={np.median(ed_mp)/100:.2f}")
    print(f"  Exit diff (pips): mean={xd_mp.mean()/100:.2f} median={np.median(xd_mp)/100:.2f}")
    pd_mp = np.array(pnl_diff)  # already in MP units, 100 MP = 1 pip
    print(f"  PnL diff (pips): mean={pd_mp.mean()/100:.2f} median={np.median(pd_mp)/100:.2f}")
    print(f"  Duration diff (bars): mean={dd.mean():.1f} median={np.median(dd):.1f}")
    print(f"  Direction match: {np.mean(dir_match):.0%}")
    print(f"  Z-score match: {np.mean(z_match):.0%}")

# Show the trade z≈-2.307
target = -2.307
print(f"\n  Trades near z≈{target}:")
for bt_time in sorted(bt_trades.keys()):
    bt = bt_trades[bt_time]
    if abs(bt['z'] - target) < 0.1:
        lv = lv_by_time.get(bt_time, None)
        lv_str = f"LV: entry={lv['entry']:.5f} exit={lv['exit']:.5f} pnl={lv['pnl']/10000:+.2f}" if lv else "LV: MISSING"
        print(f"    {bt_time}: BT z={bt['z']:.3f} entry={bt['entry']/10000:.5f} exit={bt['exit']/10000:.5f} "
              f"pnl={bt['pnl']/10000:+.2f} dur={bt['exit_bar']}bars | {lv_str}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
