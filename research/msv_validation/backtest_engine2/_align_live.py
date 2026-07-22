"""
Alignment verification: live code path (BarBuilder + PairState) vs backtest code.
Feeds Exness ticks through both paths on the SAME data, compares signals.
"""
import numpy as np, pandas as pd, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'paper_trade' / 'strategies' / 'm1_z_reversal'))
import paper_trade.core.config as cfg_mod
cfg_mod.register = lambda n, c: None

from strategy import CONFIG, PairState

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


def backtest_signal_list(b, z_thresh=2.0, atr_pctl=0.25):
    """Run backtest code and return list of signal dicts (no PnL, just signal detection)."""
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    atr_gate = atr.shift(1).rolling(100, min_periods=10).quantile(atr_pctl).bfill()
    atr_pass = atr > atr_gate
    valid = z.notna() & atr.notna() & (z.abs() > z_thresh) & atr_pass
    idxs = np.where(valid)[0]
    z_vals = z.values; atr_vals = atr.values
    signals = []
    for pos in idxs:
        direction = -1 if z_vals[pos] > 0 else 1
        signals.append({
            'bar_idx': pos,
            'bar_time': b.index[pos],
            'direction': direction,
            'z_score': z_vals[pos],
            'atr': atr_vals[pos],
            'close': b['close'].values[pos],
        })
    return signals


def live_signal_list(ticks_df, pair, b_backtest, z_thresh=2.0, atr_pctl=0.25):
    """Feed ticks through live code path (BarBuilder + PairState). Pre-seed from backtest bars.

    Args:
        ticks_df: raw tick DataFrame with 'B' (bid) column
        pair: symbol name
        b_backtest: M1 bar DataFrame from backtest (for pre-seeding)
    """
    ps = PairState(pair, {**CONFIG, 'z_thresh': z_thresh, 'atr_pctl': atr_pctl})

    # Pre-seed from first 60 backtest bars (same warmup as backtest)
    seed_count = min(60, len(b_backtest) - 10)
    for i in range(seed_count):
        bar = b_backtest.iloc[i]
        ps.seed_bar({
            'open': bar['open'], 'high': bar['high'],
            'low': bar['low'], 'close': bar['close'],
            'time': int(bar.name.timestamp()),
        })

    # Find the first tick that belongs to a bar AFTER the seed period
    seed_end_time = int(b_backtest.index[seed_count - 1].timestamp())
    raw_bids = ticks_df['B'].values
    raw_asks = ticks_df['A'].values
    raw_times = ticks_df.index.astype(np.int64) // 10**9
    signals = []
    prev_bar_min = -1

    for i in range(len(raw_bids)):
        bid = raw_bids[i]
        ask = raw_asks[i]
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid
        t = int(raw_times[i])
        if t <= seed_end_time:
            continue  # skip ticks during seed period
        sig = ps.update(mid, t)
        if sig is not None:
            bar_min = t // 60
            if bar_min == prev_bar_min:
                continue
            prev_bar_min = bar_min
            signals.append(sig)
    return signals


t0 = time.time()

print("=" * 75)
print("ALIGNMENT VERIFICATION: Live code path vs Backtest code path")
print("=" * 75)

for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
    print(f"\n--- {pair} ---")
    t = load(pair)

    # ─── Backtest path: resample ticks to M1 bars, run backtest ───
    b = t['MP'].resample('1min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    bt_signals = backtest_signal_list(b)

    # ─── Live path: feed ticks through BarBuilder + PairState ───
    live_signals = live_signal_list(t, pair, b)

    # ─── Compare ───
    print(f"  Backtest signals: {len(bt_signals)}")
    print(f"  Live path signals: {len(live_signals)}")

    # Align by bar time
    bt_by_time = {s['bar_time']: s for s in bt_signals}
    live_by_time = {}
    for s in live_signals:
        t_s = pd.Timestamp(s['bar_time'], unit='s') if isinstance(s['bar_time'], (int, np.integer)) else s['bar_time']
        live_by_time[t_s] = s

    matched = 0
    mismatched_dir = 0
    mismatched_z = 0
    bt_only = 0
    live_only = 0

    for bt_time, bt_s in bt_by_time.items():
        if bt_time in live_by_time:
            matched += 1
            lv_s = live_by_time[bt_time]
            if bt_s['direction'] != lv_s['direction']:
                mismatched_dir += 1
            z_diff = abs(bt_s['z_score'] - lv_s['z_score'])
            if z_diff > 0.01:
                mismatched_z += 1
        else:
            bt_only += 1

    for t_s in live_by_time:
        if t_s not in bt_by_time:
            live_only += 1

    print(f"  Matched: {matched} / {len(bt_signals)}")
    if mismatched_dir:
        print(f"  *** DIRECTION MISMATCH: {mismatched_dir} ***")
    if mismatched_z:
        print(f"  *** Z-SCORE MISMATCH (>0.01): {mismatched_z} ***")
    if bt_only:
        print(f"  Backtest-only signals (not in live): {bt_only}")
    if live_only:
        print(f"  Live-only signals (not in backtest): {live_only}")

    if matched > 0:
        # Show first 3 matched signals
        print(f"  First 3 matched signals:")
        shown = 0
        for bt_time, bt_s in bt_by_time.items():
            if bt_time in live_by_time and shown < 3:
                lv_s = live_by_time[bt_time]
                print(f"    BT: z={bt_s['z_score']:.3f} dir={bt_s['direction']:+d} atr={bt_s['atr']:.4f}")
                print(f"    LV: z={lv_s['z_score']:.3f} dir={lv_s['direction']:+d} atr={lv_s['atr']:.4f}")
                shown += 1

print(f"\nTotal: {time.time()-t0:.1f}s")
