"""Multi-pair simultaneous trading with leg filter."""
import pandas as pd, numpy as np
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"

ALL_PAIRS = ['gbpnzd', 'eurnzd', 'gbpaud', 'euraud', 'gbpcad', 'audnzd']
LEG_MAP = {
    'gbpnzd': ('gbpusd', 'nzdusd'), 'eurnzd': ('eurusd', 'nzdusd'),
    'gbpaud': ('gbpusd', 'audusd'), 'euraud': ('eurusd', 'audusd'),
    'gbpcad': ('gbpusd', 'usdcad'), 'audnzd': ('audusd', 'nzdusd'),
}
SPREAD_MAP = {'gbpnzd': 5.0, 'eurnzd': 4.0, 'gbpaud': 4.0, 'euraud': 3.0, 'gbpcad': 4.0, 'audnzd': 3.0}

# Load all data
data = {}
needed = set(ALL_PAIRS)
for v in LEG_MAP.values():
    needed.add(v[0]); needed.add(v[1])
for p in needed:
    data[p] = pd.read_parquet(DATA_DIR / f'{p}.parquet').set_index('timestamp').astype(float)


def z_score(close, window=50):
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    return (ret - mu) / sigma, ret

# Precompute all z-scores
z_scores = {}
for p in list(ALL_PAIRS) + ['gbpusd', 'nzdusd', 'eurusd', 'audusd', 'usdcad']:
    z_scores[p], _ = z_score(data[p]['close'])


def leg_filter(pair, leg_thresh=0.5):
    base, quote = LEG_MAP[pair]
    mask = (z_scores[base].abs() < leg_thresh) & (z_scores[quote].abs() < leg_thresh)
    return z_scores[pair].where(mask & z_scores[pair].notna())


def backtest(pair, z_values, z_thresh=4.0, spread_pips=5.0, max_bars=54,
             stop_a=6.0, trig_a=0.7, gap_a=0.05, session_hours=None):
    c = data[pair]['close']; h = data[pair]['high']; l_ = data[pair]['low']
    atr = (h - l_).shift(1).rolling(20).mean().clip(1e-10)
    z_f = z_values.copy(); z_f[z_f.abs() < z_thresh] = np.nan

    valid = z_f.notna()
    idxs = np.where(valid.values if hasattr(valid, 'values') else valid)[0] if hasattr(z_f, 'notna') else np.where(valid)[0]

    trades, in_trade = [], -1
    for pos in idxs:
        if pos <= in_trade or pos + 2 >= len(data[pair]): continue
        hour = data[pair].index[pos].hour
        if session_hours is not None and hour not in session_hours: continue
        direction = -1 if z_f.iloc[pos] > 0 else 1
        entry = c.iloc[pos]; atr_v = atr.iloc[pos]
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(data[pair]): break
            if direction == 1:
                if h.iloc[bp] > best: best = h.iloc[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l_.iloc[bp] <= sl:
                    pnl = (sl - entry) - spread_pips * 0.0001
                    trades.append({'pair': pair, 'pnl_pips': pnl * 10000, 'hour': hour})
                    in_trade = bp; break
            else:
                if l_.iloc[bp] < best: best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h.iloc[bp] >= sl:
                    pnl = (entry - sl) - spread_pips * 0.0001
                    trades.append({'pair': pair, 'pnl_pips': pnl * 10000, 'hour': hour})
                    in_trade = bp; break
        else:
            eb = min(pos + max_bars, len(data[pair]) - 1)
            pnl = (c.iloc[eb] - entry) * direction - spread_pips * 0.0001
            trades.append({'pair': pair, 'pnl_pips': pnl * 10000, 'hour': hour})
            in_trade = eb
    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ── Test: Multi-pair with strict leg filter ──
print("=" * 70)
print("MULTI-PAIR TEST: Strict leg filter (legs < 0.5) across all hours")
print("=" * 70)

configs = [
    ('Very strict', 0.5, 4.0, 6.0, 0.7, 0.05),
    ('Strict', 0.5, 3.5, 5.0, 0.7, 0.05),
    ('Moderate', 1.0, 3.5, 4.0, 0.7, 0.05),
    ('Relaxed', 1.0, 3.0, 4.0, 0.5, 0.1),
    ('Original Sydney', 1.5, 2.5, 5.0, 0.7, 0.05),
]

for cfg_name, leg_t, zt, stop_a, trig_a, gap_a in configs:
    print(f"\n--- {cfg_name}: leg<{leg_t}, z>{zt}, stop={stop_a}, trig={trig_a}, gap={gap_a} ---")
    all_trades = []
    for pair in ALL_PAIRS:
        z_f = leg_filter(pair, leg_t)
        t = backtest(pair, z_f, z_thresh=zt, spread_pips=SPREAD_MAP[pair],
                     stop_a=stop_a, trig_a=trig_a, gap_a=gap_a)
        if len(t):
            wr = (t.pnl_pips > 0).mean()
            net = t.pnl_pips.sum()
            all_trades.append(t)
            print(f"  {pair.upper():7s}: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p  avg={t.pnl_pips.mean():>+5.1f}p")

    if all_trades:
        combined = pd.concat(all_trades)
        wr = (combined.pnl_pips > 0).mean()
        net = combined.pnl_pips.sum()
        n_days = (data[ALL_PAIRS[0]].index[-1] - data[ALL_PAIRS[0]].index[0]).total_seconds() / 86400
        tpd = len(combined) / (n_days / 30)  # trades per month
        print(f"  {'TOTAL':7s}: n={len(combined):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p  avg={combined.pnl_pips.mean():>+5.1f}p  ~{tpd:.0f}/mo")

# ── Show per-day trade count ──
print("\n" + "=" * 70)
print("DAILY TRADE BREAKDOWN (Original Sydney config = Leg<1.5, z>2.5, Sydney)")
print("=" * 70)

configs = [
    ('Leg<1.5 Sydney only',  1.5, 2.5, 5.0, 0.7, 0.05, range(21, 24)),
    ('Leg<0.5 All hours',    0.5, 4.0, 6.0, 0.7, 0.05, None),
    ('Leg<1.0 All hours',    1.0, 3.5, 5.0, 0.7, 0.05, None),
]
for cfg_name, leg_t, zt, stop_a, trig_a, gap_a, sess in configs:
    all_trades = []
    for pair in ALL_PAIRS:
        z_f = leg_filter(pair, leg_t)
        t = backtest(pair, z_f, z_thresh=zt, spread_pips=SPREAD_MAP[pair],
                     stop_a=stop_a, trig_a=trig_a, gap_a=gap_a,
                     session_hours=sess)
        if len(t): all_trades.append(t)
    combined = pd.concat(all_trades) if all_trades else pd.DataFrame()
    if len(combined):
        n_days = (data[ALL_PAIRS[0]].index[-1] - data[ALL_PAIRS[0]].index[0]).total_seconds() / 86400
        tpd = len(combined) / n_days * 30
        wr = (combined.pnl_pips > 0).mean()
        net = combined.pnl_pips.sum()
        print(f"  {cfg_name:25s}: {len(combined):>4d} trades, WR={wr:>5.1%}, "
              f"PnL={net:>+7.1f}p, ~{tpd:.1f}/mo (${net*0.10:.1f})")

# ── Check: Can we trade multiple pairs concurrently? ──
print("\n" + "=" * 70)
print("OVERLAP CHECK: Do signals cluster on same bars across pairs?")
print("(If yes, can't trade all simultaneously with 0.01 lot each)")
print("=" * 70)

pair_list = ['gbpnzd', 'gbpaud', 'euraud']
for leg_t, zt, label in [(1.5, 2.5, 'Sydney (Leg<1.5)'), (0.5, 4.0, 'Strict (Leg<0.5)')]:
    print(f"\n  {label}:")
    pair_signals = {}
    for pair in pair_list:
        z_f = leg_filter(pair, leg_t)
        z_f[z_f.abs() < zt] = np.nan
        signals = z_f.notna().astype(int)
        pair_signals[pair] = signals
        print(f"    {pair.upper()}: {signals.sum()} signal bars")

    # Count bars with signals from multiple pairs
    signal_matrix = pd.DataFrame(pair_signals)
    concurrent = signal_matrix.sum(axis=1)
    multi = (concurrent > 1).sum()
    print(f"    Bars with 2+ signals: {multi}")
    if multi > 0:
        print(f"    Overlap rate: {multi/signal_matrix.sum().sum():.1%}")
