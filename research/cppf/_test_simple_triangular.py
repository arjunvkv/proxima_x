"""Test simple triangular filter: skip trade if either leg is extreme."""
import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"
SPREAD = {'gbpnzd': 5.0, 'eurnzd': 4.0, 'gbpaud': 4.0, 'euraud': 3.0, 'gbpcad': 4.0, 'audnzd': 3.0}

# Load all needed pairs
pairs_needed = ['gbpnzd', 'eurnzd', 'gbpaud', 'euraud', 'gbpcad', 'audnzd',
                'gbpusd', 'nzdusd', 'eurusd', 'audusd', 'usdcad']
data = {}
for p in pairs_needed:
    df = pd.read_parquet(DATA_DIR / f'{p}.parquet').set_index('timestamp').astype(float)
    data[p] = df


def z_score(close, window=50):
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    z = (ret - mu) / sigma
    return z, ret


def simple_leg_filter(z_cross, z_base, z_quote, leg_thresh=1.5):
    """Skip if either leg z-score exceeds threshold (broad flow)."""
    mask = (z_base.abs() < leg_thresh) & (z_quote.abs() < leg_thresh)
    result = pd.Series(np.nan, index=z_cross.index)
    result[mask] = z_cross[mask]
    return result


def backtest(pair, z_values, spread_pips, max_bars=54,
             stop_a=3.0, trig_a=0.5, gap_a=0.1,
             session_hours=None):
    c = data[pair]['close']
    h = data[pair]['high']
    l_ = data[pair]['low']
    atr = (h - l_).shift(1).rolling(20).mean().clip(1e-10)

    valid = z_values.notna()
    if hasattr(valid, 'values'):
        idxs = np.where(valid.values)[0]
    else:
        idxs = np.where(valid)[0]

    trades = []
    in_trade_until = -1
    for pos in idxs:
        if pos <= in_trade_until:
            continue
        if pos + 2 >= len(data[pair]):
            continue

        hour = data[pair].index[pos].hour
        if session_hours is not None and hour not in session_hours:
            continue

        direction = -1 if z_values.iloc[pos] > 0 else 1
        entry = c.iloc[pos]
        atr_v = atr.iloc[pos]
        s = stop_a * atr_v
        tg = trig_a * atr_v
        gp = gap_a * atr_v

        best = entry
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(data[pair]):
                break
            if direction == 1:
                if h.iloc[bp] > best:
                    best = h.iloc[bp]
                sl = entry - s
                if best - entry > tg:
                    sl = best - gp
                if l_.iloc[bp] <= sl:
                    pnl = (sl - entry) - spread_pips * 0.0001
                    trades.append(pnl * 10000)
                    in_trade_until = bp
                    break
            else:
                if l_.iloc[bp] < best:
                    best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg:
                    sl = best + gp
                if h.iloc[bp] >= sl:
                    pnl = (entry - sl) - spread_pips * 0.0001
                    trades.append(pnl * 10000)
                    in_trade_until = bp
                    break
        else:
            eb = min(pos + max_bars, len(data[pair]) - 1)
            pnl = (c.iloc[eb] - entry) * direction - spread_pips * 0.0001
            trades.append(pnl * 10000)
            in_trade_until = eb

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame({'pnl': trades})


# Map pairs to legs
LEG_MAP = {
    'gbpnzd': ('gbpusd', 'nzdusd'), 'eurnzd': ('eurusd', 'nzdusd'),
    'gbpaud': ('gbpusd', 'audusd'), 'euraud': ('eurusd', 'audusd'),
    'gbpcad': ('gbpusd', 'usdcad'), 'audnzd': ('audusd', 'nzdusd'),
}

print("=== SIMPLE LEG FILTER TEST ===")
print(f"{'Pair':>7s} | {'Config':>25s} | {'n':>4s} {'WR':>6s} {'PnL':>8s} {'avgW':>6s} {'avgL':>6s}")
print("-" * 75)

for pair in LEG_MAP:
    base, quote = LEG_MAP[pair]
    z_cross, _ = z_score(data[pair]['close'])
    z_base, _ = z_score(data[base]['close'])
    z_quote, _ = z_score(data[quote]['close'])
    z_filtered = simple_leg_filter(z_cross, z_base, z_quote, leg_thresh=1.5)
    spread = SPREAD[pair]

    for cfg_name, stop_a, trig_a, gap_a, zt, sess in [
        ('Default all hours', 3.0, 0.5, 0.1, 2.5, None),
        ('Default Sydney', 3.0, 0.5, 0.1, 2.5, range(21, 24)),
        ('Optimized Sydney', 5.0, 0.7, 0.05, 2.5, range(21, 24)),
    ]:
        z_final = z_filtered.copy()
        z_final[z_final.abs() < zt] = np.nan
        t = backtest(pair, z_final, spread_pips=spread,
                     stop_a=stop_a, trig_a=trig_a, gap_a=gap_a,
                     session_hours=sess)
        if len(t) == 0:
            print(f"{pair.upper():>7s} | {cfg_name:>25s}:  0 trades")
        else:
            wr = (t['pnl'] > 0).mean()
            net = t['pnl'].sum()
            avg_w = t.loc[t['pnl'] > 0, 'pnl'].mean() if (t['pnl'] > 0).any() else 0
            avg_l = t.loc[t['pnl'] <= 0, 'pnl'].mean() if (t['pnl'] <= 0).any() else 0
            print(f"{pair.upper():>7s} | {cfg_name:>25s}: {len(t):>4d} "
                  f"{wr:>5.1%} {net:>+8.1f}p {avg_w:>+6.1f}p {avg_l:>+6.1f}p")

print("\n=== WINNER: GBPNZD Simple Leg Filter + Sydney ===")
pair = 'gbpnzd'
z_cross, _ = z_score(data[pair]['close'])
z_base, _ = z_score(data['gbpusd']['close'])
z_quote, _ = z_score(data['nzdusd']['close'])
z_f = simple_leg_filter(z_cross, z_base, z_quote, 1.5)
z_f[z_f.abs() < 2.5] = np.nan
t = backtest(pair, z_f, spread_pips=5.0, stop_a=5.0, trig_a=0.7, gap_a=0.05, session_hours=range(21, 24))
wr = (t['pnl'] > 0).mean()
net = t['pnl'].sum()
print(f"  Trades: {len(t)}")
print(f"  WR: {wr:.1%}")
print(f"  Net PnL: {net:.1f} pips")
# At 0.01 lot, 1 pip on GBPNZD ≈ $0.10
print(f"  Est profit at 0.01 lot: ${net * 0.10:.2f}")
print(f"  Est profit at 0.10 lot: ${net * 1.00:.2f}")
