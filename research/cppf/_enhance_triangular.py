"""Enhance the triangular filter: buff WR, break session dependency."""
import pandas as pd, numpy as np
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"
SPREAD = "gbpnzd"

# Load data
data = {}
for p in ['gbpnzd', 'gbpusd', 'nzdusd', 'eurusd', 'audusd', 'usdcad', 'gbpaud', 'gbpcad']:
    data[p] = pd.read_parquet(DATA_DIR / f'{p}.parquet').set_index('timestamp').astype(float)


def z_score(close, window=50):
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    z = (ret - mu) / sigma
    return z, ret


def backtest_generic(pair, z_values, spread_pips=5.0, max_bars=54,
                     stop_a=3.0, trig_a=0.5, gap_a=0.1,
                     session_hours=None):
    """Generic backtest returning trade DataFrame with full details."""
    c = data[pair]['close']; h = data[pair]['high']; l_ = data[pair]['low']
    atr = (h - l_).shift(1).rolling(20).mean().clip(1e-10)

    valid = z_values.notna()
    if hasattr(valid, 'values'):
        idxs = np.where(valid.values)[0]
    else:
        idxs = np.where(valid)[0]

    trades = []
    in_trade_until = -1
    for pos in idxs:
        if pos <= in_trade_until: continue
        if pos + 2 >= len(data[pair]): continue
        hour = data[pair].index[pos].hour
        if session_hours is not None and hour not in session_hours: continue

        direction = -1 if z_values.iloc[pos] > 0 else 1
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
                    trades.append({'pnl_pips': pnl * 10000, 'z': float(z_values.iloc[pos]),
                                   'hour': hour, 'dir': direction, 'exit_reason': 'stop',
                                   'bars_held': j})
                    in_trade_until = bp; break
            else:
                if l_.iloc[bp] < best: best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h.iloc[bp] >= sl:
                    pnl = (entry - sl) - spread_pips * 0.0001
                    trades.append({'pnl_pips': pnl * 10000, 'z': float(z_values.iloc[pos]),
                                   'hour': hour, 'dir': direction, 'exit_reason': 'stop',
                                   'bars_held': j})
                    in_trade_until = bp; break
        else:
            eb = min(pos + max_bars, len(data[pair]) - 1)
            pnl = (c.iloc[eb] - entry) * direction - spread_pips * 0.0001
            trades.append({'pnl_pips': pnl * 10000, 'z': float(z_values.iloc[pos]),
                           'hour': hour, 'dir': direction, 'exit_reason': 'expiry',
                           'bars_held': eb - pos})
            in_trade_until = eb

    if not trades: return pd.DataFrame()
    return pd.DataFrame(trades)


# ── Compute z-scores ──
z_gbpnzd, _ = z_score(data['gbpnzd']['close'])
z_gbpusd, _ = z_score(data['gbpusd']['close'])
z_nzdusd, _ = z_score(data['nzdusd']['close'])
z_eurusd, _ = z_score(data['eurusd']['close'])
z_audusd, _ = z_score(data['audusd']['close'])
z_usdcad, _ = z_score(data['usdcad']['close'])

print("=" * 70)
print("DARK ENHANCEMENT: Making triangular filter work across ALL sessions")
print("=" * 70)

# ── Experiment 1: Optimize leg threshold and z threshold ──
print("\n[1] Leg threshold sweep (all hours, z=2.5 fixed):")
for leg_t in [0.5, 1.0, 1.5, 2.0, 2.5]:
    mask = (z_gbpusd.abs() < leg_t) & (z_nzdusd.abs() < leg_t)
    z_filt = z_gbpnzd.where(mask & z_gbpnzd.notna())
    z_filt[z_filt.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_filt)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  Leg<{leg_t:3.1f}: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p  "
              f"avg={t.pnl_pips.mean():>+6.1f}p")

# ── Experiment 2: Directional leg check ──
print("\n[2] Directional leg filter (all hours):")
print("    Only fade when BOTH legs move in opposite directions to cross:")
print("    e.g., GBPNZD up, GBPUSD up, NZDUSD down → SKIP (explained by legs)")
print("    e.g., GBPNZD up, GBPUSD up, NZDUSD up → FADE (anomalous!)")

# Define: fade when legs move in same direction (anomalous cross move)
for leg_t in [0.5, 1.0, 1.5, 2.0]:
    z_g = z_gbpnzd
    z_b = z_gbpusd
    z_q = z_nzdusd
    
    # Both legs same direction as each other = cross should be flat
    # Fade when: cross extreme AND legs moving same direction
    mask_anomaly = (z_b * z_q > 0) & (z_b.abs() >= leg_t) & (z_q.abs() >= leg_t)
    # Also fade when both legs are quiet (original filter)
    mask_quiet = (z_b.abs() < leg_t) & (z_q.abs() < leg_t)
    
    z_filt = z_g.where(mask_anomaly | mask_quiet)
    z_filt[z_filt.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_filt)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  Leg<{leg_t:3.1f}+anomaly: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p")

# ── Experiment 3: Multi-pair confirmation ──
print("\n[3] Multi-pair cross confirmation (all hours):")
print("    Check if OTHER GBP crosses also show the same signal")
# If GBPNZD is extreme AND GBPAUD is not extreme → GBPNZD specific → FADE
# If GBPNZD is extreme AND GBPAUD is also extreme → broad GBP flow → SKIP

z_gbpaud, _ = z_score(data['gbpaud']['close'])
z_gbpcad, _ = z_score(data['gbpcad']['close'])

for thresh in [1.0, 1.5, 2.0]:
    # Skip if any other GBP cross is also extreme (broad GBP flow)
    other_extreme = (z_gbpaud.abs() >= thresh) | (z_gbpcad.abs() >= thresh)
    z_filt = z_gbpnzd.where(~other_extreme & z_gbpnzd.notna())
    z_filt[z_filt.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_filt)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  Skip if other GBP cross >{thresh:3.1f}: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p")

# ── Experiment 4: Parameter optimization with leg filter ──
print("\n[4] Parameter sweep on best filter (all hours):")
mask = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
z_filt = z_gbpnzd.where(mask & z_gbpnzd.notna())

results = []
for zt, stop_a, trig_a, gap_a in product(
    [2.0, 2.5, 3.0, 3.5, 4.0], [3.0, 4.0, 5.0, 6.0],
    [0.3, 0.5, 0.7, 1.0], [0.05, 0.1, 0.2]):
    z_f = z_filt.copy()
    z_f[z_f.abs() < zt] = np.nan
    t = backtest_generic('gbpnzd', z_f, stop_a=stop_a, trig_a=trig_a, gap_a=gap_a)
    if len(t) < 20: continue
    net = t.pnl_pips.sum()
    wr = (t.pnl_pips > 0).mean()
    results.append({'z': zt, 'stop': stop_a, 'trig': trig_a, 'gap': gap_a,
                    'n': len(t), 'wr': wr, 'pnl': net})

df = pd.DataFrame(results)
top = df.sort_values('wr', ascending=False).head(10)
print(f"  Top 10 by WR (all hours):")
print(f"  {'z':>4s} {'stop':>5s} {'trig':>5s} {'gap':>5s}  {'n':>4s} {'WR':>6s} {'PnL':>8s}")
for _, r in top.iterrows():
    print(f"  {r['z']:>4.1f} {r['stop']:>5.1f} {r['trig']:>5.1f} {r['gap']:>5.2f}  "
          f"{int(r['n']):>4d} {r['wr']:>5.1%} {r['pnl']:>+8.1f}")

# Also show top by PnL
top_pnl = df.sort_values('pnl', ascending=False).head(10)
print(f"  Top 10 by PnL (all hours):")
for _, r in top_pnl.iterrows():
    print(f"  {r['z']:>4.1f} {r['stop']:>5.1f} {r['trig']:>5.1f} {r['gap']:>5.2f}  "
          f"{int(r['n']):>4d} {r['wr']:>5.1%} {r['pnl']:>+8.1f}")

# ── Experiment 5: Adaptive z threshold by session ──
print("\n[5] Adaptive strategy by session:")
mask = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
z_filt = z_gbpnzd.where(mask & z_gbpnzd.notna())

print(f"  {'Session':>15s} | {'z_t':>4s} {'stop':>5s} {'trig':>5s} {'gap':>5s} | "
      f"{'n':>4s} {'WR':>6s} {'PnL':>8s} {'avg':>6s}")
for sess_name, sess_hours in [
    ('Sydney', range(21, 24)), ('Asian', range(0, 8)),
    ('London', range(8, 16)), ('NY', range(13, 21))]:
    best = {'wr': 0, 'pnl': -999}
    for zt, stop_a, trig_a, gap_a in product(
        [2.0, 2.5, 3.0, 3.5], [3.0, 4.0, 5.0],
        [0.3, 0.5, 0.7], [0.05, 0.1, 0.2]):
        z_f = z_filt.copy()
        z_f[z_f.abs() < zt] = np.nan
        t = backtest_generic('gbpnzd', z_f, stop_a=stop_a,
                             trig_a=trig_a, gap_a=gap_a,
                             session_hours=sess_hours)
        if len(t) < 5: continue
        net = t.pnl_pips.sum()
        wr = (t.pnl_pips > 0).mean()
        if net > best['pnl']:
            best = {'zt': zt, 'stop': stop_a, 'trig': trig_a, 'gap': gap_a,
                    'n': len(t), 'wr': wr, 'pnl': net}
    if best['pnl'] > -999:
        print(f"  {sess_name:>15s} | {best['zt']:>4.1f} {best['stop']:>5.1f} "
              f"{best['trig']:>5.1f} {best['gap']:>5.2f} | {best['n']:>4d} "
              f"{best['wr']:>5.1%} {best['pnl']:>+8.1f}p "
              f"{best['pnl']/best['n']:>+6.1f}p")

# ── Experiment 6: Z-score of z-score (second derivative) ──
print("\n[6] Z-score of z-score (rate of change filter):")
# If z-score is accelerating (z increases from t-1 to t), the move has momentum
# If z-score is decelerating (z decreases from t-1 to t), the move is fading
z_delta = z_gbpnzd.diff()
z_delta_z, _ = z_score(z_delta.dropna().rename('dz'), window=20)
z_delta_z = z_delta_z.reindex(z_gbpnzd.index, method='ffill')

# Fade only when z is PEAKING (z_delta_z < 0 means downward acceleration of z)
# i.e., z was going up but now starting to slow down → imminent reversal
for cond_name, cond in [
    ('Z accelerating (momentum)', z_delta_z > 1.5),
    ('Z decelerating (reversal)', z_delta_z < -1.5),
]:
    mask_legs = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
    z_f = z_gbpnzd.where(mask_legs & cond & z_gbpnzd.notna())
    z_f[z_f.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_f)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  {cond_name:30s}: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p "
              f"avg={t.pnl_pips.mean():>+6.1f}p")

# ── Experiment 7: Multi-pair volume of legs (how many pairs are extreme) ──
print("\n[7] Market breadth filter (how many pairs are extreme):")
all_pairs_z = {
    'gbpusd': z_gbpusd, 'nzdusd': z_nzdusd, 'eurusd': z_eurusd,
    'audusd': z_audusd, 'usdcad': z_usdcad, 'gbpaud': z_gbpaud, 'gbpcad': z_gbpcad,
}
for n_extreme_thresh in [0, 1, 2, 3]:
    # Count how many of the 7 pairs have |z| > 2.0
    extreme_count = sum((z.abs() > 2.0).astype(int) for z in all_pairs_z.values())
    # If MANY pairs are extreme → broad market stress → SKIP (trending)
    # If FEW pairs are extreme → individual pair dislocation → FADE
    mask_few = extreme_count <= n_extreme_thresh
    mask_legs = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
    z_f = z_gbpnzd.where(mask_few & mask_legs & z_gbpnzd.notna())
    z_f[z_f.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_f)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  <= {n_extreme_thresh} extreme pairs: n={len(t):>4d}  "
              f"WR={wr:>5.1%}  PnL={net:>+8.1f}p")

# ── Experiment 8: ATR regime filter ──
print("\n[8] ATR regime filter:")
atr_20 = (data['gbpnzd']['high'] - data['gbpnzd']['low']).rolling(20).mean()
atr_50 = (data['gbpnzd']['high'] - data['gbpnzd']['low']).rolling(50).mean()
atr_ratio = atr_20 / atr_50  # >1 = volatility expanding, <1 = volatility contracting

for regime_name, cond in [
    ('Low vol (contrarian works)', atr_ratio < 0.8),
    ('Normal vol', (atr_ratio >= 0.8) & (atr_ratio <= 1.2)),
    ('High vol (trending)', atr_ratio > 1.2),
]:
    mask_legs = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
    z_f = z_gbpnzd.where(mask_legs & cond & z_gbpnzd.notna())
    z_f[z_f.abs() < 2.5] = np.nan
    t = backtest_generic('gbpnzd', z_f)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        print(f"  {regime_name:30s}: n={len(t):>4d}  WR={wr:>5.1%}  PnL={net:>+8.1f}p")

# ── Experiment 9: Combined best from each session ──
print("\n[9] Combined session-adaptive strategy (best config per session):")
total_pnl = 0
total_trades = 0
total_wins = 0
for sess_name, sess_hours in [
    ('Sydney', range(21, 24)), ('Asian', range(0, 8)),
    ('London', range(8, 16)), ('NY', range(13, 21))]:
    best = {'pnl': -999}
    for zt, stop_a, trig_a, gap_a in product(
        [2.0, 2.5, 3.0, 3.5], [3.0, 4.0, 5.0, 6.0],
        [0.3, 0.5, 0.7], [0.05, 0.1, 0.2]):
        mask_legs = (z_gbpusd.abs() < 1.5) & (z_nzdusd.abs() < 1.5)
        z_f = z_gbpnzd.where(mask_legs & z_gbpnzd.notna())
        z_f[z_f.abs() < zt] = np.nan
        t = backtest_generic('gbpnzd', z_f, stop_a=stop_a,
                             trig_a=trig_a, gap_a=gap_a,
                             session_hours=sess_hours)
        if len(t) < 3: continue
        net = t.pnl_pips.sum()
        if net > best['pnl']:
            best = {'zt': zt, 'stop': stop_a, 'trig': trig_a, 'gap': gap_a,
                    'n': len(t), 'wr': (t.pnl_pips > 0).mean(), 'pnl': net, 't': t}
    if best['pnl'] > -999:
        total_pnl += best['pnl']
        total_trades += best['n']
        total_wins += int(best['n'] * best['wr'])
        print(f"  {sess_name:>15s}: z>{best['zt']:.1f} stop={best['stop']:.1f} "
              f"trig={best['trig']:.1f} gap={best['gap']:.2f}  "
              f"n={best['n']:>4d} WR={best['wr']:>5.1%} PnL={best['pnl']:>+8.1f}p avg={best['pnl']/best['n']:>+5.1f}p")

overall_wr = total_wins / total_trades if total_trades else 0
print(f"  {'COMBINED':>15s}: n={total_trades:>4d} WR={overall_wr:>5.1%} PnL={total_pnl:>+8.1f}p")
