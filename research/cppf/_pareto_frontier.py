"""Pareto frontier: find configs hitting 65%+ WR at maximum trade frequency."""
import pandas as pd, numpy as np
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"
ALL_PAIRS = [    'gbpnzd', 'eurnzd', 'gbpaud', 'euraud', 'gbpcad', 'audnzd',
    'eurgbp', 'gbpchf']
LEG_MAP = {
    'gbpnzd': ('gbpusd', 'nzdusd'), 'eurnzd': ('eurusd', 'nzdusd'),
    'gbpaud': ('gbpusd', 'audusd'), 'euraud': ('eurusd', 'audusd'),
    'gbpcad': ('gbpusd', 'usdcad'), 'audnzd': ('audusd', 'nzdusd'),
    'eurgbp': ('eurusd', 'gbpusd'), 'gbpjpy': ('gbpusd', 'usdjpy'),
    'eurjpy': ('eurusd', 'usdjpy'), 'audjpy': ('audusd', 'usdjpy'),
    'nzdjpy': ('nzdusd', 'usdjpy'), 'gbpchf': ('gbpusd', 'usdchf'),
}
SPREAD_MAP = {
    'gbpnzd': 5.0, 'eurnzd': 4.0, 'gbpaud': 4.0, 'euraud': 3.0,
    'gbpcad': 4.0, 'audnzd': 3.0, 'eurgbp': 2.0, 'gbpchf': 4.0,
}
JPY_PAIRS = {'gbpjpy', 'eurjpy', 'audjpy', 'nzdjpy', 'chfjpy'}

# Deduplicate
ALL_PAIRS = list(dict.fromkeys(ALL_PAIRS))

# Load data
data = {}
needed = set(ALL_PAIRS)
for v in LEG_MAP.values():
    needed.add(v[0]); needed.add(v[1])
for p in sorted(needed):
    f = DATA_DIR / f'{p}.parquet'
    if f.exists():
        data[p] = pd.read_parquet(f).set_index('timestamp').astype(float)
    else:
        print(f"  WARN: {p}.parquet not found")

available = [p for p in ALL_PAIRS if p in data]
print(f"Loaded {len(data)} symbols, {len(available)} tradeable pairs")


def z_score(close, window=50):
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    return (ret - mu) / sigma, ret

# Precompute z-scores
z_scores = {}
for p in list(available) + list(set(n for v in LEG_MAP.values() for n in v)):
    if p in data:
        z_scores[p], _ = z_score(data[p]['close'])


def backtest_simple(pair, z_val, z_thresh=2.5, leg_thresh=1.5,
                    max_bars=54, session_hours=None,
                    stop_a=5.0, trig_a=0.7, gap_a=0.05):
    """Simple backtest for fast scanning."""
    pair_data = data[pair]
    c = pair_data['close']; h = pair_data['high']; l_ = pair_data['low']
    atr = (h - l_).shift(1).rolling(20).mean().clip(1e-10)
    base, quote = LEG_MAP[pair]

    # Apply leg filter
    valid = z_val.copy()
    leg_mask = (z_scores[base].abs() < leg_thresh) & (z_scores[quote].abs() < leg_thresh)
    valid[~(valid.notna() & leg_mask)] = np.nan
    valid[valid.abs() < z_thresh] = np.nan

    idxs = np.where(valid.notna().values)[0] if hasattr(valid, 'values') else np.where(valid.notna())[0]
    trades, in_trade = [], -1
    for pos in idxs:
        if pos <= in_trade or pos + 2 >= len(pair_data): continue
        hour = pair_data.index[pos].hour
        if session_hours is not None and hour not in session_hours: continue
        direction = -1 if valid.iloc[pos] > 0 else 1
        entry = c.iloc[pos]
        atr_v = atr.iloc[pos]
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(pair_data): break
            if direction == 1:
                if h.iloc[bp] > best: best = h.iloc[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l_.iloc[bp] <= sl:
                    trades.append({'pair': pair, 'pnl_pips': (sl - entry) * 10000 - SPREAD_MAP[pair],
                                   'hour': hour, 'bar': pos})
                    in_trade = bp; break
            else:
                if l_.iloc[bp] < best: best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h.iloc[bp] >= sl:
                    trades.append({'pair': pair, 'pnl_pips': (entry - sl) * 10000 - SPREAD_MAP[pair],
                                   'hour': hour, 'bar': pos})
                    in_trade = bp; break
        else:
            eb = min(pos + max_bars, len(pair_data) - 1)
            pnl = (c.iloc[eb] - entry) * direction * 10000 - SPREAD_MAP[pair]
            trades.append({'pair': pair, 'pnl_pips': pnl, 'hour': hour, 'bar': pos})
            in_trade = eb
    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1: Full grid sweep on GBPNZD with all combos
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: Grid sweep GBPNZD — Pareto frontier of WR vs trade count")
print("=" * 70)

z_threshs = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
leg_threshs = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
stops = [3.0, 5.0, 6.0]
trigs = [0.3, 0.5, 0.7]
gaps = [0.05, 0.1]

results = []
total = len(z_threshs) * len(leg_threshs) * len(stops) * len(trigs) * len(gaps)
count = 0
z_val = z_scores['gbpnzd']
for zt, lt, st, tr, gp in product(z_threshs, leg_threshs, stops, trigs, gaps):
    count += 1
    if count % 200 == 0:
        print(f"  {count}/{total}...")
    t = backtest_simple('gbpnzd', z_val, z_thresh=zt, leg_thresh=lt,
                        stop_a=st, trig_a=tr, gap_a=gp)
    if len(t) > 0:
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        results.append({'z': zt, 'leg': lt, 'stop': st, 'trig': tr,
                        'gap': gp, 'n': len(t), 'WR': wr, 'PnL': net})

df = pd.DataFrame(results)
print(f"\nTotal configs tested: {len(results)}")
print(f"Configs with WR >= 65%: {(df.WR >= 0.65).sum()}")

# Show the Pareto frontier
best_by_trades = df[df.WR >= 0.65].sort_values('n', ascending=False).head(20)
print(f"\nTop 20 by trade count (WR >= 65%):")
for _, r in best_by_trades.iterrows():
    print(f"  z={r.z:.1f} leg<{r.leg:.1f} stop={r.stop:.0f} trig={r.trig:.1f} "
          f"gap={r.gap:.2f}  n={int(r.n):>4d}  WR={r.WR:.1%}  PnL={r.PnL:>+7.1f}p")

# What's the MAX trade count at each WR threshold?
print(f"\nMax trades at each WR level:")
for wr_target in [0.60, 0.65, 0.70, 0.75, 0.80]:
    candidates = df[df.WR >= wr_target]
    if len(candidates):
        best = candidates.loc[candidates.n.idxmax()]
        print(f"  WR >= {wr_target:.0%}: max n={int(best.n):>4d}  "
              f"(z={best.z:.1f} leg<{best.leg:.1f} stop={best.stop:.0f} "
              f"trig={best.trig:.1f} gap={best.gap:.2f})")

# What's the MAX WR at each trade count threshold?
print(f"\nMax WR at each trade count:")
for ntarget in [20, 30, 50, 100, 200, 500]:
    candidates = df[df.n >= ntarget]
    if len(candidates):
        best = candidates.loc[candidates.WR.idxmax()]
        print(f"  n >= {ntarget:>4d}: max WR={best.WR:.1%}  PnL={best.PnL:>+7.1f}p  "
              f"(z={best.z:.1f} leg<{best.leg:.1f})")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 2: Multi-pair with relaxed configs
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: Multi-pair candidate configs (best from sweep)")
print("=" * 70)

candidate_configs = [
    ('Leg<0.3 z>5.0',        0.3, 5.0, 5.0, 0.7, 0.05),
    ('Leg<0.5 z>4.0',        0.5, 4.0, 6.0, 0.7, 0.05),
    ('Leg<0.5 z>3.0',        0.5, 3.0, 5.0, 0.7, 0.1),
    ('Leg<0.7 z>3.5',        0.7, 3.5, 5.0, 0.7, 0.05),
    ('Leg<0.7 z>3.0',        0.7, 3.0, 5.0, 0.5, 0.1),
    ('Leg<1.0 z>3.5',        1.0, 3.5, 5.0, 0.7, 0.05),
    ('Leg<1.0 z>3.0',        1.0, 3.0, 5.0, 0.5, 0.1),
    ('Leg<1.5 z>2.5 Sess',   1.5, 2.5, 5.0, 0.7, 0.05),  # Sydney only
]

for cfg_name, leg_t, zt, stop_a, trig_a, gap_a in candidate_configs:
    is_sydney = 'Sess' in cfg_name
    sess = range(21, 24) if is_sydney else None
    all_trades = []
    for pair in available:
        if pair not in z_scores: continue
        t = backtest_simple(pair, z_scores[pair], z_thresh=zt, leg_thresh=leg_t,
                            stop_a=stop_a, trig_a=trig_a, gap_a=gap_a,
                            session_hours=sess, max_bars=54)
        if len(t):
            all_trades.append(t)

    if all_trades:
        combined = pd.concat(all_trades)
        n_days = (data[available[0]].index[-1] - data[available[0]].index[0]).total_seconds() / 86400
        tpd = len(combined) / n_days
        wr = (combined.pnl_pips > 0).mean()
        net = combined.pnl_pips.sum()
        avg_p = combined.pnl_pips.mean()
        wr_display = f"{wr:.1%}"
        good = (combined.pnl_pips > 0).sum()
        bad = (combined.pnl_pips <= 0).sum()
        print(f"\n  {cfg_name:25s}: n={len(combined):>4d}  WR={wr_display}  "
              f"PnL={net:>+7.1f}p  avg={avg_p:>+5.1f}p  "
              f"~{tpd*30:.0f}/mo  win:{good} loss:{bad}")
    else:
        print(f"\n  {cfg_name:25s}: NO TRADES")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 3: GBPUSD z-score filter (the leg pair itself)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 3: Can we trade GBPUSD directly with leg filter?")
print("(Instead of cross pairs, trade the MAJOR when crosses are dislocated)")
print("=" * 70)

# For GBPUSD: "legs" are... well, there are no legs.
# But we can check if GBPNZD/EURAUD etc are extreme → signal that GBPUSD is
# about to revert (the leg overshoots).

# Idea: When GBPNZD z>3.0 and legs are quiet, fade the CROSS not the major.
# That's what we're already doing.

# But what if we trade GBPUSD DIRECTLY when CROSSES of GBP are extreme?
# e.g., GBPNZD z>4.0 AND GBPAUD z>4.0 → fade GBPUSD
# This would mean: "the GBP crosses are dislocating in the same direction,
# so fade the GBP leg that caused both"

# Let's check: if multiple GBP crosses fire same direction
print("\n  Idea: When MULTIPLE GBP crosses show same-direction signal, "
      "fade GBPUSD\n")

gbp_crosses = ['gbpnzd', 'gbpaud', 'gbpcad', 'eurgbp']  # eurgbp has reverse sign
filtered = [p for p in gbp_crosses if p in z_scores]
for leg_t in [0.5, 0.7, 1.0]:
    for zt in [3.0, 3.5, 4.0]:
        all_signals = []
        for pair in filtered:
            base, quote = LEG_MAP[pair]
            leg_mask = (z_scores[base].abs() < leg_t) & (z_scores[quote].abs() < leg_t)
            z_f = z_scores[pair].where(leg_mask & z_scores[pair].notna())
            # Direction: positive z = cross went UP (e.g., GBPNZD up)
            sig = (z_f.abs() >= zt).astype(int)
            # Store direction (1 = buy GBPUSD, -1 = sell GBPUSD)
            dir_sig = z_f.where(z_f.notna(), 0).apply(lambda x: 1 if x >= zt else (-1 if x <= -zt else 0))
            all_signals.append(dir_sig)

        if len(all_signals) >= 2:
            signal_df = pd.DataFrame(all_signals).T
            signal_df.columns = filtered
            # Count how many GBP crosses fire in same direction
            multi_same = signal_df[(signal_df.abs().sum(axis=1) >= 2)].copy()
            if len(multi_same) > 0:
                print(f"  leg<{leg_t} z>{zt}: {len(multi_same)} bars with 2+ GBP cross signals")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 4: Can we find ANY cross pair achieving 30 trades/day at 65%?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 4: What trade density is PHYSICALLY POSSIBLE?")
print("(Maximum possible trades ignoring WR, per config)")
print("=" * 70)

for leg_t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    for zt in [2.0, 3.0, 4.0]:
        total_trades = 0
        for pair in available:
            if pair not in z_scores: continue
            base, quote = LEG_MAP[pair]
            leg_mask = (z_scores[base].abs() < leg_t) & (z_scores[quote].abs() < leg_t)
            z_f = z_scores[pair].where(leg_mask & z_scores[pair].notna())
            signals = (z_f.abs() >= zt).sum()
            total_trades += signals
        n_days = (data[available[0]].index[-1] - data[available[0]].index[0]).total_seconds() / 86400
        tpd = total_trades / n_days
        print(f"  leg<{leg_t:.1f} z>{zt:.0f}: {total_trades:>4d} signal bars across {len(available)} pairs, ~{tpd:.1f}/day")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 5: What about shorter z-score windows?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 5: Different z-score windows")
print("(Shorter window = faster signals = more trades)")
print("=" * 70)

for window in [10, 20, 50, 100]:
    z_fast, _ = z_score(data['gbpnzd']['close'], window=window)
    t = backtest_simple('gbpnzd', z_fast, z_thresh=3.0, leg_thresh=0.7,
                        stop_a=5.0, trig_a=0.7, gap_a=0.05)
    if len(t):
        wr = (t.pnl_pips > 0).mean()
        net = t.pnl_pips.sum()
        n_days = (data['gbpnzd'].index[-1] - data['gbpnzd'].index[0]).total_seconds() / 86400
        tpd = len(t) / n_days
        print(f"  window={window:>4d}: n={len(t):>4d}  WR={wr:.1%}  "
              f"PnL={net:>+7.1f}p  avg={t.pnl_pips.mean():>+5.1f}p  "
              f"~{tpd*30:.0f}/mo")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 6: Cumulative returns of best found configs
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 6: Daily PnL series for top configs")
print("=" * 70)

for cfg_name, leg_t, zt, stop_a, trig_a, gap_a in [
    ('Leg<0.5 z>4.0',        0.5, 4.0, 6.0, 0.7, 0.05),
    ('Leg<0.5 z>3.0',        0.5, 3.0, 5.0, 0.7, 0.1),
    ('Leg<0.7 z>3.5',        0.7, 3.5, 5.0, 0.7, 0.05),
]:
    all_trades = []
    for pair in available:
        if pair not in z_scores: continue
        t = backtest_simple(pair, z_scores[pair], z_thresh=zt, leg_thresh=leg_t,
                            stop_a=stop_a, trig_a=trig_a, gap_a=gap_a)
        if len(t): all_trades.append(t)

    if all_trades:
        combined = pd.concat(all_trades)
        wr = (combined.pnl_pips > 0).mean()
        net = combined.pnl_pips.sum()
        n_days = (data[available[0]].index[-1] - data[available[0]].index[0]).total_seconds() / 86400
        tpd = len(combined) / n_days
        max_win = combined.pnl_pips.max()
        max_loss = combined.pnl_pips.min()
        std_pnl = combined.pnl_pips.std()
        sharpe = combined.pnl_pips.mean() / std_pnl * np.sqrt(252 * tpd) if std_pnl > 0 else 0
        max_dd_pct = 0
        cum = combined.pnl_pips.cumsum()
        peak = cum.expanding().max()
        dd = cum - peak
        max_dd_pct = dd.min()
        print(f"\n  {cfg_name:25s}:")
        print(f"    n={len(combined):>4d}  WR={wr:.1%}  PnL={net:>+7.1f}p  "
              f"avg={combined.pnl_pips.mean():>+5.1f}p  ~{tpd*30:.0f}/mo")
        print(f"    MaxWin={max_win:>+5.1f}p  MaxLoss={max_loss:>+5.1f}p  "
              f"Std={std_pnl:.1f}")
        print(f"    Sharpe={sharpe:.2f}  MaxDD={max_dd_pct:.1f}p")
        pct_pos = (combined.pnl_pips > 0).sum() / len(combined)
        print(f"    Best consecutive samples: demo")
