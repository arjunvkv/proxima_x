"""Test cross-pair correlation lag.
Hypothesis: when EURUSD moves, GBPUSD follows within 1-5 seconds (lag, not instant)."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

# Load overlapping pairs
print("=== Loading tick data ===")
eusd = pd.read_csv(DATA / "EURUSD_mt5.csv", parse_dates=['time'])
gbpusd = pd.read_csv(DATA / "GBPUSD_mt5.csv", parse_dates=['time'])
print(f"  EURUSD: {len(eusd):,} ticks, {eusd.time.min()} → {eusd.time.max()}")
print(f"  GBPUSD: {len(gbpusd):,} ticks, {gbpusd.time.min()} → {gbpusd.time.max()}")

# Resample to 50ms bars for fine-grained alignment
print("\n=== Aligning at 50ms ===")
eusd['mid'] = (eusd.bid + eusd.ask) / 2
gbpusd['mid'] = (gbpusd.bid + gbpusd.ask) / 2

eusd_50 = eusd.set_index('time')[['mid']].resample('50ms').last().dropna().rename(columns={'mid': 'eurusd'})
gbpusd_50 = gbpusd.set_index('time')[['mid']].resample('50ms').last().dropna().rename(columns={'mid': 'gbpusd'})
aligned = eusd_50.join(gbpusd_50, how='inner')
print(f"  Aligned 50ms bars: {len(aligned):,}")
print(f"  Range: {aligned.index.min()} → {aligned.index.max()}")

# Compute 1-bar returns
aligned['eur_ret'] = aligned.eurusd.pct_change() * 10000  # in bps
aligned['gbp_ret'] = aligned.gbpusd.pct_change() * 10000

# Test cross-correlation with lag
print("\n=== Cross-Correlation: EURUSD ↔ GBPUSD ===")
for lag_ms in [50, 100, 200, 300, 500, 1000, 2000, 5000]:
    lag_bars = lag_ms // 50
    if lag_bars == 0: continue
    col = f'gbp_lag_{lag_ms}'
    aligned[col] = aligned.gbp_ret.shift(-lag_bars)
    mask = (aligned.eur_ret.abs() > 0.5)  # only meaningful moves
    corr = aligned.loc[mask, 'eur_ret'].corr(aligned.loc[mask, col])
    print(f"  corr(EURUSD(t), GBPUSD(t+{lag_ms:4d}ms)) = {corr:+.3f}")

# Trade simulation: when EURUSD moves, enter GBPUSD in same direction
print("\n=== Simulated Trade: EURUSD move → enter GBPUSD ===")
results = []
for lag_ms in [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 30000]:
    lag_bars = lag_ms // 50
    if lag_bars == 0: continue
    # Entry: EURUSD has a significant move (> 1 bps = ~1 pip)
    entry_mask = (aligned.eur_ret.abs() > 1.0)
    entry_idx = aligned[entry_mask].index
    if len(entry_idx) < 5: continue

    wins = 0
    pips = []
    lags = []
    for idx in entry_idx[:10000]:  # limit for speed
        entry_eur = aligned.eurusd.loc[idx]
        entry_gbp = aligned.gbpusd.loc[idx]
        direction = np.sign(aligned.eur_ret.loc[idx])

        # Find first tick after lag_ms in alignment
        target_time = idx + pd.Timedelta(milliseconds=lag_ms)
        after = aligned[aligned.index >= target_time]
        if len(after) == 0: continue
        exit_idx = after.index[0]
        exit_gbp = aligned.gbpusd.loc[exit_idx]

        ret = (exit_gbp - entry_gbp) / entry_gbp * 10000 * np.sign(direction)
        pips.append(ret)
        if ret > 0: wins += 1
    if len(pips) < 5: continue
    wr = wins / len(pips)
    avg = np.mean(pips)
    print(f"  lag={lag_ms:5d}ms: WR={wr*100:.1f}% avg={avg:+.2f}bps N={len(pips):,}")

# Tick rate burst test
print("\n=== Tick Rate Burst Test (EURUSD) ===")
# Count ticks per second
tick_counts = eusd.set_index('time').resample('1s').size()
tick_rate = tick_counts.values
burst_mask = tick_rate > np.percentile(tick_rate, 95)  # top 5% tick rate
print(f"  Total seconds: {len(tick_rate):,}")
print(f"  Burst seconds (top 5%): {burst_mask.sum():,}")
print(f"  Mean tick rate: {tick_rate.mean():.1f}/s")
print(f"  Burst tick rate: {tick_rate[burst_mask].mean():.1f}/s")

# After a burst, does price revert?
burst_prices = []
for i in range(len(tick_rate)):
    if burst_mask[i]:
        t = tick_counts.index[i]
        # Price at burst
        before = eusd[eusd.time <= t].iloc[-1]
        # Price 60s after burst
        after_t = t + pd.Timedelta(seconds=60)
        after = eusd[eusd.time >= after_t]
        if len(after) == 0: continue
        e = after.iloc[0]
        ret = (e.mid - before.mid) / before.mid * 10000
        burst_prices.append(ret)

if burst_prices:
    bp = np.array(burst_prices)
    print(f"  Burst events: {len(bp):,}")
    print(f"  Post-burst 60s avg return: {bp.mean():+.2f} bps")
    print(f"  Post-burst 60s positive: {(bp > 0).mean()*100:.1f}%")
    print(f"  Post-burst 60s negative: {(bp < 0).mean()*100:.1f}%")

print("\n=== DONE ===")
