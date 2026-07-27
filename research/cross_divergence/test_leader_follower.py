"""Test multi-pair leader-follower dynamics and volume-filtered tick exhaustion."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

print("=== Loading tick data ===")
all_dfs = {}
for name, f in [("EURUSD","EURUSD_mt5.csv"), ("GBPUSD","GBPUSD_mt5.csv"),
                 ("USDJPY","USDJPY_mt5_v2.csv"), ("EURJPY","EURJPY_mt5_v2.csv"),
                 ("GBPJPY","GBPJPY_mt5_v2.csv")]:
    path = DATA / f
    if not path.exists(): continue
    df = pd.read_csv(path, parse_dates=['time'])
    df['mid'] = (df.bid + df.ask) / 2
    df = df[['time','mid']].set_index('time')
    all_dfs[name] = df
    print(f"  {name}: {len(df):,} ticks, {df.index.min()} → {df.index.max()}")

aligned = None
for name, df in all_dfs.items():
    s = df.resample('1s').last().dropna().rename(columns={'mid': name})
    if aligned is None: aligned = s
    else: aligned = aligned.join(s, how='outer')
aligned = aligned.ffill(limit=1).dropna()
print(f"\n  Aligned 1s bars: {len(aligned):,} ({aligned.index.min()} → {aligned.index.max()})")

rets = aligned.pct_change().dropna() * 10000
has = {c: c in rets.columns for c in ['EURUSD','GBPUSD','USDJPY','EURJPY','GBPJPY']}
print(f"  Available: {[k for k,v in has.items() if v]}")

def sharp_move_test(leader, follower, thresholds, lags, rets):
    print(f"\n=== {leader} Sharp Move → {follower} Follow ===")
    for thresh in thresholds:
        for lag_s in lags:
            mask = rets[leader].abs() > thresh
            leader_dir = np.sign(rets[leader])
            fwd = rets[follower].shift(-lag_s)
            valid = mask & fwd.notna()
            n = valid.sum()
            if n < 10: continue
            same_dir = (np.sign(fwd) == leader_dir)[valid].mean()
            avg_ret = fwd[valid].mean() * np.sign(leader_dir[valid]).mean()
            print(f"  {leader}>{thresh:3.1f}bps → {follower}+{lag_s:2d}s: "
                  f"same_dir={same_dir*100:.0f}% N={n:,} avg={avg_ret:+.2f}bps")

# Test all leader-follower pairs
thresholds = [0.5, 1.0, 2.0, 5.0]
lags = [1, 2, 3, 5, 10, 30, 60]

if has['EURUSD'] and has['GBPUSD']:
    sharp_move_test('EURUSD', 'GBPUSD', thresholds, lags, rets)

if has['USDJPY'] and has['EURJPY']:
    sharp_move_test('USDJPY', 'EURJPY', thresholds, lags, rets)

if has['EURJPY'] and has['GBPJPY']:
    sharp_move_test('EURJPY', 'GBPJPY', thresholds, lags, rets)

if has['EURUSD'] and has['USDJPY'] and has['EURJPY']:
    print("\n=== EURJPY Implied vs Actual Deviation ===")
    rets['eurjpy_implied'] = rets.EURUSD + rets.USDJPY
    rets['eurjpy_delta'] = rets.EURJPY - rets.eurjpy_implied
    delta = rets.eurjpy_delta
    print(f"  EURJPY dev from implied: mean={delta.mean():+.3f}bps, std={delta.std():.3f}bps, "
          f"max_dev={delta.abs().max():.2f}bps")
    # After deviation, does EURJPY mean-revert?
    for lag_s in lags:
        fwd = delta.shift(-lag_s)
        # Mean reversion: if delta positive → EURJPY overpriced → expect negative return
        reversals = (np.sign(delta) != np.sign(fwd))[delta.abs() > 0.5]
        n = reversals.sum()
        if n < 10: continue
        wr = reversals.mean()
        print(f"  EURJPY dev→revert+{lag_s:2d}s: WR={wr*100:.0f}% N={n:,}")

# Cross-correlation heatmap
print("\n=== Cross-Correlations at Equal Time ===")
pairs = [(l, f) for l in ['EURUSD','USDJPY','EURJPY','GBPJPY','GBPUSD'] if l in rets.columns
         for f in ['EURUSD','USDJPY','EURJPY','GBPJPY','GBPUSD'] if f in rets.columns and l < f]
for l, f in pairs:
    corr = rets[l].corr(rets[f])
    print(f"  corr({l}, {f}) = {corr:+.3f}")

# Volume-filtered tick exhaustion on EURUSD
if Path(DATA / "EURUSD_mt5.csv").exists():
    print("\n=== EURUSD Volume-Filtered Tick Exhaustion ===")
    raw = pd.read_csv(DATA / "EURUSD_mt5.csv", parse_dates=['time'])
    raw['mid'] = (raw.bid + raw.ask) / 2
    mids = raw.mid.values.astype(float)
    vol = raw.volume.values.astype(float)
    dirs = np.sign(np.diff(mids))
    for vol_rank in ['all','top50','top25','top10']:
        vol_thresh = np.percentile(vol[1:], 50) if vol_rank == 'top50' else \
                     np.percentile(vol[1:], 75) if vol_rank == 'top25' else \
                     np.percentile(vol[1:], 90) if vol_rank == 'top10' else 0
        lookahead = 10
        for streak in [3, 4, 5]:
            wins = 0; total = 0
            for i in range(streak, len(dirs) - lookahead):
                if dirs[i-streak:i].sum() == streak:
                    if vol_rank == 'all' or vol[i] >= vol_thresh:
                        total += 1
                        fwd = dirs[i:i+lookahead]
                        if fwd.sum() < 0: wins += 1
            if total > 0:
                print(f"  streak≥{streak}, {vol_rank:7s}: WR={wins/total*100:.1f}% N={total:,}")

print("\n=== DONE ===")
