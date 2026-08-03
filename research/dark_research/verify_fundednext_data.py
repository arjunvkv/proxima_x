#!/usr/bin/env python3
"""Verify FundedNext data quality."""
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(__file__)
pairs = ["eurjpy", "eurusd", "gbpjpy"]
pair_names = ["EURJPY", "EURUSD", "GBPJPY"]

for p, pn in zip(pairs, pair_names):
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()

    print(f"\n{pn}:")
    print(f"  Bars: {len(df)}")
    print(f"  Period: {df.index[0]} - {df.index[-1]}")
    print(f"  Close: mean={df['close'].mean():.5f} min={df['close'].min():.5f} max={df['close'].max():.5f}")

    # Check for gaps
    gaps = df.index.to_series().diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta(minutes=2)]
    print(f"  Gaps > 2min: {len(large_gaps)}")
    if len(large_gaps) > 0:
        print(f"  Largest gap: {large_gaps.max()}")

    # Check first bar timestamp alignment
    print(f"  First bar: {df.index[0]}")
    print(f"  Second: {df.index[1]}")
    print(f"  Minute offset of first bar: {df.index[0].minute}.{df.index[0].second}")

    # Check for NaN/zero prices
    for col in ["open", "high", "low", "close"]:
        bad = (df[col] == 0).sum() | df[col].isna().sum()
        if bad > 0:
            print(f"  WARNING: {bad} {col} values are 0 or NaN")

    # Check price continuity (returns shouldn't be absurd)
    rets = np.abs(np.diff(np.log(df["close"].values)))
    extreme = np.sum(rets > 0.01)  # >1% in 1 minute
    print(f"  1-min returns >1%: {extreme} ({extreme/len(rets)*100:.3f}%)")
    extreme2 = np.sum(rets > 0.02)
    print(f"  1-min returns >2%: {extreme2} ({extreme2/len(rets)*100:.3f}%)")

print("\n=== Cross-pair alignment ===")
# Check that timestamps match exactly
times = {}
for p in pairs:
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    t = pd.to_datetime(pd.DataFrame(d)["time"], unit="s")
    times[p] = set(t)

for p1 in pairs:
    for p2 in pairs:
        if p1 < p2:
            overlap = times[p1] & times[p2]
            print(f"  {p1} x {p2}: {len(overlap)}/{len(times[p1])} common ({len(overlap)/len(times[p1])*100:.1f}%)")
