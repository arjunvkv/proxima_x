"""Quick verification that two-pointer matches searchsorted approach."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
import pandas as pd, numpy as np
from pathlib import Path

D = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")
d = pd.read_csv(D / "EURUSD_Raw_Spread_2025_10.zip", names=["E","S","Ts","B","A"],
                skiprows=1, header=None, dtype={"Ts": str, "B": float, "A": float}, nrows=50000)
d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
d = d.dropna(subset=["Ts"]); d["ts_s"] = d["Ts"].astype(np.int64) // 10**9

m = (d["B"].values + d["A"].values) / 2
ts = d["ts_s"].values; n = len(d)
print(f"Testing with {n} ticks")

# Old: searchsorted per tick
t0 = time.time()
e1 = []; i = 0
while i < n:
    end = min(int(np.searchsorted(ts, ts[i] + 10, side="right")), n)
    w = m[i:end]
    if len(w) < 2: i += 1; continue
    hp = (np.max(w) - w[0]) * 10000
    lp = abs((np.min(w) - w[0]) * 10000)
    if max(hp, lp) >= 5:
        d1 = 1 if hp >= lp else -1
        ei = i + (np.argmax(w) if d1 == 1 else np.argmin(w))
        e1.append(ei); i = ei
    else: i += 1
t1 = time.time()
print(f"Old (searchsorted): {len(e1)} impulses in {(t1-t0)*1000:.0f}ms")

# New: two-pointer
t0 = time.time()
e2 = []; i = 0; end = 0
while i < n:
    if i >= end: end = i + 1
    while end < n and ts[end] <= ts[i] + 10: end += 1
    if end - i >= 2:
        w = m[i:end]
        hp = (np.max(w) - m[i]) * 10000
        lp = abs((np.min(w) - m[i]) * 10000)
        if max(hp, lp) >= 5:
            d1 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d1 == 1 else np.argmin(w))
            e2.append(ei); i = ei; continue
    i += 1
t1 = time.time()
print(f"New (two-pointer): {len(e2)} impulses in {(t1-t0)*1000:.0f}ms")

match = sum(1 for a, b in zip(e1, e2) if a == b)
print(f"Matched impulses: {match}/{len(e1)}")
if match < len(e1):
    diff = [(a, b) for a, b in zip(e1, e2) if a != b][:5]
    print(f"First {len(diff)} diffs: {diff}")
