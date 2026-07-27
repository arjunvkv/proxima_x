"""Verify live v2z_bar trades against backtest data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime

PAIR = "GBPNZD"
Z_THRESH = 2.5

# Decode timestamps
fills = {
    "GBPNZD": 1784830140,
    "EURNZD": 1784830201,
    "GBPAUD": 1784830261,
    "EURAUD": 1784830321,
    "GBPCAD": 1784830381,
    "AUDNZD": 1784830441,
}
print("=== Trade Entry Times (UTC) ===")
for p, ts in sorted(fills.items(), key=lambda x: x[1]):
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    print(f"  {p}: {ts} = {dt}")

print("\n=== GBPNZD Bar Context Around Entry ===")
df = pd.read_parquet("research/phase_dislocation/dukascopy_data/gbpnzd.parquet")
if "timestamp" in df.columns:
    df = df.set_index("timestamp")
df.index = pd.to_datetime(df.index)

ts_target = pd.to_datetime(fills["GBPNZD"], unit="s")
t0 = ts_target - pd.Timedelta(minutes=10)
t1 = ts_target + pd.Timedelta(minutes=10)
window = df.loc[t0:t1]
print(f"Bars from {t0} to {t1}:")
for idx, row in window.iterrows():
    marker = " <-- ENTRY" if idx == ts_target else ""
    print(f"  {idx} O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}{marker}")

# Compute z-scores and ATR for the entry bar
c = df["close"].values
h = df["high"].values
l = df["low"].values
n = len(df)

# Find index of entry bar
entry_idx = df.index.get_indexer([ts_target], method="pad")[0]
print(f"\n=== Entry bar index = {entry_idx} ===")
print(f"Bar timestamp: {df.index[entry_idx]}")

# Z-score computation (same as backtest)
ret = c[entry_idx] - c[entry_idx - 1]
rw = []
for k in range(entry_idx - 51, entry_idx - 1):
    rw.append(c[k] - c[k - 1])
mu = sum(rw) / len(rw)
var = sum((x - mu) ** 2 for x in rw) / (len(rw) - 1)
sig = var ** 0.5
z = (ret - mu) / sig if sig > 1e-10 else 0
print(f"Return: {ret:.8f}")
print(f"Mean of 50 prior: {mu:.8f}")
print(f"Std dev: {sig:.8f}")
print(f"Z-score: {z:.4f}")
print(f"Signal: {'SHORT' if z > Z_THRESH else 'LONG' if z < -Z_THRESH else 'NONE'} (z>{Z_THRESH})")

# ATR
rng = df["high"].values - df["low"].values
atr = rng[entry_idx - 20:entry_idx].mean()
print(f"ATR: {atr:.6f}")

# Trailing stop params
s = 0.15 * atr
tg = 0.20 * atr
gp = 0.10 * atr
entry_price = c[entry_idx]
direction = -1
best = entry_price
stop = entry_price - s

print(f"\n=== Stop Calculation ===")
print(f"Entry: {entry_price:.5f}")
print(f"Initial stop: {stop:.5f} (dist={s:.6f})")
print(f"Trigger: {tg:.6f}, Gap: {gp:.6f}")

# Check next bar
if entry_idx + 1 < n:
    nb = entry_idx + 1
    print(f"\n=== Next Bar ===")
    print(f"  {df.index[nb]} O={df.iloc[nb]['open']:.5f} H={df.iloc[nb]['high']:.5f} L={df.iloc[nb]['low']:.5f} C={df.iloc[nb]['close']:.5f}")
    if h[nb] > best:
        best = h[nb]
    if best - entry_price > tg:
        stop = best - gp
        print(f"  Triggered! Best={best:.5f}, Stop moved to {stop:.5f}")
    if l[nb] <= stop:
        print(f"  STOP HIT at {stop:.5f}")
    else:
        print(f"  No stop hit")

print("\n=== Live Trade Log ===")
log = pd.read_csv("paper_trade/live/logs/v2z_bar_1784830093.csv")
for _, row in log.iterrows():
    ts = row["timestamp"]
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    print(f"  {dt} | {row['type']:5s} | {row['pair']:7s} | {row['detail']}")
