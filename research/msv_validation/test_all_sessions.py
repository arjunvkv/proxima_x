"""Check if the exhaustion signal works outside Asia."""
import sys, numpy as np
from datetime import datetime, timedelta, timezone
from collections import deque
sys.path.insert(0, "currency_decomposition")
import MetaTrader5 as mt5
from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

PAIRS = list(BASE_CURRENCY_MAP.keys())[:16]
HORIZON = 30
ROLLING = 500

end = datetime.now()
start = end - timedelta(days=120)
all_data = {}
for p in PAIRS:
    r = mt5.copy_rates_range(p, mt5.TIMEFRAME_M5, start, end)
    if r is not None and len(r) > 0:
        all_data[p] = r
N = min(len(v) for v in all_data.values())
print(f"Data: {len(all_data)} pairs, {N} bars")

ms = MarketStateVector()
dh = deque(maxlen=ROLLING)

def hour_to_session(h):
    if h < 7:   return "ASIA"
    if h < 12:  return "LONDON"
    if h < 16:  return "LDN_NY"
    if h < 21:  return "NY"
    return "NY_LATE"

records = {s: [] for s in ["ASIA","LONDON","LDN_NY","NY","NY_LATE"]}

for idx in range(N):
    rets = {}
    for p in all_data:
        if idx == 0:
            rets[p] = 0.0
        else:
            c = float(all_data[p][idx]["close"])
            pv = float(all_data[p][idx-1]["close"])
            rets[p] = max(min((c/pv-1) if pv>0 else 0.0, 0.05), -0.05)
    now = float(all_data[PAIRS[0]][idx]["time"])
    snap = ms.update(rets, timestamp=now)
    dh.append(snap.network.dispersion)
    dp = sum(1 for h in dh if h < snap.network.dispersion) / max(len(dh), 1)

    pre60 = 0.0
    if idx >= 12:
        for p in all_data:
            cur = float(all_data[p][idx]["close"])
            p60 = float(all_data[p][idx-12]["close"])
            pre60 += (cur/p60 - 1) if p60 > 0 else 0.0
        pre60 /= len(all_data)

    if len(dh) >= 12:
        dv = snap.network.dispersion - list(dh)[-12]
    else:
        dv = 0.0

    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    sess = hour_to_session(dt.hour)

    # forward 30m return (ALL pairs basket)
    if idx + HORIZON < N:
        fwd = []
        for p in all_data:
            cur = float(all_data[p][idx]["close"])
            fut = float(all_data[p][idx+HORIZON]["close"])
            fwd.append((fut/cur-1) if cur>0 else 0.0)
        fwd_mu = float(np.mean(fwd))
    else:
        fwd_mu = None

    cond = (dp >= 0.95 and pre60 < -0.0002 and dv > 0)
    if cond and fwd_mu is not None:
        records[sess].append(fwd_mu)

    if (idx+1) % 5000 == 0:
        print(f"  {idx+1}/{N}")

print()
print(f"{'Session':12s} {'n':>6s} {'MeanRet':>10s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
print("-" * 52)
for sess in ["ASIA","LONDON","LDN_NY","NY","NY_LATE"]:
    vals = records[sess]
    if len(vals) < 5:
        continue
    mu = float(np.mean(vals))
    s = float(np.std(vals))
    sh = (mu/s) * np.sqrt(12*24) if s > 0 else 0.0
    pos = sum(1 for v in vals if v > 0) / len(vals) * 100
    t = mu / (s / np.sqrt(len(vals))) if s > 0 else 0.0
    print(f"{sess:12s} {len(vals):6d} {mu:+10.6f} {sh:+8.3f} {pos:5.1f}% {t:+8.2f}")

mt5.shutdown()
