"""Investigate if FundedNext tick bias/lookahead explains EURUSD edge.

Key question: If FundedNext tick data has stale bid (0% bid flags), then
mid = (stale_bid + ask)/2 would create artificial mean reversion.
When ask spikes up and returns, mid goes up then down, but it's NOT
real price action - just bid not updating.

We check:
1. Does bid actually update or is it truly stale?
2. Are 5p mid impulses caused by ask moving or bid moving?
3. Does the raw mid WR hold up if we use ONLY ask prices?
"""
import sys, numpy as np
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")
PIP = 0.0001

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    flags = np.array([t[6] for t in d], dtype=np.uint32)
    return ts, bid, ask, flags

def detect_on_price(ts, price, pip, window_s=20, detect_pips=5):
    """Generic detection on any price series (mid, ask, etc.)."""
    n = len(price)
    min_q, max_q = deque(), deque()
    ws_idx = 0
    evs = []
    thresh = detect_pips * pip
    for i in range(n):
        v = float(price[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > window_s:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = price[ws_idx]
            hp = float(max_q[0][0] - wp)
            lp = float(wp - min_q[0][0])
            span = ts[i] - ts[ws_idx]
            if span <= window_s and (hp >= thresh or lp >= thresh):
                if evs and evs[-1][0] >= ws_idx: continue
                ext_idx = max_q[0][1] if hp >= lp else min_q[0][1]
                d = 1 if hp >= lp else -1
                evs.append((ws_idx, ext_idx, d))
    return evs

def measure_fade(ev_list, ts, price, hold_s=30):
    """For each impulse, measure if price fades at hold_s.
    Uses the same price series for entry/exit - zero transaction cost.
    """
    n = len(price)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = price[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        xp = price[he]
        pnl = (xp - ep) * ed
        pnls.append(pnl)
    return np.array(pnls, dtype=np.float64)

print("Loading EURUSD...")
ts, bid, ask, flags = load("EURUSD")
mid = (bid + ask) / 2.0

# === CHECK 1: Does bid ever change? ===
bid_changes = np.sum(np.diff(bid) != 0)
ask_changes = np.sum(np.diff(ask) != 0)
both_change = np.sum((np.diff(bid) != 0) & (np.diff(ask) != 0))
only_bid = np.sum((np.diff(bid) != 0) & (np.diff(ask) == 0))
only_ask = np.sum((np.diff(bid) == 0) & (np.diff(ask) != 0))

print(f"\n=== BID/ASK Change Analysis ===")
print(f"Total ticks: {len(ts):,}")
print(f"Bid changes: {bid_changes:,} ({bid_changes/len(ts)*100:.1f}%)")
print(f"Ask changes: {ask_changes:,} ({ask_changes/len(ts)*100:.1f}%)")
print(f"Both change together: {both_change:,} ({both_change/len(ts)*100:.1f}%)")
print(f"Only bid changes: {only_bid:,} ({only_bid/len(ts)*100:.1f}%)")
print(f"Only ask changes: {only_ask:,} ({only_ask/len(ts)*100:.1f}%)")
print(f"Bid stays same while ask moves: {only_ask / max(ask_changes,1) * 100:.1f}% of ask movements")

# === CHECK 2: Detect 5p/20s on MID, ASK, BID separately ===
print(f"\n=== Detection: 5p/20s on different price series ===")
for label, price in [("MID", mid), ("ASK", ask), ("BID", bid)]:
    evs = detect_on_price(ts, price, PIP)
    raw_pnls = measure_fade(evs, ts, price, 30)
    wr = (raw_pnls > 0).mean() * 100
    avg = raw_pnls.mean() / PIP
    print(f"{label}: {len(evs):>5d} impulses  WR={wr:.1f}%  avg={avg:+.2f}p  total={raw_pnls.sum()/PIP:+.0f}p")

# === CHECK 3: Cross-detection — detect on ASK, measure fade on ASK ===
print(f"\n=== Cross-check: detect on one series, measure fade on same ===")
for detect_label, detect_price in [("MID", mid), ("ASK", ask)]:
    for measure_label, measure_price in [("MID", mid), ("ASK", ask)]:
        evs = detect_on_price(ts, detect_price, PIP)
        pnls = measure_fade(evs, ts, measure_price, 30)
        wr = (pnls > 0).mean() * 100
        avg = pnls.mean() / PIP
        print(f"Detect on {detect_label}, fade on {measure_label}: "
              f"n={len(evs)} WR={wr:.1f}% avg={avg:+.2f}p")

# === CHECK 4: What if we detect on MID but measure on ASK (real exec price)? ===
print(f"\n=== Practical check: detect on MID, simulate on ASK (entry) === ")
evs = detect_on_price(ts, mid, PIP)
pnls_ask = measure_fade(evs, ts, ask, 30)
wr_ask = (pnls_ask > 0).mean() * 100
avg_ask = pnls_ask.mean() / PIP
print(f"  WR={wr_ask:.1f}% avg={avg_ask:+.2f}p  (entry=exit both at ASK, zero cost)")

# === CHECK 5: Compare with Exness ticks ===
print(f"\n=== Comparison w/ Exness (EURUSD, Oct-Dec 2025) ===")
from pathlib import Path
import pandas as pd
TICK_DIR_EX = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")
ex_files = [
    TICK_DIR_EX / "EURUSD_Raw_Spread_2025_10.zip",
    TICK_DIR_EX / "EURUSD_Raw_Spread_2025_11.zip",
    TICK_DIR_EX / "EURUSD_Raw_Spread_2025_12.zip",
]
ex_dfs = []
for f in ex_files:
    d = pd.read_csv(f, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                    dtype={"Ts":str,"B":np.float64,"A":np.float64})
    d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                             format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    ex_dfs.append(d.dropna(subset=["Ts"]))
ex = pd.concat(ex_dfs).sort_values("Ts").reset_index(drop=True)
ex_mid = (ex["B"].values + ex["A"].values) / 2.0
ex_ts = ex["Ts"].astype(np.int64).values // 10**9

# Detect on exness mid
evs_ex = detect_on_price(ex_ts, ex_mid, PIP)
# Raw WR on Exness (mid price, zero cost)
pnls_ex_raw = measure_fade(evs_ex, ex_ts, ex_mid, 30)
wr_ex = (pnls_ex_raw > 0).mean() * 100
avg_ex = pnls_ex_raw.mean() / PIP
print(f"Exness EURUSD (65 days):")
print(f"  Ticks: {len(ex):,}")
print(f"  {len(evs_ex)} impulses  Raw WR={wr_ex:.1f}%  avg={avg_ex:+.2f}p")

# But does Exness also show bid staleness?
ex_bid = ex["B"].values.astype(np.float64)
ex_ask = ex["A"].values.astype(np.float64)
ex_bid_changes = np.sum(np.diff(ex_bid) != 0)
ex_ask_changes = np.sum(np.diff(ex_ask) != 0)
print(f"  Exness bid changes: {ex_bid_changes:,} ({ex_bid_changes/len(ex)*100:.1f}%)")
print(f"  Exness ask changes: {ex_ask_changes:,} ({ex_ask_changes/len(ex)*100:.1f}%)")

# Now detect/mesure on Exness ASK only
evs_ex_ask = detect_on_price(ex_ts, ex_ask, PIP)
pnls_ex_ask = measure_fade(evs_ex_ask, ex_ts, ex_ask, 30)
wr_ex_ask = (pnls_ex_ask > 0).mean() * 100
avg_ex_ask = pnls_ex_ask.mean() / PIP
print(f"  Detect/measure on ASK only: n={len(evs_ex_ask)} WR={wr_ex_ask:.1f}% avg={avg_ex_ask:+.2f}p")

print(f"\n=== CONCLUSIONS ===")
