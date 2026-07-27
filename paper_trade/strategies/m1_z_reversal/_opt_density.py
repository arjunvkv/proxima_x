"""Ultra-optimized: 30 events/day @ 65%+ WR via micro-trend filter + grid."""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_META = {
    "EURUSD": {"pip": 0.0001, "cost": 0.00003, "spread_p": 0.3},
    "EURJPY": {"pip": 0.01, "cost": 0.006, "spread_p": 0.6},
}

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False), format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect_fast(ticks, pip_size, thresh_pips=5, thresh_sec=10):
    """O(n) two-pointer sliding window detect."""
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size
    evs = []; end = 0; start = 0

    while start < n:
        if end < start + 1: end = start + 1
        while end < n and ts[end] - ts[start] < thresh_sec:
            end += 1
        if end - start < 2: start += 1; continue

        w = mid[start:end]
        w_min = np.min(w); w_max = np.max(w)
        hp = w_max - w[0]; lp = w[0] - w_min

        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = start + (np.argmax(w) if d2 == 1 else np.argmin(w))
            evs.append({"extreme_idx": ei, "direction": d2,
                        "price_extreme": mid[ei], "time": ts[start]})
            start = ei; end = ei
        else:
            start += 1

    return pd.DataFrame(evs)

def micro_trend(events, ticks, lookback_s=60):
    """For each event, compute price change over lookback_s before impulse start.
    Returns array: +1 if trend goes WITH fade direction, -1 if against, 0 if flat."""
    bids = ticks["B"].values; asks = ticks["A"].values; mid = (bids + asks) / 2
    ts = ticks["ts_s"].values; n = len(ts)
    results = np.full(len(events), np.nan)

    for i, (_, ev) in enumerate(events.iterrows()):
        ei = int(ev["extreme_idx"])
        if ei < 3: continue
        start_t = ts[ei] - lookback_s
        sj = int(np.searchsorted(ts, start_t, side="right"))
        if sj >= ei - 1: continue
        mid_before = mid[sj]; mid_at_event = mid[ei]
        change = mid_at_event - mid_before  # price change over lookback
        impulse_dir = ev["direction"]       # +1 = price went UP in impulse
        fade_dir = -impulse_dir

        # If trend is in fade direction → strong setup
        if change * fade_dir > 0:
            results[i] = 1  # WITH trend
        elif change * fade_dir < 0:
            results[i] = -1  # AGAINST trend
        else:
            results[i] = 0  # flat
    return results

def sim_batch(events_data, ticks, pip, cost, holds, trend_filter=None):
    """Sim all hold times + trend filter in one pass per event."""
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ts)

    n_ev = len(events_data); n_hold = len(holds)
    pnl = np.full((n_ev, n_hold), np.nan)
    trend_ok = np.ones(n_ev, dtype=bool)  # always True if no filter

    for ev_i in range(n_ev):
        ed = -events_data["direction"].iloc[ev_i]
        ei = int(events_data["extreme_idx"].iloc[ev_i]) + 1
        if ei >= n - 1: continue
        ep = asks[ei] if ed == 1 else bids[ei]; et = ts[ei]

        # Pre-compute exit for longest hold
        max_h = max(holds); max_end = min(int(np.searchsorted(ts, et + max_h, side="right")), n)
        if max_end <= ei + 1: continue

        for hi, h in enumerate(holds):
            hold_end = min(int(np.searchsorted(ts, et + h, side="right")), n)
            if hold_end >= n: continue
            xp = bids[hold_end] if ed == 1 else asks[hold_end]
            pnl[ev_i, hi] = (xp - ep) * ed - cost

    return pnl

# ── Config ──
THRESHOLDS = [3, 4, 5, 6, 7]
WINDOWS = [10, 15, 20]
HOLDS = [30, 60, 120]
LOOKBACKS = [30, 60, 120]

t0 = time.time()
print(f"\n{'='*80}", flush=True)
print(f"OPTIMIZED DENSITY TEST — 30 events/day @ 65%+ WR")
print(f"{'='*80}", flush=True)

# Load all data once (EURUSD and EURJPY)
data = {}
for pair in ["EURUSD", "EURJPY"]:
    t_load = time.time()
    data[pair] = load(pair, [(2025,10),(2025,11),(2025,12)])
    print(f"  Loaded {pair}: {len(data[pair]):,} ticks in {time.time()-t_load:.1f}s", flush=True)

# ── PHASE 1: Raw density vs WR (no filter) ──
print(f"\n{'─'*80}", flush=True)
print("PHASE 1: Raw density × WR (threshold × window × hold)")
print(f"{'─'*80}", flush=True)

results = []
for pair in ["EURUSD", "EURJPY"]:
    meta = PAIR_META[pair]; t = data[pair]; pip = meta["pip"]

    for tp in THRESHOLDS:
        for ws in WINDOWS:
            if pair == "EURJPY" and tp < 5: continue  # EURJPY needs higher threshold
            t_d = time.time()
            ev = detect_fast(t, pip, tp, ws)
            if len(ev) < 50: continue

            pnl = sim_batch(ev, t, pip, meta["cost"], HOLDS)

            for hi, hold in enumerate(HOLDS):
                p = pnl[:, hi]
                p = p[~np.isnan(p)] / pip
                n = len(p); wr = (p > 0).mean() * 100
                tdays = 65  # approximate
                results.append({"pair":pair, "tp":tp, "ws":ws, "hold":hold,
                                "n":n, "wr":wr, "gross":p.sum(), "avg":p.mean(),
                                "per_day":n/65, "filter":"none"})

# Display top results
print(f"\n{'Pair':>6s} {'Thr':>3s} {'Win':>3s} {'Hold':>4s} | {'n(3mo)':>7s} {'n/day':>5s} | {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s}", flush=True)
print(f"{'─'*60}", flush=True)

# Group by pair and show best per (threshold, window)
for pair in ["EURUSD", "EURJPY"]:
    sub = [r for r in results if r["pair"]==pair and r["filter"]=="none"]
    sub.sort(key=lambda x: -x["wr"])
    seen = set()
    for r in sub:
        key = (r["tp"], r["ws"], r["hold"])
        if key not in seen and r["n"] >= 50:
            seen.add(key)
            print(f" {pair:>6s} {r['tp']:>3d}p {r['ws']:>3d}s {r['hold']:>4d}s | {r['n']:>7d} {r['per_day']:>5.1f} | {r['wr']:>5.1f}% {r['gross']:>+9.1f}p {r['avg']:>+7.2f}p", flush=True)

# ── PHASE 2: Micro-trend filter ──
print(f"\n{'─'*80}", flush=True)
print("PHASE 2: Micro-trend filter (lookback 30-120s)")
print("-> Filters: WITH trend (fade goes same direction as prior 60s trend)")
print(f"{'─'*80}", flush=True)

# Focus on high-density configs: 3p, 4p with various windows
for pair in ["EURUSD", "EURJPY"]:
    meta = PAIR_META[pair]; t = data[pair]; pip = meta["pip"]
    configs = [(3,10),(3,15),(3,20),(4,10),(4,15),(4,20)] if pair == "EURUSD" else [(5,15),(7,10),(7,15),(10,10)]

    for tp, ws in configs:
        ev = detect_fast(t, pip, tp, ws)
        if len(ev) < 50: continue

        for lb in LOOKBACKS:
            trend = micro_trend(ev, t, lb)
            has_trend = ~np.isnan(trend)

            for hi, hold in enumerate(HOLDS):
                pnl = sim_batch(ev, t, pip, meta["cost"], [hold])
                p_all = pnl[:, 0][~np.isnan(pnl[:, 0])] / pip

                # Filter: WITH trend only
                mask_with = (trend == 1) & ~np.isnan(pnl[:, 0])
                p_with = pnl[:, 0][mask_with] / pip

                # Filter: AGAINST trend only
                mask_against = (trend == -1) & ~np.isnan(pnl[:, 0])
                p_against = pnl[:, 0][mask_against] / pip

                for p, label, mask in [(p_all, "all", slice(None)),
                                        (p_with, f"with_trend_{lb}s", mask_with),
                                        (p_against, f"against_trend_{lb}s", mask_against)]:
                    if len(p) < 20: continue
                    n = len(p); wr = (p > 0).mean() * 100
                    results.append({"pair":pair, "tp":tp, "ws":ws, "hold":hold,
                                    "n":n, "wr":wr, "gross":p.sum(), "avg":p.mean(),
                                    "per_day":n/65, "filter":label})

# Display best filtered results
print(f"\n{'='*80}", flush=True)
print("BEST CONFIGURATIONS (any filter, n≥30)")
print(f"{'='*80}", flush=True)
print(f" {'Pair':>6s} {'Config':>14s} {'Hold':>4s} {'Filter':>20s} | {'n(3mo)':>7s} {'n/day':>5s} | {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s}", flush=True)
print(f"{'─'*80}", flush=True)

best = [r for r in results if r["n"] >= 30]
best.sort(key=lambda x: -x["wr"])

for r in best[:30]:
    cfg = f"{r['tp']}p/{r['ws']}s"
    print(f" {r['pair']:>6s} {cfg:>14s} {r['hold']:>4d}s {r['filter']:>20s} | {r['n']:>7d} {r['per_day']:>5.1f} | {r['wr']:>5.1f}% {r['gross']:>+9.1f}p {r['avg']:>+7.2f}p", flush=True)

# ── PHASE 3: Combined multi-pair + multi-config ──
print(f"\n{'─'*80}", flush=True)
print("PHASE 3: Combined density (EURUSD + EURJPY best configs)")
print(f"{'─'*80}", flush=True)

# Pick best configs from above and combine non-overlapping events
usd_configs = [(3,10,30),(3,20,30),(4,15,30),(4,20,30)]  # (tp, ws, hold)
jpy_configs = [(5,15,30),(7,10,30),(10,10,30)]

all_events_used = set()  # (pair, extreme_idx) to avoid overlap
combined_pnls = []

for pair, configs in [("EURUSD", usd_configs), ("EURJPY", jpy_configs)]:
    meta = PAIR_META[pair]; t = data[pair]; pip = meta["pip"]
    for tp, ws, hold in configs:
        ev = detect_fast(t, pip, tp, ws)
        # Filter to non-overlapping events
        new_events = ev[~ev["extreme_idx"].isin(
            [ei for (p, ei) in all_events_used if p == pair])]
        if len(new_events) == 0: continue

        pnl = sim_batch(new_events, t, pip, meta["cost"], [hold])
        p = pnl[:, 0][~np.isnan(pnl[:, 0])] / pip
        combined_pnls.extend(p.tolist())
        for ei in new_events["extreme_idx"].values:
            all_events_used.add((pair, ei))

if len(combined_pnls) > 0:
    p = np.array(combined_pnls)
    n = len(p); wr = (p > 0).mean() * 100
    max_l = 0; cur = 0
    for v in p:
        if v < 0: cur += 1
        else: cur = 0
        max_l = max(max_l, cur)
    print(f" Combined 4 EURUSD + 3 EURJPY configs: n={n} WR={wr:.1f}% Gross={p.sum():+.1f}p n/day={n/65:.1f} MaxL={max_l}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
