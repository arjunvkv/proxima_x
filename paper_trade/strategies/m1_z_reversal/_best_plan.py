"""Fast vectorized sweep: highest-WR FundedNext config."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

PAIR_META = {
    "EURUSD": {"pip": 0.0001, "cost": 0.00003},
    "EURJPY": {"pip": 0.01, "cost": 0.006},
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

def detect_events(ticks, pip_size, thresh_pips, thresh_sec=10):
    mid = (ticks["B"] + ticks["A"]).values / 2.0
    ts = ticks["ts_s"].values; n = len(ticks)
    raw = thresh_pips * pip_size; evs = []; i = 0
    while i < n:
        end = min(int(np.searchsorted(ts, ts[i] + thresh_sec, side="right")), n)
        w = mid[i:end]
        if len(w) < 2: i += 1; continue
        hp = np.max(w) - w[0]; lp = w[0] - np.min(w)
        if max(hp, lp) >= raw:
            d2 = 1 if hp >= lp else -1
            ei = i + (np.argmax(w) if d2 == 1 else np.argmin(w))
            evs.append({"time": ts[i], "extreme_idx": ei, "direction": d2, "price_extreme": mid[ei]})
            i = ei
        else: i += 1
    return pd.DataFrame(evs)

def sim_config(events, ticks, pip, cost_per_trade, hold_s, stop_pips=0, retrace_pips=0, entry_delay=1):
    bids = ticks["B"].values; asks = ticks["A"].values; ts = ticks["ts_s"].values; n = len(ts)
    pnls = np.full(len(events), np.nan)
    for ev_i, (_, ev) in enumerate(events.iterrows()):
        ed = -ev["direction"]
        ei = int(ev["extreme_idx"]) + entry_delay
        if ei >= n - 2: continue
        entry_price = asks[ei] if ed == 1 else bids[ei]
        entry_time = ts[ei]
        if retrace_pips > 0:
            raw_r = retrace_pips * pip
            rj = None; max_scan = min(ei + 2000, n)
            for j in range(ei, max_scan):
                px = asks[j] if ed == 1 else bids[j]
                if (ed == 1 and px <= ev["price_extreme"] - raw_r) or \
                   (ed == -1 and px >= ev["price_extreme"] + raw_r):
                    rj = j; break
            if rj is None: continue
            entry_price = asks[rj] if ed == 1 else bids[rj]; entry_time = ts[rj]; ei = rj
        hold_end_t = entry_time + hold_s
        win_end = min(int(np.searchsorted(ts, hold_end_t, side="right")), n)
        if win_end <= ei + 1: continue
        stop_idx = None
        if stop_pips > 0:
            raw_s = stop_pips * pip
            window = bids[ei:win_end] if ed == 1 else asks[ei:win_end]
            hit = np.where(window <= entry_price - raw_s)[0] if ed == 1 else np.where(window >= entry_price + raw_s)[0]
            if len(hit) > 0: stop_idx = ei + hit[0]
        exit_idx = stop_idx if (stop_idx is not None and ts[stop_idx] <= hold_end_t) else int(np.searchsorted(ts, hold_end_t, side="right"))
        if exit_idx >= n: continue
        exit_px = bids[exit_idx] if ed == 1 else asks[exit_idx]
        pnls[ev_i] = (exit_px - entry_price) * ed - cost_per_trade
    return pnls

# ══════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════
PAIRS = ["EURUSD", "EURJPY"]
IMPULSES = [5, 7, 10]
HOLDS = [60, 120, 180]
STOPS = [0, 2, 3, 5, 8]
RETRACES = [0.0, 0.1, 0.2]
# Limit EURJPY events to keep runtime manageable
MAX_EVENTS = 600

print("=" * 80)
print("BEST PLAN: FundedNext-compatible sweep")
print("=" * 80)

all_rows = []
pair_data = {}  # cache (pair, month_range) -> (t, ev_dict)

for pair in PAIRS:
    meta = PAIR_META[pair]; pip = meta["pip"]
    print(f"\n{pair} — loading Oct-Dec...")
    t_full = load(pair, [(2025,10),(2025,11),(2025,12)])

    for imp in IMPULSES:
        ev = detect_events(t_full, pip, imp, 10)
        if len(ev) < 30: continue
        print(f"  impulse={imp}p — {len(ev)} events", end="", flush=True)

        for hold in HOLDS:
            for stop in STOPS:
                for ret in RETRACES:
                    pnls = sim_config(ev, t_full, pip, meta["cost"], hold, stop, ret)
                    valid = ~np.isnan(pnls); n = valid.sum()
                    if n < 15: continue
                    p = pnls[valid]; wins = (p > 0).sum()
                    all_rows.append({
                        "pair": pair, "imp": imp, "hold": hold,
                        "stop": stop, "retrace": ret,
                        "n": n, "wr": wins / n * 100, "gross": p.sum() / pip, "avg": p.mean() / pip,
                    })
        print(f" done")

    # Cache for monthly breakdown
    pair_data[(pair, "full")] = (t_full, {imp: detect_events(t_full, pip, imp, 10) for imp in IMPULSES})

df = pd.DataFrame(all_rows)

# ── Rankings ──
for pair in PAIRS:
    sub = df[(df["pair"]==pair) & (df["n"]>=30)].sort_values("wr", ascending=False)
    print(f"\n{'='*70}")
    print(f"TOP 20 — {pair} (n≥30, by WR)")
    print(f"{'='*70}")
    print(f" {'Rank':>4s} {'Imp':>3s} {'Hold':>4s} {'Stop':>4s} {'Retr':>4s} | {'n':>4s} {'WR':>5s} {'Gross(p)':>9s} {'Avg(p)':>7s}")
    print(f" {'-'*48}")
    for i, (_, r) in enumerate(sub.head(20).iterrows()):
        print(f" {i+1:>4d} {r['imp']:>3.0f} {r['hold']:>4.0f} {r['stop']:>4.0f} {r['retrace']:>4.1f} | {r['n']:>4.0f} {r['wr']:>5.1f}% {r['gross']:>+9.1f}p {r['avg']:>+7.2f}p")

print(f"\n{'='*80}")
print("TOP 10 CROSS-PAIR (n≥30, by WR)")
print(f"{'='*80}")
for i, (_, r) in enumerate(df[df["n"]>=30].sort_values("wr", ascending=False).head(10).iterrows()):
    print(f" {i+1:>2d}. {r['pair']:>6s} imp={r['imp']:>2.0f}p hold={r['hold']:>3.0f}s stop={r['stop']:>2.0f}p ret={r['retrace']:.1f}: n={r['n']:>4.0f} WR={r['wr']:>5.1f}% {r['gross']:>+8.1f}p")

# ══════════════════════════════════════════════════════════════════════
# MONTHLY + WALK-FORWARD (using cached data)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("MONTHLY + WALK-FORWARD — Best FundedNext configs (hold≥120s, stop≤2)")
print(f"{'='*80}")

candidates = [
    ("EURUSD", 7, 120, 0, 0.0),
    ("EURUSD", 5, 120, 0, 0.0),
    ("EURUSD", 5, 120, 2, 0.0),
    ("EURJPY", 10, 120, 0, 0.0),
    ("EURJPY", 10, 120, 0, 0.0),
]

for pair in ["EURUSD", "EURJPY"]:
    meta = PAIR_META[pair]; pip = meta["pip"]
    # Load all months once
    t_oct = load(pair, [(2025,10)])
    t_nov = load(pair, [(2025,11)])
    t_dec = load(pair, [(2025,12)])
    t_train = pd.concat([t_oct, t_nov], ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t_train["ts_s"] = t_train["Ts"].astype(np.int64) // 10**9

    for imp, hold, stop, ret in [
        (7, 120, 0, 0.0), (5, 120, 0, 0.0), (5, 120, 2, 0.0)
    ] if pair == "EURUSD" else [(10, 120, 0, 0.0), (10, 180, 0, 0.0)]:
        ev_oct = detect_events(t_oct, pip, imp, 10)
        ev_nov = detect_events(t_nov, pip, imp, 10)
        ev_dec = detect_events(t_dec, pip, imp, 10)
        ev_train = detect_events(t_train, pip, imp, 10)

        print(f"\n  {pair} {imp}p/10s hold={hold}s stop={stop}p retrace={ret}:")

        for mlab, ev_m, t_m in [("Oct",ev_oct,t_oct),("Nov",ev_nov,t_nov),("Dec",ev_dec,t_dec)]:
            pnls = sim_config(ev_m, t_m, pip, meta["cost"], hold, stop, ret)
            p = pnls[~np.isnan(pnls)] / pip
            if len(p) == 0: continue
            n = len(p); w = (p > 0).sum()
            print(f"      {mlab}: n={n:>3d} WR={w/n*100:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p")

        for mlab, ev_m, t_m in [("Train",ev_train,t_train),("Test",ev_dec,t_dec)]:
            pnls = sim_config(ev_m, t_m, pip, meta["cost"], hold, stop, ret)
            p = pnls[~np.isnan(pnls)] / pip
            if len(p) == 0: continue
            n = len(p); w = (p > 0).sum()
            print(f"      {mlab}: n={n:>3d} WR={w/n*100:.1f}% Gross={p.sum():+7.1f}p Avg={p.mean():+7.2f}p")

        # Max consecutive losses
        all_p = []
        for ev_m, t_m in [(ev_oct,t_oct),(ev_nov,t_nov),(ev_dec,t_dec)]:
            pnls = sim_config(ev_m, t_m, pip, meta["cost"], hold, stop, ret)
            all_p.append(pnls[~np.isnan(pnls)] / pip)
        if all_p:
            all_p_c = np.concatenate([x for x in all_p if len(x) > 0])
            if len(all_p_c) > 0:
                cur = 0; cur_dd = 0; max_loss = 0; max_dd = 0
                for v in all_p_c:
                    if v < 0: cur += 1; cur_dd += abs(v)
                    else: cur = 0; cur_dd = 0
                    max_loss = max(max_loss, cur); max_dd = max(max_dd, cur_dd)
                print(f"      Max cons loss: {max_loss} ({max_dd:.1f}p)")
