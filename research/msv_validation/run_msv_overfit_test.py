"""MSV Anti-Overfitting — optimized.
Processes data ONCE, then applies all parameter filters against cached records.
"""

import sys, os, time, hashlib, random, json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

PAIRS = list(BASE_CURRENCY_MAP.keys())[:16]
ROLLING = 500
TOTAL_DAYS = 120

MSV_V1 = {
    "session": "ASIA",
    "dispersion_percentile": 0.95,
    "previous_return_threshold": -0.0002,
    "velocity_threshold": 0.0,
    "rolling_window": 500,
    "exit_minutes": 30,
}
MSV_V1_HASH = hashlib.md5(str(MSV_V1).encode()).hexdigest()[:8]
print(f"MSV v1 hash: {MSV_V1_HASH}")

def load_data():
    end = datetime.now()
    start = end - timedelta(days=TOTAL_DAYS)
    all_data = {}
    for pair in PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def build_records(all_data):
    """Process once, store all metadata for every bar."""
    N = min(len(v) for v in all_data.values())
    print(f"Processing {N} bars once...")

    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=1500)  # max rolling window we'll test

    records = []
    t0 = time.time()
    for idx in range(N):
        rets = {}
        for p in all_data:
            if idx == 0:
                rets[p] = 0.0
            else:
                c = float(all_data[p][idx]["close"])
                pv = float(all_data[p][idx - 1]["close"])
                rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        now = float(all_data[list(all_data.keys())[0]][idx]["time"])
        snap = ms.update(rets, timestamp=now)
        dh.append(snap.network.dispersion)

        # Precompute percentile for different window sizes
        # We'll compute per-window on the fly during filtering

        pre60 = 0.0
        if idx >= 12:
            for p in all_data:
                cur = float(all_data[p][idx]["close"])
                p60 = float(all_data[p][idx - 12]["close"])
                pre60 += (cur / p60 - 1) if p60 > 0 else 0.0
            pre60 /= len(all_data)

        if len(dh) >= 12:
            dv = snap.network.dispersion - list(dh)[-12]
        else:
            dv = 0.0

        # Forward 30m return
        fwd_30 = None
        if idx + 30 < N:
            vals = []
            for p in all_data:
                cur = float(all_data[p][idx]["close"])
                fut = float(all_data[p][idx + 30]["close"])
                vals.append((fut / cur - 1) if cur > 0 else 0.0)
            fwd_30 = float(np.mean(vals))

        dt = datetime.fromtimestamp(now, tz=timezone.utc)

        records.append({
            "idx": idx, "ts": now, "hour": dt.hour, "wd": dt.weekday(),
            "disp": snap.network.dispersion, "pre60": pre60,
            "dv": dv, "fwd_30": fwd_30,
            "dh_snapshot": list(dh),  # copy of dispersion history for percentile calc
        })

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * N - elapsed
            print(f"  {idx+1}/{N} (ETA: {eta:.0f}s)")

    return records

def compute_pct(disp, history, window):
    h = history[-min(window, len(history)):]
    if len(h) < 10:
        return 0.5
    return sum(1 for x in h if x < disp) / len(h)

def get_signals(records, pct_thresh=0.95, decl_thresh=-0.0002, vel_thresh=0.0, window=500):
    """Filter records by all MSV v1 conditions."""
    vals = []
    for r in records:
        if r["fwd_30"] is None:
            continue
        if r["hour"] >= 7:
            continue
        dp = compute_pct(r["disp"], r["dh_snapshot"], window)
        if dp < pct_thresh:
            continue
        if r["pre60"] > decl_thresh:
            continue
        if r["dv"] <= vel_thresh:
            continue
        vals.append(r["fwd_30"])
    return vals

def sharpe_stats(vals):
    if len(vals) < 3:
        return 0.0, 0.0, 0.0, 0
    mu = float(np.mean(vals))
    s = float(np.std(vals))
    sh = (mu / s) * np.sqrt(12 * 24) if s > 0 else 0.0
    t = mu / (s / np.sqrt(len(vals))) if s > 0 else 0.0
    pos = sum(1 for v in vals if v > 0) / len(vals) * 100
    return sh, t, pos, len(vals)

def main():
    all_data = load_data()
    records = build_records(all_data)
    print(f"Built {len(records)} records — {min(len(v) for v in all_data.values())} bars")

    # ──────────────────────────────────────────────────────────
    # TEST 1: OUT-OF-SAMPLE FREEZE (75/25 split)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 1: OUT-OF-SAMPLE FREEZE (75/25 SPLIT)")
    print("=" * 70)

    split = int(len(records) * 0.75)
    train_recs = records[:split]
    test_recs = records[split:]

    print(f"  Train: {len(train_recs)} bars")
    print(f"  Test:  {len(test_recs)} bars")

    for label, recs in [("TRAIN", train_recs), ("TEST (frozen)", test_recs)]:
        vals = get_signals(recs)
        sh, t, pos, n = sharpe_stats(vals)
        print(f"  {label:20s} n={n:4d}  sharpe={sh:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # ──────────────────────────────────────────────────────────
    # TEST 2: PARAMETER SENSITIVITY
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 2: PARAMETER SENSITIVITY")
    print("=" * 70)

    print("\n  A. Dispersion percentile:")
    print(f"  {'Pct':>6s} {'n':>5s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
    print(f"  {'-'*35}")
    for pct in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]:
        vals = get_signals(records, pct_thresh=pct)
        sh, t, pos, n = sharpe_stats(vals)
        if n < 3: continue
        print(f"  {pct:5.2f}   {n:5d}  {sh:+8.3f}  {pos:5.1f}%  {t:+8.2f}")

    print("\n  B. Previous decline threshold:")
    print(f"  {'Decline':>10s} {'n':>5s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
    print(f"  {'-'*40}")
    for decl in [-0.01, -0.002, -0.001, -0.0005, -0.0002, -0.0001]:
        vals = get_signals(records, decl_thresh=decl)
        sh, t, pos, n = sharpe_stats(vals)
        if n < 3: continue
        print(f"  {decl:+8.1e}  {n:5d}  {sh:+8.3f}  {pos:5.1f}%  {t:+8.2f}")

    print("\n  C. Rolling window size:")
    print(f"  {'Window':>7s} {'n':>5s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
    print(f"  {'-'*37}")
    for w in [200, 300, 500, 750, 1000, 1500]:
        vals = get_signals(records, window=w)
        sh, t, pos, n = sharpe_stats(vals)
        if n < 3: continue
        print(f"  {w:5d}   {n:5d}  {sh:+8.3f}  {pos:5.1f}%  {t:+8.2f}")

    print("\n  D. Velocity threshold:")
    print(f"  {'Vel':>10s} {'n':>5s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
    print(f"  {'-'*40}")
    for vel_label, vel_val in [("ANY (no filter)", -999), ("V > -1e-7", -1e-7),
                                ("V > 0", 0), ("V > 1e-7", 1e-7), ("V > 1e-6", 1e-6)]:
        vals = get_signals(records, vel_thresh=vel_val)
        sh, t, pos, n = sharpe_stats(vals)
        if n < 3: continue
        print(f"  {vel_label:12s}  {n:5d}  {sh:+8.3f}  {pos:5.1f}%  {t:+8.2f}")

    # ──────────────────────────────────────────────────────────
    # TEST 3: INTERACTION SURFACE (27 models)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 3: PARAMETER INTERACTION SURFACE")
    print("=" * 70)
    print(f"\n  {'Pct':>5s} {'Decl':>8s} {'n':>5s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s}")
    print(f"  {'-'*45}")

    for pct in [0.90, 0.95, 0.97]:
        for decl in [-0.0002, -0.0005, -0.001]:
            vals = get_signals(records, pct_thresh=pct, decl_thresh=decl)
            sh, t, pos, n = sharpe_stats(vals)
            if n < 3: continue
            print(f"  {pct:4.2f}  {decl:+8.1e}  {n:5d}  {sh:+8.3f}  {pos:5.1f}%  {t:+8.2f}")

    # ──────────────────────────────────────────────────────────
    # TEST 4: MONTE CARLO PERMUTATION
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 4: MONTE CARLO PERMUTATION")
    print("=" * 70)

    actual_vals = get_signals(records)
    actual_sh, _, _, n_actual = sharpe_stats(actual_vals)
    print(f"  Actual: n={n_actual} Sharpe={actual_sh:+.3f}")

    # Get all ASIA bars for sampling pool
    asia_vals = [r["fwd_30"] for r in records if r["hour"] < 7 and r["fwd_30"] is not None]
    print(f"  ASIA pool: {len(asia_vals)} bars")

    n_perm = 10000
    perm_sharpes = []
    for p in range(n_perm):
        sampled = random.choices(asia_vals, k=n_actual)
        ps, _, _, _ = sharpe_stats(sampled)
        perm_sharpes.append(ps)
        if (p + 1) % 2000 == 0:
            print(f"    Permutation {p+1}/{n_perm}")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.sum(perm_sharpes >= actual_sh) / n_perm
    print(f"\n  p-value: {p_value:.6f}")
    print(f"  Mean random Sharpe: {np.mean(perm_sharpes):+.3f}")
    print(f"  Max random Sharpe:  {np.max(perm_sharpes):+.3f}")
    print(f"  Rank: {(actual_sh > perm_sharpes).sum()}/{n_perm}")

    # ──────────────────────────────────────────────────────────
    # TEST 5: RANDOM PARAMETER SEARCH
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 5: RANDOM PARAMETER SEARCH (1000 configs)")
    print("=" * 70)

    random.seed(42)
    rand_sharpes_all = []
    rand_sharpes_50 = []  # >=50 events
    for trial in range(1000):
        pct = random.uniform(0.80, 0.99)
        decl = -random.uniform(0.0001, 0.01)
        vals = get_signals(records, pct_thresh=pct, decl_thresh=decl)
        sh, _, _, n = sharpe_stats(vals)
        if n >= 3:
            rand_sharpes_all.append({"sharpe": sh, "n": n, "pct": pct, "decl": decl})
        if n >= 50:
            rand_sharpes_50.append({"sharpe": sh, "n": n, "pct": pct, "decl": decl})
        if (trial + 1) % 200 == 0:
            print(f"    Trial {trial+1}/1000")

    for label, rs, min_n in [("all (n>=3)", rand_sharpes_all, 3),
                              ("robust (n>=50)", rand_sharpes_50, 50)]:
        if not rs:
            print(f"\n  Random configs ({label}): none found")
            continue
        rsh = np.array([r["sharpe"] for r in rs])
        print(f"\n  Random configs ({label}): {len(rs)}")
        print(f"    Median Sharpe: {np.median(rsh):+.3f}")
        print(f"    90th pct:      {np.percentile(rsh, 90):+.3f}")
        print(f"    99th pct:      {np.percentile(rsh, 99):+.3f}")
        print(f"    Max:           {np.max(rsh):+.3f}")
        print(f"    MSV v1 rank:   {(actual_sh > rsh).sum()}/{len(rs)}")

    rand_sharpes_50_arr = np.array([r["sharpe"] for r in rand_sharpes_50]) if rand_sharpes_50 else np.array([])

    # ──────────────────────────────────────────────────────────
    # VERDICT
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    r50_rank = f"{(actual_sh > rand_sharpes_50_arr).sum()}/{len(rand_sharpes_50_arr)}" if len(rand_sharpes_50_arr) > 0 else "N/A"
    print(f"""
MSV v1: ASIA + P>0.95 + pre60<-0.0002 + V>0
Hash: {MSV_V1_HASH}

OOS freeze:     see TEST 1 above
Sensitivity:    see TEST 2 above
Interaction:    see TEST 3 above
Permutation:    p={p_value:.6f}
Random robust:  MSV v1 rank = {r50_rank} of configs with n>=50
""")

    mt5.shutdown()

if __name__ == "__main__":
    main()
