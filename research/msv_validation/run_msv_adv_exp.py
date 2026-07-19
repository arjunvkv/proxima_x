"""MSV Advanced Experiments — Round 2.
Implements ChatGPT's 5 priority experiments:
1. 6-month walk-forward survival test
2. Dispersion velocity/acceleration state machine
3. Currency factor portfolio
4. Session decomposition
5. MSV energy cycle state machine
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

# 12 liquid pairs (ChatGPT suggestion) — using BASE_CURRENCY_MAP keys
PAIRS = [
    "EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "USDJPY", "USDCHF", "USDCAD",
]
CURRENCIES = ["EUR", "GBP", "USD", "JPY", "AUD", "CAD", "CHF", "NZD"]
HORIZONS = [5, 15, 30, 60]
TOTAL_DAYS = 120  # 4 months for faster iteration
TRAIN_DAYS = 10
TEST_DAYS = 5
ROLLING_WINDOW = 500

def load_all(pairs, days=TOTAL_DAYS):
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"Loading {days}d M5 data ({len(pairs)} pairs)...")
    all_data = {}
    for pair in pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    N = min(len(v) for v in all_data.values())
    print(f"  Loaded {len(all_data)} pairs, {N} bars")
    return all_data, N

def session_hour(ts):
    """Return session label from Unix timestamp."""
    dt = datetime.fromtimestamp(float(ts))
    h = dt.hour
    # FX sessions (rough): Asia=0-8, London=7-16, NY=12-21
    if 0 <= h < 7:
        return "ASIA"
    elif 7 <= h < 12:
        return "LONDON"
    elif 12 <= h < 16:
        return "LONDON_NY"
    elif 16 <= h < 21:
        return "NY"
    else:
        return "NY_LATE"

def rolling_pct(value, history):
    if len(history) < 10:
        return 0.5
    return sum(1 for h in history if h < value) / len(history)

def process_series(all_data, N, pair_subset=None):
    """Run MSV over all bars, return feature records."""
    pairs = pair_subset or list(all_data.keys())
    ms = MarketStateVector(history_size=50)

    disp_hist = deque(maxlen=ROLLING_WINDOW)
    records = []

    for idx in range(N):
        returns = {}
        for pair in all_data:
            if idx == 0:
                ret = 0.0
            else:
                prev = float(all_data[pair][idx - 1]["close"])
                curr = float(all_data[pair][idx]["close"])
                ret = (curr / prev - 1) if prev > 0 else 0.0
            ret = np.clip(ret, -0.05, 0.05)
            returns[pair] = ret

        now = float(all_data[pairs[0]][idx]["time"])
        snapshot = ms.update(returns, timestamp=now)

        disp = snapshot.network.dispersion
        agree = snapshot.network.agreement
        shock = abs(snapshot.residual.residual_shock)
        disp_hist.append(disp)
        disp_pct = rolling_pct(disp, list(disp_hist))

        # Dispersion velocity/acceleration
        if len(disp_hist) >= 12:
            disp_vel = disp - list(disp_hist)[-12]
        else:
            disp_vel = 0.0
        if len(disp_hist) >= 24:
            prev_vel = list(disp_hist)[-12] - list(disp_hist)[-24]
            disp_accel = disp_vel - prev_vel
        else:
            disp_accel = 0.0

        # Currency strengths
        strengths = {c: n.level for c, n in snapshot.currencies.items()}

        # Forward returns
        fwd = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx >= N:
                fwd[h] = None
                continue
            fwd[h] = {}
            for pair in pairs:
                cur = float(all_data[pair][idx]["close"])
                fut = float(all_data[pair][fwd_idx]["close"])
                fwd[h][pair] = (fut / cur - 1) if cur > 0 else 0.0

        records.append({
            "idx": idx, "ts": now,
            "session": session_hour(now),
            "disp": disp, "disp_pct": disp_pct,
            "disp_vel": disp_vel, "disp_accel": disp_accel,
            "agree": agree, "shock": shock,
            "strengths": strengths,
            "fwd": fwd,
        })

        if (idx + 1) % 2000 == 0:
            print(f"    Processed {idx+1}/{N}")

    return records

def walk_forward(records, train_bars, test_bars):
    """Run walk-forward windows."""
    print(f"\n=== WALK-FORWARD (train={train_bars} test={test_bars}) ===")
    windows = []
    start = train_bars + 1  # need +1 for enough history
    while start + test_bars <= len(records):
        windows.append((start - train_bars, start, start + test_bars))
        start += test_bars

    print(f"  Windows: {len(windows)}")
    results = []
    for w_idx, (t0, t1, t2) in enumerate(windows):
        test_recs = records[t1:t2]
        # Train threshold: 95th percentile of disp_pct in train window
        train_disp = [r["disp_pct"] for r in records[t0:t1]]
        if len(train_disp) < 10:
            continue
        p95 = np.percentile(train_disp, 95)
        p80 = np.percentile(train_disp, 80)
        p20 = np.percentile(train_disp, 20)

        for h in HORIZONS:
            for label, cond, thr in [
                ("EXTREME_DISP", lambda r: r["disp_pct"] >= p95, p95),
                ("HIGH_DISP", lambda r: p80 <= r["disp_pct"] < p95, p80),
                ("MID_DISP", lambda r: p20 <= r["disp_pct"] < p80, p20),
                ("LOW_DISP", lambda r: r["disp_pct"] < p20, p20),
            ]:
                vals = []
                for r in test_recs:
                    if r["fwd"][h] is None:
                        continue
                    if cond(r):
                        basket = np.mean([r["fwd"][h][p] for p in PAIRS])
                        vals.append(basket)
                if len(vals) < 3:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
                pos = sum(1 for v in vals if v > 0) / len(vals) * 100
                t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
                results.append({
                    "window": w_idx, "regime": label, "horizon": h,
                    "n": len(vals), "mean_ret": mu, "sharpe": sharpe,
                    "pos_pct": pos, "t_stat": t,
                })
    return results

def session_analysis(records):
    """Break down EXTREME_DISP signals by session."""
    print(f"\n=== SESSION DECOMPOSITION ===")
    all_disp = [r["disp_pct"] for r in records if r["fwd"][5] is not None]
    if len(all_disp) < 10:
        return
    p95 = np.percentile(all_disp, 95)
    print(f"  95th percentile threshold: {p95:.4f}")

    sessions = ["ASIA", "LONDON", "LONDON_NY", "NY", "NY_LATE"]
    for sess in sessions:
        vals = [r for r in records if r["session"] == sess and r["fwd"][5] is not None]
        extreme = [r for r in vals if r["disp_pct"] >= p95]
        extreme_pct = len(extreme) / max(len(vals), 1) * 100
        print(f"\n  {sess:12s}: {len(vals):5d} total, {len(extreme):5d} extreme ({extreme_pct:.1f}%)")
        if len(extreme) < 3:
            continue
        for h in HORIZONS:
            fwd_vals = [np.mean([r["fwd"][h][p] for p in PAIRS]) for r in extreme if r["fwd"][h] is not None]
            if len(fwd_vals) < 3:
                continue
            mu = float(np.mean(fwd_vals))
            sigma = float(np.std(fwd_vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in fwd_vals if v > 0) / len(fwd_vals) * 100
            t = mu / (sigma / np.sqrt(len(fwd_vals))) if sigma > 0 else 0.0
            print(f"    {h:3d}m  n={len(fwd_vals):4d}  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

def currency_factor_portfolio(records):
    """Build long-strong-currencies / short-weak-currencies portfolio."""
    print(f"\n=== CURRENCY FACTOR PORTFOLIO ===")
    all_disp = [r["disp_pct"] for r in records if r["fwd"][5] is not None]
    if len(all_disp) < 10:
        return
    p95 = np.percentile(all_disp, 95)

    # Build currency return matrix
    # For each record, compute how each currency performed
    ccy_returns = defaultdict(list)
    for r in records:
        if r["fwd"][30] is None:
            continue
        strengths = r["strengths"]
        sorted_ccy = sorted(strengths.items(), key=lambda x: x[1])
        if len(sorted_ccy) < 4:
            continue
        weak2 = sorted_ccy[:2]
        strong2 = sorted_ccy[-2:]

        # Long strong2, short weak2 via their pairs
        long_pnl = []
        short_pnl = []
        for p in PAIRS:
            base, quote = BASE_CURRENCY_MAP[p]
            fwd_ret = r["fwd"][30][p]
            # Long strong currencies: go long pairs where strong is base
            for ccy, _ in strong2:
                if base == ccy:
                    long_pnl.append(fwd_ret)
                elif quote == ccy:
                    short_pnl.append(fwd_ret)  # shorting this pair = longing the quote currency
        factor_pnl = (np.mean(long_pnl) - np.mean(short_pnl)) if long_pnl and short_pnl else 0.0
        ccy_returns["ALL"].append(factor_pnl)
        if r["disp_pct"] >= p95:
            ccy_returns["EXTREME"].append(factor_pnl)
        elif r["disp_pct"] < np.percentile(all_disp, 20):
            ccy_returns["LOW"].append(factor_pnl)

    for label, vals in ccy_returns.items():
        if len(vals) < 3:
            continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
        print(f"  {label:10s}  n={len(vals):5d}  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

def velocity_state_machine(records):
    """Test dispersion velocity/acceleration states."""
    print(f"\n=== DISPERSION VELOCITY STATE MACHINE ===")
    all_disp = [r["disp_pct"] for r in records if r["fwd"][5] is not None]
    if len(all_disp) < 10:
        return
    p95 = np.percentile(all_disp, 95)

    states = [
        ("HIGH_DISP+V_INC", lambda r: r["disp_pct"] >= p95 and r["disp_vel"] > 0),
        ("HIGH_DISP+V_DEC", lambda r: r["disp_pct"] >= p95 and r["disp_vel"] <= 0),
        ("LOW_DISP+V_INC", lambda r: r["disp_pct"] < 0.2 and r["disp_vel"] > 0),
        ("MID_DISP+V_INC", lambda r: 0.2 <= r["disp_pct"] < p95 and r["disp_vel"] > 0),
        ("MID_DISP+V_DEC", lambda r: 0.2 <= r["disp_pct"] < p95 and r["disp_vel"] <= 0),
    ]

    for label, cond in states:
        n = sum(1 for r in records if cond(r) and r["fwd"][5] is not None)
        if n < 5:
            continue
        print(f"\n  {label:20s} (n={n}):")
        for h in HORIZONS:
            vals = [np.mean([r["fwd"][h][p] for p in PAIRS])
                    for r in records if cond(r) and r["fwd"][h] is not None]
            if len(vals) < 5:
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"    {h:3d}m  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

def main():
    t0 = time.time()

    all_data, N = load_all(PAIRS, TOTAL_DAYS)
    print(f"Processing {N} bars...")
    records = process_series(all_data, N)

    # ── FULL-SAMPLE BENCHMARK (EXP 1 baseline) ──
    print(f"\n{'='*70}")
    print("FULL-SAMPLE BENCHMARK (120 days)")
    print("="*70)
    all_disp = [r["disp_pct"] for r in records if r["fwd"][5] is not None]
    if len(all_disp) >= 10:
        p95 = float(np.percentile(all_disp, 95))
        p80 = float(np.percentile(all_disp, 80))
        p20 = float(np.percentile(all_disp, 20))
        print(f"  Dispersion thresholds: P20={p20:.4f} P80={p80:.4f} P95={p95:.4f}")

        for h in HORIZONS:
            print(f"\n  Horizon={h}m:")
            for label, cond in [
                (f"LOW_DISP  (<P20={p20:.4f})", lambda r: r["disp_pct"] < p20),
                (f"MID_DISP  (P20-P95)", lambda r: p20 <= r["disp_pct"] < p95),
                (f"HIGH_DISP (>P95={p95:.4f})", lambda r: r["disp_pct"] >= p95),
            ]:
                vals = [np.mean([r["fwd"][h][p] for p in PAIRS])
                        for r in records if r["fwd"][h] is not None and cond(r)]
                if len(vals) < 5:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
                pos = sum(1 for v in vals if v > 0) / len(vals) * 100
                t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
                print(f"    {label:35s}  n={len(vals):5d}  mean={mu:+.6f}  "
                      f"sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # ── EXP 1: WALK-FORWARD ──
    wf_results = walk_forward(records, train_bars=TRAIN_DAYS*288, test_bars=TEST_DAYS*288)
    if wf_results:
        print(f"\n  Walk-forward summary ({len(wf_results)} regime-window-horizon combos):")
        for regime in ["EXTREME_DISP", "HIGH_DISP", "MID_DISP", "LOW_DISP"]:
            rows = [r for r in wf_results if r["regime"] == regime]
            if not rows:
                continue
            print(f"\n  {regime}:")
            for h in HORIZONS:
                hrows = [r for r in rows if r["horizon"] == h]
                if not hrows:
                    continue
                sharpes = [r["sharpe"] for r in hrows]
                pos_wins = sum(1 for s in sharpes if s > 0)
                mean_sharpe = float(np.mean(sharpes))
                print(f"    {h:3d}m  windows={len(hrows):2d}  "
                      f"profitable={pos_wins}/{len(hrows)}  "
                      f"mean_sharpe={mean_sharpe:+.2f}  "
                      f"best={max(sharpes):+.2f}  worst={min(sharpes):+.2f}")

    # ── EXP 2: VELOCITY STATE MACHINE ──
    velocity_state_machine(records)

    # ── EXP 3: CURRENCY FACTOR ──
    currency_factor_portfolio(records)

    # ── EXP 4: SESSION DECOMPOSITION ──
    session_analysis(records)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    mt5.shutdown()

if __name__ == "__main__":
    main()
