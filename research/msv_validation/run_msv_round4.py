"""MSV Round 4: Continuation vs direction, event decay, polarization, tradable portfolio.
Goal: stress-test the Asian liquidity imbalance mechanism.
"""

import sys, os, time, numpy as np
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
HORIZONS = [5, 10, 15, 30, 45, 60, 90, 120]
ROLLING_WINDOW = 500
TOTAL_DAYS = 120

def load_data():
    end = datetime.now()
    start = end - timedelta(days=TOTAL_DAYS)
    all_data = {}
    for pair in PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def rolling_pct(value, history):
    if len(history) < 10: return 0.5
    return sum(1 for h in history if h < value) / len(history)

def session_info(ts):
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    h = dt.hour
    if h < 7: return "ASIA", "ASIA_OPEN" if h < 2 else "ASIA_MID" if h < 5 else "ASIA_CLOSE"
    if h < 12: return "LONDON", "LONDON_OPEN" if h < 9 else "LONDON_MID"
    if h < 16: return "LONDON_NY", "LONDON_NY"
    if h < 21: return "NY", "NY_OPEN" if h < 18 else "NY_MID"
    return "NY_LATE", "NY_LATE"

def main():
    all_data = load_data()
    N = min(len(v) for v in all_data.values())
    print(f"Loaded {len(all_data)} pairs, {N} bars ({N/288:.1f} days)")

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

        now = float(all_data[PAIRS[0]][idx]["time"])
        snapshot = ms.update(returns, timestamp=now)
        disp = snapshot.network.dispersion
        disp_hist.append(disp)
        disp_pct = rolling_pct(disp, list(disp_hist))

        if len(disp_hist) >= 12:
            disp_vel = disp - list(disp_hist)[-12]
        else:
            disp_vel = 0.0

        # Pre-state: return in last 60 min (12 bars)
        pre60 = 0.0
        if idx >= 12:
            for pair in all_data:
                pre_ret = (float(all_data[pair][idx]["close"]) / float(all_data[pair][idx - 12]["close"]) - 1)
                pre60 += pre_ret
            pre60 /= len(all_data)

        # Polarization: largest |strength| / sum(|strengths|)
        strengths = {c: n.level for c, n in snapshot.currencies.items()}
        s_abs = np.array([abs(v) for v in strengths.values()])
        pol = float(np.max(s_abs) / max(np.sum(s_abs), 1e-12))

        # Cross DER
        cross_der = float(np.mean([abs(v) for v in returns.values()]))

        # WLS currency ranking
        sorted_ccy = sorted(strengths.items(), key=lambda x: x[1])
        strong2 = [c for c, _ in sorted_ccy[-2:]]
        weak2 = [c for c, _ in sorted_ccy[:2]]

        sess, _ = session_info(now)

        # Forward returns for many horizons
        fwd = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx >= N:
                fwd[h] = None
                continue
            vals = []
            for pair in all_data:
                cur = float(all_data[pair][idx]["close"])
                fut = float(all_data[pair][fwd_idx]["close"])
                vals.append((fut / cur - 1) if cur > 0 else 0.0)
            fwd[h] = float(np.mean(vals))

        # Tradable portfolio returns: long strong2 via their pairs, short weak2 via their pairs
        tradable_pnl = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx >= N:
                tradable_pnl[h] = None
                continue
            long_pnl, short_pnl = [], []
            for pair in all_data:
                base, quote = BASE_CURRENCY_MAP[pair]
                cur = float(all_data[pair][idx]["close"])
                fut = float(all_data[pair][fwd_idx]["close"])
                ret = (fut / cur - 1) if cur > 0 else 0.0
                if base in strong2:
                    long_pnl.append(ret)
                if quote in strong2:
                    short_pnl.append(-ret)  # shorting pair = long quote
                if base in weak2:
                    short_pnl.append(-ret)  # short pair = long base which is weak? Wait...
                    # Actually we want to SHORT the weak currencies, so we go SHORT pairs where weak is base
                if quote in weak2:
                    long_pnl.append(ret)   # long pair where weak is quote = shorting weak currency
            # Actually, let me redo this more carefully:
            # Long strong currencies: go LONG pairs where strong is BASE, go SHORT pairs where strong is QUOTE
            # Short weak currencies: go SHORT pairs where weak is BASE, go LONG pairs where weak is QUOTE
            # But this is complex. Let me simplify:
            # Portfolio = sum over all pairs of (strong_base - strong_quote - weak_base + weak_quote) / 4 * ret
            pair_weight = {}
            ip = 0
            for pair in all_data:
                base, quote = BASE_CURRENCY_MAP[pair]
                w = 0.0
                if base in strong2: w += 1.0
                if quote in strong2: w -= 1.0
                if base in weak2: w -= 1.0
                if quote in weak2: w += 1.0
                pair_weight[pair] = w
            # Normalize
            total_w = sum(abs(w) for w in pair_weight.values())
            if total_w > 0:
                pnl = 0.0
                for pair in all_data:
                    cur = float(all_data[pair][idx]["close"])
                    fut = float(all_data[pair][fwd_idx]["close"])
                    ret = (fut / cur - 1) if cur > 0 else 0.0
                    pnl += (pair_weight[pair] / total_w) * ret
                tradable_pnl[h] = pnl
            else:
                tradable_pnl[h] = 0.0

        records.append({
            "idx": idx, "session": sess,
            "disp_pct": disp_pct, "disp_vel": disp_vel,
            "pol": pol, "cross_der": cross_der,
            "pre60": pre60,
            "strong2": strong2, "weak2": weak2,
            "fwd": fwd, "tradable": tradable_pnl,
        })

        if (idx + 1) % 4000 == 0:
            print(f"  {idx+1}/{N}")

    print(f"\nAnalyzing {len(records)} records...")

    # ── EXP 1: Continuation vs New Direction ──
    print(f"\n{'='*70}")
    print("EXP 1: CONTINUATION vs NEW DIRECTION (Asia, Extreme Disp)")
    print('='*70)

    all_disp = [r["disp_pct"] for r in records if r["fwd"][30] is not None]
    p95 = float(np.percentile(all_disp, 95)) if len(all_disp) >= 20 else 0.95

    asia_extreme = [r for r in records if r["session"] == "ASIA"
                    and r["disp_pct"] >= p95 and r["fwd"][30] is not None]

    print(f"\n  Asia + Extreme Disp (n={len(asia_extreme)}), P95={p95:.3f}")
    print(f"\n  Pre-state direction buckets:")
    for pre_label, pre_cond in [
        ("PREVIOUS_UP", lambda r: r["pre60"] > 0.0002),
        ("PREVIOUS_DOWN", lambda r: r["pre60"] < -0.0002),
        ("PREVIOUS_FLAT", lambda r: abs(r["pre60"]) <= 0.0002),
    ]:
        subset = [r for r in asia_extreme if pre_cond(r)]
        if len(subset) < 3:
            continue
        print(f"\n    {pre_label:15s} (n={len(subset)}):")
        for h in [15, 30, 60]:
            vals = [r["fwd"][h] for r in subset if r["fwd"][h] is not None]
            if len(vals) < 3: continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"      {h:3d}m  mean={mu:+.6f}  sharpe={sharpe:+.3f}  "
                  f"pos%={pos:5.1f}%  t={t:+.2f}")

    # ── EXP 2: Event Decay Curve ──
    print(f"\n{'='*70}")
    print("EXP 2: EVENT DECAY CURVE")
    print('='*70)

    for label, records_subset, n_label in [
        ("ASIA_EXTREME", asia_extreme, len(asia_extreme)),
    ]:
        if n_label < 5:
            continue
        print(f"\n  {label} (n={n_label}):")
        print(f"  {'Horizon':>8s} {'MeanRet':>10s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s} {'n':>6s}")
        print(f"  {'-'*48}")
        for h in HORIZONS:
            vals = [r["fwd"][h] for r in records_subset if r["fwd"][h] is not None]
            if len(vals) < 3: continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"  {h:4d}m    {mu:+10.6f}  {sharpe:+8.3f}  {pos:5.1f}%  {t:+8.2f}  {len(vals):5d}")

    # ── EXP 2b: Tradable Portfolio Decay ──
    print(f"\n  Tradable Portfolio (long strong2 / short weak2):")
    print(f"  {'Horizon':>8s} {'MeanRet':>10s} {'Sharpe':>8s} {'Pos%':>6s} {'t':>8s} {'n':>6s}")
    print(f"  {'-'*48}")
    for h in HORIZONS:
        vals = [r["tradable"][h] for r in asia_extreme if r["tradable"][h] is not None]
        if len(vals) < 3: continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
        print(f"  {h:4d}m    {mu:+10.6f}  {sharpe:+8.3f}  {pos:5.1f}%  {t:+8.2f}  {len(vals):5d}")

    # ── EXP 3: Network Polarization ──
    print(f"\n{'='*70}")
    print("EXP 3: NETWORK POLARIZATION")
    print('='*70)

    pol_vals = [r["pol"] for r in asia_extreme]
    if len(pol_vals) >= 10:
        pol_p50 = float(np.median(pol_vals))
        print(f"\n  Median polarization: {pol_p50:.3f}")
        for pol_label, pol_cond in [
            ("LOW_POL (<median)", lambda r: r["pol"] < pol_p50),
            ("HIGH_POL (>=median)", lambda r: r["pol"] >= pol_p50),
        ]:
            subset = [r for r in asia_extreme if pol_cond(r)]
            if len(subset) < 3: continue
            print(f"\n    {pol_label:20s} (n={len(subset)}):")
            for h in [15, 30, 60]:
                vals = [r["fwd"][h] for r in subset if r["fwd"][h] is not None]
                if len(vals) < 3: continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
                pos = sum(1 for v in vals if v > 0) / len(vals) * 100
                t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
                print(f"      {h:3d}m  mean={mu:+.6f}  sharpe={sharpe:+.3f}  "
                      f"pos%={pos:5.1f}%  t={t:+.2f}")

    # ── EXP 4: Velocity × Polarization interaction ──
    print(f"\n{'='*70}")
    print("EXP 4: VELOCITY × POLARIZATION (Asia Extreme)")
    print('='*70)

    for v_label, v_cond in [
        ("V+INC", lambda r: r["disp_vel"] > 0),
        ("V_DEC", lambda r: r["disp_vel"] <= 0),
    ]:
        for p_label, p_cond in [
            ("+LOW_POL", lambda r: r["pol"] < 0.5),
            ("+HIGH_POL", lambda r: r["pol"] >= 0.5),
        ]:
            subset = [r for r in asia_extreme if v_cond(r) and p_cond(r)]
            if len(subset) < 3: continue
            vals = [r["fwd"][30] for r in subset if r["fwd"][30] is not None]
            if len(vals) < 3: continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"    {v_label:8s} {p_label:12s} n={len(vals):4d}  "
                  f"30m mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # ── COMPARISON: ASIA basket vs tradable portfolio ──
    print(f"\n{'='*70}")
    print("COMPARISON: Basket vs Tradable Portfolio (Asia Extreme, 30m)")
    print('='*70)
    basket_vals = [r["fwd"][30] for r in asia_extreme if r["fwd"][30] is not None]
    trade_vals = [r["tradable"][30] for r in asia_extreme if r["tradable"][30] is not None]

    for label, vals in [("BASKET (equal-weight pairs)", basket_vals),
                         ("TRADABLE (long strong2/short weak2)", trade_vals)]:
        if len(vals) < 3: continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
        print(f"  {label:40s}  n={len(vals):4d}  mean={mu:+.6f}  "
              f"sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
