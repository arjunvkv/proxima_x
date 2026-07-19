"""MSV Research Validation Agent — Experiment Loop.
Implements ChatGPT's suggested experiments to test whether
MarketStateVector contains predictive information.
"""

import sys, os, time, json, math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
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

ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())
HORIZONS = [5, 15, 30, 60]
DAYS = 14
ROLLING_WINDOW = 500

def load_all_data(pairs, days=DAYS):
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"Loading M5 data: {start.date()} to {end.date()} ({days} days)")
    all_data = {}
    for pair in pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    print(f"Loaded {len(all_data)} pairs, common bars: {min(len(v) for v in all_data.values())}")
    return all_data

def rolling_percentile(value, history):
    if len(history) < 10:
        return 0.5
    count = sum(1 for h in history if h < value)
    return count / len(history)

def main():
    pairs = ALL_PAIRS
    all_data = load_all_data(pairs)
    N = min(len(v) for v in all_data.values())

    ms = MarketStateVector(history_size=50)
    records = []

    disp_history = deque(maxlen=ROLLING_WINDOW)
    agree_history = deque(maxlen=ROLLING_WINDOW)
    shock_history = deque(maxlen=ROLLING_WINDOW)
    energy_history = deque(maxlen=ROLLING_WINDOW)

    for idx in range(N):
        returns = {}
        for pair in all_data:
            bar = all_data[pair][idx]
            if idx == 0:
                ret = 0.0
            else:
                prev = float(all_data[pair][idx - 1]["close"])
                curr = float(all_data[pair][idx]["close"])
                ret = (curr / prev - 1) if prev > 0 else 0.0
            ret = np.clip(ret, -0.05, 0.05)
            returns[pair] = ret

        now = float(bar["time"])
        snapshot = ms.update(returns, timestamp=now)

        reg = ms.regime(snapshot)
        rs = ms.risk_score(snapshot)
        disp = snapshot.network.dispersion
        agree = snapshot.network.agreement
        shock = abs(snapshot.residual.residual_shock)
        energy = snapshot.residual.residual_energy

        # Track rolling histories
        disp_history.append(disp)
        agree_history.append(agree)
        shock_history.append(shock)
        energy_history.append(energy)

        # Percentile-based features
        disp_pct = rolling_percentile(disp, list(disp_history))
        agree_pct = rolling_percentile(agree, list(agree_history))
        shock_pct = rolling_percentile(shock, list(shock_history))
        energy_pct = rolling_percentile(energy, list(energy_history))

        # Dispersion velocity
        if len(disp_history) >= 12:
            disp_vel = disp - list(disp_history)[-12]
        else:
            disp_vel = 0.0
        if len(disp_history) >= 24:
            prev_vel = list(disp_history)[-12] - list(disp_history)[-24]
            disp_accel = disp_vel - prev_vel
        else:
            disp_accel = 0.0

        # --- Forward returns: pair-specific AND basket ---
        fwd_ret_basket = {}
        fwd_ret_pair = {}
        fwd_vol_basket = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx >= N:
                fwd_ret_basket[h] = None
                fwd_ret_pair[h] = {}
                fwd_vol_basket[h] = None
                continue

            basket_pnl = []
            pair_pnl = {}
            for pair in all_data:
                fwd_close = float(all_data[pair][fwd_idx]["close"])
                cur_close = float(all_data[pair][idx]["close"])
                r = (fwd_close / cur_close - 1) if cur_close > 0 else 0.0
                pair_pnl[pair] = r
                basket_pnl.append(r)

            fwd_ret_basket[h] = float(np.mean(basket_pnl))
            fwd_ret_pair[h] = pair_pnl
            fwd_vol_basket[h] = float(np.std(basket_pnl)) if len(basket_pnl) > 1 else 1e-10

        records.append({
            "idx": idx, "ts": now,
            "regime": reg, "risk": rs,
            "disp": disp, "disp_pct": disp_pct,
            "disp_vel": disp_vel, "disp_accel": disp_accel,
            "agree": agree, "agree_pct": agree_pct,
            "shock": shock, "shock_pct": shock_pct,
            "energy": energy, "energy_pct": energy_pct,
            "fwd_ret": fwd_ret_basket,
            "fwd_ret_pair": fwd_ret_pair,
            "fwd_vol": fwd_vol_basket,
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{N}")

    print(f"\nTotal records: {len(records)}")

    # =========================================================
    # EXPERIMENT 1: Percentile-based regime normalization
    # =========================================================
    print("\n" + "="*70)
    print("EXP 1: PERCENTILE-BASED REGIME CLASSIFICATION")
    print("="*70)

    for percentile_group, label in [
        (lambda r: r["disp_pct"] < 0.2, "LOW_DISP (<20%)"),
        (lambda r: 0.2 <= r["disp_pct"] < 0.8, "MID_DISP (20-80%)"),
        (lambda r: r["disp_pct"] >= 0.8, "HIGH_DISP (>80%)"),
        (lambda r: r["disp_pct"] >= 0.95, "EXTREME_DISP (>95%)"),
    ]:
        n = sum(1 for r in records if percentile_group(r) and r["fwd_ret"][5] is not None)
        if n < 5:
            continue
        print(f"\n  {label} (n={n}):")
        for h in HORIZONS:
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and percentile_group(r)]
            if len(vals) < 5:
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"    {h:3d}m  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # =========================================================
    # EXPERIMENT 2: State transition prediction
    # =========================================================
    print("\n" + "="*70)
    print("EXP 2: STATE TRANSITION PREDICTION")
    print("="*70)

    # Define regime states based on percentiles
    for h in [15, 30]:
        trans_matrix = {}
        for i in range(len(records) - h - 1):
            cur_reg = "HIGH_DISP" if records[i]["disp_pct"] >= 0.8 else \
                      "LOW_DISP" if records[i]["disp_pct"] < 0.2 else "MID_DISP"
            fut_reg = "HIGH_DISP" if records[i + h]["disp_pct"] >= 0.8 else \
                      "LOW_DISP" if records[i + h]["disp_pct"] < 0.2 else "MID_DISP"
            if cur_reg not in trans_matrix:
                trans_matrix[cur_reg] = {}
            trans_matrix[cur_reg][fut_reg] = trans_matrix[cur_reg].get(fut_reg, 0) + 1

        print(f"\n  P(next_state | current_state) at {h}m horizon:")
        for cur in ["LOW_DISP", "MID_DISP", "HIGH_DISP"]:
            if cur not in trans_matrix:
                continue
            total = sum(trans_matrix[cur].values())
            if total == 0:
                continue
            trans_pcts = {fut: c/total*100 for fut, c in trans_matrix[cur].items()}
            print(f"    {cur:12s} -> {trans_pcts}")

    # =========================================================
    # EXPERIMENT 3: Pair-specific WLS prediction
    # =========================================================
    print("\n" + "="*70)
    print("EXP 3: PAIR-SPECIFIC WLS PREDICTION")
    print("="*70)

    # Build currency strengths from each record
    pair_ics = {p: [] for p in ALL_PAIRS}
    for r in records:
        if r["fwd_ret"][5] is None or r["idx"] < 5:
            continue
        snapshot = None  # We don't store strengths directly
        # We need to store the currency strengths from each MSV update
        # Let me check: r doesn't have snapshot. Need to re-derive.
        # Actually, we recorded fwd_ret_pair which has pair-level forward returns
        # But for the WLS prediction, we need the WLS-implied pair move

    # Re-run storing WLS predicted pair moves
    print("\n  Re-running with pair-specific WLS predictions stored...")
    ms2 = MarketStateVector(history_size=50)
    pair_data = []
    for idx in range(N):
        returns = {}
        for pair in all_data:
            bar = all_data[pair][idx]
            if idx == 0:
                ret = 0.0
            else:
                prev = float(all_data[pair][idx - 1]["close"])
                curr = float(all_data[pair][idx]["close"])
                ret = (curr / prev - 1) if prev > 0 else 0.0
            ret = np.clip(ret, -0.05, 0.05)
            returns[pair] = ret
        now = float(all_data[pairs[0]][idx]["time"])
        snapshot = ms2.update(returns, timestamp=now)

        # Get WLS currency strengths from the snapshot
        strengths = {c: n.level for c, n in snapshot.currencies.items()}

        # WLS-implied pair moves
        implied = {}
        for sym, (base, quote) in BASE_CURRENCY_MAP.items():
            implied[sym] = strengths.get(base, 0.0) - strengths.get(quote, 0.0)

        # Forward pair returns
        fwd = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx < N:
                fwd[h] = {
                    p: float(all_data[p][fwd_idx]["close"]) / float(all_data[p][idx]["close"]) - 1
                    for p in all_data
                }
            else:
                fwd[h] = None

        pair_data.append({"implied": implied, "fwd": fwd, "idx": idx})

        if (idx + 1) % 500 == 0:
            print(f"    Re-processed {idx+1}/{N}")

    # Compute IC per pair
    print("\n  IC(WLS implied → actual return) per pair:")
    pair_ics = {}
    for sym in ALL_PAIRS:
        implied_list = []
        actual_list = []
        for p in pair_data:
            if p["fwd"][5] is None:
                continue
            implied_list.append(p["implied"].get(sym, 0.0))
            actual_list.append(p["fwd"][5].get(sym, 0.0))
        if len(implied_list) < 10:
            continue
        if np.std(implied_list) > 0 and np.std(actual_list) > 0:
            ic = float(np.corrcoef(implied_list, actual_list)[0, 1])
        else:
            ic = 0.0
        pair_ics[sym] = ic

    sorted_ics = sorted(pair_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    for sym, ic in sorted_ics:
        print(f"    {sym:8s} IC={ic:+.4f}")

    best_ic = max(abs(v) for v in pair_ics.values()) if pair_ics else 0.0
    print(f"\n  Best |IC|: {best_ic:.4f}")
    print(f"  Mean |IC|: {np.mean([abs(v) for v in pair_ics.values()]):.4f}")

    # =========================================================
    # EXPERIMENT 6: Volatility-normalized returns
    # =========================================================
    print("\n" + "="*70)
    print("EXP 6: VOLATILITY-NORMALIZED RETURNS")
    print("="*70)

    for disp_cond, label in [
        (lambda r: True, "ALL"),
        (lambda r: r["disp_pct"] < 0.2, "LOW_DISP"),
        (lambda r: r["disp_pct"] >= 0.8, "HIGH_DISP"),
    ]:
        n = sum(1 for r in records if disp_cond(r) and r["fwd_ret"][5] is not None)
        if n < 5:
            continue
        print(f"\n  {label}:")
        for h in HORIZONS:
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and disp_cond(r)]
            vols = [r["fwd_vol"][h] for r in records
                    if r["fwd_ret"][h] is not None and disp_cond(r)]
            if len(vals) < 5:
                continue
            # Volatility-normalized: return / vol
            norm_vals = [v / max(vol, 1e-10) for v, vol in zip(vals, vols)]
            mu_norm = float(np.mean(norm_vals))
            sigma_norm = float(np.std(norm_vals))
            sharpe_norm = mu_norm / sigma_norm * np.sqrt(12 * 24) if sigma_norm > 0 else 0.0
            pos_norm = sum(1 for v in norm_vals if v > 0) / len(norm_vals) * 100
            t_norm = mu_norm / (sigma_norm / np.sqrt(len(norm_vals))) if sigma_norm > 0 else 0.0
            print(f"    {h:3d}m  n={len(vals):5d}  norm_sharpe={sharpe_norm:+.3f}  "
                  f"norm_pos%={pos_norm:5.1f}%  norm_t={t_norm:+.2f}")

    # =========================================================
    # EXPERIMENT 4: Opportunity ranking model
    # =========================================================
    print("\n" + "="*70)
    print("EXP 4: OPPORTUNITY RANKING SCORE")
    print("="*70)

    # Use the pair_data from experiment 3 which has WLS implied moves
    print("\n  Top-ranked pairs per bar vs forward return:")
    rank_hits = {h: [] for h in HORIZONS}
    for p in pair_data:
        if p["fwd"][5] is None:
            continue

        # Rank pairs by |implied| (WLS edge magnitude)
        implied_pairs = sorted(p["implied"].items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = set(sym for sym, _ in implied_pairs[:3])
        bottom3 = set(sym for sym, _ in implied_pairs[-3:])

        for h in HORIZONS:
            if p["fwd"][h] is None:
                continue
            # Did top WLS pairs outperform bottom WLS pairs?
            top_ret = np.mean([p["fwd"][h].get(sym, 0.0) for sym in top3])
            bottom_ret = np.mean([p["fwd"][h].get(sym, 0.0) for sym in bottom3])
            spread = top_ret - bottom_ret
            rank_hits[h].append(spread)

    for h in HORIZONS:
        spreads = rank_hits[h]
        if len(spreads) < 5:
            continue
        mu = float(np.mean(spreads))
        sigma = float(np.std(spreads))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos_pct = sum(1 for s in spreads if s > 0) / len(spreads) * 100
        t = mu / (sigma / np.sqrt(len(spreads))) if sigma > 0 else 0.0
        print(f"    {h:3d}m  n={len(spreads):5d}  mean_spread={mu:+.6f}  "
              f"sharpe={sharpe:+.3f}  pos%={pos_pct:5.1f}%  t={t:+.2f}")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "="*70)
    print("SUMMARY — All Experiments")
    print("="*70)

    mt5.shutdown()

if __name__ == "__main__":
    main()
