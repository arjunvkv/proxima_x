"""DPL-18B: Execution Layer Research.
Signal locked: Entry = sign(TPI_200). Optimize exits.

Experiments:
1. Fixed hold baseline (hold=1,2,3,6,12) — expectancy curve
2. MFE/MAE profiling per pair, per state, per TPI magnitude
3. Hazard exit — edge half-life (accuracy by bar in trade)
4. Volatility-conditioned exits — SAF/SIT/VEM-based adaptive exits

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl18b_execution.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_to_bars_vectorized,
    compute_directional_labels_vect,
)
from research.volatility_physics.vpl_1.core.target_engine import (
    compute_returns, compute_crf, realized_variance_from_log_returns,
)
from research.volatility_physics.vpl_1.core.sit_engine import compute_sit
from research.volatility_physics.vpl_1.core.vcm_engine import compute_vcm

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]

def load_data(symbol):
    """Load ticks, M5 bars, compute TPI + state variables. Returns bar-level arrays."""
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    close = m5["close"]
    high = m5["close"] if "high" not in m5 else m5["high"]
    low = m5["close"] if "low" not in m5 else m5["low"]
    ts_m5 = m5["timestamp"]
    n = len(close)

    # Align ticks to bars
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n - 1)

    # TPI features
    tpi_feats = compute_tpi(ticks["mid"])
    bar_feats = aggregate_to_bars_vectorized(tpi_feats["tpi_200"], tick_bar_idx, n)
    bar_feats_50 = aggregate_to_bars_vectorized(tpi_feats.get("tpi_50", np.full(len(ticks["mid"]), np.nan)), tick_bar_idx, n)

    # State variables
    r = compute_returns(close)
    rv = realized_variance_from_log_returns(r, 24)
    crf = compute_crf(close, high, low)
    saf_raw = crf["crf"]
    sit_out = compute_sit(close, high, low, r, rv, saf_raw)
    sit_raw = sit_out["instability"]
    vcm_out = compute_vcm(r, rv)
    vem_raw = vcm_out["vcm"]
    vem = np.full(n, np.nan); vem[1:] = vem_raw
    saf, sit = saf_raw.copy(), sit_raw.copy()

    # Clip to valid range (bars with TPI)
    first_valid = np.where(~np.isnan(bar_feats))[0]
    if len(first_valid) == 0:
        return None
    s = first_valid[0]
    # Need up to n-12 for 12-bar forward returns
    e = min(first_valid[-1] + 1, n - 12)
    return {
        "symbol": symbol, "n": e - s, "close": close[s:e],
        "tpi_200": bar_feats[s:e], "tpi_50": bar_feats_50[s:e],
        "saf": saf[s:e], "sit": sit[s:e], "vem": vem[s:e],
    }

def forward_returns(close, horizons=(1, 2, 3, 6, 12)):
    """Compute forward log returns for each horizon."""
    log_close = np.log(close)
    n = len(close)
    results = {}
    for h in horizons:
        if h >= n: continue
        fwd = log_close[h:] - log_close[:n-h]
        results[h] = np.full(n, np.nan)
        results[h][:n-h] = fwd
    return results

def directional_labels_from_returns(fwd_ret):
    """Convert forward returns to {1, -1} labels."""
    labels = np.full(len(fwd_ret), np.nan)
    labels[fwd_ret > 0] = 1
    labels[fwd_ret < 0] = -1
    return labels

def experiment_1_fixed_hold(data, horizons=(1, 2, 3, 6, 12)):
    """Fixed hold baseline: expectancy, hit rate, mean return for each hold horizon."""
    tpi = data["tpi_200"]
    close = data["close"]
    n = len(tpi)
    fwd = forward_returns(close, horizons)

    results = {}
    for h in horizons:
        if h not in fwd: continue
        fwd_r = fwd[h]
        labels = directional_labels_from_returns(fwd_r)
        # TPI zero-threshold entry
        valid = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(labels)
        if np.sum(valid) < 10: continue
        tpi_v, label_v, ret_v = tpi[valid], labels[valid], fwd_r[valid]
        pred = np.where(tpi_v > 0, 1, -1)
        hit_rate = float(np.mean(pred == label_v))
        # Expectancy: mean return when correct minus mean return when wrong
        correct_ret = np.mean(ret_v[pred == label_v]) if np.sum(pred == label_v) > 0 else 0.0
        wrong_ret = np.mean(ret_v[pred != label_v]) if np.sum(pred != label_v) > 0 else 0.0
        expectancy = float(correct_ret) - float(abs(wrong_ret))
        avg_ret_when_correct = float(correct_ret)
        avg_ret_when_wrong = float(wrong_ret)

        results[h] = {
            "hit_rate": hit_rate,
            "expectancy": expectancy,
            "avg_ret_correct": avg_ret_when_correct,
            "avg_ret_wrong": avg_ret_when_wrong,
            "n_trades": int(np.sum(valid)),
        }
    return results

def experiment_2_mfe_mae(data):
    """MFE/MAE profiling by TPI magnitude quartile, SAF/SIT/VEM state."""
    tpi = data["tpi_200"]
    close = data["close"]
    saf = data["saf"]
    sit = data["sit"]
    vem = data["vem"]
    n = len(tpi)

    # Forward return over 3 bars
    fwd_r = forward_returns(close, (3,))[3]
    labels = directional_labels_from_returns(fwd_r)
    tpi_mag = np.abs(tpi)
    tpi_sign = np.where(tpi > 0, 1, np.where(tpi < 0, -1, 0)).astype(float)

    valid = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(labels)
    if np.sum(valid) < 10:
        return None

    results = {}

    # By TPI magnitude quartile
    mag_valid = tpi_mag[valid]
    quartiles = np.percentile(mag_valid, [25, 50, 75])
    def q_idx(v, qs):
        if v <= qs[0]: return 0
        if v <= qs[1]: return 1
        if v <= qs[2]: return 2
        return 3

    for label, arr, name in [
        ("tpi_mag_quartile", tpi_mag, "TPI Mag Q"),
        ("saf_decile", saf, "SAF"),
        ("sit_decile", sit, "SIT"),
        ("vem_decile", vem, "VEM"),
    ]:
        arr_valid = arr[valid]
        deciles = np.percentile(arr_valid[~np.isnan(arr_valid)], [10, 20, 30, 40, 50, 60, 70, 80, 90])
        grp_results = {}
        for g in range(10 if "decile" in label else 4):
            if "quartile" in label:
                if g == 0: mask = tpi_mag <= quartiles[0]
                elif g == 1: mask = (tpi_mag > quartiles[0]) & (tpi_mag <= quartiles[1])
                elif g == 2: mask = (tpi_mag > quartiles[1]) & (tpi_mag <= quartiles[2])
                else: mask = tpi_mag > quartiles[2]
            else:
                lo = -np.inf if g == 0 else deciles[g-1]
                hi = np.inf if g == 9 else deciles[g]
                mask = (arr >= lo) & (arr < hi)

            gv = mask[valid] & ~np.isnan(fwd_r[valid])
            if np.sum(gv) < 5: continue
            tpi_g = tpi_sign[valid][gv]
            ret_g = fwd_r[valid][gv]
            pred_g = np.where(tpi_g > 0, 1, -1)
            label_g = labels[valid][gv]
            hit = float(np.mean(pred_g == label_g))
            mfe = float(np.mean(ret_g[pred_g == label_g])) if np.sum(pred_g == label_g) > 0 else 0.0
            mae = float(np.mean(np.abs(ret_g[pred_g != label_g]))) if np.sum(pred_g != label_g) > 0 else 0.0
            grp_results[f"g{g}"] = {"hit_rate": hit, "mfe": mfe, "mae": mae, "n": int(np.sum(gv))}
        results[label] = grp_results

    # Overall MFE/MAE
    pred = np.where(tpi_sign[valid] > 0, 1, -1)
    label_v = labels[valid]
    ret_v = fwd_r[valid]
    results["overall"] = {
        "hit_rate": float(np.mean(pred == label_v)),
        "mfe": float(np.mean(ret_v[pred == label_v])) if np.sum(pred == label_v) > 0 else 0.0,
        "mae": float(np.mean(np.abs(ret_v[pred != label_v]))) if np.sum(pred != label_v) > 0 else 0.0,
        "n": int(np.sum(valid)),
    }

    return results

def experiment_3_hazard_exit(data):
    """Edge half-life: accuracy by bar-in-trade."""
    tpi = data["tpi_200"]
    close = data["close"]
    n = len(tpi)
    fwd = forward_returns(close, range(1, 13))

    results = {}
    for h in range(1, 13):
        if h not in fwd: continue
        fwd_r = fwd[h]
        labels = directional_labels_from_returns(fwd_r)
        valid = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(labels)
        if np.sum(valid) < 10: continue
        tpi_v, label_v = tpi[valid], labels[valid]
        pred = np.where(tpi_v > 0, 1, -1)
        acc = float(np.mean(pred == label_v))
        results[f"h{h}"] = {"accuracy": acc, "n": int(np.sum(valid))}

    return results

def experiment_4_vol_conditioned(data):
    """Test volatility-conditioned exits: compare fixed hold vs state-adaptive hold.

    Strategy: 
      Baseline = TPI_200 > 0 entry, fixed 3-bar hold
      Adaptive = if Low SAF + High SIT → hold 6 bars (expansion expected)
                 elif High SAF + High VEM → hold 1 bar (exhaustion expected)
                 else → hold 3 bars
    """
    tpi = data["tpi_200"]
    close = data["close"]
    saf = data["saf"]
    sit = data["sit"]
    vem = data["vem"]
    n = len(tpi)

    # Compute state deciles
    saf_valid = saf[~np.isnan(saf)]
    sit_valid = sit[~np.isnan(sit)]
    vem_valid = vem[~np.isnan(vem)]
    saf_lo = np.percentile(saf_valid, 33) if len(saf_valid) > 0 else 0
    saf_hi = np.percentile(saf_valid, 67) if len(saf_valid) > 0 else 1
    sit_lo = np.percentile(sit_valid, 33) if len(sit_valid) > 0 else 0
    sit_hi = np.percentile(sit_valid, 67) if len(sit_valid) > 0 else 1
    vem_lo = np.percentile(vem_valid, 33) if len(vem_valid) > 0 else 0
    vem_hi = np.percentile(vem_valid, 67) if len(vem_valid) > 0 else 1

    # Baseline: fixed 3-bar hold
    fwd_3 = forward_returns(close, (3,))[3]
    valid_3 = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(fwd_3)
    tpi_v3 = tpi[valid_3]
    pred_3 = np.where(tpi_v3 > 0, 1, -1)
    ret_3 = fwd_3[valid_3]
    baseline_expectancy = float(np.mean(ret_3[pred_3 == 1]) - np.mean(ret_3[pred_3 == -1]))

    # Adaptive: choose hold based on state
    fwd_all = forward_returns(close, range(1, 13))
    adaptive_returns = []
    for i in range(n - 12):
        if np.isnan(tpi[i]) or tpi[i] == 0: continue
        if np.isnan(saf[i]) or np.isnan(sit[i]) or np.isnan(vem[i]): continue

        # Determine hold based on state
        expansion = saf[i] < saf_lo and sit[i] > sit_hi  # Low SAF + High SIT = expansion expected
        exhaustion = saf[i] > saf_hi and vem[i] > vem_hi  # High SAF + High VEM = exhaustion
        hold = 6 if expansion else (1 if exhaustion else 3)

        fwd_r = fwd_all.get(hold)
        if fwd_r is None or np.isnan(fwd_r[i]): continue
        pred = 1 if tpi[i] > 0 else -1
        adaptive_returns.append(fwd_r[i] * pred)

    adaptive_expectancy = float(np.mean(adaptive_returns)) if adaptive_returns else 0.0

    return {
        "baseline_3bar_expectancy": baseline_expectancy,
        "adaptive_expectancy": adaptive_expectancy,
        "adaptive_n": len(adaptive_returns),
        "baseline_n": int(np.sum(valid_3)),
        "params": {"saf_lo": float(saf_lo), "saf_hi": float(saf_hi),
                    "sit_lo": float(sit_lo), "sit_hi": float(sit_hi),
                    "vem_lo": float(vem_lo), "vem_hi": float(vem_hi)},
    }

if __name__ == "__main__":
    print("=" * 70)
    print("DPL-18B: Execution Layer Research")
    print("=" * 70)

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d: all_data[sym] = d

    results = {}

    for sym, data in all_data.items():
        print(f"\n{'='*70}")
        print(f" {sym}")
        print(f"{'='*70}")

        # Experiment 1: Fixed hold baseline
        print(f"\n  EXPERIMENT 1 — Fixed Hold Baseline")
        print(f"  {'Hold':>6s}  {'HitRate':>8s}  {'Expectancy':>12s}  {'AvgWin':>10s}  {'AvgLoss':>10s}  {'Trades':>8s}")
        e1 = experiment_1_fixed_hold(data)
        for h in sorted(e1.keys()):
            r = e1[h]
            print(f"  {h:>6d}  {r['hit_rate']:>8.4f}  {r['expectancy']:>+12.6f}  {r['avg_ret_correct']:>+10.6f}  {r['avg_ret_wrong']:>+10.6f}  {r['n_trades']:>8d}")
        results[sym] = {"exp1_fixed_hold": e1}

        # Experiment 2: MFE/MAE profiling
        print(f"\n  EXPERIMENT 2 — MFE/MAE Profiling")
        e2 = experiment_2_mfe_mae(data)
        if e2:
            ov = e2.get("overall", {})
            print(f"  Overall:  hit={ov.get('hit_rate',0):.4f}  MFE={ov.get('mfe',0):+.6f}  MAE={ov.get('mae',0):.6f}  n={ov.get('n',0)}")
            for group_key in ["tpi_mag_quartile", "saf_decile", "sit_decile", "vem_decile"]:
                if group_key in e2:
                    print(f"  {group_key}:")
                    for gk, gv in sorted(e2[group_key].items()):
                        print(f"    {gk}: hit={gv['hit_rate']:.4f}  MFE={gv['mfe']:+.6f}  MAE={gv['mae']:.6f}  n={gv['n']}")
        results[sym]["exp2_mfe_mae"] = e2

        # Experiment 3: Hazard exit
        print(f"\n  EXPERIMENT 3 — Edge Half-Life")
        print(f"  {'Bar':>4s}  {'Accuracy':>10s}  {'N':>8s}")
        e3 = experiment_3_hazard_exit(data)
        for h in sorted(e3.keys(), key=lambda x: int(x[1:])):
            r = e3[h]
            print(f"  {h:>4s}  {r['accuracy']:>10.4f}  {r['n']:>8d}")
        results[sym]["exp3_hazard"] = e3

        # Experiment 4: Vol-conditioned exits
        print(f"\n  EXPERIMENT 4 — Volatility-Conditioned Exits")
        e4 = experiment_4_vol_conditioned(data)
        print(f"  Baseline 3-bar expectancy: {e4['baseline_3bar_expectancy']:+.6f} (n={e4['baseline_n']})")
        print(f"  Adaptive expectancy:        {e4['adaptive_expectancy']:+.6f} (n={e4['adaptive_n']})")
        print(f"  Lift: {e4['adaptive_expectancy'] - e4['baseline_3bar_expectancy']:+.6f}")
        results[sym]["exp4_vol_conditioned"] = e4

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Best hold by expectancy (all pairs):")
    for sym, r in results.items():
        e1 = r.get("exp1_fixed_hold", {})
        best_h = max(e1.keys(), key=lambda h: e1[h]["expectancy"]) if e1 else None
        if best_h:
            print(f"  {sym:8s}:  h{best_h}  expectancy={e1[best_h]['expectancy']:+.6f}  hit={e1[best_h]['hit_rate']:.4f}")

    print(f"\n  Vol-conditioned exit lift:")
    for sym, r in results.items():
        e4 = r.get("exp4_vol_conditioned", {})
        bl = e4.get("baseline_3bar_expectancy", 0)
        ad = e4.get("adaptive_expectancy", 0)
        print(f"  {sym:8s}:  baseline={bl:+.6f}  adaptive={ad:+.6f}  lift={ad-bl:+.6f}")

    with open(os.path.join(REPORT_DIR, "dpl18b_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDPL-18B -> dpl18b_results.json")
