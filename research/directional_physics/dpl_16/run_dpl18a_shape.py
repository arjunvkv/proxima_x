"""DPL-18A: TPI Shape Refinement.
Tests: do TPI curvature / concentration / terminal bias improve over raw TPI_200?

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl18a_shape.py
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
from scipy.ndimage import uniform_filter1d

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]
W = 200  # TPI window

def _up_down(mid):
    """Return padded cumsum arrays for easy rolling window computation.
    up[i] = 1 if tick i is an up-tick (price increase from i-1 to i).
    Cumsum has len(mid)+1 so cum[hi]-cum[lo] = sum over [lo, hi).
    """
    delta = np.diff(mid)
    # First tick has no direction, pad with 0
    up = np.concatenate([[0], (delta > 1e-8).astype(np.float64)])
    down = np.concatenate([[0], (delta < -1e-8).astype(np.float64)])
    net = up - down
    total = up + down
    # Pad with 0 at start so cum[hi] - cum[lo] = sum over [lo, hi)
    # cum has length n+1, indices 0..n
    cum_net = np.concatenate([[0], np.cumsum(net)])
    cum_total = np.concatenate([[0], np.cumsum(total)])
    return cum_net, cum_total

def compute_shape_curvature(mid, w=W):
    """TPI Curvature: ΔTPI and Δ²TPI between first/second half of window.
    
    For each tick position i, splits window [i-w+1, i] into first half and
    second half. Curvature = TPI_second_half - TPI_first_half.
    Positive = pressure accelerating (recent ticks more imbalanced).
    """
    n = len(mid)
    cum_net, cum_total = _up_down(mid)
    hw = w // 2
    
    curvature = np.full(n, np.nan)
    curvature_accel = np.full(n, np.nan)
    if n <= w: return curvature, curvature_accel
    
    # First half: ticks [i-w+1, i-hw]  → cum indices [i-w, i-hw)
    f_lo = np.maximum(0, np.arange(n) - w)
    f_hi = np.clip(np.arange(n) - hw, 0, n)
    fn = cum_net[f_hi] - cum_net[f_lo]
    ft = cum_total[f_hi] - cum_total[f_lo]
    tpi_first = np.where(ft > 0, fn / ft, np.nan)
    
    # Second half: ticks [i-hw+1, i] → cum indices [i-hw, i+1)
    # cum_net has n+1 elements (padded), so cum_net[i+1] is valid for i < n
    s_lo = np.maximum(0, np.arange(n) - hw)
    s_hi = np.arange(n) + 1  # valid since cum has n+1 elements
    sn = cum_net[s_hi] - cum_net[s_lo]
    st = cum_total[s_hi] - cum_total[s_lo]
    tpi_second = np.where(st > 0, sn / st, np.nan)
    
    curvature = tpi_second - tpi_first
    if n > 2:
        curvature_accel[2:] = curvature[2:] - 2 * curvature[1:-1] + curvature[:-2]
    return curvature, curvature_accel

def compute_shape_terminal(mid, w=W, w_small=50):
    """TPI Terminal Bias: small-window TPI minus large-window TPI.
    Positive = recent ticks (last 50) are more bullish than broader 200-tick window.
    """
    n = len(mid)
    cum_net, cum_total = _up_down(mid)  # both have length n+1
    
    # Large window (w): cum indices [i-w, i+1)
    lo_l = np.maximum(0, np.arange(n) - w)
    hi_l = np.arange(n) + 1
    net_l = cum_net[hi_l] - cum_net[lo_l]
    tot_l = cum_total[hi_l] - cum_total[lo_l]
    tpi_l = np.where(tot_l > 0, net_l / tot_l, np.nan)
    
    # Small window (w_small): cum indices [i-w_small, i+1)
    lo_s = np.maximum(0, np.arange(n) - w_small)
    hi_s = np.arange(n) + 1
    net_s = cum_net[hi_s] - cum_net[lo_s]
    tot_s = cum_total[hi_s] - cum_total[lo_s]
    tpi_s = np.where(tot_s > 0, net_s / tot_s, np.nan)
    
    valid = ~np.isnan(tpi_s) & ~np.isnan(tpi_l)
    bias = np.full(n, np.nan)
    bias[valid] = tpi_s[valid] - tpi_l[valid]
    return bias

def compute_shape_concentration(mid, w=W, n_subs=5):
    """TPI Concentration: entropy of sub-window imbalance distribution.
    Low entropy = concentrated (bursty imbalance). High entropy = evenly distributed.
    Uses vectorized computation instead of loops.
    """
    n = len(mid)
    cum_net, _ = _up_down(mid)  # length n+1
    
    sub_w = w // n_subs
    if n <= w or sub_w < 3:
        return np.full(n, np.nan)
    
    n_subs = w // sub_w  # adjust
    sub_vals = np.zeros((n_subs, n))
    
    for s in range(n_subs):
        lo = np.arange(n) - w + s * sub_w
        hi = np.arange(n) - w + (s + 1) * sub_w
        if s == n_subs - 1:
            hi = np.arange(n) + 1  # last sub-window goes to i
        lo = np.clip(lo, 0, n).astype(int)
        hi = np.clip(hi, 0, n).astype(int)
        sub_vals[s] = cum_net[hi] - cum_net[lo]
    
    abs_vals = np.abs(sub_vals)
    sum_abs = np.sum(abs_vals, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(sum_abs > 0, abs_vals / sum_abs, 0)
        logp = np.where(p > 0, np.log2(p), 0)
        entropy = -np.sum(p * logp, axis=0)
        max_e = np.log2(n_subs)
        sub_entropy = np.where(sum_abs > 0, entropy / max_e, np.nan)
    
    return sub_entropy

def compute_all_shapes(mid):
    """Compute all TPI shape features at tick level."""
    results = {}
    results["tpi_raw"] = compute_tpi(mid)["tpi_200"]
    
    curv, curv_accel = compute_shape_curvature(mid)
    results["tpi_curvature"] = curv
    results["tpi_curvature_accel"] = curv_accel
    
    results["tpi_terminal_bias"] = compute_shape_terminal(mid)
    results["tpi_concentration"] = compute_shape_concentration(mid)
    
    return results

def accuracy_by_median(feature, labels, min_samples=50):
    """Median-split directional accuracy."""
    valid = ~np.isnan(feature) & ~np.isnan(labels)
    if np.sum(valid) < min_samples:
        return np.nan, 0
    fv, lv = feature[valid], labels[valid]
    med = np.nanmedian(fv)
    pred = np.where(fv > med, 1, -1)
    acc = float(np.mean(pred == lv))
    return acc, int(np.sum(valid))

def accuracy_by_zero_threshold(feature, labels, min_samples=50):
    """Zero-threshold accuracy: feature > 0 predicts UP, < 0 predicts DOWN."""
    valid = (feature != 0) & ~np.isnan(feature) & ~np.isnan(labels)
    if np.sum(valid) < min_samples:
        return np.nan, 0
    fv, lv = feature[valid], labels[valid]
    pred = np.where(fv > 0, 1, -1)
    acc = float(np.mean(pred == lv))
    return acc, int(np.sum(valid))

if __name__ == "__main__":
    print("=" * 65)
    print("DPL-18A: TPI Shape Refinement")
    print("=" * 65)
    
    all_results = {}
    for sym in SYMBOLS:
        print(f"\n{'='*65}")
        print(f"Processing {sym}...")
        
        ticks = load_ticks(sym)
        m5 = load_m5(sym)
        close = m5["close"]
        ts_m5 = m5["timestamp"]
        n_bars = len(close)
        
        # Align ticks to bars
        starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
        tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
        tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)
        
        # Compute shape features at tick level
        shapes = compute_all_shapes(ticks["mid"])
        
        # Aggregate to bars
        bar_shapes = {}
        for key, feat in shapes.items():
            bar_shapes[key] = aggregate_to_bars_vectorized(feat, tick_bar_idx, n_bars)
        
        # Labels
        labels = compute_directional_labels_vect(close)
        
        # Validate: TPI should match
        print(f"  Tick TPI_200: {np.nanmean(shapes['tpi_raw']):.6f}")
        print(f"  Bar TPI_200:  {np.nanmean(bar_shapes['tpi_raw']):.6f}")
        
        # Test each feature
        sym_results = {}
        for key, feat in bar_shapes.items():
            # Both median-split and zero-threshold
            acc_m, n_m = accuracy_by_median(feat, labels)
            acc_z, n_z = accuracy_by_zero_threshold(feat, labels)
            sym_results[key] = {
                "median_split": {"accuracy": acc_m, "n": n_m},
                "zero_threshold": {"accuracy": acc_z, "n": n_z},
            }
            if not np.isnan(acc_m):
                print(f"  {key:25s}  median={acc_m:.4f}(n={n_m})  zero-thresh={acc_z:.4f}(n={n_z})")
        
        all_results[sym] = sym_results
    
    # Summary
    print(f"\n{'='*65}")
    print(f"SUMMARY: INCREMENTAL LIFT OVER RAW TPI_200 (zero-threshold)")
    print(f"{'='*65}")
    features = ["tpi_curvature", "tpi_curvature_accel", "tpi_terminal_bias", "tpi_concentration"]
    for sym in SYMBOLS:
        if sym not in all_results: continue
        r = all_results[sym]
        baseline = r.get("tpi_raw", {}).get("zero_threshold", {}).get("accuracy", np.nan)
        print(f"\n  {sym:8s}  Baseline(tpi_200): {baseline:.4f}")
        for feat in features:
            acc = r.get(feat, {}).get("zero_threshold", {}).get("accuracy", np.nan)
            if not np.isnan(acc) and not np.isnan(baseline):
                lift = acc - baseline
                print(f"           {feat:25s}  {acc:.4f}  lift={lift:+.4f}")
    
    # Save
    with open(os.path.join(REPORT_DIR, "dpl18a_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDPL-18A -> dpl18a_results.json")
