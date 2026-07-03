"""DPL-17B: Lead/Lag Causality + Cell Significance.
Tests: does TPI predict state transitions? Are grid cells statistically valid?

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl17b_causality.py
"""
import sys, os, json, warnings
import numpy as np
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_features_to_bars,
    compute_directional_labels_vect,
)
from research.volatility_physics.vpl_1.core.target_engine import (
    compute_returns, compute_crf, realized_variance_from_log_returns,
    compute_expansion_labels,
)
from research.volatility_physics.vpl_1.core.sit_engine import compute_sit
from research.volatility_physics.vpl_1.core.vcm_engine import compute_vcm

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]
TPI_KEY = "tpi_200"

def prepare(symbol):
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    close = m5["close"]
    high = m5["close"] if "high" not in m5 else m5["high"]
    low = m5["close"] if "low" not in m5 else m5["low"]
    ts_m5 = m5["timestamp"]
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    n_bars = len(starts)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)
    tpi_feats = compute_tpi(ticks["mid"])
    bar_feats = aggregate_features_to_bars(tpi_feats, tick_bar_idx, n_bars)
    tpi = bar_feats[TPI_KEY]
    r = compute_returns(close)
    rv = realized_variance_from_log_returns(r, 24)
    crf = compute_crf(close, high, low)
    saf_raw = crf["crf"]
    sit_out = compute_sit(close, high, low, r, rv, saf_raw)
    sit_raw = sit_out["instability"]
    vcm_out = compute_vcm(r, rv)
    vem_raw = vcm_out["vcm"]
    n = len(close)
    vem = np.full(n, np.nan); vem[1:] = vem_raw
    saf, sit = saf_raw.copy(), sit_raw.copy()
    labels_3 = compute_directional_labels_vect(close, n_bars=3)

    # Expansion labels
    exp = compute_expansion_labels(close)
    expand_3 = np.full(n, np.nan)
    key = "expand_1.5_3"
    if key in exp:
        expand_3 = exp[key]["label"]

    first_valid = np.where(~np.isnan(tpi))[0]
    if len(first_valid) == 0: return None
    s, e = first_valid[0], min(first_valid[-1] + 1, n)
    return {"symbol": symbol, "n": e - s, "ts": ts_m5[s:e], "close": close[s:e],
            "tpi": tpi[s:e], "saf": saf[s:e], "sit": sit[s:e], "vem": vem[s:e],
            "labels_3": labels_3[s:e], "expand_3": expand_3[s:e],
            "close_full": close, "saf_full": saf_raw, "sit_full": sit_raw}

def wilson_ci(acc, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0: return 0, 0
    p = acc * n / n
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0, centre - margin), min(1, centre + margin)

# ====================== CELL SIGNIFICANCE ======================
def cell_significance(data):
    """For each SAF×SIT cell: accuracy, n, confidence interval."""
    saf, sit, tpi, labels = data["saf"], data["sit"], data["tpi"], data["labels_3"]
    valid = ~np.isnan(saf) & ~np.isnan(sit) & ~np.isnan(tpi) & ~np.isnan(labels)
    if np.sum(valid) < 100: return None
    saf_v, sit_v, tpi_v, lv = saf[valid], sit[valid], tpi[valid], labels[valid]
    n = len(saf_v)
    saf_ranks = np.argsort(np.argsort(saf_v))
    sit_ranks = np.argsort(np.argsort(sit_v))
    d_size = n // 10
    si_idx = np.clip(saf_ranks // d_size, 0, 9) if d_size > 0 else np.zeros(n, dtype=int)
    ti_idx = np.clip(sit_ranks // d_size, 0, 9) if d_size > 0 else np.zeros(n, dtype=int)
    overall_med = np.nanmedian(tpi_v)
    grid_acc, grid_n, grid_ci = np.full((10, 10), np.nan), np.zeros((10, 10), dtype=int), [[(0, 0)]*10 for _ in range(10)]
    for d in range(10):
        for t in range(10):
            mask = (si_idx == d) & (ti_idx == t)
            cnt = int(np.sum(mask))
            grid_n[d, t] = cnt
            if cnt < 10: continue
            pred = np.where(tpi_v[mask] > overall_med, 1, -1)
            acc = float(np.mean(pred == lv[mask]))
            grid_acc[d, t] = acc
            grid_ci[d][t] = wilson_ci(acc, cnt)
    return grid_acc, grid_n, grid_ci

def print_grid_with_counts(grid, counts, title):
    print(f"\n  {title}:")
    header = "      " + "".join(f" SIT{i:3d}" for i in range(10))
    print(header)
    for d in range(9, -1, -1):
        row = f" SAF{d:2d} "
        for t in range(10):
            v, c = grid[d, t], counts[d, t]
            if not np.isnan(v):
                row += f" {v*100:3.0f}"
            else:
                row += "  --"
        print(row)
    print(f"\n  Sample counts:")
    for d in range(9, -1, -1):
        row = f" SAF{d:2d} "
        for t in range(10):
            c = counts[d, t]
            row += f" {c:4d}" if c > 0 else "    -"
        print(row)

def print_highlights(grid, counts, symbol):
    """Print best cells meeting significance thresholds."""
    print(f"\n  --- {symbol}: Significant cells (n >= 100) ---")
    for d in range(10):
        for t in range(10):
            c = counts[d, t]
            if c < 100: continue
            v = grid[d, t]
            if not np.isnan(v):
                lo, hi = wilson_ci(v, c)
                mark = " ***" if v > 0.60 else ""
                print(f"    SAF{d}×SIT{t}: acc={v:.3f}  n={c}  CI=[{lo:.3f}, {hi:.3f}]{mark}")

# ====================== LEAD/LAG CAUSALITY ======================
def lead_lag(data):
    """Test if TPI predicts state transitions."""
    saf, sit, tpi, expand = data["saf"], data["sit"], data["tpi"], data["expand_3"]
    n = len(tpi)
    saf_high = saf > np.nanmedian(saf)
    tpi_pos = tpi > 0
    tpi_neg = tpi < 0
    sit_p66 = np.nanpercentile(sit, 66)
    sit_high = sit > sit_p66

    results = {}

    # A) Future SAF transition: P(High SAF -> Low SAF next bar)
    saf_transition = np.full(n, np.nan)
    for i in range(n - 1):
        if saf_high[i] and not np.isnan(saf_high[i+1]):
            saf_transition[i] = 1.0 if not saf_high[i+1] else 0.0

    for name, cond in [("TPI>0", tpi_pos), ("TPI<0", tpi_neg), ("All", np.ones(n, bool))]:
        mask = cond & ~np.isnan(saf_transition)
        if np.sum(mask) < 50: continue
        tr = saf_transition[mask]
        rate = float(np.mean(tr))
        results[f"SAF_high_to_low_{name}"] = {"transition_rate": rate, "n": int(np.sum(mask))}

    # B) Future SIT increase
    sit_up = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isnan(sit[i]) and not np.isnan(sit[i+1]):
            sit_up[i] = 1.0 if sit[i+1] > sit[i] else 0.0

    for name, cond in [("TPI>0", tpi_pos), ("TPI<0", tpi_neg)]:
        mask = cond & ~np.isnan(sit_up)
        if np.sum(mask) < 50: continue
        rate = float(np.mean(sit_up[mask]))
        results[f"SIT_increase_{name}"] = {"rate": rate, "n": int(np.sum(mask))}

    # C) Future volatility expansion
    for name, cond in [("TPI>0", tpi_pos), ("TPI<0", tpi_neg),
                        ("TPI>0+HighSAF", tpi_pos & saf_high),
                        ("TPI<0+HighSAF", tpi_neg & saf_high)]:
        mask = cond & ~np.isnan(expand)
        if np.sum(mask) < 30: continue
        exp_rate = float(np.nanmean(expand[mask]))
        base_rate = float(np.nanmean(expand[~np.isnan(expand)]))
        results[f"expansion_{name}"] = {"rate": exp_rate, "base_rate": base_rate,
                                         "lift": exp_rate / base_rate if base_rate > 0 else 0,
                                         "n": int(np.sum(mask))}

    # D) SAF transition + TPI direction: does TPI predict expansion after absorption?
    after_abs = np.full(n, np.nan)
    for i in range(1, n):
        if saf_high[i-1] and not saf_high[i]:
            after_abs[i] = float(tpi_pos[i])
    mask = ~np.isnan(after_abs) & ~np.isnan(expand)
    if np.sum(mask) > 30:
        both = after_abs[mask] > 0.5
        exp_both = expand[mask]
        rate_tpi = float(np.nanmean(exp_both[both])) if np.sum(both) > 10 else 0
        rate_not = float(np.nanmean(exp_both[~both])) if np.sum(~both) > 10 else 0
        results["expansion_after_absorption_TPIpos"] = {"rate": rate_tpi, "n": int(np.sum(both))}
        results["expansion_after_absorption_TPIneg"] = {"rate": rate_not, "n": int(np.sum(~both))}

    return results

def tpi_persistence_accuracy(data, labels):
    """Refined: TPI sign accuracy with absolute thresholds instead of median split."""
    tpi = data["tpi"]
    # Use zero as threshold (natural: TPI>0 means more up-ticks than down-ticks)
    pred = np.where(tpi > 0, 1, -1)
    # Remove zeros (TPI == 0 exactly)
    non_zero = tpi != 0
    valid = non_zero & ~np.isnan(labels)
    if np.sum(valid) < 50: return np.nan, 0
    acc = float(np.mean(pred[valid] == labels[valid]))
    return acc, int(np.sum(valid))

if __name__ == "__main__":
    print("=" * 60)
    print("DPL-17B: Lead/Lag Causality + Cell Significance")
    print("=" * 60)

    all_data = {}
    for sym in SYMBOLS:
        d = prepare(sym)
        if d: all_data[sym] = d

    # TPI sign baseline (zero-threshold)
    print(f"\n{'='*60}")
    print(f"TPI SIGN BASELINE (zero-threshold, 3-bar)")
    print(f"{'='*60}")
    for sym, d in all_data.items():
        acc, n = tpi_persistence_accuracy(d, d["labels_3"])
        if n > 0:
            print(f"  {sym:8s}  TPI>0 dir_acc={acc:.4f}  n={n}")

    # Cell significance audit
    print(f"\n{'='*60}")
    print(f"CELL SIGNIFICANCE AUDIT (SAF × SIT)")
    print(f"{'='*60}")
    all_cells = {}
    for sym, d in all_data.items():
        grid, counts, ci = cell_significance(d)
        if grid is not None:
            print(f"\n  {sym} — TPI Accuracy by State:")
            print_grid_with_counts(grid, counts, f"")
            print_highlights(grid, counts, sym)
            all_cells[sym] = {"acc": grid.tolist(), "counts": counts.tolist()}

    # Lead/lag causality
    print(f"\n{'='*60}")
    print(f"LEAD/LAG CAUSALITY")
    print(f"{'='*60}")
    causality = {}
    for sym, d in all_data.items():
        r = lead_lag(d)
        causality[sym] = r
        print(f"  {sym}:")
        for k, v in r.items():
            if isinstance(v, dict):
                parts = [f"{kk}={vv:.4f}" for kk, vv in v.items()]
                print(f"    {k:45s}  {'  '.join(parts)}")

    # Save
    report = {"cell_significance": all_cells, "lead_lag": causality}
    with open(os.path.join(REPORT_DIR, "dpl17b_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDPL-17B -> dpl17b_results.json")
