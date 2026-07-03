"""DPL-17: Flow × Volatility Interaction.
Tests: where does TPI_200 survive and where does it die across volatility states.

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl17_flow_vol.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_features_to_bars,
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
TPI_KEY = "tpi_200"

def prepare(symbol):
    """Load ticks + M5, compute TPI_200 + VPL states, return aligned arrays."""
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    ts_m5 = m5["timestamp"]
    close, high, low = m5["close"], m5.get("high"), m5.get("low")
    if high is None:
        high = close
        low = close

    # Tick-to-bar alignment
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    n_bars = len(starts)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)

    # TPI
    tpi_feats = compute_tpi(ticks["mid"])
    bar_feats = aggregate_features_to_bars(tpi_feats, tick_bar_idx, n_bars)
    tpi = bar_feats[TPI_KEY]

    # VPL states
    r = compute_returns(close)
    rv = realized_variance_from_log_returns(r, 24)
    crf = compute_crf(close, high, low)
    saf_raw = crf["crf"]

    sit_out = compute_sit(close, high, low, r, rv, saf_raw)
    sit_raw = sit_out["instability"]
    jerk_raw = sit_out["jerk"]

    vcm_out = compute_vcm(r, rv)
    vem_raw = vcm_out["vcm"]

    # VEM needs padding (return-based, shift by 1)
    n = len(close)
    vem = np.full(n, np.nan)
    vem[1:] = vem_raw
    saf = saf_raw.copy()
    sit = sit_raw.copy()
    jerk = jerk_raw.copy()

    # Directional labels
    labels_1 = compute_directional_labels_vect(close, n_bars=1)
    labels_3 = compute_directional_labels_vect(close, n_bars=3)

    # Trim to overlap where TPI has data
    first_valid = np.where(~np.isnan(tpi))[0]
    if len(first_valid) == 0:
        return None
    s, e = first_valid[0], min(first_valid[-1] + 1, n)
    tpi = tpi[s:e]; saf = saf[s:e]; sit = sit[s:e]; jerk = jerk[s:e]
    vem = vem[s:e]; labels_1 = labels_1[s:e]; labels_3 = labels_3[s:e]
    ts = ts_m5[s:e]; close_s = close[s:e]

    return {
        "symbol": symbol, "n": len(tpi),
        "ts": ts, "close": close_s,
        "tpi": tpi, "saf": saf, "sit": sit, "jerk": jerk, "vem": vem,
        "labels_1": labels_1, "labels_3": labels_3,
    }

def tpi_acc(tpi, labels):
    """TPI directional accuracy via median split."""
    valid = ~np.isnan(tpi) & ~np.isnan(labels)
    if np.sum(valid) < 30:
        return np.nan, 0
    fv, lv = tpi[valid], labels[valid]
    med = np.nanmedian(fv)
    pred = np.where(fv > med, 1, -1)
    return float(np.mean(pred == lv)), int(np.sum(valid))

def decile_accuracy(data, state_name, state_arr, labels, n_deciles=10):
    """Phase 1: Accuracy(TPI | state decile)."""
    valid = ~np.isnan(state_arr) & ~np.isnan(data["tpi"]) & ~np.isnan(labels)
    if np.sum(valid) < 100:
        return None
    sv, tv, lv = state_arr[valid], data["tpi"][valid], labels[valid]
    order = np.argsort(sv)
    d_size = len(order) // n_deciles
    results = {}
    for d in range(n_deciles):
        s = d * d_size
        e = len(order) if d == n_deciles - 1 else (d + 1) * d_size
        idx = order[s:e]
        med = np.nanmedian(tv[idx])
        pred = np.where(tv[idx] > med, 1, -1)
        acc = float(np.mean(pred == lv[idx]))
        results[f"{state_name}_d{d}"] = {"accuracy": acc, "n": len(idx)}
    return results

def state_grid_accuracy(data, labels):
    """Phase 2: SAF decile × SIT decile grid."""
    valid = ~np.isnan(data["saf"]) & ~np.isnan(data["sit"]) & ~np.isnan(data["tpi"]) & ~np.isnan(labels)
    if np.sum(valid) < 100:
        return None
    saf_v, sit_v, tpi_v, lv = data["saf"][valid], data["sit"][valid], data["tpi"][valid], labels[valid]
    saf_order = np.argsort(saf_v)
    sit_order = np.argsort(sit_v)
    n = len(saf_v)
    d_size = n // 10
    grid = np.full((10, 10), np.nan)
    counts = np.zeros((10, 10), dtype=int)
    overall_med = np.nanmedian(tpi_v)
    for i in range(n):
        si = min(saf_order[i] // d_size, 9)
        ti = min(sit_order[i] // d_size, 9)
        si = int(np.searchsorted(saf_order, i) // d_size) if d_size > 0 else 0
        ti = int(np.searchsorted(sit_order, i) // d_size) if d_size > 0 else 0
    # Vectorized version:
    saf_ranks = np.argsort(np.argsort(saf_v))
    sit_ranks = np.argsort(np.argsort(sit_v))
    si_idx = np.clip(saf_ranks // d_size, 0, 9) if d_size > 0 else np.zeros(n, dtype=int)
    ti_idx = np.clip(sit_ranks // d_size, 0, 9) if d_size > 0 else np.zeros(n, dtype=int)
    for d in range(10):
        for t in range(10):
            mask = (si_idx == d) & (ti_idx == t)
            if np.sum(mask) < 10:
                continue
            pred = np.where(tpi_v[mask] > overall_med, 1, -1)
            grid[d, t] = float(np.mean(pred == lv[mask]))
            counts[d, t] = int(np.sum(mask))
    return grid, counts

def print_grid(grid, counts, title):
    print(f"\n  {title}:")
    header = "      " + "".join(f" SIT{i:2d}" for i in range(10))
    print(header)
    for d in range(10):
        row = f" SAF{d:2d} "
        for t in range(10):
            v = grid[d, t]
            if not np.isnan(v):
                row += f" {v*100:3.0f}"
            else:
                row += "  --"
        print(row)

def persistence_interaction(data, labels):
    """Phase 3: TPI accuracy conditioned on SIT persistence."""
    n = len(data["tpi"])
    sit_high = data["sit"] > np.nanpercentile(data["sit"], 66)
    sit_persist1 = sit_high.copy().astype(float)  # 1 bar
    sit_persist2 = np.full(n, np.nan)  # ≥2 bars
    sit_persist3 = np.full(n, np.nan)  # ≥3 bars
    streak = 0
    for i in range(n):
        if np.isnan(sit_high[i]):
            streak = 0
            continue
        if sit_high[i]:
            streak += 1
        else:
            streak = 0
        sit_persist2[i] = 1.0 if streak >= 2 else 0.0
        sit_persist3[i] = 1.0 if streak >= 3 else 0.0
    results = {}
    for name, cond in [("1_bar", sit_persist1), ("2_bars", sit_persist2), ("3_bars", sit_persist3)]:
        mask = cond > 0.5
        if np.sum(mask) < 30:
            continue
        acc, n_s = tpi_acc(data["tpi"][mask], labels[mask])
        results[name] = {"accuracy": acc, "n": n_s}
    return results

def tpi_persistence(data, labels):
    """Phase 4: TPI accuracy conditioned on sustained TPI direction."""
    n = len(data["tpi"])
    tpi_pos = data["tpi"] > 0
    tpi_p1 = tpi_pos.copy()
    tpi_p2 = np.full(n, False)
    tpi_p3 = np.full(n, False)
    streak = 0
    for i in range(n):
        if np.isnan(tpi_pos[i]):
            streak = 0; continue
        streak = streak + 1 if tpi_pos[i] else 0
        tpi_p2[i] = streak >= 2
        tpi_p3[i] = streak >= 3
    results = {}
    for name, cond in [("1_bar", tpi_p1), ("2_bars", tpi_p2), ("3_bars", tpi_p3)]:
        mask = cond
        if np.sum(mask) < 30:
            continue
        # Directional accuracy: do positive TPI bars with persistence predict UP?
        valid = ~np.isnan(labels[mask])
        if np.sum(valid) < 30:
            continue
        lv = labels[mask][valid]
        up_acc = float(np.mean(lv > 0))
        results[name] = {"up_accuracy": up_acc, "n": int(np.sum(valid))}
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("DPL-17: Flow × Volatility Interaction")
    print("=" * 60)

    all_data = {}
    for sym in SYMBOLS:
        d = prepare(sym)
        if d:
            all_data[sym] = d
            base_acc, n = tpi_acc(d["tpi"], d["labels_3"])
            print(f"  {sym}: {d['n']:,} bars, TPI baseline acc={base_acc:.4f} (n={n})")

    report = {"baseline": {sym: tpi_acc(d["tpi"], d["labels_3"])[0] for sym, d in all_data.items()}}

    # Phase 1: Accuracy by state decile
    print(f"\n{'='*60}")
    print(f"PHASE 1: CONDITIONAL ACCURACY BY STATE DECILE")
    print(f"{'='*60}")
    for state_name in ["saf", "sit", "vem"]:
        print(f"\n  --- {state_name.upper()} ---")
        for sym, d in all_data.items():
            r = decile_accuracy(d, state_name, d[state_name], d["labels_3"])
            if r:
                vals = [v["accuracy"] for v in r.values()]
                mn, mx = min(vals), max(vals)
                print(f"    {sym:8s}  range=[{mn:.4f}, {mx:.4f}]  spread={mx-mn:.4f}")
                report.setdefault(f"decile_{state_name}", {})[sym] = r

    # Phase 2: SAF × SIT grid
    print(f"\n{'='*60}")
    print(f"PHASE 2: STATE TOPOLOGY (SAF × SIT)")
    print(f"{'='*60}")
    for sym, d in all_data.items():
        grid, counts = state_grid_accuracy(d, d["labels_3"])
        if grid is not None:
            print(f"\n  {sym}:")
            print_grid(grid, counts, "TPI Accuracy %")
            # Find best/worst regions
            valid_mask = ~np.isnan(grid)
            if valid_mask.any():
                best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
                worst_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
                print(f"    Best: SAF{best_idx[0]}×SIT{best_idx[1]} = {grid[best_idx]:.3f}")
                print(f"    Worst: SAF{worst_idx[0]}×SIT{worst_idx[1]} = {grid[worst_idx]:.3f}")
            report.setdefault("grid", {})[sym] = {
                "grid_vals": grid.tolist() if grid is not None else None,
                "counts": counts.tolist() if counts is not None else None,
            }

    # Phase 3: SIT persistence
    print(f"\n{'='*60}")
    print(f"PHASE 3: SIT PERSISTENCE INTERACTION")
    print(f"{'='*60}")
    for sym, d in all_data.items():
        r = persistence_interaction(d, d["labels_3"])
        print(f"  {sym}:")
        for k, v in r.items():
            if v["n"] > 0:
                print(f"    SIT high {k:8s}  acc={v['accuracy']:.4f}  n={v['n']}")
        report.setdefault("sit_persistence", {})[sym] = r

    # Phase 4: TPI persistence
    print(f"\n{'='*60}")
    print(f"PHASE 4: TPI PERSISTENCE")
    print(f"{'='*60}")
    for sym, d in all_data.items():
        r = tpi_persistence(d, d["labels_3"])
        print(f"  {sym}:")
        for k, v in r.items():
            if v["n"] > 0:
                print(f"    TPI pos {k:8s}  up_acc={v['up_accuracy']:.4f}  n={v['n']}")
        report.setdefault("tpi_persistence", {})[sym] = r

    with open(os.path.join(REPORT_DIR, "dpl17_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDPL-17 results -> dpl17_results.json")
