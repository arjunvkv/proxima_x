"""DPL-16F/G/H: Robustness audits for TPI signal.
Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl16_robustness.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_all_msl_features, aggregate_features_to_bars,
    compute_directional_labels_vect,
    evaluate_directional_accuracy,
)

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# We only care about TPI_200 as the primary candidate
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]
FEATURES_KEY = "tpi_200"

def prepare_data(symbol):
    """Load ticks, align to M5, return overlapping features + labels."""
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], m5["timestamp"])
    n_bars = len(starts)

    tick_bar_idx = np.searchsorted(m5["timestamp"], tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)

    features = compute_all_msl_features(ticks["mid"], ticks["timestamp"])
    bar_features = aggregate_features_to_bars(features, tick_bar_idx, n_bars)

    # Only use bars that have valid feature data (the overlapping range)
    feat = bar_features[FEATURES_KEY]
    first_valid = np.where(~np.isnan(feat))[0]
    if len(first_valid) == 0:
        print(f"  {symbol}: NO VALID DATA")
        return None
    s, e = first_valid[0], first_valid[-1] + 1
    return {
        "symbol": symbol,
        "m5_ts": m5["timestamp"][s:e],
        "m5_close": m5["close"][s:e],
        "tpi": feat[s:e],
        "n": e - s,
    }

def acc_from_tpi(tpi, labels, min_samples=50):
    """Directional accuracy using above/below median split."""
    valid = ~np.isnan(tpi) & ~np.isnan(labels)
    if np.sum(valid) < min_samples:
        return np.nan, 0
    fv = tpi[valid]
    lv = labels[valid]
    med = np.nanmedian(fv)
    pred = np.where(fv > med, 1, -1)
    return float(np.mean(pred == lv)), int(np.sum(valid))


def horizon_decay(symbol, data):
    """DPL-16F: Test TPI_200 accuracy at 1,2,3,6,12 bar horizons."""
    horizons = [1, 2, 3, 6, 12]
    results = {}
    for h in horizons:
        labels = compute_directional_labels_vect(data["m5_close"], n_bars=h)
        n_use = min(len(data["tpi"]), len(labels))
        acc, n = acc_from_tpi(data["tpi"][:n_use], labels[:n_use])
        results[f"h{h}"] = {"accuracy": acc, "n": n}
        if not np.isnan(acc):
            mark = " >>>" if acc > 0.52 else ""
            print(f"    {symbol:8s}  h={h:2d}  acc={acc:.4f}  n={n}{mark}")
    return results


def purged_walk_forward(symbol, data, train_bars=8640, test_bars=2880):
    """DPL-16G: 3-month train / 1-month test rolling windows.
    8640 M5 bars ≈ 30 days * 24h * 12 bars/h
    2880 M5 bars ≈ 10 days * 24h * 12
    Total train+test ≈ 11520 bars ≈ 40 days per window
    """
    n = len(data["tpi"])
    labels = compute_directional_labels_vect(data["m5_close"], n_bars=3)
    n_use = min(n, len(labels))
    tpi = data["tpi"][:n_use]
    labels = labels[:n_use]

    window_size = train_bars + test_bars
    step = test_bars
    results = []

    for start in range(0, n_use - window_size, step):
        train_s = start
        train_e = start + train_bars
        test_s = train_e
        test_e = test_s + test_bars

        train_tpi = tpi[train_s:train_e]
        train_lbl = labels[train_s:train_e]
        test_tpi = tpi[test_s:test_e]
        test_lbl = labels[test_s:test_e]

        valid_t = ~np.isnan(train_tpi) & ~np.isnan(train_lbl)
        if np.sum(valid_t) < 100:
            continue
        med = np.nanmedian(train_tpi[valid_t])

        valid_te = ~np.isnan(test_tpi) & ~np.isnan(test_lbl)
        if np.sum(valid_te) < 20:
            continue
        pred = np.where(test_tpi[valid_te] > med, 1, -1)
        acc = np.mean(pred == test_lbl[valid_te])
        results.append(acc)

    if results:
        print(f"    {symbol:8s}  windows={len(results)}  mean_acc={np.mean(results):.4f}  "
              f"min={np.min(results):.4f}  max={np.max(results):.4f}  "
              f"positive_windows={np.mean([r > 0.5 for r in results]):.0%}")
    return results


def temporal_stability(symbol, data):
    """DPL-16H: Check TPI accuracy by half-year periods."""
    from datetime import datetime, timezone
    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in data["m5_ts"]]
    labels = compute_directional_labels_vect(data["m5_close"], n_bars=3)
    n_use = min(len(data["tpi"]), len(labels))
    tpi = data["tpi"][:n_use]
    labels = labels[:n_use]
    times = times[:n_use]

    valid = ~np.isnan(tpi) & ~np.isnan(labels)
    tpi = tpi[valid]
    labels = labels[valid]
    times = [t for t, v in zip(times, valid) if v]
    med = np.nanmedian(tpi)

    # Group by half-year
    periods = {}
    for t, f, l in zip(times, tpi, labels):
        key = f"{t.year}-H{1 if t.month <= 6 else 2}"
        if key not in periods:
            periods[key] = {"tpi": [], "lbl": []}
        periods[key]["tpi"].append(f)
        periods[key]["lbl"].append(l)

    results = {}
    for period in sorted(periods.keys()):
        p = periods[period]
        fv = np.array(p["tpi"])
        lv = np.array(p["lbl"])
        pred = np.where(fv > med, 1, -1)
        acc = np.mean(pred == lv)
        results[period] = {"accuracy": acc, "n": len(fv)}
        mark = " >>>" if acc > 0.52 else ""
        print(f"    {symbol:8s}  {period:8s}  acc={acc:.4f}  n={len(fv)}{mark}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DPL-16: Robustness Audits (F/G/H)")
    print("=" * 60)

    all_data = {}
    for sym in SYMBOLS:
        d = prepare_data(sym)
        if d:
            all_data[sym] = d
            print(f"  {sym}: {d['n']:,} overlapping bars")

    # DPL-16F: Horizon decay
    print(f"\n{'='*60}")
    print(f"DPL-16F: HORIZON DECAY AUDIT")
    print(f"{'='*60}")
    horizon_results = {}
    for sym in SYMBOLS:
        if sym in all_data:
            h = horizon_decay(sym, all_data[sym])
            horizon_results[sym] = h

    # DPL-16G: Purged walk-forward
    print(f"\n{'='*60}")
    print(f"DPL-16G: PURGED WALK-FORWARD")
    print(f"{'='*60}")
    walk_results = {}
    for sym in SYMBOLS:
        if sym in all_data:
            w = purged_walk_forward(sym, all_data[sym])
            walk_results[sym] = w

    # DPL-16H: Temporal stability
    print(f"\n{'='*60}")
    print(f"DPL-16H: TEMPORAL STABILITY")
    print(f"{'='*60}")
    temporal_results = {}
    for sym in SYMBOLS:
        if sym in all_data:
            t = temporal_stability(sym, all_data[sym])
            temporal_results[sym] = t

    # Save results
    report = {
        "horizon_decay": horizon_results,
        "walk_forward": {k: {"mean_acc": float(np.mean(v)) if v else None,
                             "min": float(np.min(v)) if v else None,
                             "max": float(np.max(v)) if v else None,
                             "windows": len(v)} for k, v in walk_results.items()},
        "temporal": temporal_results,
    }
    with open(os.path.join(REPORT_DIR, "dpl16_robustness.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nRobustness results saved -> dpl16_robustness.json")
