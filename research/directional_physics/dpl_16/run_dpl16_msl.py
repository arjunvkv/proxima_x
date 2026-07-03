import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_all_msl_features, aggregate_features_to_bars,
    aggregate_to_bars_vectorized, compute_directional_labels_vect,
    evaluate_directional_accuracy, compute_accuracy_by_quintile,
    cross_pair_evaluation, normalize,
)

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS_TICK = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]
FEATURE_GROUPS = {
    "tpi": ["tpi_50", "tpi_100", "tpi_200", "tpi_persistence_50", "tpi_persistence_100", "tpi_persistence_200", "tpi_accel"],
    "ssf": ["sweep_intensity_20", "sweep_intensity_50", "sweep_intensity_100"],
    "rap": ["rap_100", "rap_200"],
    "mff": ["alt_10", "alt_30", "rev_10", "rev_30", "failed_push_10", "failed_push_30"],
    "tfc": ["tfc_50", "tfc_100"],
}
ALL_FEATURE_NAMES = sum(FEATURE_GROUPS.values(), [])

def run_symbol(symbol):
    print(f"\n{'='*60}")
    print(f"DPL-16: {symbol}")
    print(f"{'='*60}")

    ticks = load_ticks(symbol, stride=1)
    m5 = load_m5(symbol)
    print(f"  Ticks: {ticks['n']:,} | M5 bars: {m5['n']:,}")

    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], m5["timestamp"])
    n_bars = len(starts)

    # Assign each tick to its bar index
    tick_bar_idx = np.searchsorted(m5["timestamp"], tick_sec, side='right') - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)

    # Compute tick-level features
    features = compute_all_msl_features(ticks["mid"], ticks["timestamp"])

    # Aggregate to bar level using vectorized groupby
    bar_features = aggregate_features_to_bars(features, tick_bar_idx, n_bars)

    # Directional labels
    labels_1 = compute_directional_labels_vect(m5["close"], n_bars=1)
    labels_3 = compute_directional_labels_vect(m5["close"], n_bars=3)

    n_use = min(n_bars, len(labels_1), len(labels_3))
    for k in bar_features:
        bar_features[k] = bar_features[k][:n_use]
    labels_1 = labels_1[:n_use]
    labels_3 = labels_3[:n_use]

    fm = np.column_stack([bar_features[k] for k in ALL_FEATURE_NAMES])

    acc_1 = evaluate_directional_accuracy(fm, labels_1, ALL_FEATURE_NAMES, min_samples=50)
    acc_3 = evaluate_directional_accuracy(fm, labels_3, ALL_FEATURE_NAMES, min_samples=50)

    print(f"\n  --- Directional Accuracy (1-bar) ---")
    best_1 = ("", 0.0)
    for fn in ALL_FEATURE_NAMES:
        r = acc_1.get(fn, {})
        acc = r.get("accuracy", np.nan)
        n = r.get("n", 0)
        if not np.isnan(acc):
            mark = " >>>" if acc > 0.53 else ""
            print(f"    {fn:30s}  acc={acc:.4f}  n={n}{mark}")
            if acc > best_1[1]:
                best_1 = (fn, acc)
        else:
            print(f"    {fn:30s}  ERR: {r.get('error', 'unknown')}")

    print(f"\n  --- Directional Accuracy (3-bar) ---")
    best_3 = ("", 0.0)
    for fn in ALL_FEATURE_NAMES:
        r = acc_3.get(fn, {})
        acc = r.get("accuracy", np.nan)
        n = r.get("n", 0)
        if not np.isnan(acc):
            mark = " >>>" if acc > 0.53 else ""
            print(f"    {fn:30s}  acc={acc:.4f}  n={n}{mark}")
            if acc > best_3[1]:
                best_3 = (fn, acc)
        else:
            print(f"    {fn:30s}  ERR: {r.get('error', 'unknown')}")

    return {
        "symbol": symbol,
        "n_ticks": ticks["n"],
        "n_bars": n_use,
        "accuracy_1bar": acc_1,
        "accuracy_3bar": acc_3,
        "best_1bar": best_1,
        "best_3bar": best_3,
        "fm": fm,
        "labels_3": labels_3,
        "bar_features": bar_features,
    }


def run_quintile_analysis(symbol, bar_features, labels_3):
    print(f"\n  --- Quintile Analysis ({symbol}) ---")
    results = {}
    for fn in ALL_FEATURE_NAMES:
        q = compute_accuracy_by_quintile(bar_features[fn], labels_3, n_quintiles=5)
        if q:
            results[fn] = q
            if abs(q["q4"]["accuracy"] - q["q0"]["accuracy"]) > 0.02:
                print(f"    {fn:30s}  Q0={q['q0']['accuracy']:.4f}  Q4={q['q4']['accuracy']:.4f}  spread={abs(q['q4']['accuracy'] - q['q0']['accuracy']):.4f}")
    return results


def run_combined_predictor(symbol, bar_features, labels_3):
    signals_avg = np.zeros(len(labels_3))
    count = 0
    for fn in ALL_FEATURE_NAMES:
        feat = bar_features[fn]
        valid = ~np.isnan(feat) & ~np.isnan(labels_3)
        if np.sum(valid) < 100:
            continue
        median = np.nanmedian(feat[valid])
        sig = np.where(feat > median, 1, -1)
        sig[~valid] = 0
        signals_avg += sig
        count += 1
    if count > 0:
        signals_avg /= count
        valid = ~np.isnan(signals_avg) & ~np.isnan(labels_3)
        if np.sum(valid) > 0:
            acc = np.mean(np.sign(signals_avg[valid]) == labels_3[valid])
            print(f"  {symbol:8s}  Combined acc={acc:.4f}  n={int(np.sum(valid))}")
            return acc
    return np.nan


def run_cross_year(symbol, bar_features, labels_3):
    m5 = load_m5(symbol)
    from datetime import datetime, timezone
    m5_times = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in m5["timestamp"][:len(labels_3)]])
    years = sorted(set(t.year for t in m5_times))

    print(f"\n  --- {symbol}: Cross-year ---")
    fm_all = np.column_stack([bar_features[k] for k in ALL_FEATURE_NAMES])
    for year in years:
        mask = np.array([t.year == year for t in m5_times])
        if np.sum(mask) < 50:
            continue
        fm_y = fm_all[mask]
        lb_y = labels_3[mask]
        acc = evaluate_directional_accuracy(fm_y, lb_y, ALL_FEATURE_NAMES, min_samples=10)
        vals = [v.get("accuracy", 0) for v in acc.values() if not np.isnan(v.get("accuracy", np.nan))]
        best = max(vals) if vals else 0
        print(f"    {year}: {int(np.sum(mask)):,} bars, best_acc={best:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("DPL-16: Microstructure Shadow Layer (MSL)")
    print("=" * 60)

    all_results = {}
    for symbol in SYMBOLS_TICK:
        r = run_symbol(symbol)
        all_results[symbol] = r

    # Cross-pair
    print(f"\n{'='*60}")
    print(f"CROSS-PAIR STABILITY")
    print(f"{'='*60}")
    pair_data = {sym: all_results[sym]["accuracy_3bar"] for sym in SYMBOLS_TICK}
    cross = cross_pair_evaluation(pair_data, ALL_FEATURE_NAMES)
    stable = 0
    for fn in sorted(cross.keys()):
        c = cross[fn]
        tag = "STABLE" if c["stable"] else "UNSTABLE"
        if c["stable"]: stable += 1
        print(f"  {fn:30s}  {tag:8s}  mean={c['mean_acc']:.4f}  [{c['min_acc']:.4f}-{c['max_acc']:.4f}]")
    print(f"  Stable: {stable}/{len(cross)}")

    # Cross-year (EURJPY only)
    print(f"\n{'='*60}")
    print(f"CROSS-YEAR STABILITY (EURJPY)")
    print(f"{'='*60}")
    run_cross_year("EURJPY", all_results["EURJPY"]["bar_features"], all_results["EURJPY"]["labels_3"])

    # Quintile analysis
    print(f"\n{'='*60}")
    print(f"QUINTILE ANALYSIS (EURJPY)")
    print(f"{'='*60}")
    run_quintile_analysis("EURJPY", all_results["EURJPY"]["bar_features"], all_results["EURJPY"]["labels_3"])

    # Combined predictor
    print(f"\n{'='*60}")
    print(f"COMBINED MSL PREDICTOR (3-bar)")
    print(f"{'='*60}")
    for sym in SYMBOLS_TICK:
        run_combined_predictor(sym, all_results[sym]["bar_features"], all_results[sym]["labels_3"])

    # Summary
    print(f"\n{'='*60}")
    print(f"BEST FEATURE PER SYMBOL (3-bar)")
    print(f"{'='*60}")
    for sym in SYMBOLS_TICK:
        fn, acc = all_results[sym]["best_3bar"]
        if acc > 0:
            print(f"  {sym:8s}  {fn:30s}  acc={acc:.4f}")

    # Save summary JSON (without large arrays)
    summary = {"purpose": "DPL-16 MSL: Directional accuracy from microstructure features"}
    for sym in SYMBOLS_TICK:
        acc = all_results[sym]["accuracy_3bar"]
        summary[sym] = {k: v for k, v in acc.items() if isinstance(v, dict) and not np.isnan(v.get("accuracy", np.nan))}
        summary[f"{sym}_best"] = list(all_results[sym]["best_3bar"])

    summary["cross_pair"] = cross

    with open(os.path.join(REPORT_DIR, "dpl16_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved -> dpl16_summary.json")
    print("DPL-16 MSL analysis complete.")
