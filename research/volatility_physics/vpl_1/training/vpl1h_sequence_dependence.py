import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, realized_variance_from_log_returns, SYMBOLS_FULL,
)
from core.sit_engine import compute_sit
from core.vcm_engine import compute_vcm

np.set_printoptions(precision=4, suppress=True)


def build_features(symbol):
    data = load_m5(symbol)
    c, h, l, ts = data["close"], data["high"], data["low"], data["timestamp"]
    r = compute_returns(c)
    rv = realized_variance_from_log_returns(r, 24)
    sav = compute_crf(c, h, l)
    saf_val = sav["crf"]
    sit_out = compute_sit(c, h, l, r, rv, saf_val)
    sit_val = sit_out["instability"]
    return c, saf_val, sit_val, ts


def analyze_sequence(symbol):
    c, saf, sit, ts = build_features(symbol)
    r = compute_returns(c)
    exp = compute_expansion_labels(c, baseline_window=24, horizons=(12,), thresholds=(1.5,))
    label = exp["expand_1.5_12"]["label"]
    n_min = min(len(saf), len(label))
    saf, sit_val, lab, ts_f = saf[:n_min], sit[:n_min], label[:n_min], ts[:n_min]
    keep = ~(np.isnan(saf) | np.isnan(sit_val) | np.isnan(lab))
    saf, sit_val, lab = saf[keep], sit_val[keep], lab[keep]

    uncond = float(np.mean(lab))

    saf_z = (saf - np.mean(saf)) / max(np.std(saf), 1e-12)
    sit_z = (sit_val - np.mean(sit_val)) / max(np.std(sit_val), 1e-12)

    saf_high = saf_z > 0
    sit_high = sit_z > 0

    state_codes = saf_high.astype(int) * 2 + sit_high.astype(int)
    labels = {0: "LOW_SAF+LOW_SIT", 1: "LOW_SAF+HIGH_SIT",
              2: "HIGH_SAF+LOW_SIT", 3: "HIGH_SAF+HIGH_SIT"}

    n = len(state_codes)
    lookback = 10

    results = {}
    for final_state in range(4):
        state_name = labels[final_state]
        count = 0
        entry = {
            "state": state_name, "uncond": uncond,
            "paths": {}
        }
        path_results = {}
        for i in range(lookback, n):
            if state_codes[i] != final_state:
                continue
            for look in [3, 5, 10]:
                if i < look:
                    continue
                path_key = tuple(state_codes[i - look:i].tolist())
                if path_key not in path_results:
                    path_results[path_key] = {"count": 0, "total": 0.0}
                path_results[path_key]["count"] += 1
                path_results[path_key]["total"] += lab[i]
                count += 1
        sorted_paths = sorted(path_results.items(), key=lambda x: x[1]["count"], reverse=True)
        top_paths = []
        for path_key, pdata in sorted_paths[:10]:
            freq = pdata["total"] / max(pdata["count"], 1)
            top_paths.append({
                "path": [labels[s] for s in path_key],
                "count": pdata["count"],
                "freq": round(float(freq), 4),
                "lift": round(float(freq / max(uncond, 0.001)), 4),
            })
        entry["total_count"] = count
        entry["top_paths"] = top_paths
        results[state_name] = entry

    return {"symbol": symbol, "n": n, "unconditional": round(uncond, 4), "states": results}


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS_FULL[:2]:  # Test only EURJPY and NAS100 first
        print(f"\n=== VPL-1H: {sym} ===")
        t0 = time.time()
        res = analyze_sequence(sym)
        all_results[sym] = res
        elapsed = time.time() - t0
        print(f"  N={res['n']} ({elapsed:.1f}s)")
        print(f"  Unconditional: {res['unconditional']:.4f}")
        for state_name, state_data in res["states"].items():
            print(f"\n  State: {state_name} (n={state_data['total_count']})")
            for path in state_data["top_paths"][:5]:
                print(f"    path={path['path'][-3:]}  freq={path['freq']:.4f}  lift={path['lift']:.4f}x  cnt={path['count']}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1h_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
