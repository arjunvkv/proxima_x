import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, compute_decile_expansion_profile,
    realized_variance_from_log_returns, normalize, SYMBOLS_FULL,
)
from core.sit_engine import compute_sit, compute_grid_profile

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def analyze_symbol(symbol):
    data = load_m5(symbol)
    c, h, l = data["close"], data["high"], data["low"]
    r = compute_returns(c)
    n_close = len(c)
    n_ret = len(r)

    rv = realized_variance_from_log_returns(r, 24)
    rv_close = np.full(n_close, np.nan)
    rv_close[1:] = rv

    sav = compute_crf(c, h, l)
    saf_val = sav["crf"]
    saf_dec = compute_crf_deciles(saf_val)

    sit_out = compute_sit(c, h, l, r, rv, saf_val)
    I, J, A = sit_out["instability"], sit_out["jerk"], sit_out["acceleration"]

    sit_dec = compute_crf_deciles(I)
    jerk_dec = compute_crf_deciles(J)
    accel_dec = compute_crf_deciles(A)

    exp = compute_expansion_labels(c, baseline_window=24, horizons=(12,), thresholds=(1.5,))
    label_key = "expand_1.5_12"
    label = exp[label_key]["label"]

    max_valid = min(len(sit_dec), len(label))
    sit_dec_t = sit_dec[:max_valid]
    jerk_dec_t = jerk_dec[:max_valid]
    accel_dec_t = accel_dec[:max_valid]
    saf_dec_t = saf_dec[:max_valid]
    label_t = label[:max_valid]

    # 1. SIT deciles alone
    sit_profile = compute_decile_expansion_profile(sit_dec_t, label_t)

    # 2. Jerk deciles alone
    jerk_profile = compute_decile_expansion_profile(jerk_dec_t, label_t)

    # 3. Acceleration deciles alone
    accel_profile = compute_decile_expansion_profile(accel_dec_t, label_t)

    # 4. SAF × SIT grid
    saf_sit_grid, saf_sit_count = compute_grid_profile(saf_dec_t, sit_dec_t, label_t)

    # 5. SAF × Jerk grid
    saf_jerk_grid, saf_jerk_count = compute_grid_profile(saf_dec_t, jerk_dec_t, label_t)

    # 6. SAF × Acceleration grid
    saf_accel_grid, saf_accel_count = compute_grid_profile(saf_dec_t, accel_dec_t, label_t)

    unconditional = np.nanmean(label)

    result = {
        "symbol": symbol,
        "n": n_close,
        "unconditional_expansion": round(float(unconditional), 4),
        "sit_decile_profile": [
            {"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]}
            for p in sit_profile
        ],
        "jerk_decile_profile": [
            {"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]}
            for p in jerk_profile
        ],
        "accel_decile_profile": [
            {"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]}
            for p in accel_profile
        ],
        "saf_sit_grid": saf_sit_grid.tolist(),
        "saf_sit_counts": saf_sit_count.tolist(),
        "saf_jerk_grid": saf_jerk_grid.tolist(),
        "saf_jerk_counts": saf_jerk_count.tolist(),
        "saf_accel_grid": saf_accel_grid.tolist(),
        "saf_accel_counts": saf_accel_count.tolist(),
    }
    return result


def print_grid(grid, counts, title):
    print(f"  {title}:")
    print(f"       S0    S1    S2    S3    S4    S5    S6    S7    S8    S9")
    for si in range(10):
        row = f"  SAF{si}:"
        for tj in range(10):
            v = grid[si][tj]
            row += f" {v:.3f}" if not np.isnan(v) else "  nan"
        row += f" | avg={np.nanmean(grid[si]):.4f}"
        print(row)
    print(f"  Column avg:", end="")
    for tj in range(10):
        vals = [grid[si][tj] for si in range(10) if not np.isnan(grid[si][tj])]
        avg = np.mean(vals) if vals else np.nan
        print(f" {avg:.3f}" if not np.isnan(avg) else "  nan", end="")
    print()


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS_FULL:
        print(f"\n=== VPL-1C: {sym} ===")
        t0 = time.time()
        res = analyze_symbol(sym)
        all_results[sym] = res
        elapsed = time.time() - t0
        print(f"  N={res['n']} ({elapsed:.1f}s)")
        print(f"  Unconditional expansion (1.5x, N=12): {res['unconditional_expansion']:.4f}")

        print(f"  SIT decile profile:")
        for p in res["sit_decile_profile"]:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")

        print(f"  Jerk decile profile:")
        for p in res["jerk_decile_profile"]:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")

        print(f"  Accel decile profile:")
        for p in res["accel_decile_profile"]:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")

        grid = np.array(res["saf_sit_grid"])
        counts = np.array(res["saf_sit_counts"])
        print_grid(grid, counts, "SAF × SIT grid (expansion freq)")

        grid_j = np.array(res["saf_jerk_grid"])
        print_grid(grid_j, np.array(res["saf_jerk_counts"]), "SAF × Jerk grid")

        grid_a = np.array(res["saf_accel_grid"])
        print_grid(grid_a, np.array(res["saf_accel_counts"]), "SAF × Accel grid")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1c_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
