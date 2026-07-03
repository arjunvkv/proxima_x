import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, compute_decile_expansion_profile,
    compute_base_expansion_rate, compute_forward_variance_expansion,
    realized_variance_from_log_returns, SYMBOLS_FULL,
)

np.set_printoptions(precision=4, suppress=True)


def analyze_symbol(symbol):
    data = load_m5(symbol)
    c, h, l = data["close"], data["high"], data["low"]
    r = compute_returns(c)
    base_rates = compute_base_expansion_rate(c)
    baseline_n = 24
    crf = compute_crf(c, h, l)
    crf_dec = compute_crf_deciles(crf["crf"])
    labels = compute_expansion_labels(c, baseline_window=baseline_n)
    exp_profiles = {}
    for key in sorted(labels.keys()):
        label = labels[key]["label"]
        max_valid = min(len(crf_dec), len(label))
        exp_profiles[key] = compute_decile_expansion_profile(
            crf_dec[:max_valid], label[:max_valid]
        )
    top_bottom = {}
    for key in sorted(exp_profiles.keys()):
        profile = exp_profiles[key]
        top = profile[9]["freq"] if profile[9]["count"] > 0 else 0
        bot = profile[0]["freq"] if profile[0]["count"] > 0 else 0
        ratio = top / bot if bot > 0 else np.nan
        top_bottom[key] = {"top_decile_freq": top, "bottom_decile_freq": bot, "ratio": ratio}
    fwd = compute_forward_variance_expansion(r, baseline_window=baseline_n, forward_horizons=(12,))
    result = {
        "symbol": symbol,
        "n": len(c),
        "date_range": f"{data['timestamp'][0]} - {data['timestamp'][-1]}",
        "base_expansion_rates": {k: round(float(v), 4) for k, v in base_rates.items()},
        "forward_expansion": {
            str(N): {
                "mean_er": round(float(v["mean_er"]), 4),
                "p50": round(float(v["p50"]), 4),
                "p90": round(float(v["p90"]), 4),
            }
            for N, v in fwd.items()
        },
        "crf_decile_top_bottom_ratio": {
            k: {
                "top_decile_freq": round(float(v["top_decile_freq"]), 4),
                "bottom_decile_freq": round(float(v["bottom_decile_freq"]), 4),
                "ratio": round(float(v["ratio"]), 4) if not np.isnan(v["ratio"]) else None,
            }
            for k, v in top_bottom.items()
        },
        "expansion_profiles": {
            k: [
                {"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]}
                for p in profile
            ]
            for k, profile in exp_profiles.items()
        },
    }
    return result


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS_FULL:
        print(f"\n=== VPL-1A: {sym} ===")
        t0 = time.time()
        res = analyze_symbol(sym)
        all_results[sym] = res
        elapsed = time.time() - t0
        print(f"  N={res['n']} ({elapsed:.1f}s)")
        print(f"  Base expansion rates:")
        for k, v in res["base_expansion_rates"].items():
            print(f"    {k}: {v:.4f}")
        print(f"  Forward expansion:")
        for N_str, v in res["forward_expansion"].items():
            print(f"    N={N_str}: mean_er={v['mean_er']:.4f}, p50={v['p50']:.4f}, p90={v['p90']:.4f}")
        print(f"  CRF decile lifts:")
        for k, v in res["crf_decile_top_bottom_ratio"].items():
            print(f"    {k}: top={v['top_decile_freq']:.4f} bot={v['bottom_decile_freq']:.4f} ratio={v['ratio']}")
        profile_key = sorted(res["expansion_profiles"].keys())[0]
        profile = res["expansion_profiles"][profile_key]
        print(f"  Decile profile ({profile_key}):")
        for p in profile:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1a_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
