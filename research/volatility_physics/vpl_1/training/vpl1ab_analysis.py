import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, compute_decile_expansion_profile,
    compute_forward_variance_expansion, realized_variance_from_log_returns,
    normalize, SYMBOLS_FULL,
)
from core.vcm_engine import compute_vcm

np.set_printoptions(precision=4, suppress=True)


def compute_contraction_labels(close, baseline_window=24, horizons=(12,), thresholds=(0.75, 0.50)):
    r = compute_returns(close)
    n = len(r)
    result = {}
    for N in horizons:
        bv = realized_variance_from_log_returns(r, baseline_window)
        fv = realized_variance_from_log_returns(r, N)
        er = np.full(n, np.nan)
        max_i = n - N - 1
        for i in range(baseline_window, max_i):
            if bv[i] > 0 and not np.isnan(bv[i]) and not np.isnan(fv[i + 1]):
                er[i] = fv[i + 1] / bv[i]
        for thresh in thresholds:
            label = np.full(n, np.nan)
            valid = ~np.isnan(er)
            label[valid] = (er[valid] < thresh).astype(np.float64)
            result[f"contract_{thresh}_{N}"] = {"er": er, "label": label}
    return result


def analyze_symbol(symbol):
    data = load_m5(symbol)
    c, h, l = data["close"], data["high"], data["low"]
    r = compute_returns(c)
    n_close = len(c)
    n_ret = len(r)

    # SAF (renamed from CRF)
    sav = compute_crf(c, h, l)
    saf_val = sav["crf"]
    saf_dec = compute_crf_deciles(saf_val)

    # Contraction labels (returns space)
    cont = compute_contraction_labels(c, baseline_window=24, horizons=(12,), thresholds=(0.75,))
    cont_profiles = {}
    for key in sorted(cont.keys()):
        label = cont[key]["label"]
        max_valid = min(len(saf_dec), len(label))
        cont_profiles[key] = compute_decile_expansion_profile(saf_dec[:max_valid], label[:max_valid])

    # RV for VCM
    rv = realized_variance_from_log_returns(r, 24)
    rv_close = np.full(n_close, np.nan)
    rv_close[1:] = rv

    # Expansion labels (original)
    exp = compute_expansion_labels(c, baseline_window=24, horizons=(12,), thresholds=(1.5,))
    exp_profiles = {}
    for key in sorted(exp.keys()):
        label = exp[key]["label"]
        max_valid = min(len(saf_dec), len(label))
        exp_profiles[key] = compute_decile_expansion_profile(saf_dec[:max_valid], label[:max_valid])

    # VCM computation (on returns space)
    vcm_out = compute_vcm(r, rv)
    vcm_val = vcm_out["vcm"]
    vcm_close = np.full(n_close, np.nan)
    vcm_close[1:] = vcm_val
    vcm_dec = compute_crf_deciles(vcm_close)

    # VCM expansion profiles
    vcm_profiles = {}
    for key in sorted(exp.keys()):
        label = exp[key]["label"]
        max_valid = min(len(vcm_dec), len(label))
        vcm_profiles[key] = compute_decile_expansion_profile(vcm_dec[:max_valid], label[:max_valid])

    # Summary stats
    saf_top_bot = {}
    for key in sorted(cont_profiles.keys()):
        profile = cont_profiles[key]
        top = profile[9]["freq"] if profile[9]["count"] > 0 else 0
        bot = profile[0]["freq"] if profile[0]["count"] > 0 else 0
        saf_top_bot[key] = {"D9_freq": top, "D0_freq": bot, "ratio_D9_D0": top / bot if bot > 0 else None}

    vcm_top_bot = {}
    for key in sorted(vcm_profiles.keys()):
        profile = vcm_profiles[key]
        top = profile[9]["freq"] if profile[9]["count"] > 0 else 0
        bot = profile[0]["freq"] if profile[0]["count"] > 0 else 0
        vcm_top_bot[key] = {"D9_freq": top, "D0_freq": bot, "ratio_D9_D0": top / bot if bot > 0 else None}

    result = {
        "symbol": symbol,
        "n": n_close,
        "saf_contraction_profiles": {
            k: [{"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]} for p in profile]
            for k, profile in cont_profiles.items()
        },
        "saf_contraction_lift": saf_top_bot,
        "saf_expansion_profiles": {
            k: [{"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]} for p in profile]
            for k, profile in exp_profiles.items()
        },
        "vcm_profiles": {
            k: [{"decile": p["decile"], "freq": round(float(p["freq"]), 4), "count": p["count"]} for p in profile]
            for k, profile in vcm_profiles.items()
        },
        "vcm_lift": vcm_top_bot,
        "vcm_stats": {
            "vcm_mean": float(np.nanmean(vcm_val)),
            "vcm_std": float(np.nanstd(vcm_val)),
            "vmr_mean": float(np.nanmean(vcm_out["vmr"])),
            "burst_rate": float(np.nanmean(vcm_out["burst"])),
            "mean_duration": float(np.nanmean(vcm_out["duration"])),
        },
    }
    return result


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS_FULL:
        print(f"\n=== VPL-1AB: {sym} ===")
        t0 = time.time()
        res = analyze_symbol(sym)
        all_results[sym] = res
        elapsed = time.time() - t0
        print(f"  N={res['n']} ({elapsed:.1f}s)")

        print(f"  SAF CONTRACTION (contract_0.75_12):")
        prof = res["saf_contraction_profiles"]["contract_0.75_12"]
        for p in prof:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")
        lift = res["saf_contraction_lift"]["contract_0.75_12"]
        print(f"    D9/D0 ratio: {lift['ratio_D9_D0']}")

        print(f"  SAF EXPANSION (expand_1.5_12):")
        prof = res["saf_expansion_profiles"]["expand_1.5_12"]
        for p in prof:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")

        print(f"  VCM EXPANSION (expand_1.5_12):")
        prof = res["vcm_profiles"]["expand_1.5_12"]
        for p in prof:
            print(f"    D{p['decile']}: freq={p['freq']:.4f} (n={p['count']})")
        lift = res["vcm_lift"]["expand_1.5_12"]
        print(f"    D9/D0 ratio: {lift['ratio_D9_D0']}")

        print(f"  VCM stats: burst_rate={res['vcm_stats']['burst_rate']:.4f} mean_dur={res['vcm_stats']['mean_duration']:.2f}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1ab_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
