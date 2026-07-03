"""OMS-3: Cohort Synchronization Test — does synchronization count predict directional strength?"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.observable_market_state.oms_core import OMSCore, SYMBOLS
from research.directional_state.dsr_core import WalkForwardValidator, HORIZON_KEYS, HORIZONS

HORIZON_NAMES = ["H5", "H20", "H50"]
HORIZON_IDX = {hk: i for i, (h, hk) in enumerate(zip(HORIZONS, HORIZON_KEYS)) if hk in HORIZON_NAMES}
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def bucket_analysis(oms, sym):
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    marker = oms.marker_present(sym)

    n = min(len(sync), len(fut_ret), len(marker))
    sync, fut_ret, marker = sync[:n], fut_ret[:n], marker[:n]

    results = {}
    for bucket in range(1, 6):
        mask = sync == bucket
        cnt = int(np.sum(mask))
        if cnt < 10:
            results[f"sync_{bucket}"] = {"count": cnt, "insufficient": True}
            continue
        entry = {"count": cnt}
        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            fr = fut_ret[mask, col]
            valid = ~np.isnan(fr)
            if np.sum(valid) < 5:
                entry[hk] = {"p_up": np.nan, "mean_ret": np.nan, "std_ret": np.nan}
                continue
            p_up = float(np.mean(fr[valid] > 0))
            mean_ret = float(np.mean(fr[valid]))
            std_ret = float(np.std(fr[valid]))
            entry[hk] = {"p_up": round(p_up, 4), "mean_ret": round(mean_ret, 6), "std_ret": round(std_ret, 6)}
        results[f"sync_{bucket}"] = entry

    monotonicity = {}
    for hk in HORIZON_NAMES:
        p_ups = []
        valid_buckets = []
        for b in range(1, 6):
            r = results.get(f"sync_{b}", {})
            if r.get("insufficient") or r.get(hk, {}).get("p_up") is None:
                continue
            p_ups.append(r[hk]["p_up"])
            valid_buckets.append(b)
        if len(p_ups) >= 3:
            diffs = np.diff(p_ups)
            monotonicity[hk] = {
                "strictly_increasing": bool(np.all(diffs > 0)),
                "non_decreasing": bool(np.all(diffs >= 0)),
                "strictly_decreasing": bool(np.all(diffs < 0)),
                "n_buckets": len(p_ups),
                "buckets": valid_buckets,
                "p_up_sequence": [round(p, 4) for p in p_ups],
            }
    return results, monotonicity


def leading_indicator_test(oms, sym, lags=[5, 10]):
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(fut_ret))
    sync, fut_ret = sync[:n], fut_ret[:n]
    results = {}
    for lag in lags:
        lag_results = {}
        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            aligned_sync = sync[:-lag] if lag > 0 else sync
            aligned_ret = fut_ret[lag:, col] if lag > 0 else fut_ret[:, col]
            n_aligned = min(len(aligned_sync), len(aligned_ret))
            as_ = aligned_sync[:n_aligned]
            ar = aligned_ret[:n_aligned]
            valid = ~np.isnan(ar)
            as_, ar = as_[valid], ar[valid]
            bucket_results = {}
            for b in range(1, 6):
                m = as_ == b
                cnt = int(np.sum(m))
                if cnt < 10:
                    continue
                p_up = float(np.mean(ar[m] > 0))
                bucket_results[f"sync_{b}"] = {"count": cnt, "p_up": round(p_up, 4)}
            lag_results[hk] = bucket_results
        results[f"lag_{lag}"] = lag_results
    return results


def magnitude_test(oms, sym):
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(fut_ret))
    sync, fut_ret = sync[:n], fut_ret[:n]
    results = {}
    for hk in HORIZON_NAMES:
        bucket_results = {}
        for b in range(1, 6):
            mask = sync == b
            cnt = int(np.sum(mask))
            if cnt < 10:
                continue
            fr = fut_ret[mask, HORIZON_IDX[hk]]
            valid = ~np.isnan(fr)
            if np.sum(valid) < 5:
                continue
            mean_abs_ret = float(np.mean(np.abs(fr[valid])))
            mean_ret = float(np.mean(fr[valid]))
            bucket_results[f"sync_{b}"] = {
                "count": cnt,
                "mean_abs_return": round(mean_abs_ret, 6),
                "mean_return": round(mean_ret, 6),
            }
        results[hk] = bucket_results
    return results


def regime_interaction_test(oms, sym):
    sync = oms.synchronization_count(sym)
    regime = oms.get_regime(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(regime), len(fut_ret))
    sync, regime, fut_ret = sync[:n], regime[:n], fut_ret[:n]
    results = {}
    for hk in HORIZON_NAMES:
        h_results = {}
        for r in range(3):
            r_mask = regime == r
            sync_r = sync[r_mask]
            fr = fut_ret[r_mask, HORIZON_IDX[hk]]
            n_valid = min(len(sync_r), len(fr))
            sync_r, fr = sync_r[:n_valid], fr[:n_valid]
            valid = ~np.isnan(fr)
            sync_r, fr = sync_r[valid], fr[valid]
            bucket_results = {}
            for b in range(1, 6):
                m = sync_r == b
                cnt = int(np.sum(m))
                if cnt < 5:
                    continue
                p_up = float(np.mean(fr[m] > 0))
                mean_ret = float(np.mean(fr[m]))
                bucket_results[f"sync_{b}"] = {"count": cnt, "p_up": round(p_up, 4), "mean_ret": round(mean_ret, 6)}
            h_results[f"regime_{r}"] = bucket_results
        results[hk] = h_results
    return results


def threshold_detection(oms, sym):
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(fut_ret))
    sync, fut_ret = sync[:n], fut_ret[:n]
    results = {}
    for hk in HORIZON_NAMES:
        fr = fut_ret[:, HORIZON_IDX[hk]]
        valid = ~np.isnan(fr)
        s, f = sync[valid], fr[valid]
        thresholds = {}
        for threshold in range(1, 6):
            above = s >= threshold
            below = s < threshold
            cnt_above = int(np.sum(above))
            cnt_below = int(np.sum(below))
            if cnt_above < 5 or cnt_below < 5:
                continue
            p_up_above = float(np.mean(f[above] > 0))
            p_up_below = float(np.mean(f[below] > 0))
            mean_ret_above = float(np.mean(f[above]))
            mean_ret_below = float(np.mean(f[below]))
            diff = p_up_above - p_up_below
            thresholds[f"threshold_{threshold}"] = {
                "cnt_above": cnt_above,
                "cnt_below": cnt_below,
                "p_up_above": round(p_up_above, 4),
                "p_up_below": round(p_up_below, 4),
                "diff": round(diff, 4),
                "mean_ret_above": round(mean_ret_above, 6),
                "mean_ret_below": round(mean_ret_below, 6),
            }
        results[hk] = thresholds
    return results


def walkforward_model_comparison(oms, sym):
    sync = oms.synchronization_count(sym)
    marker = oms.marker_present(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(marker), len(fut_ret))
    sync, marker, fut_ret = sync[:n], marker[:n], fut_ret[:n]

    # Try year-based walk-forward; fallback to index-based chronological splits
    try:
        validator = WalkForwardValidator(oms.dsr)
        years = validator.prepare(sym)
        years = years[:n]
        splits = []
        for train_name, test_name in validator.SPLITS:
            train_mask, test_mask = validator.split(sym, train_name, test_name)
            splits.append((train_mask[:n], test_mask[:n]))
    except Exception:
        print(f"    [WF fallback: using index-based splits for {sym}]")
        split_pts = [int(n * 0.6), int(n * 0.8)]
        idx = np.arange(n)
        splits = [
            (idx < split_pts[0], (idx >= split_pts[0]) & (idx < split_pts[1])),
            ((idx >= split_pts[1] - (n - split_pts[1])) & (idx < split_pts[1]), idx >= split_pts[1]),
        ]

    split_results = []
    split_names = [f"train{i+1}_vs_test{i+1}" for i in range(len(splits))]
    for split_idx, (train_mask, test_mask) in enumerate(splits):
        split_name = split_names[split_idx]

        split_entry = {"split": split_name}
        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            fr = fut_ret[:, col]
            valid = ~np.isnan(fr)
            fr_valid = fr.copy()
            fr_valid[~valid] = 0

            test_idx = np.where(test_mask & valid)[0]
            if len(test_idx) < 20:
                continue

            y_true = (fr[test_idx] > 0).astype(float) if np.sum(valid) > 0 else np.zeros(len(test_idx))

            models = {
                "residual_sign_only": np.sign(marker[test_idx].astype(float) - 0.5),
                "sync_count_only": sync[test_idx].astype(float) / 5.0,
                "sync_plus_sign": (sync[test_idx].astype(float) / 5.0 + marker[test_idx].astype(float)) / 2.0,
            }

            if np.nanstd(sync[test_idx]) == 0:
                models["sync_count_only"] = np.full(len(test_idx), 0.5)
                models["sync_plus_sign"] = marker[test_idx].astype(float)

            h_results = {}
            for mname, pred in models.items():
                pred_bin = (pred > 0.5).astype(float)
                accuracy = float(np.mean(pred_bin == y_true)) if len(y_true) > 0 else 0
                n_test = len(y_true)
                h_results[mname] = {"accuracy": round(accuracy, 4), "n_test": n_test}
            split_entry[hk] = h_results
        split_results.append(split_entry)
    return split_results


def bootstrap_monotonicity_test(oms, sym, n_bootstrap=1000):
    """Test if P(up) monotonicity is statistically significant."""
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(fut_ret))
    sync, fut_ret = sync[:n], fut_ret[:n]
    results = {}
    for hk in HORIZON_NAMES:
        col = HORIZON_IDX[hk]
        fr = fut_ret[:, col]
        valid = ~np.isnan(fr)
        s, f = sync[valid], fr[valid]
        observed_p_ups = []
        for b in range(1, 6):
            m = s == b
            cnt = int(np.sum(m))
            if cnt < 10:
                observed_p_ups.append(np.nan)
            else:
                observed_p_ups.append(float(np.mean(f[m] > 0)))
        observed_diffs = np.diff([p for p in observed_p_ups if not np.isnan(p)])
        observed_monotonic = bool(np.all(observed_diffs >= 0)) if len(observed_diffs) > 0 else False

        bootstrap_count = 0
        total_valid = 0
        idx = np.arange(len(s))
        for _ in range(n_bootstrap):
            np.random.shuffle(idx)
            boot_p_ups = []
            for b in range(1, 6):
                m = s == b
                cnt = int(np.sum(m))
                if cnt < 10:
                    continue
                boot_p_up = float(np.mean(f[idx][m] > 0))
                boot_p_ups.append(boot_p_up)
            if len(boot_p_ups) >= 3:
                total_valid += 1
                boot_diffs = np.diff(boot_p_ups)
                if np.all(boot_diffs >= 0):
                    bootstrap_count += 1
        p_value = bootstrap_count / max(total_valid, 1)
        results[hk] = {
            "observed_p_ups": [round(p, 4) if not np.isnan(p) else None for p in observed_p_ups],
            "observed_monotonic_non_decreasing": observed_monotonic,
            "bootstrap_p_value": round(p_value, 4),
            "n_bootstrap_valid": total_valid,
        }
    return results


def correlation_analysis(oms, sym):
    """Correlation between sync count and future returns."""
    sync = oms.synchronization_count(sym)
    fut_ret = oms.get_future_returns(sym)
    n = min(len(sync), len(fut_ret))
    sync, fut_ret = sync[:n], fut_ret[:n]
    results = {}
    for hk in HORIZON_NAMES:
        col = HORIZON_IDX[hk]
        fr = fut_ret[:, col]
        valid = ~np.isnan(fr)
        if np.sum(valid) < 20:
            continue
        r_pearson = float(np.corrcoef(sync[valid], fr[valid])[0, 1]) if np.std(sync[valid]) > 0 else 0
        r_spearman = float(np.corrcoef(sync[valid], np.argsort(fr[valid]))[0, 1]) if np.std(sync[valid]) > 0 else 0
        results[hk] = {
            "pearson_r": round(r_pearson, 4),
            "spearman_r": round(r_spearman, 4),
            "n": int(np.sum(valid)),
        }
    return results


def main():
    print("=" * 72)
    print("OMS-3: Cohort Synchronization Test")
    print("=" * 72)

    oms = OMSCore()
    oms.load_all()
    print(f"\nLoaded OMS core. Cross-asset sync mean: {oms.cross_asset_sync_index().mean():.2f}/5")

    report = {"metadata": {"cross_asset_sync_mean": float(oms.cross_asset_sync_index().mean())}}
    md_lines = [
        "# OMS-3: Cohort Synchronization Test",
        "",
        f"**Cross-asset sync index mean:** {oms.cross_asset_sync_index().mean():.2f}/5 assets",
        "",
        "## Research Questions",
        "1. Does the NUMBER of synchronized assets predict directional strength?",
        "2. Is P(up) monotonic in synchronization count?",
        "3. Can synchronization count REPLACE residual sign as the predictive variable?",
        "4. Is synchronization a leading indicator?",
        "5. Does synchronization predict the MAGNITUDE of moves?",
        "6. Are there critical synchronization thresholds?",
        "",
    ]

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {sym}")
        print(f"{'='*60}")

        bucket_res, monotonicity = bucket_analysis(oms, sym)
        leading = leading_indicator_test(oms, sym)
        magnitude = magnitude_test(oms, sym)
        regime_test = regime_interaction_test(oms, sym)
        thresholds = threshold_detection(oms, sym)
        wf_compare = walkforward_model_comparison(oms, sym)
        bootstrap = bootstrap_monotonicity_test(oms, sym)
        correlation = correlation_analysis(oms, sym)

        sym_report = {
            "bucket_analysis": bucket_res,
            "monotonicity": monotonicity,
            "bootstrap_monotonicity": bootstrap,
            "leading_indicator": leading,
            "magnitude": magnitude,
            "thresholds": thresholds,
            "walkforward_model_comparison": wf_compare,
            "regime_interaction": regime_test,
            "correlation": correlation,
        }
        report[sym] = sym_report

        md_lines.append(f"## {sym}")
        md_lines.append("")
        md_lines.append("### Bucket Analysis: P(up) per Synchronization Level")
        md_lines.append("")
        header = f"| Sync Count | {' | '.join([f'{hk}' for hk in HORIZON_NAMES])} |"
        sep = f"|{':---|' * (1 + len(HORIZON_NAMES))}"
        md_lines.append(header)
        md_lines.append(sep)
        for b in range(1, 6):
            r = bucket_res.get(f"sync_{b}", {})
            if r.get("insufficient"):
                md_lines.append(f"| {b} | insufficient data |")
                continue
            vals = []
            for hk in HORIZON_NAMES:
                hd = r.get(hk, {})
                p = hd.get("p_up")
                if p is not None:
                    vals.append(f"{p:.3f}")
                else:
                    vals.append("N/A")
            md_lines.append(f"| {b} | {' | '.join(vals)} |")
        md_lines.append("")

        md_lines.append("### Monotonicity Test")
        md_lines.append("")
        for hk in HORIZON_NAMES:
            m = monotonicity.get(hk, {})
            bm = bootstrap.get(hk, {})
            if m:
                md_lines.append(f"- **{hk}**: strictly_increasing={m.get('strictly_increasing')}, "
                               f"non_decreasing={m.get('non_decreasing')}, p_value={bm.get('bootstrap_p_value', 'N/A')}")
                md_lines.append(f"  - P(up) sequence: {m.get('p_up_sequence')}")
        md_lines.append("")

        md_lines.append("### Correlation (Sync Count vs Future Return)")
        md_lines.append("")
        for hk in HORIZON_NAMES:
            c = correlation.get(hk, {})
            if c:
                md_lines.append(f"- **{hk}**: Pearson r={c.get('pearson_r')}, Spearman r={c.get('spearman_r')}")
        md_lines.append("")

        md_lines.append("### Walk-Forward Model Comparison")
        md_lines.append("")
        md_lines.append("| Split | Horizon | residual_sign | sync_count | sync+sign |")
        md_lines.append("|:---|:---|:---|:---|:---|")
        for wf in wf_compare:
            split_name = wf["split"]
            for hk in HORIZON_NAMES:
                hd = wf.get(hk, {})
                if not hd:
                    continue
                acc_sign = hd.get("residual_sign_only", {}).get("accuracy", "N/A")
                acc_sync = hd.get("sync_count_only", {}).get("accuracy", "N/A")
                acc_both = hd.get("sync_plus_sign", {}).get("accuracy", "N/A")
                md_lines.append(f"| {split_name} | {hk} | {acc_sign} | {acc_sync} | {acc_both} |")
        md_lines.append("")

        md_lines.append("### Leading Indicator (Sync at t-lag predicts direction at t)")
        md_lines.append("")
        for lag_key, lag_data in leading.items():
            md_lines.append(f"#### {lag_key}")
            md_lines.append("")
            for hk in HORIZON_NAMES:
                hd = lag_data.get(hk, {})
                if not hd:
                    continue
                vals = []
                for b in range(1, 6):
                    bd = hd.get(f"sync_{b}", {})
                    if bd:
                        vals.append(f"{b}: P(up)={bd.get('p_up', 'N/A')}(n={bd.get('count', 0)})")
                if vals:
                    md_lines.append(f"- **{hk}**: {' | '.join(vals)}")
            md_lines.append("")

        md_lines.append("### Magnitude: Mean |Return| per Sync Bucket")
        md_lines.append("")
        header = f"| Sync Count | {' | '.join([f'{hk} |Return|' for hk in HORIZON_NAMES])} |"
        sep = f"|{':---|' * (1 + 2 * len(HORIZON_NAMES))}"
        md_lines.append(header)
        md_lines.append(sep)
        for b in range(1, 6):
            vals = []
            for hk in HORIZON_NAMES:
                hd = magnitude.get(hk, {}).get(f"sync_{b}", {})
                if hd:
                    vals.append(f"{hd.get('mean_abs_return', 'N/A')}")
                else:
                    vals.append("N/A")
            md_lines.append(f"| {b} | {' | '.join(vals)} |")
        md_lines.append("")

        md_lines.append("### Threshold Detection")
        md_lines.append("")
        for hk in HORIZON_NAMES:
            hd = thresholds.get(hk, {})
            if not hd:
                continue
            md_lines.append(f"**{hk}**:")
            for thr_key, thr_data in hd.items():
                md_lines.append(f"- {thr_key}: P(up) above={thr_data.get('p_up_above'):.3f} vs below={thr_data.get('p_up_below'):.3f}, "
                               f"diff={thr_data.get('diff'):.3f}, mean_ret_above={thr_data.get('mean_ret_above')}")
        md_lines.append("")

        md_lines.append("### Regime Interaction (Sync x Regime -> P(up))")
        md_lines.append("")
        for hk in HORIZON_NAMES:
            hd = regime_test.get(hk, {})
            if not hd:
                continue
            for reg_key, reg_data in hd.items():
                if not reg_data:
                    continue
                vals = []
                for b in range(1, 6):
                    bd = reg_data.get(f"sync_{b}", {})
                    if bd:
                        vals.append(f"{b}: P(up)={bd.get('p_up', 'N/A')}(n={bd.get('count', 0)})")
                if vals:
                    md_lines.append(f"- **{hk} {reg_key}**: {' | '.join(vals)}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        print(f"  Buckets: {len(bucket_res)} levels")
        print(f"  Monotonicity (H5): {monotonicity.get('H5', {}).get('non_decreasing', 'N/A')}")
        print(f"  Bootstrap p (H5): {bootstrap.get('H5', {}).get('bootstrap_p_value', 'N/A')}")

        for wf in wf_compare:
            split_name = wf["split"]
            for hk in HORIZON_NAMES:
                hd = wf.get(hk, {})
                if hd:
                    acc_sign = hd.get("residual_sign_only", {}).get("accuracy", 0)
                    acc_sync = hd.get("sync_count_only", {}).get("accuracy", 0)
                    acc_both = hd.get("sync_plus_sign", {}).get("accuracy", 0)
                    print(f"  {split_name} {hk}: sign={acc_sign:.3f} sync={acc_sync:.3f} both={acc_both:.3f}")

    json_path = REPORT_DIR / "oms3_cohort_synchronization.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_path = REPORT_DIR / "OMS3_COHORT_SYNCHRONIZATION.md"
    md_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {md_path}")

    print("\n" + "=" * 72)
    print("OMS-3 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
