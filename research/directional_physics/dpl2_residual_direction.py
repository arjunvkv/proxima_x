"""
DPL-2: Residual Direction Hypothesis
Test whether residual sign determines direction in high-ES states.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

results = {}
for sym in SYMBOLS:
    d = DPLData(sym)
    es = d.es.copy()
    fut_ret = d.fut_ret

    sym_res = {}
    for residual_type in ["xgboost", "random_forest", "linear"]:
        resid = d.residuals.get(residual_type)
        if resid is None:
            continue

        type_res = {}
        for hi, h in enumerate(HORIZONS):
            fwd = fut_ret[:, hi]
            valid = ~np.isnan(resid) & ~np.isnan(es) & ~np.isnan(fwd)
            es_v = es[valid]
            resid_v = resid[valid]
            fwd_v = fwd[valid]

            # High ES states only
            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr
            if np.sum(high_es) < 10:
                continue

            resid_high = resid_v[high_es]
            fwd_high = fwd_v[high_es]

            # Split by residual sign
            pos_mask = resid_high > 0
            neg_mask = resid_high < 0
            n_pos, n_neg = int(np.sum(pos_mask)), int(np.sum(neg_mask))

            if n_pos < 3 or n_neg < 3:
                continue

            p_up_pos = float(np.mean(fwd_high[pos_mask] > 0))
            p_up_neg = float(np.mean(fwd_high[neg_mask] > 0))
            mean_pos = float(np.mean(fwd_high[pos_mask]))
            mean_neg = float(np.mean(fwd_high[neg_mask]))

            # Directional accuracy of residual sign
            pred_up = resid_high > 0
            actual_up = fwd_high > 0
            directional_acc = float(np.mean(pred_up == actual_up))

            # Correlation: residual magnitude vs directional strength
            corr_mag_dir, _ = pearsonr(np.abs(resid_high), fwd_high * np.sign(resid_high))

            type_res[HORIZON_LABELS[h]] = {
                "n_high_es": int(np.sum(high_es)),
                "n_pos_residual": n_pos, "n_neg_residual": n_neg,
                "p_up_given_pos_residual": round(p_up_pos, 4),
                "p_up_given_neg_residual": round(p_up_neg, 4),
                "mean_return_pos_residual": round(mean_pos, 6),
                "mean_return_neg_residual": round(mean_neg, 6),
                "directional_accuracy": round(directional_acc, 4),
                "corr_mag_vs_directional_strength": round(float(corr_mag_dir), 4),
            }

        # Cross-horizon directional accuracy
        accs = [v["directional_accuracy"] for v in type_res.values()]
        type_res["meta"] = {
            "mean_directional_accuracy": round(float(np.mean(accs)), 4) if accs else None,
            "residual_type": residual_type,
        }
        sym_res[residual_type] = type_res

    results[sym] = sym_res

# Transfer assessment
all_accs = []
for sym in SYMBOLS:
    for rt in ["xgboost", "random_forest", "linear"]:
        meta = results.get(sym, {}).get(rt, {}).get("meta", {})
        if meta.get("mean_directional_accuracy"):
            all_accs.append(meta["mean_directional_accuracy"])

output = {"experiment": "DPL-2", "title": "Residual Direction Hypothesis",
           "per_symbol": results,
           "cross_asset_mean_accuracy": round(float(np.mean(all_accs)), 4) if all_accs else None,
           "cross_asset_std_accuracy": round(float(np.std(all_accs)), 4) if all_accs else None}
out_path = Path(__file__).parent / "reports" / "dpl2_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-2 complete -> {out_path}")
