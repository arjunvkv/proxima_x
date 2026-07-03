"""
DPL-1: Does ES Predict Magnitude Or Direction?
Measure corr(ES, future_return) vs corr(ES, abs(future_return))
across 5 symbols × 5 horizons (H5, H10, H20, H40, H80).
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

results = {}
for sym in SYMBOLS:
    d = DPLData(sym)
    es = d.es.copy()
    valid = ~np.isnan(es) & ~np.isnan(d.fut_ret).any(axis=1)
    es_v, fr = es[valid], d.fut_ret[valid]

    sym_res = {}
    for hi, h in enumerate(HORIZONS):
        fwd = fr[:, hi]
        abs_fwd = np.abs(fwd)
        m = ~np.isnan(fwd) & ~np.isnan(abs_fwd)
        if np.sum(m) < 30:
            continue
        r_sign, _ = pearsonr(es_v[m], fwd[m])
        r_abs, _ = pearsonr(es_v[m], abs_fwd[m])
        sp_sign, _ = spearmanr(es_v[m], fwd[m])
        sp_abs, _ = spearmanr(es_v[m], abs_fwd[m])

        # Bootstrap test: is r_abs > r_sign?
        n_boot = 500
        diff_boot = []
        idx = np.where(m)[0]
        for _ in range(n_boot):
            boot = np.random.choice(idx, size=len(idx), replace=True)
            r1 = np.corrcoef(es_v[boot], fwd[boot])[0, 1]
            r2 = np.corrcoef(es_v[boot], abs_fwd[boot])[0, 1]
            diff_boot.append(r2 - r1)
        diff_boot = np.array(diff_boot)
        p_magnitude = np.mean(diff_boot > 0)

        sym_res[HORIZON_LABELS[h]] = {
            "n": int(np.sum(m)),
            "pearson_sign": round(float(r_sign), 4),
            "pearson_abs": round(float(r_abs), 4),
            "spearman_sign": round(float(sp_sign), 4),
            "spearman_abs": round(float(sp_abs), 4),
            "abs_greater_than_sign": bool(r_abs > r_sign),
            "p_magnitude_dominant": round(float(p_magnitude), 3),
        }

    # Classification
    abs_wins = sum(1 for v in sym_res.values() if v["abs_greater_than_sign"])
    total_h = len(sym_res)
    if abs_wins / max(total_h, 1) >= 0.8:
        cls = "MAGNITUDE_ONLY"
    elif abs_wins / max(total_h, 1) <= 0.2:
        cls = "DIRECTIONAL"
    else:
        cls = "MIXED"

    results[sym] = {"by_horizon": sym_res, "classification": cls,
                     "abs_wins": abs_wins, "total_horizons": total_h}

# Summary classification
all_cls = [v["classification"] for v in results.values()]
output = {"experiment": "DPL-1", "title": "ES: Magnitude vs Direction",
           "per_symbol": results, "final_classification": all_cls}
out_path = Path(__file__).parent / "reports" / "dpl1_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-1 complete -> {out_path}")
print(json.dumps({"classification": all_cls}, indent=2))
