"""
DPL-4: Energy Gradient Theory
Does gradient (change in ES) predict release direction?
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS, compute_gradient, compute_acceleration, compute_curvature
import warnings
warnings.filterwarnings("ignore")

results = {}
for sym in SYMBOLS:
    d = DPLData(sym)
    es = d.es.copy()
    fut_ret = d.fut_ret

    g1 = compute_gradient(es, window=5)
    g2 = compute_acceleration(es, window=5)
    curv = compute_curvature(es, window=5)
    log_es = np.log(np.abs(es) + 1e-12)
    rel_grad = np.full_like(es, np.nan)
    rel_grad[5:] = (es[5:] - es[:-5]) / (np.abs(es[:-5]) + 1e-12)

    sym_res = {}
    for hi, h in enumerate(HORIZONS):
        fwd = fut_ret[:, hi]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(g1) & ~np.isnan(g2)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        g1_v = g1[valid]
        g2_v = g2[valid]
        curv_v = curv[valid]
        rel_v = rel_grad[valid]
        fwd_v = fwd[valid]
        dir_v = (fwd_v > 0).astype(float)

        # High ES states
        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        if np.sum(high_es) < 10:
            continue

        # Rising gradient vs falling gradient in high ES
        rising = g1_v[high_es] > 0
        falling = g1_v[high_es] < 0

        p_up_rising = float(np.mean(dir_v[high_es][rising])) if np.sum(rising) > 3 else None
        p_up_falling = float(np.mean(dir_v[high_es][falling])) if np.sum(falling) > 3 else None
        mean_fwd_rising = float(np.mean(fwd_v[high_es][rising])) if np.sum(rising) > 3 else None
        mean_fwd_falling = float(np.mean(fwd_v[high_es][falling])) if np.sum(falling) > 3 else None

        # Correlation: gradient vs direction
        corr_g1_dir, _ = pearsonr(g1_v[high_es], dir_v[high_es])
        corr_g2_dir, _ = pearsonr(g2_v[high_es], dir_v[high_es])
        corr_curv_dir, _ = pearsonr(curv_v[high_es], dir_v[high_es])
        corr_rel_dir, _ = pearsonr(rel_v[high_es], dir_v[high_es])
        corr_es_dir, _ = pearsonr(es_v[high_es], dir_v[high_es])

        # Lead-lag: does gradient front-run returns?
        max_lag = 10
        lead_lag = {}
        for lag in range(1, max_lag + 1):
            if lag >= len(g1_v[high_es]):
                continue
            c, _ = pearsonr(g1_v[high_es][:-lag], dir_v[high_es][lag:])
            lead_lag[f"lag_{lag}"] = round(float(c), 4)

        sym_res[HORIZON_LABELS[h]] = {
            "n_high_es": int(np.sum(high_es)),
            "p_up_rising_gradient": round(p_up_rising, 4) if p_up_rising is not None else None,
            "p_up_falling_gradient": round(p_up_falling, 4) if p_up_falling is not None else None,
            "mean_fwd_rising": round(mean_fwd_rising, 6) if mean_fwd_rising is not None else None,
            "mean_fwd_falling": round(mean_fwd_falling, 6) if mean_fwd_falling is not None else None,
            "corr_gradient_direction": round(float(corr_g1_dir), 4),
            "corr_acceleration_direction": round(float(corr_g2_dir), 4),
            "corr_curvature_direction": round(float(corr_curv_dir), 4),
            "corr_relative_gradient_direction": round(float(corr_rel_dir), 4),
            "corr_es_direction_high_es": round(float(corr_es_dir), 4),
            "gradient_beats_es": bool(abs(corr_g1_dir) > abs(corr_es_dir)),
            "lead_lag": lead_lag,
        }

    results[sym] = sym_res

output = {"experiment": "DPL-4", "title": "Energy Gradient Theory",
           "per_symbol": results}
out_path = Path(__file__).parent / "reports" / "dpl4_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-4 complete -> {out_path}")
