"""
DPL-3: Memory Positioning Hypothesis
Does direction depend on price location relative to memory clusters?
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

results = {}
for sym in SYMBOLS:
    d = DPLData(sym)
    price = d.price.copy()
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret

    n = len(price)
    window = 252
    rolling_results = {hl: [] for hl in HORIZON_LABELS.values()}

    for i in range(window, n):
        if np.isnan(es[i]) or np.isnan(md[i]):
            continue
        chunk_px = price[i - window:i]
        chunk_md = md[i - window:i]

        # Identify memory clusters (2 clusters: high-density zones)
        if np.all(chunk_md == chunk_md[0]):
            continue
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=1)
            labels = kmeans.fit_predict(chunk_md.reshape(-1, 1))
            center_prices = np.array([np.mean(chunk_px[labels == l]) for l in range(2)])
            # memory_center = weighted average by density
            weights = np.array([np.mean(chunk_md[labels == l]) for l in range(2)])
            if np.sum(weights) == 0:
                continue
            memory_center = float(np.average(center_prices, weights=weights))
            memory_distance = price[i] - memory_center

            # memory_skew: asymmetry of density around current price
            above = chunk_md[chunk_px > price[i]]
            below = chunk_md[chunk_px <= price[i]]
            memory_skew = (np.mean(above) - np.mean(below)) if len(above) > 5 and len(below) > 5 else 0.0

            # memory_asymmetry: ratio of high-density points above vs below
            high_density_thr = np.nanpercentile(chunk_md, 80)
            above_high = np.sum((chunk_px > price[i]) & (chunk_md > high_density_thr))
            below_high = np.sum((chunk_px <= price[i]) & (chunk_md > high_density_thr))
            memory_asymmetry = (above_high - below_high) / max(above_high + below_high, 1)
        except Exception:
            continue

        # Test each horizon
        for hi, h in enumerate(HORIZONS):
            if i + h >= n:
                continue
            fwd = fut_ret[i, hi]
            if np.isnan(fwd):
                continue
            rolling_results[HORIZON_LABELS[h]].append({
                "memory_distance": float(memory_distance),
                "memory_skew": float(memory_skew),
                "memory_asymmetry": float(memory_asymmetry),
                "future_return": float(fwd),
                "direction": 1 if fwd > 0 else 0,
                "es": float(es[i]),
            })

    # Analyze each horizon
    sym_res = {}
    for hl, records in rolling_results.items():
        if len(records) < 30:
            continue
        dist = np.array([r["memory_distance"] for r in records])
        skew = np.array([r["memory_skew"] for r in records])
        asym = np.array([r["memory_asymmetry"] for r in records])
        dirs = np.array([r["direction"] for r in records])
        fwd = np.array([r["future_return"] for r in records])
        es_v = np.array([r["es"] for r in records])

        # Does distance predict direction?
        above = dist > 0
        below = dist < 0
        p_up_above = float(np.mean(dirs[above])) if np.sum(above) > 5 else None
        p_up_below = float(np.mean(dirs[below])) if np.sum(below) > 5 else None

        # Does asymmetry predict direction?
        pos_asym = asym > 0
        neg_asym = asym < 0
        p_up_pos_asym = float(np.mean(dirs[pos_asym])) if np.sum(pos_asym) > 5 else None
        p_up_neg_asym = float(np.mean(dirs[neg_asym])) if np.sum(neg_asym) > 5 else None

        # Informational gain: does (memory_distance) add to ES alone?
        try:
            corr_es_dir, _ = pearsonr(es_v, dirs)
            corr_dist_dir, _ = pearsonr(dist, dirs)
        except Exception:
            corr_es_dir = corr_dist_dir = 0.0

        sym_res[hl] = {
            "n": len(records),
            "p_up_above_memory_center": round(p_up_above, 4) if p_up_above is not None else None,
            "p_up_below_memory_center": round(p_up_below, 4) if p_up_below is not None else None,
            "p_up_positive_asymmetry": round(p_up_pos_asym, 4) if p_up_pos_asym is not None else None,
            "p_up_negative_asymmetry": round(p_up_neg_asym, 4) if p_up_neg_asym is not None else None,
            "corr_distance_direction": round(float(corr_dist_dir), 4),
            "corr_es_direction": round(float(corr_es_dir), 4),
            "distance_improves_es": bool(abs(corr_dist_dir) > abs(corr_es_dir)),
        }

    results[sym] = sym_res

output = {"experiment": "DPL-3", "title": "Memory Positioning Hypothesis",
           "per_symbol": results}
out_path = Path(__file__).parent / "reports" / "dpl3_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-3 complete -> {out_path}")
