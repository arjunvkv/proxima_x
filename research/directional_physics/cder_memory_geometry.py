"""
CDER: Context-Dependent Energy Release — Memory Geometry
Investigates whether memory topology determines energy release direction.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS, compute_gradient
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)


def _clean(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Only test these horizons
TEST_HORIZONS = [5, 20, 50]
TEST_HL = {h: HORIZON_LABELS[h] for h in TEST_HORIZONS}

WINDOW = 252  # rolling window size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def p_up(fwd: np.ndarray, mask: np.ndarray) -> float:
    m = mask & ~np.isnan(fwd)
    if np.sum(m) < 3:
        return 0.5
    return float(np.mean(fwd[m] > 0))


def mean_ret(fwd: np.ndarray, mask: np.ndarray) -> float:
    m = mask & ~np.isnan(fwd)
    if np.sum(m) < 3:
        return 0.0
    return float(np.mean(fwd[m]))


def explode_ratio(fwd: np.ndarray, mask: np.ndarray, thr: float = None) -> float:
    """Fraction of returns exceeding 2 sigma in magnitude."""
    m = mask & ~np.isnan(fwd)
    if np.sum(m) < 3:
        return 0.0
    if thr is None:
        thr = 2.0 * np.nanstd(fwd[~np.isnan(fwd)])
    return float(np.mean(np.abs(fwd[m]) > thr))


def info_gain_strat(es: np.ndarray, fwd: np.ndarray, mask: np.ndarray) -> dict:
    """Information gain of a mask over ES-only baseline (high-ES P(up))."""
    m = mask & ~np.isnan(es) & ~np.isnan(fwd)
    if np.sum(m) < 10:
        return {"p_up": None, "n": 0, "vs_es_high": None, "improvement": None}
    p_up_mask = float(np.mean(fwd[m] > 0))
    # ES baseline: same count of highest-ES points
    es_high_mask = ~np.isnan(es) & ~np.isnan(fwd)
    es_v = es[es_high_mask]
    fwd_v = fwd[es_high_mask]
    n = np.sum(m)
    if n > len(es_v):
        n = len(es_v)
    if n < 10:
        return {"p_up": round(p_up_mask, 4), "n": int(np.sum(m)), "vs_es_high": None, "improvement": None}
    thr_idx = len(es_v) - n
    thr = np.partition(es_v, thr_idx)[thr_idx] if thr_idx > 0 else -np.inf
    es_high_comp = fwd_v[es_v >= thr]
    p_up_es = float(np.mean(es_high_comp > 0)) if len(es_high_comp) > 3 else 0.5
    improvement = p_up_mask - p_up_es
    return {
        "p_up": round(p_up_mask, 4),
        "n": int(np.sum(m)),
        "vs_es_high_p_up": round(p_up_es, 4),
        "improvement": round(improvement, 4),
    }


def rolling_window_data(price, md, window):
    """Generator yielding (i, chunk_px, chunk_md) for each valid i >= window."""
    n = len(price)
    for i in range(window, n):
        if np.isnan(md[i]):
            continue
        chunk_px = price[i - window:i]
        chunk_md = md[i - window:i]
        if np.all(np.isnan(chunk_md)):
            continue
        yield i, chunk_px, chunk_md


# ---------------------------------------------------------------------------
# 1. Memory Gradient Analysis
# ---------------------------------------------------------------------------
def task1_memory_gradient(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    grad = compute_gradient(md, window=5)
    grad_accel = compute_gradient(grad, window=5)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(grad) & ~np.isnan(md)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        md_v = md[valid]
        grad_v = grad[valid]
        accel_v = grad_accel[valid]
        fwd_v = fwd[valid]
        dir_v = (fwd_v > 0).astype(float)

        # ES baseline (high ES)
        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        # Gradient direction
        grad_up = grad_v > 0
        grad_down = grad_v < 0

        # On all data
        p_up_grad_up = p_up(fwd_v, grad_up)
        p_up_grad_down = p_up(fwd_v, grad_down)
        corr_grad_dir, _ = pearsonr(grad_v, dir_v)

        # On high-ES only
        high_grad_up = high_es & grad_up
        high_grad_down = high_es & grad_down
        p_up_high_grad_up = p_up(fwd_v, high_grad_up)
        p_up_high_grad_down = p_up(fwd_v, high_grad_down)

        # Does gradient add info vs ES alone?
        grad_up_info = info_gain_strat(es_v, fwd_v, high_grad_up)

        # Acceleration
        corr_accel_dir, _ = pearsonr(accel_v, dir_v)

        # Energy flows toward or away from high-density?
        # If grad>0 (increasing density), check if direction is UP (toward) or DOWN (away)
        # A positive correlation means rising density -> price up (energy flows toward density)
        # A negative correlation means rising density -> price down (energy flows away)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_high_es": int(np.sum(high_es)),
            "p_up_gradient_rising": round(p_up_grad_up, 4),
            "p_up_gradient_falling": round(p_up_grad_down, 4),
            "p_up_high_es_gradient_rising": round(p_up_high_grad_up, 4),
            "p_up_high_es_gradient_falling": round(p_up_high_grad_down, 4),
            "corr_gradient_direction": round(float(corr_grad_dir), 4),
            "corr_acceleration_direction": round(float(corr_accel_dir), 4),
            "gradient_adds_info": grad_up_info,
            "energy_flow_toward_density": corr_grad_dir > 0,
            "energy_flow_away_from_density": corr_grad_dir < 0,
        }
    return results


# ---------------------------------------------------------------------------
# 2. Memory Voids
# ---------------------------------------------------------------------------
def task2_memory_voids(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    price = d.price.copy()

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]

        void_thr = np.nanpercentile(md_v, 10)
        es_high_thr = np.nanpercentile(es_v, 80)
        is_void = md_v <= void_thr
        is_high_es = es_v > es_high_thr

        # Does void alone predict?
        p_up_void = p_up(fwd_v, is_void)
        mean_ret_void = mean_ret(fwd_v, is_void)
        explode_void = explode_ratio(fwd_v, is_void)

        # Does high ES + void predict explosive moves?
        void_high_es = is_void & is_high_es
        p_up_void_high_es = p_up(fwd_v, void_high_es)
        mean_ret_void_high_es = mean_ret(fwd_v, void_high_es)
        explode_void_high_es = explode_ratio(fwd_v, void_high_es)
        void_high_es_info = info_gain_strat(es_v, fwd_v, void_high_es)

        # Non-void high ES for comparison
        nonvoid_high_es = ~is_void & is_high_es
        p_up_nonvoid_high_es = p_up(fwd_v, nonvoid_high_es)
        mean_ret_nonvoid_high_es = mean_ret(fwd_v, nonvoid_high_es)
        explode_nonvoid_high_es = explode_ratio(fwd_v, nonvoid_high_es)

        # Test: does energy preferentially release toward low-memory regions?
        # Directional asymmetry in voids
        results[TEST_HL[h]] = {
            "n_void": int(np.sum(is_void)),
            "n_void_high_es": int(np.sum(void_high_es)),
            "void_threshold_pct": round(float(void_thr), 6),
            "p_up_void": round(p_up_void, 4),
            "mean_ret_void": round(mean_ret_void, 6),
            "explode_ratio_void": round(explode_void, 4),
            "p_up_void_high_es": round(p_up_void_high_es, 4),
            "mean_ret_void_high_es": round(mean_ret_void_high_es, 6),
            "explode_ratio_void_high_es": round(explode_void_high_es, 4),
            "p_up_nonvoid_high_es": round(p_up_nonvoid_high_es, 4),
            "mean_ret_nonvoid_high_es": round(mean_ret_nonvoid_high_es, 6),
            "explode_ratio_nonvoid_high_es": round(explode_nonvoid_high_es, 4),
            "void_high_es_info_gain": void_high_es_info,
            "void_explodes_more": explode_void_high_es > explode_nonvoid_high_es,
        }
    return results


# ---------------------------------------------------------------------------
# 3. Memory Saturation
# ---------------------------------------------------------------------------
def task3_memory_saturation(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    price = d.price.copy()

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]
        px_v = price[valid]

        sat_thr = np.nanpercentile(md_v, 90)
        es_high_thr = np.nanpercentile(es_v, 80)
        is_sat = md_v >= sat_thr
        is_high_es = es_v > es_high_thr

        # Saturation alone
        p_up_sat = p_up(fwd_v, is_sat)
        mean_ret_sat = mean_ret(fwd_v, is_sat)

        # High ES + saturation: reversal?
        sat_high_es = is_sat & is_high_es
        p_up_sat_high_es = p_up(fwd_v, sat_high_es)
        mean_ret_sat_high_es = mean_ret(fwd_v, sat_high_es)

        # Non-sat high ES
        nonsat_high_es = ~is_sat & is_high_es
        p_up_nonsat_high_es = p_up(fwd_v, nonsat_high_es)
        mean_ret_nonsat_high_es = mean_ret(fwd_v, nonsat_high_es)

        sat_high_es_info = info_gain_strat(es_v, fwd_v, sat_high_es)

        # Direction relative to preceding trend
        trend = np.full_like(px_v, np.nan)
        for j in range(10, len(px_v)):
            trend[j] = np.log(px_v[j] / px_v[j - 10])

        reversal_rate = None
        if np.sum(sat_high_es) > 5:
            sat_idx = np.where(sat_high_es)[0]
            sat_idx = sat_idx[sat_idx < len(trend)]
            if len(sat_idx) > 5:
                trends_at_sat = trend[sat_idx]
                dirs_at_sat = fwd_v[sat_idx] > 0
                reversal = (trends_at_sat < 0) == dirs_at_sat
                reversal_rate = float(np.mean(reversal))

        results[TEST_HL[h]] = {
            "n_saturation": int(np.sum(is_sat)),
            "n_saturation_high_es": int(np.sum(sat_high_es)),
            "saturation_threshold_pct": round(float(sat_thr), 6),
            "p_up_saturation": round(p_up_sat, 4),
            "mean_ret_saturation": round(mean_ret_sat, 6),
            "p_up_saturation_high_es": round(p_up_sat_high_es, 4),
            "mean_ret_saturation_high_es": round(mean_ret_sat_high_es, 6),
            "p_up_nonsaturation_high_es": round(p_up_nonsat_high_es, 4),
            "mean_ret_nonsaturation_high_es": round(mean_ret_nonsat_high_es, 6),
            "saturation_reversal_rate": round(reversal_rate, 4) if reversal_rate is not None else None,
            "saturation_predicts_reversal": p_up_sat_high_es < 0.45 if p_up_sat_high_es is not None else None,
            "saturation_high_es_info_gain": sat_high_es_info,
            "saturation_reverses_relative_to_es_high": bool(
                p_up_sat_high_es is not None and p_up_nonsat_high_es is not None
                and abs(p_up_sat_high_es - 0.5) > abs(p_up_nonsat_high_es - 0.5)
                and (p_up_sat_high_es < 0.5) != (p_up_nonsat_high_es < 0.5)
            ),
        }
    return results


# ---------------------------------------------------------------------------
# 4. Memory Asymmetry
# ---------------------------------------------------------------------------
def task4_memory_asymmetry(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    price = d.price.copy()

    asym_idx = np.full_like(md, np.nan)
    simple_dist = np.full_like(md, np.nan)
    mem_above_arr = np.full_like(md, np.nan)
    mem_below_arr = np.full_like(md, np.nan)

    for i, chunk_px, chunk_md in rolling_window_data(price, md, WINDOW):
        px_i = price[i]
        above = chunk_md[chunk_px > px_i]
        below = chunk_md[chunk_px <= px_i]
        mem_above = np.nanmean(above) if len(above) > 5 else 0
        mem_below = np.nanmean(below) if len(below) > 5 else 0
        mem_above_arr[i] = mem_above
        mem_below_arr[i] = mem_below
        denom = mem_above + mem_below
        asym_idx[i] = (mem_above - mem_below) / max(denom, 1e-12)

        # Simple distance-to-center (weighted by density)
        weights = chunk_md / max(np.sum(chunk_md), 1e-12)
        center = float(np.average(chunk_px, weights=weights))
        simple_dist[i] = px_i - center

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(asym_idx) & ~np.isnan(simple_dist)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        asym_v = asym_idx[valid]
        dist_v = simple_dist[valid]
        fwd_v = fwd[valid]
        dir_v = (fwd_v > 0).astype(float)

        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        # Asymmetry test: positive asym (more density above) -> price should go down (revert toward density)
        # Negative asym -> price should go up
        pos_asym = asym_v > 0
        neg_asym = asym_v < 0
        p_up_pos_asym = p_up(fwd_v, pos_asym & high_es)
        p_up_neg_asym = p_up(fwd_v, neg_asym & high_es)
        mean_ret_pos_asym = mean_ret(fwd_v, pos_asym & high_es)
        mean_ret_neg_asym = mean_ret(fwd_v, neg_asym & high_es)

        corr_asym_dir, _ = pearsonr(asym_v[high_es], dir_v[high_es])
        corr_dist_dir, _ = pearsonr(dist_v[high_es], dir_v[high_es])

        # Does asymmetry beat simple distance?
        asym_predicts = abs(corr_asym_dir) > abs(corr_dist_dir)

        # Asymmetry as binary signal
        high_es_subset = np.where(high_es)[0]
        n_signal = max(int(len(high_es_subset) * 0.3), 10)
        if n_signal < len(high_es_subset):
            # Top 30% most asymmetric (in absolute terms)
            top_asym_idx = np.argsort(-np.abs(asym_v[high_es]))[:n_signal]
            top_asym = high_es_subset[top_asym_idx]
            p_up_top_asym = float(np.mean(dir_v[top_asym]))
            p_up_es_top = float(np.mean(dir_v[high_es_subset[:n_signal]]))
        else:
            p_up_top_asym = p_up(fwd_v, high_es)
            p_up_es_top = p_up(fwd_v, high_es)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_high_es": int(np.sum(high_es)),
            "p_up_positive_asymmetry": round(p_up_pos_asym, 4),
            "p_up_negative_asymmetry": round(p_up_neg_asym, 4),
            "mean_ret_positive_asymmetry": round(mean_ret_pos_asym, 6),
            "mean_ret_negative_asymmetry": round(mean_ret_neg_asym, 6),
            "corr_asymmetry_direction": round(float(corr_asym_dir), 4),
            "corr_distance_direction": round(float(corr_dist_dir), 4),
            "asymmetry_beats_distance": asym_predicts,
            "p_up_top_30pct_asymmetry": round(p_up_top_asym, 4),
            "p_up_top_30pct_es": round(p_up_es_top, 4),
            "top_asymmetry_improves_es": round(p_up_top_asym - p_up_es_top, 4),
        }
    return results


# ---------------------------------------------------------------------------
# 5. Memory Clustering
# ---------------------------------------------------------------------------
def task5_memory_clustering(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    price = d.price.copy()

    n = len(md)
    cluster_labels = np.full(n, -1, dtype=np.int32)

    for i, chunk_px, chunk_md in rolling_window_data(price, md, WINDOW):
        if np.nanstd(chunk_md) < 1e-12:
            continue
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=1)
            labels = kmeans.fit_predict(chunk_md.reshape(-1, 1))
            # Assign current point's cluster based on its memory density
            current_md = md[i]
            # Find which cluster center is closest
            centers = kmeans.cluster_centers_.flatten()
            cluster_labels[i] = int(np.argmin(np.abs(centers - current_md)))
        except Exception:
            continue

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & (cluster_labels >= 0)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        fwd_v = fwd[valid]
        cl_v = cluster_labels[valid]

        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        cluster_info = {}
        for c in range(3):
            c_mask = cl_v == c
            c_high_es = c_mask & high_es
            cluster_info[f"cluster_{c}"] = {
                "n": int(np.sum(c_mask)),
                "n_high_es": int(np.sum(c_high_es)),
                "p_up": round(p_up(fwd_v, c_mask), 4),
                "mean_ret": round(mean_ret(fwd_v, c_mask), 6),
                "p_up_high_es": round(p_up(fwd_v, c_high_es), 4),
                "mean_ret_high_es": round(mean_ret(fwd_v, c_high_es), 6),
            }

        # Direction flip across clusters (in high ES)
        present_clusters = [int(k.split("_")[1]) for k in cluster_info]
        p_ups_high_es = [cluster_info[f"cluster_{c}"]["p_up_high_es"] for c in present_clusters if cluster_info[f"cluster_{c}"]["n_high_es"] > 5]
        cluster_flips = []
        for i, c1 in enumerate(present_clusters):
            for c2 in present_clusters[i + 1:]:
                k1, k2 = f"cluster_{c1}", f"cluster_{c2}"
                p1, p2 = cluster_info[k1]["p_up_high_es"], cluster_info[k2]["p_up_high_es"]
                if cluster_info[k1]["n_high_es"] > 5 and cluster_info[k2]["n_high_es"] > 5:
                    cluster_flips.append({
                        "c1": c1, "c2": c2,
                        "p_up_c1": round(p1, 4), "p_up_c2": round(p2, 4),
                        "spread": round(abs(p1 - p2), 4),
                        "flip_detected": abs(p1 - p2) > 0.2,
                    })

        max_cluster_spread = max([f["spread"] for f in cluster_flips]) if cluster_flips else 0

        # ES baseline vs best cluster
        best_p_up = max(p_ups_high_es) if p_ups_high_es else 0.5
        worst_p_up = min(p_ups_high_es) if p_ups_high_es else 0.5

        results[TEST_HL[h]] = {
            "n_clustered": int(np.sum(cluster_labels >= 0)),
            "per_cluster": cluster_info,
            "cluster_directional_flips": cluster_flips,
            "max_cluster_spread_high_es": round(max_cluster_spread, 4),
            "best_cluster_p_up_high_es": round(best_p_up, 4),
            "worst_cluster_p_up_high_es": round(worst_p_up, 4),
            "any_cluster_flip": any(f["flip_detected"] for f in cluster_flips),
            "cluster_predicts_better_than_es": best_p_up > 0.55 and worst_p_up < 0.45,
        }
    return results


# ---------------------------------------------------------------------------
# 6. Memory Imbalance
# ---------------------------------------------------------------------------
def task6_memory_imbalance(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret
    price = d.price.copy()

    imbalance = np.full_like(md, np.nan)

    for i, chunk_px, chunk_md in rolling_window_data(price, md, WINDOW):
        px_i = price[i]
        above = chunk_md[chunk_px > px_i]
        below = chunk_md[chunk_px <= px_i]
        sum_above = np.nansum(above)
        sum_below = np.nansum(below)
        denom = sum_above + sum_below
        imbalance[i] = (sum_above - sum_below) / max(denom, 1e-12)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(imbalance)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        imb_v = imbalance[valid]
        fwd_v = fwd[valid]
        dir_v = (fwd_v > 0).astype(float)

        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        # Imbalance > 0: more density above -> expect price to go down (toward density)
        # Imbalance < 0: more density below -> expect price to go up
        pos_imb = imb_v > 0
        neg_imb = imb_v < 0
        strong_pos = imb_v > 0.5
        strong_neg = imb_v < -0.5

        p_up_pos_imb = p_up(fwd_v, pos_imb & high_es)
        p_up_neg_imb = p_up(fwd_v, neg_imb & high_es)
        p_up_strong_pos = p_up(fwd_v, strong_pos & high_es)
        p_up_strong_neg = p_up(fwd_v, strong_neg & high_es)
        mean_ret_pos_imb = mean_ret(fwd_v, pos_imb & high_es)
        mean_ret_neg_imb = mean_ret(fwd_v, neg_imb & high_es)

        corr_imb_dir, _ = pearsonr(imb_v[high_es], dir_v[high_es])
        corr_es_dir, _ = pearsonr(es_v[high_es], dir_v[high_es])

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_high_es": int(np.sum(high_es)),
            "p_up_positive_imbalance": round(p_up_pos_imb, 4),
            "p_up_negative_imbalance": round(p_up_neg_imb, 4),
            "p_up_strong_positive_imbalance": round(p_up_strong_pos, 4),
            "p_up_strong_negative_imbalance": round(p_up_strong_neg, 4),
            "mean_ret_positive_imbalance": round(mean_ret_pos_imb, 6),
            "mean_ret_negative_imbalance": round(mean_ret_neg_imb, 6),
            "corr_imbalance_direction": round(float(corr_imb_dir), 4),
            "corr_es_direction": round(float(corr_es_dir), 4),
            "imbalance_beats_es": bool(abs(corr_imb_dir) > abs(corr_es_dir)),
            "imbalance_direction_consistent": bool(
                round(p_up_strong_pos, 4) < 0.5 and round(p_up_strong_neg, 4) > 0.5
            ),
        }
    return results


# ---------------------------------------------------------------------------
# 7. Memory-Energy Interaction (Cross-tabulation)
# ---------------------------------------------------------------------------
def task7_memory_energy_interaction(d: DPLData, sym: str) -> dict:
    md = d.memory_density.copy()
    es = d.es.copy()
    fut_ret = d.fut_ret

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]

        # Quintiles
        es_edges = [np.nanpercentile(es_v, p) for p in [0, 20, 40, 60, 80, 100]]
        md_edges = [np.nanpercentile(md_v, p) for p in [0, 20, 40, 60, 80, 100]]

        # Fix edges for uniqueness
        for q in range(1, 6):
            if es_edges[q] <= es_edges[q - 1]:
                es_edges[q] = es_edges[q - 1] + 1e-12
            if md_edges[q] <= md_edges[q - 1]:
                md_edges[q] = md_edges[q - 1] + 1e-12

        es_q = np.zeros(len(es_v), dtype=np.int32)
        md_q = np.zeros(len(md_v), dtype=np.int32)
        for q in range(5):
            if q < 4:
                es_q[(es_v >= es_edges[q]) & (es_v < es_edges[q + 1])] = q
                md_q[(md_v >= md_edges[q]) & (md_v < md_edges[q + 1])] = q
            else:
                es_q[es_v >= es_edges[q]] = q
                md_q[md_v >= md_edges[q]] = q

        # Cross-tabulation
        cell_data = {}
        for eq in range(5):
            for mq in range(5):
                mask = (es_q == eq) & (md_q == mq)
                n = int(np.sum(mask))
                if n < 5:
                    continue
                p = float(np.mean(fwd_v[mask] > 0))
                mr = float(np.mean(fwd_v[mask]))
                cell_data[f"ES_Q{eq}_MD_Q{mq}"] = {
                    "n": n,
                    "p_up": round(p, 4),
                    "mean_ret": round(mr, 6),
                    "es_quintile": eq,
                    "memory_quintile": mq,
                }

        # Identify direction flips within each ES quintile across MD quintiles
        flips = []
        for eq in range(5):
            p_ups = {}
            for mq in range(5):
                key = f"ES_Q{eq}_MD_Q{mq}"
                if key in cell_data:
                    p_ups[mq] = cell_data[key]["p_up"]
            if len(p_ups) >= 2:
                max_p = max(p_ups.values())
                min_p = min(p_ups.values())
                if max_p - min_p > 0.2:
                    flips.append({
                        "es_quintile": eq,
                        "p_up_by_memory_quintile": {str(k): round(v, 4) for k, v in p_ups.items()},
                        "spread": round(max_p - min_p, 4),
                        "direction_flips_with_memory": True,
                    })

        # Marginal: P(up) by ES quintile alone vs ES+memory
        es_marginal = {}
        for eq in range(5):
            mask = es_q == eq
            n = int(np.sum(mask))
            if n >= 5:
                es_marginal[f"ES_Q{eq}"] = round(float(np.mean(fwd_v[mask] > 0)), 4)

        # Best and worst memory combination
        best_cell = max(cell_data.values(), key=lambda x: abs(x["p_up"] - 0.5)) if cell_data else None

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "es_marginal_p_up": es_marginal,
            "cross_tabulation": cell_data,
            "direction_flips_across_memory": flips,
            "n_flips_detected": len(flips),
            "best_cell": best_cell,
            "memory_modulates_es_direction": len(flips) > 0,
        }
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    per_symbol = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Loading {sym}...")
        d = DPLData(sym)
        print(f"  Data: {len(d.price)} bars")

        print(f"  Task 1: Memory Gradient...")
        t1 = task1_memory_gradient(d, sym)

        print(f"  Task 2: Memory Voids...")
        t2 = task2_memory_voids(d, sym)

        print(f"  Task 3: Memory Saturation...")
        t3 = task3_memory_saturation(d, sym)

        print(f"  Task 4: Memory Asymmetry...")
        t4 = task4_memory_asymmetry(d, sym)

        print(f"  Task 5: Memory Clustering...")
        t5 = task5_memory_clustering(d, sym)

        print(f"  Task 6: Memory Imbalance...")
        t6 = task6_memory_imbalance(d, sym)

        print(f"  Task 7: Memory-Energy Interaction...")
        t7 = task7_memory_energy_interaction(d, sym)

        per_symbol[sym] = {
            "memory_gradient": t1,
            "memory_voids": t2,
            "memory_saturation": t3,
            "memory_asymmetry": t4,
            "memory_clustering": t5,
            "memory_imbalance": t6,
            "memory_energy_interaction": t7,
        }

    summary = _build_summary(per_symbol)

    output = {
        "experiment": "CDER-Memory-Geometry",
        "title": "Context-Dependent Energy Release: Memory Geometry Analysis",
        "per_symbol": per_symbol,
        "summary": summary,
    }

    out_path = Path(__file__).parent / "reports" / "cder_memory_geometry.json"
    out_path.write_text(json.dumps(_clean(output), indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"CDER Memory Geometry complete -> {out_path}")


def _build_summary(per_symbol: dict) -> dict:
    """Extract key cross-cutting findings."""

    q1_gradient = {"gradient_predicts_direction": False, "gradient_sign_consistency": 0}
    q2_voids = {"voids_cause_explosive_releases": False, "void_direction_consistency": 0}
    q3_saturation = {"saturation_predicts_reversals": False, "saturation_reversal_rate": 0}
    q4_asymmetry = {"asymmetry_beats_distance": False, "asymmetry_avg_corr": 0}
    q5_clustering = {"clustering_predicts_direction": False, "cluster_spread_avg": 0}
    q6_imbalance = {"imbalance_predicts_direction": False, "imbalance_avg_corr": 0}
    q7_interaction = {"memory_modulates_es": False, "n_flips": 0}

    # Collect per-horizon, per-symbol metrics
    gradients = []
    void_spreads = []
    sat_spreads = []
    asym_corrs = []
    dist_corrs = []
    cluster_spreads = []
    imb_corrs = []
    es_corrs_asym = []
    es_corrs_imb = []
    total_flips = 0
    symb_with_flips = 0

    for sym, r in per_symbol.items():
        # Gradient
        tg = r.get("memory_gradient", {})
        for hl, v in tg.items():
            if "corr_gradient_direction" in v:
                gradients.append(v["corr_gradient_direction"])

        # Voids
        tv = r.get("memory_voids", {})
        for hl, v in tv.items():
            if "explode_ratio_void_high_es" in v and "explode_ratio_nonvoid_high_es" in v:
                void_spreads.append(v["explode_ratio_void_high_es"] - v["explode_ratio_nonvoid_high_es"])

        # Saturation
        ts = r.get("memory_saturation", {})
        for hl, v in ts.items():
            if "p_up_saturation_high_es" in v and "p_up_nonsaturation_high_es" in v:
                p_sat = v.get("p_up_saturation_high_es", 0.5)
                p_non = v.get("p_up_nonsaturation_high_es", 0.5)
                sat_spreads.append(abs(p_sat - 0.5) - abs(p_non - 0.5))
                if v.get("saturation_reversal_rate") is not None:
                    q3_saturation["saturation_reversal_rate"] = max(
                        q3_saturation["saturation_reversal_rate"],
                        v["saturation_reversal_rate"]
                    )

        # Asymmetry
        ta = r.get("memory_asymmetry", {})
        for hl, v in ta.items():
            if "corr_asymmetry_direction" in v:
                asym_corrs.append(v["corr_asymmetry_direction"])
            if "corr_distance_direction" in v:
                dist_corrs.append(v["corr_distance_direction"])
            if v.get("asymmetry_beats_distance"):
                q4_asymmetry["asymmetry_beats_distance"] = True

        # Clustering
        tc = r.get("memory_clustering", {})
        for hl, v in tc.items():
            if "max_cluster_spread_high_es" in v:
                cluster_spreads.append(v["max_cluster_spread_high_es"])
            if v.get("any_cluster_flip"):
                q5_clustering["clustering_predicts_direction"] = True

        # Imbalance
        ti = r.get("memory_imbalance", {})
        for hl, v in ti.items():
            if "corr_imbalance_direction" in v:
                imb_corrs.append(v["corr_imbalance_direction"])
            if v.get("imbalance_beats_es"):
                q6_imbalance["imbalance_predicts_direction"] = True

        # Interaction
        t7 = r.get("memory_energy_interaction", {})
        for hl, v in t7.items():
            nf = v.get("n_flips_detected", 0)
            total_flips += nf
            if nf > 0:
                symb_with_flips += 1
            if v.get("memory_modulates_es_direction"):
                q7_interaction["memory_modulates_es"] = True
        q7_interaction["n_flips"] = int(total_flips)

    # Aggregate answers
    if gradients:
        q1_gradient["gradient_predicts_direction"] = bool(np.mean(np.abs(gradients)) > 0.05)
        q1_gradient["gradient_sign_consistency"] = round(float(np.mean(gradients)), 4)

    if void_spreads:
        q2_voids["voids_cause_explosive_releases"] = bool(np.mean(void_spreads) > 0)
        q2_voids["avg_explode_delta"] = round(float(np.mean(void_spreads)), 4)

    if sat_spreads:
        avg_sat_improvement = float(np.mean(sat_spreads))
        q3_saturation["saturation_predicts_reversals"] = avg_sat_improvement > 0.05

    if asym_corrs:
        q4_asymmetry["asymmetry_avg_corr"] = round(float(np.mean(np.abs(asym_corrs))), 4)

    if cluster_spreads:
        q5_clustering["cluster_spread_avg"] = round(float(np.mean(cluster_spreads)), 4)

    if imb_corrs:
        q6_imbalance["imbalance_avg_corr"] = round(float(np.mean(np.abs(imb_corrs))), 4)

    # Best predictor: compare abs corrs
    best = "ES (energy storage) baseline"
    best_score = 0
    if gradients:
        g = float(np.mean(np.abs(gradients)))
        if g > best_score:
            best_score = g
            best = "Memory Gradient"
    if asym_corrs:
        a = float(np.mean(np.abs(asym_corrs)))
        if a > best_score:
            best_score = a
            best = "Memory Asymmetry"
    if imb_corrs:
        i = float(np.mean(np.abs(imb_corrs)))
        if i > best_score:
            best_score = i
            best = "Memory Imbalance"

    return {
        "n_symbols": len(per_symbol),
        "tested_horizons": [TEST_HL[h] for h in TEST_HORIZONS],
        "q1_memory_gradient": {
            "does_memory_gradient_predict_direction": q1_gradient["gradient_predicts_direction"],
            "avg_gradient_direction_correlation": q1_gradient["gradient_sign_consistency"],
            "interpretation": "Gradient predicts direction" if q1_gradient["gradient_predicts_direction"]
            else "Gradient does NOT reliably predict direction",
        },
        "q2_memory_voids": {
            "do_voids_cause_explosive_releases": q2_voids["voids_cause_explosive_releases"],
            "avg_explode_ratio_delta": q2_voids.get("avg_explode_delta", 0),
            "interpretation": "Voids amplify explosive releases" if q2_voids["voids_cause_explosive_releases"]
            else "Voids do NOT amplify explosive releases",
        },
        "q3_memory_saturation": {
            "does_saturation_predict_reversals": q3_saturation["saturation_predicts_reversals"],
            "max_saturation_reversal_rate": q3_saturation["saturation_reversal_rate"],
            "interpretation": "Saturation predicts reversals" if q3_saturation["saturation_predicts_reversals"]
            else "Saturation does NOT reliably predict reversals",
        },
        "q4_memory_asymmetry": {
            "asymmetry_beats_simple_distance": q4_asymmetry["asymmetry_beats_distance"],
            "avg_abs_asymmetry_correlation": q4_asymmetry["asymmetry_avg_corr"],
            "interpretation": "Asymmetry outperforms distance" if q4_asymmetry["asymmetry_beats_distance"]
            else "Simple distance performs as well or better",
        },
        "q5_memory_clustering": {
            "clustering_predicts_direction": q5_clustering["clustering_predicts_direction"],
            "avg_cluster_spread": q5_clustering["cluster_spread_avg"],
            "interpretation": "Cluster membership predicts direction" if q5_clustering["clustering_predicts_direction"]
            else "Cluster membership does NOT reliably predict direction",
        },
        "q6_memory_imbalance": {
            "imbalance_predicts_direction": q6_imbalance["imbalance_predicts_direction"],
            "avg_abs_imbalance_correlation": q6_imbalance["imbalance_avg_corr"],
            "interpretation": "Imbalance predicts direction" if q6_imbalance["imbalance_predicts_direction"]
            else "Imbalance does NOT reliably predict direction",
        },
        "q7_memory_energy_interaction": {
            "memory_modulates_es_direction": q7_interaction["memory_modulates_es"],
            "total_direction_flips_across_memory_quintiles": q7_interaction["n_flips"],
            "interpretation": "Memory modulates ES direction" if q7_interaction["memory_modulates_es"]
            else "Memory does NOT modulate ES direction",
        },
        "best_memory_geometry_predictor": best,
        "best_predictor_avg_abs_correlation": round(best_score, 4),
    }


if __name__ == "__main__":
    main()
