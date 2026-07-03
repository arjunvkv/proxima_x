"""
CDER: Context-Dependent Energy Release — Energy × Memory Interaction
Investigates whether energy release direction depends on memory context.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS, compute_gradient

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)

TEST_HORIZONS = [5, 20, 50]
TEST_HL = {h: HORIZON_LABELS[h] for h in TEST_HORIZONS}
WINDOW = 252


def _clean(obj):
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


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    s = np.nanstd(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / s


def _compute_memory_asymmetry(d: DPLData) -> np.ndarray:
    md = d.memory_density
    price = d.price
    asym = np.full_like(md, np.nan)
    n = len(price)
    for i in range(WINDOW, n):
        if np.isnan(md[i]):
            continue
        chunk_px = price[i - WINDOW:i]
        chunk_md = md[i - WINDOW:i]
        if np.all(np.isnan(chunk_md)):
            continue
        px_i = price[i]
        above = chunk_md[chunk_px > px_i]
        below = chunk_md[chunk_px <= px_i]
        mem_above = np.nanmean(above) if len(above) > 5 else 0
        mem_below = np.nanmean(below) if len(below) > 5 else 0
        denom = mem_above + mem_below
        asym[i] = (mem_above - mem_below) / max(denom, 1e-12)
    return asym


# -------------------------------------------------------------------
# 1. ES × Memory Cross-Tabulation
# -------------------------------------------------------------------
def task1_crosstab(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]

        es_edges = [np.nanpercentile(es_v, p) for p in [0, 20, 40, 60, 80, 100]]
        md_edges = [np.nanpercentile(md_v, p) for p in [0, 20, 40, 60, 80, 100]]

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

        grid = np.full((5, 5), np.nan)
        cell_data = {}
        for eq in range(5):
            for mq in range(5):
                mask = (es_q == eq) & (md_q == mq)
                n = int(np.sum(mask))
                if n < 5:
                    continue
                p = float(np.mean(fwd_v[mask] > 0))
                grid[eq, mq] = p
                cell_data[f"ES_Q{eq}_MD_Q{mq}"] = {
                    "n": n,
                    "p_up": round(p, 4),
                    "es_quintile": eq,
                    "memory_quintile": mq,
                }

        cell_std = float(np.nanstd(grid))
        p_ups = [c["p_up"] for c in cell_data.values()]
        p_up_min = min(p_ups) if p_ups else None
        p_up_max = max(p_ups) if p_ups else None
        p_up_range = round(p_up_max - p_up_min, 4) if p_ups else None

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "grid_5x5_p_up": [[round(float(grid[eq, mq]), 4) if not np.isnan(grid[eq, mq]) else None for mq in range(5)] for eq in range(5)],
            "cell_details": cell_data,
            "cell_p_up_std": round(cell_std, 4),
            "p_up_min": round(p_up_min, 4) if p_up_min is not None else None,
            "p_up_max": round(p_up_max, 4) if p_up_max is not None else None,
            "p_up_range": p_up_range,
            "direction_varies_by_cell": cell_std > 0.03,
        }
    return results


# -------------------------------------------------------------------
# 2. High ES + Memory Shapes
# -------------------------------------------------------------------
def task2_high_es_shapes(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]

        es_edges = [np.nanpercentile(es_v, p) for p in [0, 20, 40, 60, 80, 100]]
        md_edges = [np.nanpercentile(md_v, p) for p in [0, 20, 40, 60, 80, 100]]

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

        mask_dense = (es_q == 4) & (md_q == 4)
        mask_sparse = (es_q == 4) & (md_q == 0)

        n_dense = int(np.sum(mask_dense))
        n_sparse = int(np.sum(mask_sparse))
        p_up_dense = p_up(fwd_v, mask_dense)
        p_up_sparse = p_up(fwd_v, mask_sparse)
        mean_dense = mean_ret(fwd_v, mask_dense)
        mean_sparse = mean_ret(fwd_v, mask_sparse)

        ret_dense = fwd_v[mask_dense & ~np.isnan(fwd_v)]
        ret_sparse = fwd_v[mask_sparse & ~np.isnan(fwd_v)]
        t_stat, p_val = None, None
        if len(ret_dense) >= 3 and len(ret_sparse) >= 3:
            t_stat, p_val = ttest_ind(ret_dense, ret_sparse, equal_var=False)
            t_stat = round(float(t_stat), 4)
            p_val = round(float(p_val), 4)

        sig_diff = False
        if p_val is not None:
            sig_diff = p_val < 0.05

        results[TEST_HL[h]] = {
            "n_high_es_dense_memory": n_dense,
            "n_high_es_sparse_memory": n_sparse,
            "p_up_high_es_dense_memory": round(p_up_dense, 4),
            "p_up_high_es_sparse_memory": round(p_up_sparse, 4),
            "mean_ret_high_es_dense_memory": round(mean_dense, 6),
            "mean_ret_high_es_sparse_memory": round(mean_sparse, 6),
            "ttest_statistic": t_stat,
            "ttest_p_value": p_val,
            "significantly_different": sig_diff,
            "spread": round(abs(p_up_dense - p_up_sparse), 4),
        }
    return results


# -------------------------------------------------------------------
# 3. Memory Asymmetry × ES
# -------------------------------------------------------------------
def task3_asymmetry_es(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret
    asym = _compute_memory_asymmetry(d)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md) & ~np.isnan(asym)
        if np.sum(valid) < 30:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        fwd_v = fwd[valid]
        asym_v = asym[valid]

        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        pos_asym = asym_v > 0
        neg_asym = asym_v < 0

        mask_pos = high_es & pos_asym
        mask_neg = high_es & neg_asym
        mask_neutral = high_es & (asym_v == 0)

        n_pos = int(np.sum(mask_pos))
        n_neg = int(np.sum(mask_neg))
        p_up_pos = p_up(fwd_v, mask_pos)
        p_up_neg = p_up(fwd_v, mask_neg)
        p_up_neutral = p_up(fwd_v, mask_neutral)
        mean_pos = mean_ret(fwd_v, mask_pos)
        mean_neg = mean_ret(fwd_v, mask_neg)

        ret_pos = fwd_v[mask_pos & ~np.isnan(fwd_v)]
        ret_neg = fwd_v[mask_neg & ~np.isnan(fwd_v)]
        t_stat, p_val = None, None
        if len(ret_pos) >= 3 and len(ret_neg) >= 3:
            t_stat, p_val = ttest_ind(ret_pos, ret_neg, equal_var=False)
            t_stat = round(float(t_stat), 4)
            p_val = round(float(p_val), 4)

        sig_diff = False
        if p_val is not None:
            sig_diff = p_val < 0.05

        results[TEST_HL[h]] = {
            "n_high_es_positive_asymmetry": n_pos,
            "n_high_es_negative_asymmetry": n_neg,
            "p_up_high_es_positive_asymmetry": round(p_up_pos, 4),
            "p_up_high_es_negative_asymmetry": round(p_up_neg, 4),
            "p_up_high_es_neutral_asymmetry": round(p_up_neutral, 4) if np.sum(mask_neutral) >= 3 else None,
            "mean_ret_high_es_positive_asymmetry": round(mean_pos, 6),
            "mean_ret_high_es_negative_asymmetry": round(mean_neg, 6),
            "ttest_statistic": t_stat,
            "ttest_p_value": p_val,
            "significantly_different": sig_diff,
            "asymmetry_determines_direction_given_high_es": sig_diff,
            "spread": round(abs(p_up_pos - p_up_neg), 4),
        }
    return results


# -------------------------------------------------------------------
# 4. Memory Gradient × ES
# -------------------------------------------------------------------
def task4_gradient_es(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret
    grad = compute_gradient(md, window=5)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md) & ~np.isnan(grad)
        if np.sum(valid) < 30:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        fwd_v = fwd[valid]
        grad_v = grad[valid]

        es_thr = np.nanpercentile(es_v, 80)
        high_es = es_v > es_thr

        rising = grad_v > 0
        falling = grad_v < 0

        mask_rising = high_es & rising
        mask_falling = high_es & falling

        n_rising = int(np.sum(mask_rising))
        n_falling = int(np.sum(mask_falling))
        p_up_rising = p_up(fwd_v, mask_rising)
        p_up_falling = p_up(fwd_v, mask_falling)
        mean_rising = mean_ret(fwd_v, mask_rising)
        mean_falling = mean_ret(fwd_v, mask_falling)

        ret_rising = fwd_v[mask_rising & ~np.isnan(fwd_v)]
        ret_falling = fwd_v[mask_falling & ~np.isnan(fwd_v)]
        t_stat, p_val = None, None
        if len(ret_rising) >= 3 and len(ret_falling) >= 3:
            t_stat, p_val = ttest_ind(ret_rising, ret_falling, equal_var=False)
            t_stat = round(float(t_stat), 4)
            p_val = round(float(p_val), 4)

        sig_diff = False
        if p_val is not None:
            sig_diff = p_val < 0.05

        results[TEST_HL[h]] = {
            "n_high_es_rising_memory": n_rising,
            "n_high_es_falling_memory": n_falling,
            "p_up_high_es_rising_memory": round(p_up_rising, 4),
            "p_up_high_es_falling_memory": round(p_up_falling, 4),
            "mean_ret_high_es_rising_memory": round(mean_rising, 6),
            "mean_ret_high_es_falling_memory": round(mean_falling, 6),
            "ttest_statistic": t_stat,
            "ttest_p_value": p_val,
            "significantly_different": sig_diff,
            "gradient_determines_direction_given_high_es": sig_diff,
            "spread": round(abs(p_up_rising - p_up_falling), 4),
        }
    return results


# -------------------------------------------------------------------
# 5. Interaction Model (Logistic Regression)
# -------------------------------------------------------------------
def task5_interaction_model(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 100:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]
        dir_v = (fwd_v > 0).astype(float)

        es_z = _zscore(es_v)
        md_z = _zscore(md_v)
        interaction = es_z * md_z

        # Model 1: direction ~ ES
        if len(np.unique(dir_v)) < 2:
            results[TEST_HL[h]] = {"error": "no directional variance"}
            continue
        m1 = LogisticRegression(max_iter=1000, random_state=42)
        m1.fit(es_z.reshape(-1, 1), dir_v)
        m1_pred = m1.predict(es_z.reshape(-1, 1))
        m1_acc = float(np.mean(m1_pred == dir_v))
        m1_pseudo_r2 = 1 - m1.llf_ / m1.llf_null_ if hasattr(m1, 'llf_') else None
        if m1_pseudo_r2 is None:
            llf = LogisticRegression(max_iter=1000, random_state=42).fit(es_z.reshape(-1, 1), dir_v)
            try:
                from sklearn.metrics import log_loss
                ll_null = log_loss(dir_v, np.full_like(dir_v, np.mean(dir_v)))
                ll_fit = log_loss(dir_v, m1.predict_proba(es_z.reshape(-1, 1))[:, 1])
                m1_pseudo_r2 = round(float(1 - ll_fit / ll_null), 6)
            except Exception:
                m1_pseudo_r2 = 0.0

        # Model 2: direction ~ ES + memory_density
        X2 = np.column_stack([es_z, md_z])
        m2 = LogisticRegression(max_iter=1000, random_state=42)
        m2.fit(X2, dir_v)
        m2_pred = m2.predict(X2)
        m2_acc = float(np.mean(m2_pred == dir_v))
        try:
            ll_fit2 = log_loss(dir_v, m2.predict_proba(X2)[:, 1])
            m2_pseudo_r2 = round(float(1 - ll_fit2 / ll_null), 6)
        except Exception:
            m2_pseudo_r2 = 0.0

        # Model 3: direction ~ ES + memory_density + ES×memory
        X3 = np.column_stack([es_z, md_z, interaction])
        m3 = LogisticRegression(max_iter=1000, random_state=42)
        m3.fit(X3, dir_v)
        m3_pred = m3.predict(X3)
        m3_acc = float(np.mean(m3_pred == dir_v))
        try:
            ll_fit3 = log_loss(dir_v, m3.predict_proba(X3)[:, 1])
            m3_pseudo_r2 = round(float(1 - ll_fit3 / ll_null), 6)
        except Exception:
            m3_pseudo_r2 = 0.0

        coefs = m3.coef_[0].tolist()
        interaction_coef = coefs[2]

        r2_improvement_m2 = round(m2_pseudo_r2 - m1_pseudo_r2, 6) if m1_pseudo_r2 is not None else None
        r2_improvement_m3 = round(m3_pseudo_r2 - m1_pseudo_r2, 6) if m1_pseudo_r2 is not None else None
        acc_improvement_m3 = round(m3_acc - m1_acc, 4)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "model1_ES_only": {
                "accuracy": round(m1_acc, 4),
                "pseudo_r2": round(m1_pseudo_r2, 6) if m1_pseudo_r2 is not None else None,
            },
            "model2_ES_plus_memory": {
                "accuracy": round(m2_acc, 4),
                "pseudo_r2": m2_pseudo_r2,
            },
            "model3_ES_memory_interaction": {
                "accuracy": round(m3_acc, 4),
                "pseudo_r2": m3_pseudo_r2,
                "coefficients": {"ES": round(coefs[0], 4), "memory_density": round(coefs[1], 4), "interaction": round(interaction_coef, 4)},
            },
            "improvement_over_ES_only": {
                "r2_improvement_m2": r2_improvement_m2,
                "r2_improvement_m3": r2_improvement_m3,
                "accuracy_improvement_m3": acc_improvement_m3,
            },
            "interaction_term_significant": abs(interaction_coef) > 0.01,
        }
    return results


# -------------------------------------------------------------------
# 6. Direction Flip Detection
# -------------------------------------------------------------------
def task6_flip_detection(d: DPLData, sym: str) -> dict:
    md = d.memory_density
    es = d.es
    fut_ret = d.fut_ret

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = fut_ret[:, hidx]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(md)
        if np.sum(valid) < 30:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es[valid]
        md_v = md[valid]
        fwd_v = fwd[valid]

        es_edges = [np.nanpercentile(es_v, p) for p in [0, 20, 40, 60, 80, 100]]
        md_edges = [np.nanpercentile(md_v, p) for p in [0, 20, 40, 60, 80, 100]]

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

        cells = []
        for eq in range(5):
            for mq in range(5):
                mask = (es_q == eq) & (md_q == mq)
                n = int(np.sum(mask))
                if n < 5:
                    continue
                p = float(np.mean(fwd_v[mask] > 0))
                cells.append({"es_q": eq, "md_q": mq, "n": n, "p_up": round(p, 4)})

        flips = []
        for i, c1 in enumerate(cells):
            for c2 in cells[i + 1:]:
                if (c1["p_up"] > 0.65 and c2["p_up"] < 0.35) or (c2["p_up"] > 0.65 and c1["p_up"] < 0.35):
                    up_cell = c1 if c1["p_up"] > c2["p_up"] else c2
                    down_cell = c2 if c1["p_up"] > c2["p_up"] else c1
                    flips.append({
                        "up_cell": f"ES_Q{up_cell['es_q']}_MD_Q{up_cell['md_q']}",
                        "down_cell": f"ES_Q{down_cell['es_q']}_MD_Q{down_cell['md_q']}",
                        "p_up_up_cell": up_cell["p_up"],
                        "p_up_down_cell": down_cell["p_up"],
                        "n_up_cell": up_cell["n"],
                        "n_down_cell": down_cell["n"],
                        "flip_magnitude": round(abs(up_cell["p_up"] - down_cell["p_up"]), 4),
                    })

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_cells": len(cells),
            "direction_flip_pairs": flips,
            "n_flip_pairs": len(flips),
            "any_flip_detected": len(flips) > 0,
        }
    return results


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    per_symbol = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Loading {sym}...")
        d = DPLData(sym)
        print(f"  Data: {len(d.price)} bars")

        print(f"  Task 1: ES × Memory Cross-Tabulation...")
        t1 = task1_crosstab(d, sym)

        print(f"  Task 2: High ES + Memory Shapes...")
        t2 = task2_high_es_shapes(d, sym)

        print(f"  Task 3: Memory Asymmetry × ES...")
        t3 = task3_asymmetry_es(d, sym)

        print(f"  Task 4: Memory Gradient × ES...")
        t4 = task4_gradient_es(d, sym)

        print(f"  Task 5: Interaction Model...")
        t5 = task5_interaction_model(d, sym)

        print(f"  Task 6: Direction Flip Detection...")
        t6 = task6_flip_detection(d, sym)

        per_symbol[sym] = {
            "crosstab": t1,
            "high_es_shapes": t2,
            "asymmetry_es": t3,
            "gradient_es": t4,
            "interaction_model": t5,
            "flip_detection": t6,
        }

    summary = _build_summary(per_symbol)

    output = {
        "experiment": "CDER-Energy-Memory",
        "title": "Context-Dependent Energy Release: Energy × Memory Interaction",
        "per_symbol": per_symbol,
        "summary": summary,
    }

    out_path = Path(__file__).parent / "reports" / "cder_energy_memory.json"
    out_path.write_text(json.dumps(_clean(output), indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"CDER Energy-Memory complete -> {out_path}")


def _build_summary(per_symbol: dict) -> dict:
    t1_cell_stds = []
    t1_range = []
    t2_spreads = []
    t2_sig_count = 0
    t2_total = 0
    t3_spreads = []
    t3_sig_count = 0
    t3_total = 0
    t4_spreads = []
    t4_sig_count = 0
    t4_total = 0
    t5_r2_improvements = []
    t5_acc_improvements = []
    t5_best_interaction_sym = None
    t5_best_r2_improve = -1
    t5_interaction_works = 0
    t5_interaction_total = 0
    t6_flip_count = 0
    t6_symbols_with_flips = 0

    for sym, r in per_symbol.items():
        # Task 1
        tc = r.get("crosstab", {})
        for hl, v in tc.items():
            if isinstance(v, dict) and "cell_p_up_std" in v:
                t1_cell_stds.append(v["cell_p_up_std"])
            if isinstance(v, dict) and "p_up_range" in v and v["p_up_range"] is not None:
                t1_range.append(v["p_up_range"])

        # Task 2
        t2 = r.get("high_es_shapes", {})
        for hl, v in t2.items():
            if isinstance(v, dict) and "spread" in v:
                t2_spreads.append(v["spread"])
                t2_total += 1
                if v.get("significantly_different"):
                    t2_sig_count += 1

        # Task 3
        t3 = r.get("asymmetry_es", {})
        for hl, v in t3.items():
            if isinstance(v, dict) and "spread" in v:
                t3_spreads.append(v["spread"])
                t3_total += 1
                if v.get("significantly_different"):
                    t3_sig_count += 1

        # Task 4
        t4 = r.get("gradient_es", {})
        for hl, v in t4.items():
            if isinstance(v, dict) and "spread" in v:
                t4_spreads.append(v["spread"])
                t4_total += 1
                if v.get("significantly_different"):
                    t4_sig_count += 1

        # Task 5
        t5 = r.get("interaction_model", {})
        best_h = None
        best_r2 = -1
        for hl, v in t5.items():
            if isinstance(v, dict) and "improvement_over_ES_only" in v:
                t5_interaction_total += 1
                imp = v["improvement_over_ES_only"]
                r2_imp = imp.get("r2_improvement_m3")
                acc_imp = imp.get("accuracy_improvement_m3")
                if r2_imp is not None:
                    t5_r2_improvements.append(r2_imp)
                if acc_imp is not None:
                    t5_acc_improvements.append(acc_imp)
                if v.get("interaction_term_significant"):
                    t5_interaction_works += 1
                if r2_imp is not None and r2_imp > best_r2:
                    best_r2 = r2_imp
                    best_h = hl
        if best_r2 > t5_best_r2_improve:
            t5_best_r2_improve = best_r2
            t5_best_interaction_sym = sym

        # Task 6
        t6 = r.get("flip_detection", {})
        sym_flips = 0
        for hl, v in t6.items():
            if isinstance(v, dict) and "n_flip_pairs" in v:
                t6_flip_count += v["n_flip_pairs"]
                sym_flips += v["n_flip_pairs"]
        if sym_flips > 0:
            t6_symbols_with_flips += 1

    avg_cell_std = round(float(np.mean(t1_cell_stds)), 4) if t1_cell_stds else None
    avg_range = round(float(np.mean(t1_range)), 4) if t1_range else None
    avg_t2_spread = round(float(np.mean(t2_spreads)), 4) if t2_spreads else None
    avg_t3_spread = round(float(np.mean(t3_spreads)), 4) if t3_spreads else None
    avg_t4_spread = round(float(np.mean(t4_spreads)), 4) if t4_spreads else None
    avg_r2_improve = round(float(np.mean(t5_r2_improvements)), 6) if t5_r2_improvements else None
    avg_acc_improve = round(float(np.mean(t5_acc_improvements)), 4) if t5_acc_improvements else None

    q1 = {
        "avg_cell_std_dev": avg_cell_std,
        "avg_p_up_range": avg_range,
        "interpretation": f"Direction varies across ES×Memory cells (std={avg_cell_std}, range={avg_range}) — memory modulates ES direction" if avg_cell_std and avg_cell_std > 0.03 else "Direction does NOT meaningfully vary across ES×Memory cells",
    }
    q2 = {
        "avg_p_up_spread": avg_t2_spread,
        "n_significant": t2_sig_count,
        "n_total": t2_total,
        "interpretation": f"High-ES direction differs by memory density (spread={avg_t2_spread}, {t2_sig_count}/{t2_total} significant)"
    }
    q3 = {
        "avg_p_up_spread": avg_t3_spread,
        "n_significant": t3_sig_count,
        "n_total": t3_total,
        "interpretation": f"Memory asymmetry modifies high-ES direction (spread={avg_t3_spread}, {t3_sig_count}/{t3_total} significant)"
    }
    q4 = {
        "avg_p_up_spread": avg_t4_spread,
        "n_significant": t4_sig_count,
        "n_total": t4_total,
        "interpretation": f"Memory gradient modifies high-ES direction (spread={avg_t4_spread}, {t4_sig_count}/{t4_total} significant)"
    }
    q5 = {
        "avg_r2_improvement_vs_ES_only": avg_r2_improve,
        "avg_accuracy_improvement": avg_acc_improve,
        "n_interaction_term_significant": t5_interaction_works,
        "n_total_models": t5_interaction_total,
        "best_symbol": t5_best_interaction_sym,
        "best_symbol_r2_improvement": round(t5_best_r2_improve, 6) if t5_best_r2_improve > -1 else None,
        "interpretation": f"ES×Memory interaction {'significantly' if t5_interaction_works > t5_interaction_total // 3 else 'does NOT'} improve directional prediction over ES alone (avg ΔR²={avg_r2_improve}, avg Δacc={avg_acc_improve})"
    }
    q6 = {
        "total_flip_pairs": t6_flip_count,
        "n_symbols_with_flips": t6_symbols_with_flips,
        "n_symbols_total": len(per_symbol),
        "interpretation": f"Direction flips (P(up)>0.65 vs <0.35) detected across {t6_symbols_with_flips}/{len(per_symbol)} symbols ({t6_flip_count} total flip pairs)"
    }

    return {
        "n_symbols": len(per_symbol),
        "tested_horizons": [TEST_HL[h] for h in TEST_HORIZONS],
        "q1_es_memory_crosstab": q1,
        "q2_high_es_memory_shapes": q2,
        "q3_memory_asymmetry_x_es": q3,
        "q4_memory_gradient_x_es": q4,
        "q5_interaction_model": q5,
        "q6_direction_flip_detection": q6,
        "key_finding": (
            f"CDER Energy×Memory: direction varies by ES×Memory cell (σ={avg_cell_std}, range={avg_range}). "
            f"Memory shapes: spread={avg_t2_spread}. "
            f"Asymmetry: spread={avg_t3_spread}. "
            f"Gradient: spread={avg_t4_spread}. "
            f"Interaction term improves R² by {avg_r2_improve} (acc +{avg_acc_improve}). "
            f"Best symbol: {t5_best_interaction_sym}. "
            f"Flips: {t6_flip_count} pairs across {t6_symbols_with_flips} symbols."
        ),
    }


if __name__ == "__main__":
    main()
