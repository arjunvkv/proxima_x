"""
CDER: Context-Dependent Energy Release — Multi-Timeframe Context Fields
Investigates whether directional release depends on multi-timeframe ES context.
Hypothesis: sign inversions are actually timeframe conflicts.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS, compute_gradient

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)

TEST_HORIZONS = [5, 20, 50]
TEST_HL = {h: HORIZON_LABELS[h] for h in TEST_HORIZONS}

DIR = {"UP": 1, "DOWN": 0}
DIR_LABELS = ["DOWN", "UP"]


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


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    s = np.full_like(x, np.nan)
    for i in range(window, len(x)):
        s[i] = np.nanmean(x[i - window:i])
    return s


def _p_up(fwd: np.ndarray, mask: np.ndarray) -> float:
    m = mask & ~np.isnan(fwd)
    if np.sum(m) < 3:
        return 0.5
    return float(np.mean(fwd[m] > 0))


def _mean_ret(fwd: np.ndarray, mask: np.ndarray) -> float:
    m = mask & ~np.isnan(fwd)
    if np.sum(m) < 3:
        return 0.0
    return float(np.mean(fwd[m]))


# ===================================================================
# 1. Nested ES Context
# ===================================================================
def task1_nested_es_context(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    es_fast = es_raw
    es_medium = _sma(es_raw, 20)
    es_slow = _sma(es_raw, 50)
    es_macro = _sma(es_raw, 100)

    grad_fast = compute_gradient(es_fast, window=5)
    grad_medium = compute_gradient(es_medium, window=5)
    grad_slow = compute_gradient(es_slow, window=5)
    grad_macro = compute_gradient(es_macro, window=10)

    fast_dir = np.full(len(es_fast), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(float)
    medium_dir = np.full(len(es_medium), np.nan)
    medium_dir[5:] = (grad_medium[5:] > 0).astype(float)
    slow_dir = np.full(len(es_slow), np.nan)
    slow_dir[5:] = (grad_slow[5:] > 0).astype(float)
    macro_dir = np.full(len(es_macro), np.nan)
    macro_dir[10:] = (grad_macro[10:] > 0).astype(float)

    results = {}
    for hi, h in enumerate(HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_fast) & ~np.isnan(es_medium) & ~np.isnan(es_slow)
                 & ~np.isnan(es_macro) & ~np.isnan(fast_dir) & ~np.isnan(medium_dir)
                 & ~np.isnan(slow_dir) & ~np.isnan(macro_dir) & ~np.isnan(fwd))
        if np.sum(valid) < 30:
            results[HORIZON_LABELS[h]] = {"error": "insufficient data"}
            continue

        es_f = es_fast[valid]
        es_m = es_medium[valid]
        es_s = es_slow[valid]
        es_ma = es_macro[valid]
        fd = fast_dir[valid]
        md = medium_dir[valid]
        sd = slow_dir[valid]
        mad = macro_dir[valid]
        fw = fwd[valid]

        # Fast rising within slow context
        es_f_thr = np.nanpercentile(es_f, 70)
        es_s_thr = np.nanpercentile(es_s, 70)

        # Fast-high-rising in slow-up vs slow-down context
        mask_fast_high_rising = (es_f > es_f_thr) & (fd == 1)
        mask_slow_up = (sd == 1)
        mask_slow_down = (sd == 0)

        mask_fhr_slow_up = mask_fast_high_rising & mask_slow_up
        mask_fhr_slow_down = mask_fast_high_rising & mask_slow_down

        p_up_fhr_up = _p_up(fw, mask_fhr_slow_up)
        p_up_fhr_down = _p_up(fw, mask_fhr_slow_down)
        n_fhr_up = int(np.sum(mask_fhr_slow_up))
        n_fhr_down = int(np.sum(mask_fhr_slow_down))

        # Fast rising with medium rising/falling
        mask_fast_high = es_f > es_f_thr
        mask_med_up = (md == 1)
        mask_med_down = (md == 0)
        mask_fh_med_up = mask_fast_high & mask_med_up
        mask_fh_med_down = mask_fast_high & mask_med_down
        p_up_fh_med_up = _p_up(fw, mask_fh_med_up)
        p_up_fh_med_down = _p_up(fw, mask_fh_med_down)

        results[HORIZON_LABELS[h]] = {
            "n_valid": int(np.sum(valid)),
            "p_up_fast_high_rising_in_slow_up": round(p_up_fhr_up, 4),
            "p_up_fast_high_rising_in_slow_down": round(p_up_fhr_down, 4),
            "n_fast_high_rising_slow_up": n_fhr_up,
            "n_fast_high_rising_slow_down": n_fhr_down,
            "nested_context_spread_fast_rising": round(abs(p_up_fhr_up - p_up_fhr_down), 4),
            "p_up_fast_high_in_medium_up": round(p_up_fh_med_up, 4),
            "p_up_fast_high_in_medium_down": round(p_up_fh_med_down, 4),
            "medium_context_spread": round(abs(p_up_fh_med_up - p_up_fh_med_down), 4),
        }

    return results


# ===================================================================
# 2. Context Field Model
# ===================================================================
def task2_context_field_model(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    es_fast = es_raw
    es_medium = _sma(es_raw, 20)
    es_slow = _sma(es_raw, 50)

    grad_fast = compute_gradient(es_fast, window=5)
    grad_medium = compute_gradient(es_medium, window=5)
    grad_slow = compute_gradient(es_slow, window=5)

    fast_dir = np.full(len(es_fast), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(int)
    medium_dir = np.full(len(es_medium), np.nan)
    medium_dir[5:] = (grad_medium[5:] > 0).astype(int)
    slow_dir = np.full(len(es_slow), np.nan)
    slow_dir[5:] = (grad_slow[5:] > 0).astype(int)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_fast) & ~np.isnan(es_medium) & ~np.isnan(es_slow)
                 & ~np.isnan(fast_dir) & ~np.isnan(medium_dir) & ~np.isnan(slow_dir)
                 & ~np.isnan(fwd))
        if np.sum(valid) < 100:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        fd = fast_dir[valid]
        md = medium_dir[valid]
        sd = slow_dir[valid]
        fw = fwd[valid]

        context_states = {}
        combos = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        for combo in combos:
            f, m, s = combo
            mask = (fd == f) & (md == m) & (sd == s)
            n = int(np.sum(mask))
            p = _p_up(fw, mask)
            mr = _mean_ret(fw, mask)
            label = f"F={'UP' if f else 'DOWN'}_M={'UP' if m else 'DOWN'}_S={'UP' if s else 'DOWN'}"
            context_states[label] = {
                "fast": "UP" if f else "DOWN",
                "medium": "UP" if m else "DOWN",
                "slow": "UP" if s else "DOWN",
                "n": n,
                "p_up": round(p, 4),
                "mean_return": round(mr, 6),
            }

        # Aligned vs conflicted
        aligned_mask = (fd == md) & (md == sd)
        conflict_mask = ((fd != md) | (md != sd)) & ~((fd == md) & (md == sd))
        triple_aligned = (fd == md) & (md == sd)
        triple_conflict = (fd != md) & (md != sd)

        p_up_aligned = _p_up(fw, aligned_mask)
        p_up_any_conflict = _p_up(fw, conflict_mask)
        p_up_triple_aligned = _p_up(fw, triple_aligned)
        p_up_triple_conflict = _p_up(fw, triple_conflict)
        n_aligned = int(np.sum(aligned_mask))
        n_conflict = int(np.sum(conflict_mask))
        n_triple_aligned = int(np.sum(triple_aligned))
        n_triple_conflict = int(np.sum(triple_conflict))

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "context_states": context_states,
            "p_up_all_timeframes_aligned": round(p_up_triple_aligned, 4),
            "p_up_any_conflict": round(p_up_any_conflict, 4),
            "p_up_triple_aligned": round(p_up_triple_aligned, 4),
            "p_up_triple_conflict": round(p_up_triple_conflict, 4),
            "n_triple_aligned": n_triple_aligned,
            "n_triple_conflict": n_triple_conflict,
            "p_up_aligned": round(p_up_aligned, 4),
            "n_aligned": n_aligned,
            "n_any_conflict": n_conflict,
            "spread_aligned_vs_conflict": round(abs(p_up_aligned - p_up_any_conflict), 4),
        }

    return results


# ===================================================================
# 3. Timeframe Conflict Detection
# ===================================================================
def task3_timeframe_conflict(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    es_fast = es_raw
    es_macro = _sma(es_raw, 100)
    grad_fast = compute_gradient(es_fast, window=5)
    grad_macro = compute_gradient(es_macro, window=10)

    fast_dir = np.full(len(es_fast), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(int)
    macro_dir = np.full(len(es_macro), np.nan)
    macro_dir[10:] = (grad_macro[10:] > 0).astype(int)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_fast) & ~np.isnan(es_macro)
                 & ~np.isnan(fast_dir) & ~np.isnan(macro_dir) & ~np.isnan(fwd))
        if np.sum(valid) < 50:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_f = es_fast[valid]
        es_ma = es_macro[valid]
        fd = fast_dir[valid]
        mad = macro_dir[valid]
        fw = fwd[valid]

        # Conflict: fast ES high AND rising, macro ES falling
        es_thr = np.nanpercentile(es_f, 80)
        conflict = (es_f > es_thr) & (fd == 1) & (mad == 0)
        no_conflict = ~((es_f > es_thr) & (fd == 1)) | (mad == 1)
        # Also: strong conflict where macro is falling but fast is high+rising
        strong_conflict = (es_f > es_thr) & (fd == 1) & (mad == 0)
        # Inverse conflict: fast is low+falling but macro is rising
        inv_conflict = (es_f < np.nanpercentile(es_f, 20)) & (fd == 0) & (mad == 1)

        p_up_conflict = _p_up(fw, conflict)
        p_up_no_conflict = _p_up(fw, no_conflict)
        p_up_strong_conflict = _p_up(fw, strong_conflict)
        p_up_inv_conflict = _p_up(fw, inv_conflict)

        n_conflict = int(np.sum(conflict))
        n_no_conflict = int(np.sum(no_conflict))
        n_strong = int(np.sum(strong_conflict))
        n_inv = int(np.sum(inv_conflict))

        ret_conflict = fw[conflict & ~np.isnan(fw)]
        ret_no_conflict = fw[no_conflict & ~np.isnan(fw)]
        t_stat, p_val = None, None
        if len(ret_conflict) >= 3 and len(ret_no_conflict) >= 3:
            t_stat, p_val = ttest_ind(ret_conflict, ret_no_conflict, equal_var=False)
            t_stat = round(float(t_stat), 4)
            p_val = round(float(p_val), 4)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_conflict_fast_up_macro_down": n_conflict,
            "n_no_conflict": n_no_conflict,
            "p_up_conflict": round(p_up_conflict, 4),
            "p_up_no_conflict": round(p_up_no_conflict, 4),
            "p_up_strong_conflict": round(p_up_strong_conflict, 4),
            "p_up_inverse_conflict": round(p_up_inv_conflict, 4),
            "n_strong_conflict": n_strong,
            "n_inverse_conflict": n_inv,
            "ttest_conflict_vs_no_conflict_stat": t_stat,
            "ttest_conflict_vs_no_conflict_p": p_val,
            "spread": round(abs(p_up_conflict - p_up_no_conflict), 4),
            "conflict_predicts_inversion": bool(p_up_conflict < 0.4 and p_up_no_conflict > 0.5),
        }

    return results


# ===================================================================
# 4. Nested Regime Analysis
# ===================================================================
def task4_nested_regime(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    states = d.states.copy().astype(int)
    es_medium = _sma(es_raw, 20)
    es_slow = _sma(es_raw, 50)
    grad_fast = compute_gradient(es_raw, window=5)
    grad_slow = compute_gradient(es_slow, window=5)

    fast_dir = np.full(len(es_raw), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(int)
    slow_dir = np.full(len(es_slow), np.nan)
    slow_dir[5:] = (grad_slow[5:] > 0).astype(int)

    # Define slow regime based on slow ES + direction
    slow_regime = np.full(len(es_raw), np.nan)
    for i in range(60, len(es_raw)):
        if np.isnan(es_slow[i]) or np.isnan(slow_dir[i]):
            continue
        es_s = es_slow[i]
        thr = np.nanpercentile(es_slow[max(0, i - 500):i + 1], 50)
        slow_regime[i] = 1 if (es_s > thr and slow_dir[i] == 1) else (0 if (es_s <= thr and slow_dir[i] == 0) else 2)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_raw) & ~np.isnan(fwd) & (states >= 0)
                 & ~np.isnan(slow_regime) & ~np.isnan(fast_dir))
        if np.sum(valid) < 50:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_v = es_raw[valid]
        s = states[valid]
        sr = slow_regime[valid]
        fd = fast_dir[valid]
        fw = fwd[valid]

        n_fast_states = int(np.max(s) + 1) if np.max(s) >= 0 else 1
        n_slow_regimes = int(np.nanmax(slow_regime[valid])) + 1

        regime_matrix = {}
        for fast_st in range(n_fast_states):
            for slow_r in range(n_slow_regimes):
                mask = (s == fast_st) & (sr == slow_r)
                n = int(np.sum(mask))
                if n < 5:
                    continue
                p = _p_up(fw, mask)
                mr = _mean_ret(fw, mask)
                key = f"FastState{fast_st}_SlowRegime{int(slow_r)}"
                regime_matrix[key] = {
                    "fast_state": int(fast_st),
                    "slow_regime": int(slow_r),
                    "n": n,
                    "p_up": round(p, 4),
                    "mean_return": round(mr, 6),
                }

        # Test: does P(up | fast_state) depend on slow regime?
        fast_state_p_up_by_slow = {}
        for fast_st in range(n_fast_states):
            p_ups = []
            for slow_r in range(n_slow_regimes):
                mask = (s == fast_st) & (sr == slow_r)
                n = int(np.sum(mask))
                if n >= 5:
                    p_ups.append(_p_up(fw, mask))
            if len(p_ups) >= 2:
                fast_state_p_up_by_slow[f"FastState{fast_st}"] = {
                    "p_up_by_slow_regime": [round(p, 4) for p in p_ups],
                    "spread": round(max(p_ups) - min(p_ups), 4),
                    "context_matters": (max(p_ups) - min(p_ups)) > 0.1,
                }

        # Also create slow regime buckets based on ES slow direction
        slow_up = sr == 1
        slow_down = sr == 0
        slow_neutral = sr == 2

        es_thr = np.nanpercentile(es_v, 70)
        fast_high = es_v > es_thr

        # P(up | high ES) in different slow regimes
        p_up_high_es_slow_up = _p_up(fw, fast_high & slow_up)
        p_up_high_es_slow_down = _p_up(fw, fast_high & slow_down)
        p_up_high_es_slow_neutral = _p_up(fw, fast_high & slow_neutral)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "n_fast_states": n_fast_states,
            "n_slow_regimes": n_slow_regimes,
            "regime_matrix": regime_matrix,
            "p_up_high_es_in_slow_up": round(p_up_high_es_slow_up, 4),
            "p_up_high_es_in_slow_down": round(p_up_high_es_slow_down, 4),
            "p_up_high_es_in_slow_neutral": round(p_up_high_es_slow_neutral, 4),
            "fast_state_dependence_on_slow": fast_state_p_up_by_slow,
        }

    return results


# ===================================================================
# 5. Hierarchical Direction Model
# ===================================================================
def task5_hierarchical_model(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    es_medium = _sma(es_raw, 20)
    es_slow = _sma(es_raw, 50)
    es_macro = _sma(es_raw, 100)

    grad_fast = compute_gradient(es_raw, window=5)
    grad_medium = compute_gradient(es_medium, window=5)
    grad_slow = compute_gradient(es_slow, window=5)
    grad_macro = compute_gradient(es_macro, window=10)

    fast_dir = np.full(len(es_raw), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(int)
    medium_dir = np.full(len(es_medium), np.nan)
    medium_dir[5:] = (grad_medium[5:] > 0).astype(int)
    slow_dir = np.full(len(es_slow), np.nan)
    slow_dir[5:] = (grad_slow[5:] > 0).astype(int)
    macro_dir = np.full(len(es_macro), np.nan)
    macro_dir[10:] = (grad_macro[10:] > 0).astype(int)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_raw) & ~np.isnan(es_medium) & ~np.isnan(es_slow) & ~np.isnan(es_macro)
                 & ~np.isnan(fast_dir) & ~np.isnan(medium_dir) & ~np.isnan(slow_dir) & ~np.isnan(macro_dir)
                 & ~np.isnan(fwd))
        if np.sum(valid) < 100:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_f = es_raw[valid]
        es_m = es_medium[valid]
        es_s = es_slow[valid]
        es_ma = es_macro[valid]
        fd = fast_dir[valid]
        md = medium_dir[valid]
        sd = slow_dir[valid]
        mad = macro_dir[valid]
        fw = fwd[valid]
        direction = (fw > 0).astype(float)

        if len(np.unique(direction)) < 2:
            results[TEST_HL[h]] = {"error": "no directional variance"}
            continue

        # --- Single Level Models ---
        # Level 1: Macro ES direction
        l1_pred = mad.astype(float)
        l1_acc = float(np.mean(l1_pred == direction)) if len(l1_pred) > 0 else 0.5

        # Level 2: Slow ES direction
        l2_pred = sd.astype(float)
        l2_acc = float(np.mean(l2_pred == direction)) if len(l2_pred) > 0 else 0.5

        # Level 3: Medium ES direction
        l3_pred = md.astype(float)
        l3_acc = float(np.mean(l3_pred == direction)) if len(l3_pred) > 0 else 0.5

        # Level 4: Fast ES direction
        l4_pred = fd.astype(float)
        l4_acc = float(np.mean(l4_pred == direction)) if len(l4_pred) > 0 else 0.5

        # --- Hierarchical Model ---
        # Hierarchy: Macro → Slow → Medium → Fast
        # Level 1 sets bias; each subsequent level adjusts within that bias
        hier_pred = np.full_like(fw, np.nan)
        # Start with macro
        hier_pred = mad.copy().astype(float)
        # Where slow disagrees with macro, reduce confidence (use medium to decide)
        for i in range(len(hier_pred)):
            if np.isnan(mad[i]) or np.isnan(sd[i]) or np.isnan(md[i]) or np.isnan(fd[i]):
                continue
            # If macro and slow agree, confidence is high → stick with macro
            if mad[i] == sd[i]:
                hier_pred[i] = mad[i]
            else:
                # Macro and slow disagree → use medium as tiebreaker
                hier_pred[i] = md[i]

        hier_acc = float(np.mean(hier_pred == direction)) if np.sum(~np.isnan(hier_pred)) > 0 else 0.5

        # --- Weighted Hierarchical ---
        # Weight: macro (0.4), slow (0.3), medium (0.2), fast (0.1)
        weighted_pred = (mad.astype(float) * 0.4 + sd.astype(float) * 0.3
                         + md.astype(float) * 0.2 + fd.astype(float) * 0.1)
        weighted_pred_bin = (weighted_pred > 0.5).astype(float)
        weighted_acc = float(np.mean(weighted_pred_bin == direction))

        # --- Logistic Regression models ---
        es_f_z = (es_f - np.nanmean(es_f)) / max(np.nanstd(es_f), 1e-12)
        es_m_z = (es_m - np.nanmean(es_m)) / max(np.nanstd(es_m), 1e-12)
        es_s_z = (es_s - np.nanmean(es_s)) / max(np.nanstd(es_s), 1e-12)
        es_ma_z = (es_ma - np.nanmean(es_ma)) / max(np.nanstd(es_ma), 1e-12)

        # Single level: Macro ES z-score
        X1 = es_ma_z.reshape(-1, 1)
        m1 = LogisticRegression(max_iter=2000, random_state=42).fit(X1, direction)
        m1_acc = float(np.mean(m1.predict(X1) == direction))
        try:
            ll_null = log_loss(direction, np.full_like(direction, np.mean(direction)))
            ll_fit1 = log_loss(direction, m1.predict_proba(X1)[:, 1])
            m1_r2 = round(float(1 - ll_fit1 / ll_null), 6)
        except Exception:
            m1_r2 = 0.0

        # Hierarchical: Macro + Slow + Medium + Fast
        X4 = np.column_stack([es_ma_z, es_s_z, es_m_z, es_f_z])
        m4 = LogisticRegression(max_iter=2000, random_state=42).fit(X4, direction)
        m4_acc = float(np.mean(m4.predict(X4) == direction))
        try:
            ll_fit4 = log_loss(direction, m4.predict_proba(X4)[:, 1])
            m4_r2 = round(float(1 - ll_fit4 / ll_null), 6)
        except Exception:
            m4_r2 = 0.0

        # Hierarchical with interactions (macro * slow)
        X5 = np.column_stack([es_ma_z, es_s_z, es_m_z, es_f_z,
                              es_ma_z * es_s_z, es_ma_z * es_m_z, es_s_z * es_m_z])
        m5 = LogisticRegression(max_iter=2000, random_state=42).fit(X5, direction)
        m5_acc = float(np.mean(m5.predict(X5) == direction))
        try:
            ll_fit5 = log_loss(direction, m5.predict_proba(X5)[:, 1])
            m5_r2 = round(float(1 - ll_fit5 / ll_null), 6)
        except Exception:
            m5_r2 = 0.0

        improvement_hier_best_single = round(m4_acc - max(l1_acc, l2_acc, l3_acc, l4_acc), 4)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "single_level_accuracy": {
                "L1_macro_ES_direction": round(l1_acc, 4),
                "L2_slow_ES_direction": round(l2_acc, 4),
                "L3_medium_ES_direction": round(l3_acc, 4),
                "L4_fast_ES_direction": round(l4_acc, 4),
            },
            "hierarchical_models": {
                "voting_hierarchy_macro_slow_medium": round(hier_acc, 4),
                "weighted_0.4macro_0.3slow_0.2med_0.1fast": round(weighted_acc, 4),
            },
            "logistic_regression_models": {
                "L1_macro_ES_zonly": {"accuracy": round(m1_acc, 4), "pseudo_r2": m1_r2},
                "L4_hier_macro_slow_med_fast": {"accuracy": round(m4_acc, 4), "pseudo_r2": m4_r2},
                "L5_hier_with_interactions": {"accuracy": round(m5_acc, 4), "pseudo_r2": m5_r2},
            },
            "improvement_hierarchical_over_best_single": improvement_hier_best_single,
            "hierarchy_improves": improvement_hier_best_single > 0,
        }

    return results


# ===================================================================
# 6. Horizon-As-Modulator
# ===================================================================
def task6_horizon_modulator(d: DPLData, sym: str) -> dict:
    es_raw = d.es.copy()
    es_fast = es_raw
    es_medium = _sma(es_raw, 20)
    es_slow = _sma(es_raw, 50)
    es_macro = _sma(es_raw, 100)

    grad_fast = compute_gradient(es_fast, window=5)
    grad_medium = compute_gradient(es_medium, window=5)
    grad_slow = compute_gradient(es_slow, window=5)
    grad_macro = compute_gradient(es_macro, window=10)

    fast_dir = np.full(len(es_fast), np.nan)
    fast_dir[5:] = (grad_fast[5:] > 0).astype(int)
    medium_dir = np.full(len(es_medium), np.nan)
    medium_dir[5:] = (grad_medium[5:] > 0).astype(int)
    slow_dir = np.full(len(es_slow), np.nan)
    slow_dir[5:] = (grad_slow[5:] > 0).astype(int)
    macro_dir = np.full(len(es_macro), np.nan)
    macro_dir[10:] = (grad_macro[10:] > 0).astype(int)

    results = {}
    for hi, h in enumerate(TEST_HORIZONS):
        hidx = HORIZONS.index(h)
        fwd = d.fut_ret[:, hidx]
        valid = (~np.isnan(es_fast) & ~np.isnan(es_medium) & ~np.isnan(es_slow) & ~np.isnan(es_macro)
                 & ~np.isnan(fast_dir) & ~np.isnan(medium_dir) & ~np.isnan(slow_dir) & ~np.isnan(macro_dir)
                 & ~np.isnan(fwd))
        if np.sum(valid) < 100:
            results[TEST_HL[h]] = {"error": "insufficient data"}
            continue

        es_f = es_fast[valid]
        es_m = es_medium[valid]
        es_s = es_slow[valid]
        es_ma = es_macro[valid]
        fd = fast_dir[valid]
        md = medium_dir[valid]
        sd = slow_dir[valid]
        mad = macro_dir[valid]
        fw = fwd[valid]
        direction = (fw > 0).astype(float)

        if len(np.unique(direction)) < 2:
            results[TEST_HL[h]] = {"error": "no directional variance"}
            continue

        # Z-score each level
        def _z(x):
            return (x - np.nanmean(x)) / max(np.nanstd(x), 1e-12)

        f_z = _z(es_f)
        m_z = _z(es_m)
        s_z = _z(es_s)
        ma_z = _z(es_ma)

        # Pearson correlation with direction for each level
        def _corr(x, y):
            m = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(m) < 5:
                return 0.0, 1.0
            try:
                r, p = pearsonr(x[m], y[m])
                return round(float(r), 4), round(float(p), 4)
            except Exception:
                return 0.0, 1.0

        corr_fast, p_fast = _corr(f_z, direction)
        corr_med, p_med = _corr(m_z, direction)
        corr_slow, p_slow = _corr(s_z, direction)
        corr_macro, p_macro = _corr(ma_z, direction)

        # Direction-only accuracy
        acc_fast_dir = float(np.mean(fd == direction))
        acc_med_dir = float(np.mean(md == direction))
        acc_slow_dir = float(np.mean(sd == direction))
        acc_macro_dir = float(np.mean(mad == direction))

        # Logistic regression: which level contributes most
        X = np.column_stack([f_z, m_z, s_z, ma_z])
        lr = LogisticRegression(max_iter=2000, random_state=42).fit(X, direction)
        coefs = {
            "fast_ES": round(float(lr.coef_[0][0]), 4),
            "medium_ES": round(float(lr.coef_[0][1]), 4),
            "slow_ES": round(float(lr.coef_[0][2]), 4),
            "macro_ES": round(float(lr.coef_[0][3]), 4),
        }
        abs_coefs = {k: abs(v) for k, v in coefs.items()}
        dominant_level = max(abs_coefs, key=abs_coefs.get)

        results[TEST_HL[h]] = {
            "n_valid": int(np.sum(valid)),
            "correlation_with_direction": {
                "fast_ES": corr_fast,
                "medium_ES": corr_med,
                "slow_ES": corr_slow,
                "macro_ES": corr_macro,
            },
            "direction_accuracy": {
                "fast_dir": round(acc_fast_dir, 4),
                "medium_dir": round(acc_med_dir, 4),
                "slow_dir": round(acc_slow_dir, 4),
                "macro_dir": round(acc_macro_dir, 4),
            },
            "logistic_regression_coefficients": coefs,
            "dominant_context_level": dominant_level,
            "dominant_coefficient": abs_coefs[dominant_level],
        }

    return results


# ===================================================================
# MAIN
# ===================================================================
def main():
    per_symbol = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Loading {sym}...")
        d = DPLData(sym)
        print(f"  Data: {len(d.price)} bars, ES shape: {d.es.shape}, fut_ret shape: {d.fut_ret.shape}")

        print(f"  Task 1: Nested ES Context...")
        t1 = task1_nested_es_context(d, sym)

        print(f"  Task 2: Context Field Model...")
        t2 = task2_context_field_model(d, sym)

        print(f"  Task 3: Timeframe Conflict Detection...")
        t3 = task3_timeframe_conflict(d, sym)

        print(f"  Task 4: Nested Regime Analysis...")
        t4 = task4_nested_regime(d, sym)

        print(f"  Task 5: Hierarchical Direction Model...")
        t5 = task5_hierarchical_model(d, sym)

        print(f"  Task 6: Horizon-As-Modulator...")
        t6 = task6_horizon_modulator(d, sym)

        per_symbol[sym] = {
            "nested_es_context": t1,
            "context_field_model": t2,
            "timeframe_conflict": t3,
            "nested_regime": t4,
            "hierarchical_model": t5,
            "horizon_modulator": t6,
        }

    summary = _build_summary(per_symbol)

    output = {
        "experiment": "CDER-MultiTimeframe",
        "title": "Context-Dependent Energy Release: Multi-Timeframe Context Fields",
        "per_symbol": per_symbol,
        "summary": summary,
    }

    out_path = OUT_DIR / "cder_multitimeframe.json"
    out_path.write_text(json.dumps(_clean(output), indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"CDER Multi-Timeframe complete -> {out_path}")


def _build_summary(per_symbol: dict) -> dict:
    # Task 1: Nested ES context
    t1_spreads_fast = []
    t1_spreads_med = []
    t1_symbols_nested_works = 0
    t1_total = 0

    # Task 2: Context field
    t2_aligned_p_ups = []
    t2_conflict_p_ups = []
    t2_spreads = []
    t2_aligned_beats_conflict = 0
    t2_total = 0

    # Task 3: Conflict detection
    t3_conflict_p_ups = []
    t3_noconflict_p_ups = []
    t3_spreads = []
    t3_inversion_count = 0
    t3_total = 0

    # Task 4: Nested regime
    t4_spreads = []
    t4_context_matters = 0
    t4_total = 0

    # Task 5: Hierarchical model
    t5_single_accs = []
    t5_hier_accs = []
    t5_improves = 0
    t5_total = 0
    t5_best_single_levels = {"macro": 0, "slow": 0, "medium": 0, "fast": 0}

    # Task 6: Horizon modulator
    t6_dominant_by_horizon = {"H5": {}, "H20": {}, "H50": {}}
    t6_dominant_counts = {"fast_ES": 0, "medium_ES": 0, "slow_ES": 0, "macro_ES": 0}
    t6_total_horizon = 0

    for sym, r in per_symbol.items():
        # Task 1
        t1 = r.get("nested_es_context", {})
        for hl, v in t1.items():
            if isinstance(v, dict) and "nested_context_spread_fast_rising" in v:
                t1_spreads_fast.append(v["nested_context_spread_fast_rising"])
                t1_spreads_med.append(v["medium_context_spread"])
                t1_total += 1
                if v["nested_context_spread_fast_rising"] > 0.05:
                    t1_symbols_nested_works += 1

        # Task 2
        t2 = r.get("context_field_model", {})
        for hl, v in t2.items():
            if isinstance(v, dict) and "p_up_triple_aligned" in v:
                t2_aligned_p_ups.append(v["p_up_triple_aligned"])
                t2_conflict_p_ups.append(v["p_up_triple_conflict"])
                t2_spreads.append(v["spread_aligned_vs_conflict"])
                t2_total += 1
                if v["p_up_triple_aligned"] > v["p_up_triple_conflict"]:
                    t2_aligned_beats_conflict += 1

        # Task 3
        t3 = r.get("timeframe_conflict", {})
        for hl, v in t3.items():
            if isinstance(v, dict) and "p_up_conflict" in v:
                t3_conflict_p_ups.append(v["p_up_conflict"])
                t3_noconflict_p_ups.append(v["p_up_no_conflict"])
                t3_spreads.append(v["spread"])
                t3_total += 1
                if v.get("conflict_predicts_inversion"):
                    t3_inversion_count += 1

        # Task 4
        t4 = r.get("nested_regime", {})
        for hl, v in t4.items():
            if isinstance(v, dict) and "p_up_high_es_in_slow_up" in v:
                spread = abs(v["p_up_high_es_in_slow_up"] - v["p_up_high_es_in_slow_down"])
                t4_spreads.append(spread)
                t4_total += 1
                if spread > 0.1:
                    t4_context_matters += 1

        # Task 5
        t5 = r.get("hierarchical_model", {})
        for hl, v in t5.items():
            if isinstance(v, dict) and "single_level_accuracy" in v:
                t5_total += 1
                single = v["single_level_accuracy"]
                best_single = max(single.values())
                t5_single_accs.append(best_single)
                hier = v["hierarchical_models"]["voting_hierarchy_macro_slow_medium"]
                t5_hier_accs.append(hier)
                if v.get("hierarchy_improves"):
                    t5_improves += 1
                # Track which single level is best
                best_lvl = max(single, key=single.get)
                if "macro" in best_lvl.lower():
                    t5_best_single_levels["macro"] += 1
                elif "slow" in best_lvl.lower():
                    t5_best_single_levels["slow"] += 1
                elif "medium" in best_lvl.lower():
                    t5_best_single_levels["medium"] += 1
                elif "fast" in best_lvl.lower():
                    t5_best_single_levels["fast"] += 1

        # Task 6
        t6 = r.get("horizon_modulator", {})
        for hl, v in t6.items():
            if isinstance(v, dict) and "dominant_context_level" in v:
                t6_total_horizon += 1
                dom = v["dominant_context_level"]
                t6_dominant_counts[dom] = t6_dominant_counts.get(dom, 0) + 1
                t6_dominant_by_horizon[hl][sym] = dom

    # Build summary
    q1 = {
        "avg_nested_context_spread_fast_rising": round(float(np.mean(t1_spreads_fast)), 4) if t1_spreads_fast else None,
        "avg_medium_context_spread": round(float(np.mean(t1_spreads_med)), 4) if t1_spreads_med else None,
        "nested_context_affects_direction": (float(np.mean(t1_spreads_fast)) > 0.05) if t1_spreads_fast else False,
        "interpretation": f"Nested ES context spreads: fast={round(float(np.mean(t1_spreads_fast)), 4) if t1_spreads_fast else None}, medium={round(float(np.mean(t1_spreads_med)), 4) if t1_spreads_med else None}",
    }

    avg_aligned = float(np.mean(t2_aligned_p_ups)) if t2_aligned_p_ups else None
    avg_conflict = float(np.mean(t2_conflict_p_ups)) if t2_conflict_p_ups else None
    q2 = {
        "avg_p_up_triple_aligned": round(avg_aligned, 4) if avg_aligned is not None else None,
        "avg_p_up_triple_conflict": round(avg_conflict, 4) if avg_conflict is not None else None,
        "avg_spread": round(float(np.mean(t2_spreads)), 4) if t2_spreads else None,
        "aligned_beats_conflict_count": t2_aligned_beats_conflict,
        "aligned_beats_conflict_ratio": round(t2_aligned_beats_conflict / max(t2_total, 1), 3),
        "interpretation": f"Triple-aligned context P(up)={avg_aligned} vs triple-conflict={avg_conflict} — {'context alignment predicts direction' if avg_aligned is not None and avg_conflict is not None and avg_aligned > avg_conflict else 'context alignment does NOT clearly predict direction'}",
    }

    avg_cp = float(np.mean(t3_conflict_p_ups)) if t3_conflict_p_ups else None
    avg_nc = float(np.mean(t3_noconflict_p_ups)) if t3_noconflict_p_ups else None
    q3 = {
        "avg_p_up_conflict": round(avg_cp, 4) if avg_cp is not None else None,
        "avg_p_up_no_conflict": round(avg_nc, 4) if avg_nc is not None else None,
        "avg_spread": round(float(np.mean(t3_spreads)), 4) if t3_spreads else None,
        "inversion_predictions": t3_inversion_count,
        "total_tests": t3_total,
        "timeframe_conflicts_predict_sign_inversions": t3_inversion_count > t3_total // 2,
        "interpretation": f"Timeframe conflict P(up)={avg_cp} vs no-conflict={avg_nc} — conflicts {'DO' if t3_inversion_count > t3_total // 2 else 'DO NOT'} predict sign inversions ({t3_inversion_count}/{t3_total} tests positive)",
    }

    q4 = {
        "avg_slow_regime_context_spread": round(float(np.mean(t4_spreads)), 4) if t4_spreads else None,
        "context_matters_count": t4_context_matters,
        "context_matters_ratio": round(t4_context_matters / max(t4_total, 1), 3),
        "interpretation": f"P(up | fast regime) depends on slow regime context (spread={round(float(np.mean(t4_spreads)), 4) if t4_spreads else None}, {t4_context_matters}/{t4_total} cases)",
    }

    avg_single = float(np.mean(t5_single_accs)) if t5_single_accs else None
    avg_hier = float(np.mean(t5_hier_accs)) if t5_hier_accs else None
    q5 = {
        "avg_best_single_level_accuracy": round(avg_single, 4) if avg_single is not None else None,
        "avg_hierarchical_model_accuracy": round(avg_hier, 4) if avg_hier is not None else None,
        "hierarchy_improves_count": t5_improves,
        "hierarchy_improves_ratio": round(t5_improves / max(t5_total, 1), 3),
        "best_single_level_distribution": t5_best_single_levels,
        "hierarchical_better_than_single": (avg_hier > avg_single) if (avg_hier is not None and avg_single is not None) else None,
        "interpretation": f"Hierarchical model accuracy={avg_hier} vs best single level={avg_single} — hierarchy {'improves' if (avg_hier is not None and avg_single is not None and avg_hier > avg_single) else 'does NOT improve'} directional prediction ({t5_improves}/{t5_total} cases)",
    }

    q6 = {
        "dominant_level_counts": t6_dominant_counts,
        "dominant_by_horizon": t6_dominant_by_horizon,
        "interpretation": f"Context level dominance: fast={t6_dominant_counts.get('fast_ES',0)}, medium={t6_dominant_counts.get('medium_ES',0)}, slow={t6_dominant_counts.get('slow_ES',0)}, macro={t6_dominant_counts.get('macro_ES',0)}",
    }

    # Find best context field model overall
    t5_best_model = "hierarchical" if (avg_hier is not None and avg_single is not None and avg_hier >= avg_single) else "single_level"
    if t2_aligned_p_ups and t2_conflict_p_ups:
        context_field_works = float(np.mean(t2_aligned_p_ups)) > float(np.mean(t2_conflict_p_ups))
    else:
        context_field_works = False

    scoring = {
        "q1_nested_es_context_spread_fast_rising": round(float(np.mean(t1_spreads_fast)), 4) if t1_spreads_fast else None,
        "q2_context_field_triple_aligned_beats_conflict": context_field_works,
        "q3_timeframe_conflicts_predict_inversions": t3_inversion_count > t3_total // 2,
        "q4_nested_regime_context_dependence": t4_context_matters > t4_total // 2,
        "q5_hierarchical_model_better_than_single": (avg_hier is not None and avg_single is not None and avg_hier > avg_single),
        "q6_dominant_level_overall": max(t6_dominant_counts, key=t6_dominant_counts.get) if t6_dominant_counts else None,
        "q6_dominant_level_by_horizon": {k: max(v, key=v.get) if v else None for k, v in t6_dominant_by_horizon.items()},
    }

    key_finding = (
        f"CDER Multi-Timeframe: "
        f"Nested ES spread={round(float(np.mean(t1_spreads_fast)), 4) if t1_spreads_fast else None}. "
        f"Context field aligned P(up)={avg_aligned} vs conflict={avg_conflict} ({'aligned wins' if context_field_works else 'no clear winner'}). "
        f"Timeframe conflicts predict inversion: {t3_inversion_count}/{t3_total}. "
        f"Nested regime context matters: {t4_context_matters}/{t4_total}. "
        f"Hierarchical accuracy={avg_hier} vs best-single={avg_single}. "
        f"Dominant context: {max(t6_dominant_counts, key=t6_dominant_counts.get) if t6_dominant_counts else None}."
    )

    return {
        "n_symbols": len(per_symbol),
        "tested_horizons": [TEST_HL[h] for h in TEST_HORIZONS],
        "q1_nested_es_context": q1,
        "q2_context_field_model": q2,
        "q3_timeframe_conflict": q3,
        "q4_nested_regime": q4,
        "q5_hierarchical_model": q5,
        "q6_horizon_modulator": q6,
        "scoring": scoring,
        "key_finding": key_finding,
    }


if __name__ == "__main__":
    main()
