"""
CDER: Information Propagation Between Assets
Investigates how ES, memory, residuals, and regime propagate across assets.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_HORIZONS = [5, 20, 50]
TEST_HL = {h: HORIZON_LABELS[h] for h in TEST_HORIZONS}
MAX_LAG = 20

RESIDUAL_KEYS = ["xgboost", "linear", "random_forest"]


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


def _lag_corr(src: np.ndarray, tgt: np.ndarray, max_lag: int = MAX_LAG) -> dict:
    """Compute cross-correlation at lags -max_lag to +max_lag (positive = src leads)."""
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    valid = ~np.isnan(src) & ~np.isnan(tgt)
    if np.sum(valid) < 30:
        return {}
    src_v, tgt_v = src[valid], tgt[valid]
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            s, t = src_v[:-lag], tgt_v[lag:]
        else:
            s, t = src_v[-lag:], tgt_v[:lag]
        if len(s) < 30:
            continue
        try:
            c, p = pearsonr(s, t)
            results[f"lag_{lag}"] = {"corr": round(float(c), 4), "p": round(float(p), 6)}
        except Exception:
            continue
    return results


def _lag_corr_direction(src_es: np.ndarray, tgt_dir: np.ndarray, max_lag: int = MAX_LAG) -> dict:
    """Cross-correlation of source ES with target direction at various lags."""
    n = min(len(src_es), len(tgt_dir))
    s, t = src_es[:n], tgt_dir[:n]
    valid = ~np.isnan(s) & ~np.isnan(t)
    if np.sum(valid) < 30:
        return {}
    s_v, t_v = s[valid], t[valid]
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            ss, tt = s_v[:-lag], t_v[lag:]
        else:
            ss, tt = s_v[-lag:], t_v[:lag]
        if len(ss) < 30:
            continue
        try:
            c, p = pearsonr(ss, tt)
            results[f"lag_{lag}"] = {"corr": round(float(c), 4), "p": round(float(p), 6)}
        except Exception:
            continue
    return results


def _optimal_lag(lag_corrs: dict) -> dict:
    """Find lag with max absolute correlation."""
    if not lag_corrs:
        return {"optimal_lag": None, "optimal_corr": None, "direction": None}
    best_lag = None
    best_abs = 0
    best_corr = 0
    for k, v in lag_corrs.items():
        c = v["corr"]
        lag = int(k.split("_")[1])
        if abs(c) > best_abs:
            best_abs = abs(c)
            best_lag = lag
            best_corr = c
    return {
        "optimal_lag": best_lag,
        "optimal_corr": round(best_corr, 4),
        "direction": "src_leads" if best_lag > 0 else "tgt_leads",
        "abs_corr": round(best_abs, 4),
    }


# ===================================================================
# Helper: aligned valid windows across two symbols
# ===================================================================
def _common_window(d1: DPLData, d2: DPLData, field: str = "es") -> tuple:
    """Get aligned arrays for a field across two symbols."""
    if field == "es":
        a1, a2 = d1.es.copy(), d2.es.copy()
    elif field == "memory_density":
        a1, a2 = d1.memory_density.copy(), d2.memory_density.copy()
    elif field == "states":
        a1, a2 = d1.states.copy().astype(int), d2.states.copy().astype(int)
    elif field == "residuals":
        a1 = d1.residuals.get("xgboost", np.full_like(d1.es, np.nan))
        a2 = d2.residuals.get("xgboost", np.full_like(d2.es, np.nan))
    elif field == "adaptive_time":
        a1, a2 = d1.adaptive_time.copy(), d2.adaptive_time.copy()
    elif field == "state_mutation":
        a1, a2 = d1.state_mutation.copy(), d2.state_mutation.copy()
    else:
        return None, None
    n = min(len(a1), len(a2))
    return a1[:n], a2[:n]


def _direction(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(float)


# ===================================================================
# 1. ES Propagation
# ===================================================================
def task1_es_propagation(all_data: dict) -> dict:
    results = {}
    for target in SYMBOLS:
        dt = all_data[target]
        t_results = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd_t = dt.fut_ret[:, hidx]
            dir_t = _direction(fwd_t)
            es_t = dt.es.copy()
            n_t = len(es_t)
            h_results = {}
            for source in SYMBOLS:
                if source == target:
                    continue
                ds = all_data[source]
                es_s = ds.es.copy()
                n_s = len(es_s)
                n = min(n_t, n_s)
                # ES -> ES lead-lag
                corr_es_es = _lag_corr(es_s[:n], es_t[:n])
                opt_es_es = _optimal_lag(corr_es_es)
                # ES -> direction lead-lag
                valid = ~np.isnan(es_s[:n]) & ~np.isnan(dir_t[:n])
                es_s_v = es_s[:n][valid]
                dir_t_v = dir_t[:n][valid]
                corr_es_dir = _lag_corr(es_s_v, dir_t_v) if np.sum(valid) >= 30 else {}
                opt_es_dir = _optimal_lag(corr_es_dir)
                # Concurrent correlation
                n_common = min(len(es_s_v), len(dir_t_v))
                c_conc, p_conc = (0.0, 1.0)
                if n_common >= 30:
                    try:
                        c_conc, p_conc = pearsonr(es_s_v[:n_common], dir_t_v[:n_common])
                    except Exception:
                        pass
                h_results[source] = {
                    "es_es_lag_corr": corr_es_es,
                    "es_es_optimal": opt_es_es,
                    "es_dir_lag_corr": corr_es_dir,
                    "es_dir_optimal": opt_es_dir,
                    "concurrent_es_dir_corr": round(float(c_conc), 4),
                    "concurrent_es_dir_p": round(float(p_conc), 6),
                }
            t_results[TEST_HL[h]] = h_results
        results[target] = t_results
    return results


# ===================================================================
# 2. Memory Propagation
# ===================================================================
def task2_memory_propagation(all_data: dict) -> dict:
    results = {}
    for target in SYMBOLS:
        dt = all_data[target]
        t_results = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd_t = dt.fut_ret[:, hidx]
            dir_t = _direction(fwd_t)
            md_t = dt.memory_density.copy()
            n_t = len(md_t)
            h_results = {}
            for source in SYMBOLS:
                if source == target:
                    continue
                ds = all_data[source]
                md_s = ds.memory_density.copy()
                n = min(n_t, len(md_s))
                # Memory -> Memory lead-lag
                corr_md_md = _lag_corr(md_s[:n], md_t[:n])
                opt_md_md = _optimal_lag(corr_md_md)
                # Memory -> direction
                valid = ~np.isnan(md_s[:n]) & ~np.isnan(dir_t[:n])
                md_s_v = md_s[:n][valid]
                dir_t_v = dir_t[:n][valid]
                corr_md_dir = _lag_corr(md_s_v, dir_t_v) if np.sum(valid) >= 30 else {}
                opt_md_dir = _optimal_lag(corr_md_dir)
                # Concurrent
                n_common = min(len(md_s_v), len(dir_t_v))
                c_conc, p_conc = (0.0, 1.0)
                if n_common >= 30:
                    try:
                        c_conc, p_conc = pearsonr(md_s_v[:n_common], dir_t_v[:n_common])
                    except Exception:
                        pass
                h_results[source] = {
                    "memory_memory_lag_corr": corr_md_md,
                    "memory_memory_optimal": opt_md_md,
                    "memory_dir_lag_corr": corr_md_dir,
                    "memory_dir_optimal": opt_md_dir,
                    "concurrent_memory_dir_corr": round(float(c_conc), 4),
                    "concurrent_memory_dir_p": round(float(p_conc), 6),
                }
            t_results[TEST_HL[h]] = h_results
        results[target] = t_results
    return results


# ===================================================================
# 3. Residual Propagation
# ===================================================================
def task3_residual_propagation(all_data: dict) -> dict:
    results = {}
    for rt in RESIDUAL_KEYS:
        rt_results = {}
        for target in SYMBOLS:
            dt = all_data[target]
            resid_t = dt.residuals.get(rt, np.full_like(dt.es, np.nan))
            n_t = len(resid_t)
            t_results = {}
            for hi, h in enumerate(TEST_HORIZONS):
                hidx = HORIZONS.index(h)
                fwd_t = dt.fut_ret[:, hidx]
                dir_t = _direction(fwd_t)
                h_results = {}
                for source in SYMBOLS:
                    if source == target:
                        continue
                    ds = all_data[source]
                    resid_s = ds.residuals.get(rt, np.full_like(ds.es, np.nan))
                    n = min(n_t, len(resid_s))
                    # Residual -> Residual lead-lag
                    corr_r_r = _lag_corr(resid_s[:n], resid_t[:n])
                    opt_r_r = _optimal_lag(corr_r_r)
                    # Residual -> direction
                    valid = ~np.isnan(resid_s[:n]) & ~np.isnan(dir_t[:n])
                    resid_s_v = resid_s[:n][valid]
                    dir_t_v = dir_t[:n][valid]
                    corr_r_dir = _lag_corr(resid_s_v, dir_t_v) if np.sum(valid) >= 30 else {}
                    opt_r_dir = _optimal_lag(corr_r_dir)
                    # Residual -> future return (continuous)
                    fwd_t_v = fwd_t[:n][valid]
                    corr_r_ret = _lag_corr(resid_s_v, fwd_t_v) if np.sum(valid) >= 30 else {}
                    opt_r_ret = _optimal_lag(corr_r_ret)
                    # Concurrent
                    n_common = min(len(resid_s_v), len(dir_t_v))
                    c_conc, p_conc = (0.0, 1.0)
                    if n_common >= 30:
                        try:
                            c_conc, p_conc = pearsonr(resid_s_v[:n_common], dir_t_v[:n_common])
                        except Exception:
                            pass
                    h_results[source] = {
                        "residual_residual_lag_corr": corr_r_r,
                        "residual_residual_optimal": opt_r_r,
                        "residual_dir_lag_corr": corr_r_dir,
                        "residual_dir_optimal": opt_r_dir,
                        "residual_ret_lag_corr": corr_r_ret,
                        "residual_ret_optimal": opt_r_ret,
                        "concurrent_residual_dir_corr": round(float(c_conc), 4),
                        "concurrent_residual_dir_p": round(float(p_conc), 6),
                    }
                t_results[TEST_HL[h]] = h_results
            rt_results[target] = t_results
        results[rt] = rt_results
    return results


# ===================================================================
# 4. Regime Propagation
# ===================================================================
def task4_regime_propagation(all_data: dict) -> dict:
    results = {}
    N_BARS = [1, 3, 5, 10, 20]
    for target in SYMBOLS:
        dt = all_data[target]
        states_t = dt.states.copy().astype(int)
        n_t = len(states_t)
        t_results = {}
        for source in SYMBOLS:
            if source == target:
                continue
            ds = all_data[source]
            states_s = ds.states.copy().astype(int)
            n = min(n_t, len(states_s))
            s_s, s_t = states_s[:n], states_t[:n]
            valid = (s_s >= 0) & (s_t >= 0)
            if np.sum(valid) < 30:
                t_results[source] = {"error": "insufficient data"}
                continue
            s_s_v, s_t_v = s_s[valid], s_t[valid]

            # Regime change points
            s_change = np.zeros(len(s_s_v))
            s_change[1:] = (s_s_v[1:] != s_s_v[:-1]).astype(float)
            t_change = np.zeros(len(s_t_v))
            t_change[1:] = (s_t_v[1:] != s_t_v[:-1]).astype(float)

            # Regime alignment (concurrent)
            aligned = float(np.mean(s_s_v == s_t_v))

            # When source changes regime, does target follow within N bars?
            change_follow = {}
            for nb in N_BARS:
                src_change_idx = np.where(s_change == 1)[0]
                if len(src_change_idx) < 3:
                    change_follow[f"within_{nb}"] = None
                    continue
                follows = 0
                total = 0
                for ci in src_change_idx:
                    start = ci + 1
                    end = min(ci + nb + 1, len(s_t_v))
                    if start >= end:
                        continue
                    # Did target change regime in the window?
                    pre_change_state = s_t_v[ci]
                    if np.any(s_t_v[start:end] != pre_change_state):
                        follows += 1
                    total += 1
                follow_rate = follows / max(total, 1)
                # Also check: did target change in same direction?
                same_dir_follows = 0
                for ci in src_change_idx:
                    if ci + 1 >= len(s_s_v) or ci >= len(s_t_v) or ci + 1 >= len(s_t_v):
                        continue
                    src_old, src_new = s_s_v[ci], s_s_v[min(ci + 1, len(s_s_v) - 1)]
                    tgt_old = s_t_v[ci]
                    tgt_new = s_t_v[min(ci + 1, len(s_t_v) - 1)]
                    if src_old != src_new and tgt_old != tgt_new:
                        # Both changed, same direction of change?
                        src_delta = src_new - src_old
                        tgt_delta = tgt_new - tgt_old
                        if np.sign(src_delta) == np.sign(tgt_delta):
                            same_dir_follows += 1
                same_dir_rate = same_dir_follows / max(total, 1)
                change_follow[f"within_{nb}"] = {
                    "follow_rate": round(float(follow_rate), 4),
                    "same_direction_rate": round(float(same_dir_rate), 4),
                    "n_source_changes": int(total),
                }

            # Regime cascade: does source leading regime predict target regime?
            lag_corr_regime = _lag_corr(s_s_v.astype(float), s_t_v.astype(float))
            opt_regime = _optimal_lag(lag_corr_regime)

            t_results[source] = {
                "regime_alignment": round(float(aligned), 4),
                "regime_lag_corr": lag_corr_regime,
                "regime_optimal_lag": opt_regime,
                "regime_change_follow": change_follow,
            }
        results[target] = t_results
    return results


# ===================================================================
# 5. Directional Information Transfer
# ===================================================================
def task5_directional_transfer(all_data: dict) -> dict:
    results = {}
    for target in SYMBOLS:
        dt = all_data[target]
        t_results = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd_t = dt.fut_ret[:, hidx]
            dir_t = _direction(fwd_t)
            es_t = dt.es.copy()
            n_t = len(es_t)
            h_results = {}
            for source in SYMBOLS:
                if source == target:
                    continue
                ds = all_data[source]
                es_s = ds.es.copy()
                fwd_s = ds.fut_ret[:, hidx]
                dir_s = _direction(fwd_s)
                n = min(n_t, len(es_s), len(dir_s))

                es_s_v = es_s[:n]
                dir_s_v = dir_s[:n]
                dir_t_v = dir_t[:n]

                valid = ~np.isnan(es_s_v) & ~np.isnan(dir_s_v) & ~np.isnan(dir_t_v)
                if np.sum(valid) < 30:
                    h_results[source] = {"error": "insufficient data"}
                    continue

                es_s_v = es_s_v[valid]
                dir_s_v = dir_s_v[valid]
                dir_t_v = dir_t_v[valid]

                # Source direction -> target direction at various lags
                dir_dir_lags = {}
                for lag in range(1, MAX_LAG // 2 + 1):
                    if lag >= len(dir_s_v):
                        continue
                    s, t = dir_s_v[:-lag], dir_t_v[lag:]
                    if len(s) < 30:
                        continue
                    acc = float(np.mean(s == t))
                    # Confusion: P(target_up | source_up, lag)
                    src_up = s == 1
                    if np.sum(src_up) >= 5:
                        p_up_given_src_up = float(np.mean(t[src_up] == 1))
                    else:
                        p_up_given_src_up = 0.5
                    src_down = s == 0
                    if np.sum(src_down) >= 5:
                        p_up_given_src_down = float(np.mean(t[src_down] == 1))
                    else:
                        p_up_given_src_down = 0.5
                    dir_dir_lags[f"lag_{lag}"] = {
                        "accuracy": round(float(acc), 4),
                        "p_target_up_given_source_up": round(float(p_up_given_src_up), 4),
                        "p_target_up_given_source_down": round(float(p_up_given_src_down), 4),
                        "n": int(len(s)),
                    }

                # Source ES direction -> target direction
                es_s_dir = _direction(es_s_v)  # ES gradient direction
                # Use actual ES level direction: ES above median = up-state
                es_thr = np.nanpercentile(es_s_v, 50)
                es_s_high = (es_s_v > es_thr).astype(float)
                es_dir_dir_lags = {}
                for lag in range(1, MAX_LAG // 2 + 1):
                    if lag >= len(es_s_high):
                        continue
                    s, t = es_s_high[:-lag], dir_t_v[lag:]
                    if len(s) < 30:
                        continue
                    acc = float(np.mean(s == t))
                    src_up = s == 1
                    if np.sum(src_up) >= 5:
                        p_up_given_es_up = float(np.mean(t[src_up] == 1))
                    else:
                        p_up_given_es_up = 0.5
                    src_down = s == 0
                    if np.sum(src_down) >= 5:
                        p_up_given_es_down = float(np.mean(t[src_down] == 1))
                    else:
                        p_up_given_es_down = 0.5
                    es_dir_dir_lags[f"lag_{lag}"] = {
                        "accuracy": round(float(acc), 4),
                        "p_target_up_given_source_es_up": round(float(p_up_given_es_up), 4),
                        "p_target_up_given_source_es_down": round(float(p_up_given_es_down), 4),
                        "n": int(len(s)),
                    }

                # Find best lag
                best_dir_lag = None
                best_dir_acc = 0
                for k, v in dir_dir_lags.items():
                    if v["accuracy"] > best_dir_acc:
                        best_dir_acc = v["accuracy"]
                        best_dir_lag = int(k.split("_")[1])

                best_es_lag = None
                best_es_acc = 0
                for k, v in es_dir_dir_lags.items():
                    if v["accuracy"] > best_es_acc:
                        best_es_acc = v["accuracy"]
                        best_es_lag = int(k.split("_")[1])

                h_results[source] = {
                    "direction_to_direction": dir_dir_lags,
                    "best_direction_lag": best_dir_lag,
                    "best_direction_accuracy": round(best_dir_acc, 4),
                    "es_high_to_direction": es_dir_dir_lags,
                    "best_es_lag": best_es_lag,
                    "best_es_accuracy": round(best_es_acc, 4),
                    "p_target_up_given_source_up_best": (
                        dir_dir_lags.get(f"lag_{best_dir_lag}", {}).get("p_target_up_given_source_up")
                        if best_dir_lag else None
                    ),
                }
            t_results[TEST_HL[h]] = h_results
        results[target] = t_results
    return results


# ===================================================================
# 6. NAS100 as Driver (shorter horizons)
# ===================================================================
def task6_nas100_driver(all_data: dict) -> dict:
    SHORT_HORIZONS = [1, 5, 20, 50]  # H1, H5, H20, H50
    SHORT_HL = {1: "H1", 5: "H5", 20: "H20", 50: "H50"}
    results = {}
    nas = all_data["NAS100"]
    es_nas = nas.es.copy()
    md_nas = nas.memory_density.copy()
    states_nas = nas.states.copy().astype(int)
    at_nas = nas.adaptive_time.copy()
    n_nas = len(es_nas)

    for target in SYMBOLS:
        if target == "NAS100":
            continue
        dt = all_data[target]
        t_results = {}
        for hi, h in enumerate(SHORT_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd_t = dt.fut_ret[:, hidx]
            dir_t = _direction(fwd_t)
            n = min(n_nas, len(dir_t))

            es_n = es_nas[:n]
            md_n = md_nas[:n]
            st_n = states_nas[:n]
            at_n = at_nas[:n]
            dir_t_v = dir_t[:n]

            valid = ~np.isnan(es_n) & ~np.isnan(dir_t_v) & ~np.isnan(md_n)
            if np.sum(valid) < 30:
                t_results[SHORT_HL[h]] = {"error": "insufficient data"}
                continue

            es_n_v = es_n[valid]
            md_n_v = md_n[valid]
            st_n_v = st_n[valid].astype(float)
            at_n_v = at_n[valid]
            dir_t_v2 = dir_t_v[valid]

            # NAS100 ES -> target direction at lags
            es_dir_lags = {}
            for lag in range(1, MAX_LAG // 2 + 1):
                if lag >= len(es_n_v):
                    continue
                s, t = es_n_v[:-lag], dir_t_v2[lag:]
                if len(s) < 30:
                    continue
                try:
                    c, p = pearsonr(s, t)
                except Exception:
                    c, p = 0.0, 1.0
                es_dir_lags[f"lag_{lag}"] = {
                    "corr": round(float(c), 4),
                    "p": round(float(p), 6),
                    "n": int(len(s)),
                }

            # NAS100 memory -> target direction
            md_dir_lags = {}
            for lag in range(1, MAX_LAG // 2 + 1):
                if lag >= len(md_n_v):
                    continue
                s, t = md_n_v[:-lag], dir_t_v2[lag:]
                if len(s) < 30:
                    continue
                try:
                    c, p = pearsonr(s, t)
                except Exception:
                    c, p = 0.0, 1.0
                md_dir_lags[f"lag_{lag}"] = {
                    "corr": round(float(c), 4),
                    "p": round(float(p), 6),
                    "n": int(len(s)),
                }

            # NAS100 state -> target direction
            st_dir_lags = {}
            for lag in range(1, MAX_LAG // 2 + 1):
                if lag >= len(st_n_v):
                    continue
                s, t = st_n_v[:-lag], dir_t_v2[lag:]
                if len(s) < 30:
                    continue
                try:
                    c, p = pearsonr(s, t)
                except Exception:
                    c, p = 0.0, 1.0
                st_dir_lags[f"lag_{lag}"] = {
                    "corr": round(float(c), 4),
                    "p": round(float(p), 6),
                    "n": int(len(s)),
                }

            # NAS100 adaptive_time -> target direction
            at_dir_lags = {}
            for lag in range(1, MAX_LAG // 2 + 1):
                if lag >= len(at_n_v):
                    continue
                s, t = at_n_v[:-lag], dir_t_v2[lag:]
                if len(s) < 30:
                    continue
                try:
                    c, p = pearsonr(s, t)
                except Exception:
                    c, p = 0.0, 1.0
                at_dir_lags[f"lag_{lag}"] = {
                    "corr": round(float(c), 4),
                    "p": round(float(p), 6),
                    "n": int(len(s)),
                }

            # Optimal lag finding
            best_es_lag = max(es_dir_lags, key=lambda k: abs(es_dir_lags[k]["corr"])) if es_dir_lags else None
            best_md_lag = max(md_dir_lags, key=lambda k: abs(md_dir_lags[k]["corr"])) if md_dir_lags else None

            # Concurrent correlation
            c_es, p_es = 0.0, 1.0
            if len(es_n_v) >= 30:
                try:
                    c_es, p_es = pearsonr(es_n_v, dir_t_v2)
                except Exception:
                    pass
            c_md, p_md = 0.0, 1.0
            if len(md_n_v) >= 30:
                try:
                    c_md, p_md = pearsonr(md_n_v, dir_t_v2)
                except Exception:
                    pass
            c_st, p_st = 0.0, 1.0
            if len(st_n_v) >= 30:
                try:
                    c_st, p_st = pearsonr(st_n_v, dir_t_v2)
                except Exception:
                    pass

            t_results[SHORT_HL[h]] = {
                "n_valid": int(np.sum(valid)),
                "nas100_es_to_target_dir_lags": es_dir_lags,
                "nas100_es_optimal_lag": best_es_lag,
                "nas100_es_optimal_corr": round(float(es_dir_lags[best_es_lag]["corr"]), 4) if best_es_lag else None,
                "nas100_memory_to_target_dir_lags": md_dir_lags,
                "nas100_memory_optimal_lag": best_md_lag,
                "nas100_memory_optimal_corr": round(float(md_dir_lags[best_md_lag]["corr"]), 4) if best_md_lag else None,
                "nas100_state_to_target_dir_lags": st_dir_lags,
                "nas100_adaptive_time_to_target_dir_lags": at_dir_lags,
                "concurrent_nas100_es_target_dir_corr": round(float(c_es), 4),
                "concurrent_nas100_es_target_dir_p": round(float(p_es), 6),
                "concurrent_nas100_memory_target_dir_corr": round(float(c_md), 4),
                "concurrent_nas100_state_target_dir_corr": round(float(c_st), 4),
            }
        results[target] = t_results
    return results


# ===================================================================
# 7. Propagation Through Context (regime-modulated transfer)
# ===================================================================
def task7_context_modulated_propagation(all_data: dict) -> dict:
    results = {}
    for target in SYMBOLS:
        dt = all_data[target]
        states_t = dt.states.copy().astype(int)
        es_t = dt.es.copy()
        n_t = len(es_t)
        t_results = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd_t = dt.fut_ret[:, hidx]
            dir_t = _direction(fwd_t)
            h_results = {}
            for source in SYMBOLS:
                if source == target:
                    continue
                ds = all_data[source]
                es_s = ds.es.copy()
                states_s = ds.states.copy().astype(int)
                n = min(n_t, len(es_s), len(states_s), len(states_t))

                es_s_v = es_s[:n]
                es_t_v = es_t[:n]
                st_s_v = states_s[:n]
                st_t_v = states_t[:n]
                dir_t_v = dir_t[:n]

                valid = (~np.isnan(es_s_v) & ~np.isnan(es_t_v) & ~np.isnan(dir_t_v)
                         & (st_s_v >= 0) & (st_t_v >= 0))
                if np.sum(valid) < 30:
                    h_results[source] = {"error": "insufficient data"}
                    continue

                es_s_v2 = es_s_v[valid]
                es_t_v2 = es_t_v[valid]
                st_s_v2 = st_s_v[valid]
                st_t_v2 = st_t_v[valid]
                dir_t_v2 = dir_t_v[valid]
                fwd_t_v2 = fwd_t[:n][valid]

                # Same regime mask
                same_regime = (st_s_v2 == st_t_v2).astype(bool)
                diff_regime = (st_s_v2 != st_t_v2).astype(bool)

                # Does propagation differ by regime alignment?
                # Source ES -> Target direction in same regime vs diff regime
                n_same = int(np.sum(same_regime))
                n_diff = int(np.sum(diff_regime))

                def _corr(x, y):
                    m = ~np.isnan(x) & ~np.isnan(y)
                    if np.sum(m) < 5:
                        return 0.0, 1.0
                    try:
                        return pearsonr(x[m], y[m])
                    except Exception:
                        return 0.0, 1.0

                c_same, p_same = _corr(es_s_v2[same_regime], dir_t_v2[same_regime])
                c_diff, p_diff = _corr(es_s_v2[diff_regime], dir_t_v2[diff_regime])

                # Same regime: source ES -> target ES
                c_es_same, p_es_same = _corr(es_s_v2[same_regime], es_t_v2[same_regime])
                c_es_diff, p_es_diff = _corr(es_s_v2[diff_regime], es_t_v2[diff_regime])

                # P(up) in same regime vs diff regime
                p_up_same_regime = float(np.mean(dir_t_v2[same_regime])) if n_same >= 3 else 0.5
                p_up_diff_regime = float(np.mean(dir_t_v2[diff_regime])) if n_diff >= 3 else 0.5

                # Global regime alignment (fraction of time in same regime)
                global_alignment = float(np.mean(same_regime))

                # Does ES from source predict target direction better in aligned regimes?
                modulation = abs(c_same) - abs(c_diff)

                h_results[source] = {
                    "n_same_regime": n_same,
                    "n_diff_regime": n_diff,
                    "global_regime_alignment": round(global_alignment, 4),
                    "same_regime_source_es_target_dir_corr": round(float(c_same), 4),
                    "same_regime_source_es_target_dir_p": round(float(p_same), 6),
                    "diff_regime_source_es_target_dir_corr": round(float(c_diff), 4),
                    "diff_regime_source_es_target_dir_p": round(float(p_diff), 6),
                    "same_regime_source_es_target_es_corr": round(float(c_es_same), 4),
                    "diff_regime_source_es_target_es_corr": round(float(c_es_diff), 4),
                    "modulation_same_vs_diff_regime": round(float(modulation), 4),
                    "propagation_stronger_in_same_regime": abs(c_same) > abs(c_diff),
                    "p_up_target_in_same_regime": round(p_up_same_regime, 4),
                    "p_up_target_in_diff_regime": round(p_up_diff_regime, 4),
                }
            t_results[TEST_HL[h]] = h_results
        results[target] = t_results
    return results


# ===================================================================
# Summary Builder
# ===================================================================
def build_summary(t1, t2, t3, t4, t5, t6, t7):
    summary = {}

    # Task 1: ES Propagation - strongest edges
    es_edges = []
    for target, tdata in t1.items():
        for hl, hdata in tdata.items():
            for source, v in hdata.items():
                opt = v.get("es_dir_optimal", {})
                if opt and opt.get("optimal_lag") is not None:
                    es_edges.append({
                        "source": source, "target": target, "horizon": hl,
                        "lag": opt["optimal_lag"], "corr": opt["optimal_corr"],
                        "direction": opt["direction"],
                    })
    es_edges_sorted = sorted(es_edges, key=lambda x: abs(x["corr"]), reverse=True)
    summary["es_propagation_strongest_edges"] = es_edges_sorted[:20]

    # NAS100 leading edges
    nas100_leads = [e for e in es_edges if e["source"] == "NAS100" and e["lag"] > 0]
    summary["nas100_es_leads"] = sorted(nas100_leads, key=lambda x: abs(x["corr"]), reverse=True)

    # Task 5: Directional transfer strongest
    dir_edges = []
    for target, tdata in t5.items():
        for hl, hdata in tdata.items():
            for source, v in hdata.items():
                if isinstance(v, dict) and "best_direction_accuracy" in v:
                    dir_edges.append({
                        "source": source, "target": target, "horizon": hl,
                        "accuracy": v["best_direction_accuracy"],
                        "lag": v["best_direction_lag"],
                        "p_up_given_src_up": v.get("p_target_up_given_source_up_best"),
                    })
    dir_edges_sorted = sorted(dir_edges, key=lambda x: x["accuracy"], reverse=True)
    summary["directional_transfer_strongest_edges"] = dir_edges_sorted[:20]

    # Task 6: NAS100 driver summary
    nas_driver_summary = {}
    for target, tdata in t6.items():
        for hl, v in tdata.items():
            if isinstance(v, dict) and "nas100_es_optimal_corr" in v:
                key = f"{target}@{hl}"
                nas_driver_summary[key] = {
                    "es_corr": v["nas100_es_optimal_corr"],
                    "es_lag": v["nas100_es_optimal_lag"],
                    "memory_corr": v["nas100_memory_optimal_corr"],
                    "memory_lag": v["nas100_memory_optimal_lag"],
                    "concurrent_es_corr": v["concurrent_nas100_es_target_dir_corr"],
                }
    summary["nas100_driver"] = nas_driver_summary

    # Task 7: Context modulation summary
    ctx_mod = []
    for target, tdata in t7.items():
        for hl, hdata in tdata.items():
            for source, v in hdata.items():
                if isinstance(v, dict) and "modulation_same_vs_diff_regime" in v:
                    ctx_mod.append({
                        "source": source, "target": target, "horizon": hl,
                        "modulation": v["modulation_same_vs_diff_regime"],
                        "propagation_stronger_in_same_regime": v["propagation_stronger_in_same_regime"],
                        "global_alignment": v["global_regime_alignment"],
                    })
    ctx_mod_sorted = sorted(ctx_mod, key=lambda x: abs(x["modulation"]), reverse=True)
    summary["context_modulation"] = ctx_mod_sorted[:20]

    # Regime propagation summary
    regime_prop = []
    for target, tdata in t4.items():
        for source, v in tdata.items():
            if isinstance(v, dict) and "regime_change_follow" in v:
                f5 = v["regime_change_follow"].get("within_5", {})
                if f5 and f5.get("follow_rate") is not None:
                    regime_prop.append({
                        "source": source, "target": target,
                        "alignment": v["regime_alignment"],
                        "follow_rate_within_5": f5["follow_rate"],
                        "same_direction_rate": f5.get("same_direction_rate"),
                    })
    regime_prop_sorted = sorted(regime_prop, key=lambda x: x["follow_rate_within_5"], reverse=True)
    summary["regime_propagation"] = regime_prop_sorted

    # Key findings
    findings = []

    # ES propagation finding
    if es_edges_sorted:
        best = es_edges_sorted[0]
        findings.append(f"Strongest ES propagation: {best['source']}->{best['target']}@{best['horizon']} lag={best['lag']} corr={best['corr']}")
    nas_leads_count = len([e for e in es_edges if e["source"] == "NAS100" and e["lag"] > 0 and abs(e["corr"]) > 0.03])
    findings.append(f"NAS100 leads {nas_leads_count} other assets via ES at positive lags")

    # Directional transfer finding
    if dir_edges_sorted and dir_edges_sorted[0]["accuracy"] > 0.5:
        bestdir = dir_edges_sorted[0]
        findings.append(f"Best directional transmission: {bestdir['source']}->{bestdir['target']}@{bestdir['horizon']} acc={bestdir['accuracy']} lag={bestdir['lag']}")

    # NAS100 driver finding
    nas_pos = sum(1 for v in nas_driver_summary.values() if v.get("concurrent_es_corr", 0) > 0.03)
    nas_neg = sum(1 for v in nas_driver_summary.values() if v.get("concurrent_es_corr", 0) < -0.03)
    findings.append(f"NAS100 ES {'positively' if nas_pos > nas_neg else 'negatively'} correlates with {max(nas_pos, nas_neg)}/{len(nas_driver_summary)} asset-direction pairs")

    # Context modulation finding
    ctx_stronger_same = sum(1 for c in ctx_mod if c["propagation_stronger_in_same_regime"])
    findings.append(f"Context modulation: propagation stronger in same regime for {ctx_stronger_same}/{len(ctx_mod)} pairs")

    # Regime cascade finding
    if regime_prop_sorted:
        avg_follow = float(np.mean([r["follow_rate_within_5"] for r in regime_prop_sorted]))
        findings.append(f"Regime cascade: avg follow rate within 5 bars = {round(avg_follow, 4)}")

    summary["key_findings"] = findings

    return summary


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("CDER: Information Propagation Between Assets")
    print("=" * 60)

    print("\nLoading all symbols...")
    all_data = {sym: DPLData(sym) for sym in SYMBOLS}
    for sym, d in all_data.items():
        print(f"  {sym}: ES len={len(d.es)}, fut_ret={d.fut_ret.shape}, residuals={list(d.residuals.keys())}")

    print("\nTask 1: ES Propagation...")
    t1 = task1_es_propagation(all_data)

    print("Task 2: Memory Propagation...")
    t2 = task2_memory_propagation(all_data)

    print("Task 3: Residual Propagation...")
    t3 = task3_residual_propagation(all_data)

    print("Task 4: Regime Propagation...")
    t4 = task4_regime_propagation(all_data)

    print("Task 5: Directional Information Transfer...")
    t5 = task5_directional_transfer(all_data)

    print("Task 6: NAS100 as Driver (shorter horizons)...")
    t6 = task6_nas100_driver(all_data)

    print("Task 7: Propagation Through Context...")
    t7 = task7_context_modulated_propagation(all_data)

    print("\nBuilding summary...")
    summary = build_summary(t1, t2, t3, t4, t5, t6, t7)

    output = {
        "experiment": "CDER-InformationPropagation",
        "title": "Context-Dependent Energy Release: Information Propagation Between Assets",
        "symbols": SYMBOLS,
        "test_horizons": [TEST_HL[h] for h in TEST_HORIZONS],
        "residual_types": RESIDUAL_KEYS,
        "task1_es_propagation": t1,
        "task2_memory_propagation": t2,
        "task3_residual_propagation": t3,
        "task4_regime_propagation": t4,
        "task5_directional_transfer": t5,
        "task6_nas100_driver": t6,
        "task7_context_modulated_propagation": t7,
        "summary": summary,
    }

    out_path = OUT_DIR / "cder_information_propagation.json"
    out_path.write_text(json.dumps(_clean(output), indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Complete -> {out_path}")
    print(f"Key findings:")
    for f in summary.get("key_findings", []):
        print(f"  • {f}")


if __name__ == "__main__":
    main()
