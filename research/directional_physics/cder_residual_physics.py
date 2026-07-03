"""
CDER: Context-Dependent Energy Release — Residual Physics
Investigates time-series structure of ES residuals beyond simple sign.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, linregress, chi2_contingency
from sklearn.linear_model import LogisticRegression

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)

TEST_HORIZONS = [5, 20, 50]
TEST_HL = {h: HORIZON_LABELS[h] for h in TEST_HORIZONS}
RESIDUAL_TYPES = ["xgboost", "linear", "random_forest"]


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


# -------------------------------------------------------------------
# 1. Residual Persistence
# -------------------------------------------------------------------
def task1_persistence(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        valid = ~np.isnan(r)
        rv = r[valid]
        n = len(rv)

        # Autocorrelation at lags 1-50
        max_lag = min(50, n // 4)
        acf = np.ones(max_lag + 1)
        mu = np.mean(rv)
        var = np.var(rv)
        if var < 1e-12:
            results[rt] = {"error": "zero variance"}
            continue
        for lag in range(1, max_lag + 1):
            acf[lag] = np.mean((rv[lag:] - mu) * (rv[:-lag] - mu)) / var

        # Find lag where ACF crosses zero (persistence timescale)
        zero_cross = max_lag
        for lag in range(1, max_lag):
            if acf[lag] <= 0:
                zero_cross = lag - 1
                break

        # Half-life: lag where ACF drops below 0.5
        half_life = 1
        for lag in range(1, max_lag):
            if acf[lag] <= 0.5:
                half_life = lag
                break

        rt_res = {
            "max_lag_computed": max_lag,
            "acf_l1": round(float(acf[1]), 4),
            "acf_l5": round(float(acf[min(5, max_lag)]), 4),
            "acf_l10": round(float(acf[min(10, max_lag)]), 4),
            "acf_l20": round(float(acf[min(20, max_lag)]), 4),
            "zero_cross_lag": int(zero_cross),
            "half_life_lag": int(half_life),
            "acf_array": [round(float(acf[l]), 4) for l in range(0, max_lag + 1)],
        }

        # Does persistent residual sign predict stronger moves?
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = d.fut_ret[:, hidx]
            es = d.es.copy()

            # Require same sign for N consecutive bars
            for n_cons in [3, 5, 10]:
                cons_mask = np.full(len(r), False)
                sign = np.sign(rv)
                sign_full = np.full(len(r), np.nan)
                sign_full[valid] = sign
                for i in range(n_cons, len(r)):
                    if np.all(sign_full[i - n_cons:i] == sign_full[i]) and ~np.isnan(sign_full[i]) and sign_full[i] != 0:
                        cons_mask[i] = True

                valid_all = cons_mask & ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(r)
                if np.sum(valid_all) < 5:
                    continue

                es_v = es[valid_all]
                fwd_v = fwd[valid_all]
                r_v = r[valid_all]
                es_thr = np.nanpercentile(es_v, 80) if np.sum(~np.isnan(es_v)) > 10 else 0
                high_es = es_v > es_thr

                pos_cons = r_v > 0
                neg_cons = r_v < 0

                key = f"H{h}_cons{n_cons}"
                rt_res[key] = {
                    "n_consistent": int(np.sum(valid_all)),
                    "n_high_es": int(np.sum(high_es)),
                    "p_up_pos_consistent": round(p_up(fwd_v, pos_cons & high_es), 4),
                    "p_up_neg_consistent": round(p_up(fwd_v, neg_cons & high_es), 4),
                    "mean_ret_pos_consistent": round(mean_ret(fwd_v, pos_cons & high_es), 6),
                    "mean_ret_neg_consistent": round(mean_ret(fwd_v, neg_cons & high_es), 6),
                }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 2. Residual Accumulation
# -------------------------------------------------------------------
def task2_accumulation(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        es = d.es.copy()
        fut_ret = d.fut_ret

        # Cumulative sum of residuals (cumulative pressure)
        cumres = np.full_like(r, np.nan)
        valid = ~np.isnan(r)
        if np.sum(valid) > 0:
            r_clean = r[valid].copy()
            r_clean = np.where(np.isnan(r_clean), 0, r_clean)
            cumvals = np.cumsum(r_clean)
            cumres[valid] = cumvals

        rt_res = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            vmask = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(cumres) & ~np.isnan(r)
            if np.sum(vmask) < 30:
                continue

            es_v = es[vmask]
            cum_v = cumres[vmask]
            fwd_v = fwd[vmask]
            r_v = r[vmask]
            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr

            # Deciles of cumulative residual
            cum_deciles = np.nanpercentile(cum_v, np.arange(0, 101, 10))
            decile_info = {}
            for di in range(10):
                lo = cum_deciles[di]
                hi_b = cum_deciles[di + 1]
                if di == 9:
                    d_mask = (cum_v >= lo) & (cum_v <= hi_b)
                else:
                    d_mask = (cum_v >= lo) & (cum_v < hi_b)
                d_mask = d_mask & high_es
                n_d = int(np.sum(d_mask))
                if n_d >= 3:
                    decile_info[f"D{di+1}"] = {
                        "n": n_d,
                        "p_up": round(p_up(fwd_v, d_mask), 4),
                        "mean_ret": round(mean_ret(fwd_v, d_mask), 6),
                        "cumres_range": [round(float(lo), 4), round(float(hi_b), 4)],
                    }

            # Direction follows accumulated residual sign?
            cum_pos = cum_v > 0
            cum_neg = cum_v < 0
            p_up_cum_pos = p_up(fwd_v, cum_pos & high_es)
            p_up_cum_neg = p_up(fwd_v, cum_neg & high_es)

            # Recent accumulation (last 20 bars)
            window = 20
            cumres_short = np.full_like(r, np.nan)
            for i in range(window, len(r)):
                chunk = r[i - window:i]
                if np.sum(~np.isnan(chunk)) > window // 2:
                    cumres_short[i] = np.nansum(chunk)

            vmask2 = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(cumres_short) & ~np.isnan(r)
            if np.sum(vmask2) >= 30:
                es_v2 = es[vmask2]
                cum_s_v = cumres_short[vmask2]
                fwd_v2 = fwd[vmask2]
                es_thr2 = np.nanpercentile(es_v2, 80)
                high_es2 = es_v2 > es_thr2
                cs_pos = cum_s_v > 0
                cs_neg = cum_s_v < 0
                p_up_cs_pos = p_up(fwd_v2, cs_pos & high_es2)
                p_up_cs_neg = p_up(fwd_v2, cs_neg & high_es2)
            else:
                p_up_cs_pos = p_up_cum_pos
                p_up_cs_neg = p_up_cum_neg

            # Correlation: cumres magnitude vs future return
            corr_cum_ret, _ = pearsonr(cum_v[high_es], fwd_v[high_es]) if np.sum(high_es) > 5 else (0, 1)
            corr_cum_dir, _ = pearsonr(cum_v[high_es], (fwd_v[high_es] > 0).astype(float)) if np.sum(high_es) > 5 else (0, 1)

            rt_res[TEST_HL[h]] = {
                "n_valid": int(np.sum(vmask)),
                "n_high_es": int(np.sum(high_es)),
                "p_up_cumres_pos": round(p_up_cum_pos, 4),
                "p_up_cumres_neg": round(p_up_cum_neg, 4),
                "p_up_recent_cumres_pos": round(p_up_cs_pos, 4),
                "p_up_recent_cumres_neg": round(p_up_cs_neg, 4),
                "corr_cumres_return": round(float(corr_cum_ret), 4),
                "corr_cumres_direction": round(float(corr_cum_dir), 4),
                "cumres_decile_analysis": decile_info,
                "accumulation_predicts_direction": bool(
                    abs(p_up_cum_pos - p_up_cum_neg) > 0.05
                ),
            }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 3. Residual Exhaustion
# -------------------------------------------------------------------
def task3_exhaustion(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        es = d.es.copy()
        fut_ret = d.fut_ret

        thr = 1.5  # sigma threshold for "extreme"
        rt_res = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(r)
            if np.sum(valid) < 30:
                continue

            es_v = es[valid]
            r_v = r[valid]
            fwd_v = fwd[valid]
            r_std = np.nanstd(r_v)
            if r_std < 1e-12:
                continue

            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr

            # Extreme residual events (positive and negative)
            extreme_pos = (r_v > thr * r_std)
            extreme_neg = (r_v < -thr * r_std)

            # What happens AFTER extreme residuals?
            # Shift forward by 1 to look at subsequent bar
            n = len(r_v)
            post_extreme_pos = np.zeros(n, dtype=bool)
            post_extreme_neg = np.zeros(n, dtype=bool)
            post_extreme_pos[:n-1] = extreme_pos[1:]
            post_extreme_neg[:n-1] = extreme_neg[1:]

            # Immediate reversal test: after extreme pos residual, does ES drop?
            es_change = np.full(n, np.nan)
            es_change[:n-1] = es_v[1:] - es_v[:-1]
            es_reversal_pos = es_change < 0
            es_reversal_neg = es_change > 0
            n_extreme_pos = int(np.sum(extreme_pos))
            n_extreme_neg = int(np.sum(extreme_neg))

            reversal_rate_pos = float(np.mean(es_reversal_pos[extreme_pos])) if n_extreme_pos > 3 else None
            reversal_rate_neg = float(np.mean(es_reversal_neg[extreme_neg])) if n_extreme_neg > 3 else None

            # Does exhaustion predict opposite direction?
            # Look at future return after extreme residual in high ES
            extreme_pos_high = extreme_pos & high_es
            extreme_neg_high = extreme_neg & high_es
            p_up_after_extreme_pos = p_up(fwd_v, extreme_pos_high)
            p_up_after_extreme_neg = p_up(fwd_v, extreme_neg_high)
            mean_ret_after_extreme_pos = mean_ret(fwd_v, extreme_pos_high)
            mean_ret_after_extreme_neg = mean_ret(fwd_v, extreme_neg_high)

            # Baseline: non-extreme high ES
            non_extreme_high = high_es & ~extreme_pos & ~extreme_neg
            p_up_baseline = p_up(fwd_v, non_extreme_high)

            # Exhaustion signal: extreme residual predicts opposite direction
            exhaustion_pos = p_up_after_extreme_pos < 0.5
            exhaustion_neg = p_up_after_extreme_neg > 0.5
            exhaustion_works = exhaustion_pos and exhaustion_neg

            # Multi-bar decay: what happens 2,3,5 bars after extreme?
            for ahead in [2, 3, 5]:
                if hi + ahead - 1 >= len(TEST_HORIZONS):
                    continue
                # Use same horizon for simplicity
                pass

            rt_res[TEST_HL[h]] = {
                "n_extreme_pos": n_extreme_pos,
                "n_extreme_neg": n_extreme_neg,
                "threshold_sigma": thr,
                "es_reversal_rate_post_pos": reversal_rate_pos,
                "es_reversal_rate_post_neg": reversal_rate_neg,
                "p_up_post_extreme_pos_high_es": round(p_up_after_extreme_pos, 4),
                "p_up_post_extreme_neg_high_es": round(p_up_after_extreme_neg, 4),
                "mean_ret_post_extreme_pos_high_es": round(mean_ret_after_extreme_pos, 6),
                "mean_ret_post_extreme_neg_high_es": round(mean_ret_after_extreme_neg, 6),
                "p_up_baseline_high_es": round(p_up_baseline, 4),
                "exhaustion_predicts_opposite_direction": exhaustion_works,
                "improvement_over_baseline_pos": round(p_up_after_extreme_pos - p_up_baseline, 4),
                "improvement_over_baseline_neg": round(p_up_after_extreme_neg - p_up_baseline, 4),
            }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 4. Residual Clustering
# -------------------------------------------------------------------
def task4_clustering(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        valid = ~np.isnan(r)
        rv = r[valid]
        n = len(rv)
        if n < 100:
            results[rt] = {"error": "insufficient data"}
            continue

        # Runs test: count sign runs vs expected
        sign = np.sign(rv)
        sign = sign[sign != 0]
        if len(sign) < 50:
            results[rt] = {"error": "too many zeros"}
            continue

        n_runs = 1 + int(np.sum(sign[1:] != sign[:-1]))
        n_pos = int(np.sum(sign > 0))
        n_neg = int(np.sum(sign < 0))
        n_total = n_pos + n_neg
        expected_runs = 1 + (2 * n_pos * n_neg) / max(n_total, 1)
        runs_z = (n_runs - expected_runs) / max(np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_total)) / (n_total ** 2 * (n_total - 1))), 1e-12) if n_total > 1 else 0

        # Positive clustering: average positive run length
        pos_runs = []
        neg_runs = []
        cur_len = 0
        cur_sign = 0
        for s in sign:
            if s == 0:
                continue
            if cur_sign == 0:
                cur_sign = s
                cur_len = 1
            elif s == cur_sign:
                cur_len += 1
            else:
                if cur_sign > 0:
                    pos_runs.append(cur_len)
                else:
                    neg_runs.append(cur_len)
                cur_sign = s
                cur_len = 1
        if cur_len > 0:
            if cur_sign > 0:
                pos_runs.append(cur_len)
            else:
                neg_runs.append(cur_len)

        avg_pos_run = float(np.mean(pos_runs)) if pos_runs else 0
        avg_neg_run = float(np.mean(neg_runs)) if neg_runs else 0
        max_pos_run = int(np.max(pos_runs)) if pos_runs else 0
        max_neg_run = int(np.max(neg_runs)) if neg_runs else 0

        # Hurst exponent (simplified: R/S method)
        def hurst_rs(x):
            n = len(x)
            if n < 100:
                return 0.5
            max_k = min(n // 4, 252)
            ks = np.logspace(np.log10(10), np.log10(max_k), 20, dtype=int)
            ks = np.unique(ks)
            rs_vals = []
            for k in ks:
                n_seg = n // k
                if n_seg < 2:
                    continue
                rs = []
                for seg in range(n_seg):
                    chunk = x[seg * k:(seg + 1) * k]
                    if np.std(chunk) < 1e-12:
                        continue
                    mean_adj = chunk - np.mean(chunk)
                    cum_dev = np.cumsum(mean_adj)
                    r = np.max(cum_dev) - np.min(cum_dev)
                    s = np.std(chunk)
                    rs.append(r / s if s > 0 else 0)
                if rs:
                    rs_vals.append(np.mean(rs))
            if len(rs_vals) < 3:
                return 0.5
            log_rs = np.log(rs_vals)
            log_k = np.log(ks[:len(rs_vals)])
            slope, _, _, _, _ = linregress(log_k, log_rs)
            return slope if not np.isnan(slope) else 0.5

        hurst = hurst_rs(rv)

        # Does clustering predict directional runs?
        es = d.es.copy()
        fut_ret = d.fut_ret
        rt_res = {
            "n_valid": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_runs": n_runs,
            "expected_runs": round(float(expected_runs), 2),
            "runs_z_score": round(float(runs_z), 4),
            "avg_pos_run_length": round(avg_pos_run, 2),
            "avg_neg_run_length": round(avg_neg_run, 2),
            "max_pos_run_length": max_pos_run,
            "max_neg_run_length": max_neg_run,
            "hurst_exponent": round(float(hurst), 4),
            "residual_clustering": "anti-persistent" if hurst < 0.45 else ("persistent" if hurst > 0.55 else "random"),
        }

        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            vmask = ~np.isnan(es) & ~np.isnan(fwd) & valid
            if np.sum(vmask) < 30:
                continue

            es_v = es[vmask]
            fwd_v = fwd[vmask]
            r_v = r[vmask]
            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr

            # Inside long runs vs outside
            long_run_mask = np.full(len(r_v), False)
            sign_r = np.sign(r_v)
            for i in range(len(r_v)):
                if i < 3:
                    continue
                if np.all(sign_r[i-2:i+1] == sign_r[i]) and sign_r[i] != 0:
                    long_run_mask[i] = True

            inside_run = long_run_mask & high_es
            outside_run = ~long_run_mask & high_es
            p_up_inside = p_up(fwd_v, inside_run)
            p_up_outside = p_up(fwd_v, outside_run)
            mean_ret_inside = mean_ret(fwd_v, inside_run)
            mean_ret_outside = mean_ret(fwd_v, outside_run)

            rt_res[TEST_HL[h]] = {
                "n_inside_run_high_es": int(np.sum(inside_run)),
                "n_outside_run_high_es": int(np.sum(outside_run)),
                "p_up_inside_residual_run": round(p_up_inside, 4),
                "p_up_outside_residual_run": round(p_up_outside, 4),
                "mean_ret_inside_residual_run": round(mean_ret_inside, 6),
                "mean_ret_outside_residual_run": round(mean_ret_outside, 6),
                "direction_amplified_inside_run": bool(abs(p_up_inside - 0.5) > abs(p_up_outside - 0.5)),
            }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 5. Residual Shocks
# -------------------------------------------------------------------
def task5_shocks(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        es = d.es.copy()
        fut_ret = d.fut_ret

        valid = ~np.isnan(r)
        rv = r[valid]
        r_std = np.nanstd(rv)
        if r_std < 1e-12:
            continue

        # Shock = |residual| > 2 sigma
        shock_mask = np.full(len(r), False)
        shock_mask[valid] = np.abs(rv) > 2 * r_std

        shock_indices = np.where(shock_mask)[0]

        rt_res = {
            "threshold_sigma": 2.0,
            "n_shocks": int(np.sum(shock_mask)),
            "shock_frequency": round(float(np.sum(shock_mask)) / max(1, len(r)), 6),
            "mean_shock_magnitude": round(float(np.mean(np.abs(r[shock_mask]))) if np.sum(shock_mask) > 0 else 0, 4),
        }

        # ES behavior before/after shocks
        if np.sum(shock_mask) > 5:
            es_before = []
            es_after = []
            for si in shock_indices:
                if si >= 5 and si < len(es) - 5:
                    es_before.append(es[si - 5:si])
                    es_after.append(es[si:si + 5])
            if es_before:
                es_before_arr = np.array(es_before)
                es_after_arr = np.array(es_after)
                rt_res["es_before_shock_mean"] = np.nanmean(es_before_arr, axis=0).tolist()
                rt_res["es_after_shock_mean"] = np.nanmean(es_after_arr, axis=0).tolist()
                rt_res["es_drop_after_shock"] = float(np.nanmean(es_after_arr[:, 0] - es_before_arr[:, -1]))
            else:
                rt_res["es_before_shock_mean"] = None
                rt_res["es_after_shock_mean"] = None
                rt_res["es_drop_after_shock"] = None
        else:
            rt_res["es_before_shock_mean"] = None
            rt_res["es_after_shock_mean"] = None
            rt_res["es_drop_after_shock"] = None

        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            vmask = ~np.isnan(es) & ~np.isnan(fwd) & valid
            if np.sum(vmask) < 30:
                continue

            es_v = es[vmask]
            fwd_v = fwd[vmask]
            s_v = shock_mask[vmask]
            r_v = r[vmask]

            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr

            # Shock then direction
            shock_high = s_v & high_es
            p_up_shock = p_up(fwd_v, shock_high)
            mean_ret_shock = mean_ret(fwd_v, shock_high)

            # Positive vs negative shocks
            pos_shock = (r_v > 2 * r_std) & high_es
            neg_shock = (r_v < -2 * r_std) & high_es
            p_up_pos_shock = p_up(fwd_v, pos_shock)
            p_up_neg_shock = p_up(fwd_v, neg_shock)
            mean_ret_pos_shock = mean_ret(fwd_v, pos_shock)
            mean_ret_neg_shock = mean_ret(fwd_v, neg_shock)

            # Non-shock high ES baseline
            nonshock_high = high_es & ~s_v
            p_up_nonshock = p_up(fwd_v, nonshock_high)

            # Breakout test: shock + high ES -> outsized move?
            breakout_ratio = float(mean_ret_shock / max(abs(mean_ret(fwd_v, nonshock_high)), 1e-12)) if np.sum(nonshock_high) > 3 else None

            rt_res[TEST_HL[h]] = {
                "n_shock_high_es": int(np.sum(shock_high)),
                "p_up_after_shock_high_es": round(p_up_shock, 4),
                "mean_ret_after_shock_high_es": round(mean_ret_shock, 6),
                "p_up_after_pos_shock": round(p_up_pos_shock, 4),
                "p_up_after_neg_shock": round(p_up_neg_shock, 4),
                "mean_ret_after_pos_shock": round(mean_ret_pos_shock, 6),
                "mean_ret_after_neg_shock": round(mean_ret_neg_shock, 6),
                "p_up_nonshock_baseline": round(p_up_nonshock, 4),
                "breakout_ratio_vs_baseline": round(breakout_ratio, 4) if breakout_ratio is not None else None,
                "shock_predicts_breakout": bool(
                    abs(p_up_shock - 0.5) > abs(p_up_nonshock - 0.5)
                ) if np.sum(nonshock_high) > 3 else None,
            }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 6. Residual Memory (Logistic Model)
# -------------------------------------------------------------------
def task6_memory_model(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        es = d.es.copy()
        fut_ret = d.fut_ret

        rt_res = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(r)
            if np.sum(valid) < 100:
                continue

            es_v = es[valid]
            r_v = r[valid]
            fwd_v = fwd[valid]
            dir_v = (fwd_v > 0).astype(float)

            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr
            if np.sum(high_es) < 30:
                continue

            # Build feature matrix: lagged residuals
            max_lag = 10
            n_samples = int(np.sum(high_es))
            idx = np.where(high_es)[0]

            X_list = []
            y_list = []
            for ii in idx:
                if ii < max_lag:
                    continue
                lags = r_v[ii - max_lag:ii]
                if np.any(np.isnan(lags)):
                    continue
                X_list.append(lags)
                y_list.append(dir_v[ii])

            if len(X_list) < 30:
                continue

            X = np.array(X_list)
            y = np.array(y_list)

            # Model 1: logistic with all lags
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            acc_lag = float(np.mean(y_pred == y))

            # Model 2: sign only (single feature: current residual sign)
            X_sign = r_v[idx[idx >= max_lag]].reshape(-1, 1)
            y_sign = dir_v[idx[idx >= max_lag]]
            X_sign_valid = ~np.isnan(X_sign.flatten())
            if np.sum(X_sign_valid) < 30:
                continue
            X_sign_clean = X_sign[X_sign_valid]
            y_sign_clean = y_sign[X_sign_valid]
            if len(np.unique(y_sign_clean)) < 2:
                continue
            clf_sign = LogisticRegression(max_iter=1000, random_state=42)
            clf_sign.fit(X_sign_clean.reshape(-1, 1), y_sign_clean)
            y_pred_sign = clf_sign.predict(X_sign_clean.reshape(-1, 1))
            acc_sign = float(np.mean(y_pred_sign == y_sign_clean))

            # Feature importance (coefficients)
            coefs = clf.coef_[0].tolist()

            # Improvement over sign alone
            improvement = round(acc_lag - acc_sign, 4)

            # Which lags matter most?
            abs_coefs = np.abs(coefs)
            best_lag = int(np.argmax(abs_coefs)) + 1  # 1-indexed

            # Reduced model: 3 best lags only
            top3_lags = np.argsort(-abs_coefs)[:3]
            X_top3 = X[:, top3_lags]
            clf3 = LogisticRegression(max_iter=1000, random_state=42)
            clf3.fit(X_top3, y)
            y_pred3 = clf3.predict(X_top3)
            acc_top3 = float(np.mean(y_pred3 == y))

            # Cross-validation (simple train/test split)
            split = int(len(X) * 0.7)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                cv_acc = acc_lag
            else:
                clf_cv = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                clf_cv.fit(X_train, y_train)
                cv_acc = float(np.mean(clf_cv.predict(X_test) == y_test))

            rt_res[TEST_HL[h]] = {
                "n_train": int(len(X)),
                "n_features": max_lag,
                "accuracy_lag_model": round(acc_lag, 4),
                "accuracy_sign_only": round(acc_sign, 4),
                "improvement_over_sign": improvement,
                "accuracy_top3_lags": round(acc_top3, 4),
                "holdout_accuracy": round(cv_acc, 4),
                "coefficients_by_lag": {f"lag_{i+1}": round(c, 4) for i, c in enumerate(coefs)},
                "most_important_lag": best_lag,
                "top3_lags": [int(x + 1) for x in top3_lags],
                "lag_model_beats_sign": acc_lag > acc_sign,
            }

        results[rt] = rt_res
    return results


# -------------------------------------------------------------------
# 7. Residual × Regime Interaction
# -------------------------------------------------------------------
def task7_regime_interaction(d: DPLData, sym: str) -> dict:
    results = {}
    for rt in RESIDUAL_TYPES:
        resid = d.residuals.get(rt)
        if resid is None:
            continue
        r = resid.copy()
        es = d.es.copy()
        states = d.states.copy().astype(int)
        fut_ret = d.fut_ret

        rt_res = {}
        for hi, h in enumerate(TEST_HORIZONS):
            hidx = HORIZONS.index(h)
            fwd = fut_ret[:, hidx]
            valid = ~np.isnan(es) & ~np.isnan(fwd) & ~np.isnan(r) & (states >= 0)
            if np.sum(valid) < 30:
                continue

            es_v = es[valid]
            r_v = r[valid]
            s_v = states[valid]
            fwd_v = fwd[valid]
            unique_s = sorted(np.unique(s_v))

            es_thr = np.nanpercentile(es_v, 80)
            high_es = es_v > es_thr

            # Cross-tabulation: residual sign × regime state
            r_sign = np.sign(r_v)
            cross_tab = {}
            for si in unique_s:
                for sign_val in [-1, 0, 1]:
                    cell_mask = (s_v == si) & (r_sign == sign_val) & high_es
                    n_cell = int(np.sum(cell_mask))
                    if n_cell < 3:
                        continue
                    label = f"S{int(si)}_resid_{'+' if sign_val > 0 else '-' if sign_val < 0 else '0'}"
                    cross_tab[label] = {
                        "n": n_cell,
                        "p_up": round(p_up(fwd_v, cell_mask), 4),
                        "mean_ret": round(mean_ret(fwd_v, cell_mask), 6),
                        "regime": int(si),
                        "residual_sign": int(sign_val),
                    }

            # Does regime modify residual directional accuracy?
            regime_acc = {}
            for si in unique_s:
                s_mask = (s_v == si) & high_es
                if np.sum(s_mask) < 5:
                    continue
                resid_sign_v = r_v[s_mask]
                fwd_sign_v = fwd_v[s_mask]
                pred_up = resid_sign_v > 0
                actual_up = fwd_sign_v > 0
                acc = float(np.mean(pred_up == actual_up)) if len(pred_up) > 3 else 0.5
                regime_acc[f"S{int(si)}"] = {
                    "n": int(np.sum(s_mask)),
                    "directional_accuracy": round(acc, 4),
                }

            # Overall residual accuracy
            all_pred = r_v > 0
            all_actual = fwd_v > 0
            base_acc = float(np.mean(all_pred[high_es] == all_actual[high_es])) if np.sum(high_es) > 3 else 0.5

            # Accuracy spread across regimes
            accs = [v["directional_accuracy"] for v in regime_acc.values()]
            acc_spread = round(float(np.max(accs) - np.min(accs)), 4) if len(accs) >= 2 else 0

            # Chi-squared test: does regime modify sign→direction relationship?
            contingency = []
            for si in unique_s:
                for sign_val in [-1, 1]:
                    cell_mask = (s_v == si) & (r_sign == sign_val) & high_es
                    if np.sum(cell_mask) < 3:
                        continue
                    n_up = int(np.sum(fwd_v[cell_mask] > 0))
                    n_down = int(np.sum(cell_mask)) - n_up
                    contingency.append([n_up, n_down])

            chi2_pval = None
            if len(contingency) >= 4:
                try:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    chi2_pval = round(float(p), 4)
                except Exception:
                    chi2_pval = None

            rt_res[TEST_HL[h]] = {
                "n_valid": int(np.sum(valid)),
                "n_high_es": int(np.sum(high_es)),
                "unique_regimes": [int(x) for x in unique_s],
                "cross_tab_residual_sign_x_regime": cross_tab,
                "residual_accuracy_by_regime": regime_acc,
                "overall_residual_accuracy": round(base_acc, 4),
                "accuracy_spread_across_regimes": acc_spread,
                "chi2_p_value": chi2_pval,
                "regime_modifies_accuracy": acc_spread > 0.05,
                "best_regime_accuracy": round(float(np.max(accs)), 4) if accs else None,
                "worst_regime_accuracy": round(float(np.min(accs)), 4) if accs else None,
            }

        results[rt] = rt_res
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
        print(f"  Data: {len(d.price)} bars, states={np.max(d.states)}")

        print(f"  Task 1: Residual Persistence...")
        t1 = task1_persistence(d, sym)

        print(f"  Task 2: Residual Accumulation...")
        t2 = task2_accumulation(d, sym)

        print(f"  Task 3: Residual Exhaustion...")
        t3 = task3_exhaustion(d, sym)

        print(f"  Task 4: Residual Clustering...")
        t4 = task4_clustering(d, sym)

        print(f"  Task 5: Residual Shocks...")
        t5 = task5_shocks(d, sym)

        print(f"  Task 6: Residual Memory Model...")
        t6 = task6_memory_model(d, sym)

        print(f"  Task 7: Residual × Regime Interaction...")
        t7 = task7_regime_interaction(d, sym)

        per_symbol[sym] = {
            "residual_persistence": t1,
            "residual_accumulation": t2,
            "residual_exhaustion": t3,
            "residual_clustering": t4,
            "residual_shocks": t5,
            "residual_memory_model": t6,
            "residual_regime_interaction": t7,
        }

    summary = _build_summary(per_symbol)

    output = {
        "experiment": "CDER-Residual-Physics",
        "title": "Context-Dependent Energy Release: Residual Physics Analysis",
        "per_symbol": per_symbol,
        "summary": summary,
    }

    out_path = Path(__file__).parent / "reports" / "cder_residual_physics.json"
    out_path.write_text(json.dumps(_clean(output), indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"CDER Residual Physics complete -> {out_path}")


def _build_summary(per_symbol: dict) -> dict:
    # Collect cross-cutting metrics
    persistence_timescales = []
    half_lives = []
    acf_l1_vals = []
    accumulation_predicts = 0
    exhaustion_works = 0
    exhaustion_total = 0
    hurst_vals = []
    shock_improves = 0
    shock_total = 0
    lag_improvements = []
    regime_modifies = 0
    regime_total = 0
    best_residual_accs = {rt: [] for rt in RESIDUAL_TYPES}
    baseline_residual_accs = {rt: [] for rt in RESIDUAL_TYPES}

    for sym, r in per_symbol.items():
        # Persistence
        tp = r.get("residual_persistence", {})
        for rt in RESIDUAL_TYPES:
            p = tp.get(rt, {})
            if "acf_l1" in p and p["acf_l1"] is not None:
                acf_l1_vals.append(p["acf_l1"])
            if "zero_cross_lag" in p:
                persistence_timescales.append(p["zero_cross_lag"])
            if "half_life_lag" in p and p["half_life_lag"] is not None:
                half_lives.append(p["half_life_lag"])

        # Accumulation
        ta = r.get("residual_accumulation", {})
        for rt in RESIDUAL_TYPES:
            ah = ta.get(rt, {})
            for hl, v in ah.items():
                if isinstance(v, dict) and v.get("accumulation_predicts_direction"):
                    accumulation_predicts += 1

        # Exhaustion
        te = r.get("residual_exhaustion", {})
        for rt in RESIDUAL_TYPES:
            eh = te.get(rt, {})
            for hl, v in eh.items():
                if isinstance(v, dict):
                    exhaustion_total += 1
                    if v.get("exhaustion_predicts_opposite_direction"):
                        exhaustion_works += 1

        # Clustering
        tc = r.get("residual_clustering", {})
        for rt in RESIDUAL_TYPES:
            ch = tc.get(rt, {})
            if "hurst_exponent" in ch:
                hurst_vals.append(ch["hurst_exponent"])

        # Shocks
        ts = r.get("residual_shocks", {})
        for rt in RESIDUAL_TYPES:
            sh = ts.get(rt, {})
            for hl, v in sh.items():
                if isinstance(v, dict):
                    shock_total += 1
                    if v.get("shock_predicts_breakout"):
                        shock_improves += 1

        # Memory model
        tm = r.get("residual_memory_model", {})
        for rt in RESIDUAL_TYPES:
            mh = tm.get(rt, {})
            for hl, v in mh.items():
                if isinstance(v, dict) and "improvement_over_sign" in v:
                    lag_improvements.append(v["improvement_over_sign"])

        # Regime interaction
        tr = r.get("residual_regime_interaction", {})
        for rt in RESIDUAL_TYPES:
            rh = tr.get(rt, {})
            for hl, v in rh.items():
                if isinstance(v, dict):
                    regime_total += 1
                    if v.get("regime_modifies_accuracy"):
                        regime_modifies += 1
                    if "overall_residual_accuracy" in v:
                        baseline_residual_accs[rt].append(v["overall_residual_accuracy"])
                    if "best_regime_accuracy" in v and v["best_regime_accuracy"] is not None:
                        best_residual_accs[rt].append(v["best_regime_accuracy"])

    # Aggregate
    avg_persistence = round(float(np.mean(persistence_timescales)), 1) if persistence_timescales else None
    avg_half_life = round(float(np.mean(half_lives)), 1) if half_lives else None
    avg_acf_l1 = round(float(np.mean(acf_l1_vals)), 4) if acf_l1_vals else None
    avg_hurst = round(float(np.mean(hurst_vals)), 4) if hurst_vals else None

    # Best residual model (highest accuracy)
    model_scores = {}
    for rt in RESIDUAL_TYPES:
        if baseline_residual_accs[rt]:
            model_scores[rt] = round(float(np.mean(baseline_residual_accs[rt])), 4)
    best_model = max(model_scores, key=model_scores.get) if model_scores else None
    best_model_acc = model_scores.get(best_model, 0) if best_model else 0

    # Best regime-enhanced accuracy
    regime_scores = {}
    for rt in RESIDUAL_TYPES:
        if best_residual_accs[rt]:
            regime_scores[rt] = round(float(np.mean(best_residual_accs[rt])), 4)
    best_regime_model = max(regime_scores, key=regime_scores.get) if regime_scores else None
    best_regime_acc = regime_scores.get(best_regime_model, 0) if best_regime_model else 0

    avg_lag_improvement = round(float(np.mean(lag_improvements)), 4) if lag_improvements else None

    # Direction prediction accuracy improvements over baseline
    # Baseline ~60% (from DPL tournament)
    baseline_directional = 0.60
    improvements = {}
    if model_scores:
        improvements["best_model_over_baseline"] = round(best_model_acc - baseline_directional, 4)
    if lag_improvements:
        improvements["lag_model_improvement_over_sign"] = avg_lag_improvement
    if best_regime_acc and baseline_residual_accs.get(best_regime_model):
        base = float(np.mean(baseline_residual_accs.get(best_regime_model, [0.6])))
        improvements["regime_improvement_over_baseline"] = round(best_regime_acc - base, 4)
        improvements["regime_improvement_over_60pct"] = round(best_regime_acc - baseline_directional, 4)

    return {
        "n_symbols": len(per_symbol),
        "tested_horizons": [TEST_HL[h] for h in TEST_HORIZONS],
        "residual_persistence": {
            "avg_persistence_timescale_bars": avg_persistence,
            "avg_half_life_bars": avg_half_life,
            "avg_acf_lag1": avg_acf_l1,
            "interpretation": f"Residual memory decays at ~{avg_persistence} bars (ACF≈0 at lag {avg_persistence})" if avg_persistence else "Unknown",
        },
        "residual_clustering": {
            "avg_hurst_exponent": avg_hurst,
            "interpretation": "Anti-persistent" if avg_hurst and avg_hurst < 0.45 else ("Persistent" if avg_hurst and avg_hurst > 0.55 else "Random walk"),
        },
        "accumulation_predicts_direction": {
            "n_cases_accumulation_helps": accumulation_predicts,
            "interpretation": "Cumulative residual pressure predicts direction" if accumulation_predicts > 5
            else "Cumulative residual does NOT reliably predict direction",
        },
        "exhaustion_predicts_reversal": {
            "exhaustion_works_ratio": round(exhaustion_works / max(exhaustion_total, 1), 4),
            "interpretation": "Extreme residuals predict opposite direction (exhaustion)" if exhaustion_works > exhaustion_total // 2
            else "Exhaustion does NOT reliably predict reversal",
        },
        "shock_breakout_analysis": {
            "shock_improves_ratio": round(shock_improves / max(shock_total, 1), 4),
            "interpretation": "2-sigma shocks predict directional breakouts" if shock_improves > shock_total // 2
            else "Shocks do NOT reliably predict breakouts",
        },
        "residual_memory_model": {
            "avg_improvement_over_sign_alone": avg_lag_improvement,
            "interpretation": "Lagged residuals improve prediction over sign alone" if avg_lag_improvement and avg_lag_improvement > 0.01
            else "Lagged residuals do NOT significantly improve over sign alone",
        },
        "residual_regime_interaction": {
            "regime_modifies_accuracy_ratio": round(regime_modifies / max(regime_total, 1), 4),
            "interpretation": "Regime state modifies residual directional accuracy" if regime_modifies > regime_total // 3
            else "Regime does NOT reliably modify residual accuracy",
        },
        "best_residual_model": {
            "model": best_model,
            "mean_directional_accuracy": best_model_acc,
        },
        "best_residual_model_with_regime": {
            "model": best_regime_model,
            "mean_regime_enhanced_accuracy": best_regime_acc,
        },
        "directional_accuracy_improvements": improvements,
        "key_finding": (
            f"Residual persistence timescale: ~{avg_persistence} bars (half-life ~{avg_half_life} bar). "
            f"Hurst={avg_hurst} ({'anti-persistent' if avg_hurst and avg_hurst < 0.45 else 'persistent' if avg_hurst and avg_hurst > 0.55 else 'random'}). "
            f"{'Accumulation predicts direction. ' if accumulation_predicts > 5 else 'Accumulation does not predict direction. '}"
            f"{'Exhaustion predicts reversal. ' if exhaustion_works > exhaustion_total // 2 else 'Exhaustion does not predict reversal. '}"
            f"{'Shocks predict breakouts. ' if shock_improves > shock_total // 2 else 'Shocks do not predict breakouts. '}"
            f"Best model: {best_model} ({best_model_acc}). "
            f"Regime enhances accuracy: {best_regime_model} ({best_regime_acc}). "
            f"Improvement over 60% baseline: {improvements.get('regime_improvement_over_60pct', 'N/A')}."
        ),
    }


if __name__ == "__main__":
    main()
