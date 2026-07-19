"""Walk-forward validation harness for WLS currency decomposition.

Answers: Do WLS currency strengths predict future pair returns?
Measures predictive skill at 5m, 15m, 30m, 60m horizons.
Jointly tunes lambda, smoothing_alpha, prior_shrink.
"""

import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .data_loader import CURRENCIES, build_design_matrix
from .metrics import predictive_skill, hit_rate, information_coefficient


@dataclass
class WLSState:
    strengths: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    prior: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    smoothed: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    smooth_counter: int = 0


def solve_wls(
    A: np.ndarray,
    pair_returns: np.ndarray,
    prior_vec: np.ndarray,
    lam: float = 0.01,
    weight_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weight_vec is None:
        weight_vec = np.ones(A.shape[0])
    weight_mat = np.diag(weight_vec)
    AtW = A.T @ weight_mat
    lhs = AtW @ A + lam * np.eye(A.shape[1])
    rhs = AtW @ pair_returns + lam * prior_vec
    x = np.linalg.solve(lhs, rhs)
    return x - np.mean(x)


def run_walk_forward(
    return_matrix: np.ndarray,
    timestamps: np.ndarray,
    pair_labels: list[str],
    design_matrix: np.ndarray,
    horizons: list[int] = None,
    lam: float = 0.01,
    window: int = 24,
    smoothing_alpha: float = 0.2,
    prior_shrink: float = 0.0,
    n_holdout: int = 2,
    verbose: bool = False,
) -> dict:
    if horizons is None:
        horizons = [1, 3, 6, 12]
    A = design_matrix
    n_pairs = A.shape[0]
    T = return_matrix.shape[0]
    min_viable = window + max(horizons) + 2
    if T < min_viable:
        return {"error": f"Need at least {min_viable} bars, got {T}"}

    state = WLSState()
    predictions_by_horizon = {h: [] for h in horizons}
    actuals_by_horizon = {h: [] for h in horizons}
    all_preds_by_horizon = {h: [] for h in horizons}
    all_actuals_by_horizon = {h: [] for h in horizons}
    strength_history = []

    n_skipped_low_pairs = 0
    n_skipped_solve_error = 0

    for t in range(window, T - max(horizons)):
        latest = return_matrix[t]
        active_mask = np.abs(latest) > 1e-12
        active_count = int(np.sum(active_mask))
        if active_count < 8:
            n_skipped_low_pairs += 1
            continue

        weight_vec = np.where(active_mask, 1.0, 1e-6)
        prior_vec = np.array([state.prior.get(c, 0.0) for c in CURRENCIES])
        try:
            strengths = solve_wls(A, latest, prior_vec, lam=lam, weight_vec=weight_vec)
        except np.linalg.LinAlgError:
            n_skipped_solve_error += 1
            continue

        strength_dict = {c: float(strengths[i]) for i, c in enumerate(CURRENCIES)}
        state.strengths = strength_dict
        shrunk = {c: (1 - prior_shrink) * strength_dict[c] for c in CURRENCIES}
        state.prior = shrunk
        if state.smooth_counter == 0:
            state.smoothed = dict(shrunk)
        else:
            for c in CURRENCIES:
                state.smoothed[c] = smoothing_alpha * shrunk[c] + (1 - smoothing_alpha) * state.smoothed[c]
        state.smooth_counter += 1
        strength_history.append(dict(state.smoothed))

        if verbose and state.smooth_counter <= 5:
            print(f"  [t={t}] active={active_count} strengths: " +
                  " ".join(f"{c}={state.smoothed[c]:+.4f}" for c in CURRENCIES[:4]))

        holdout_indices = np.random.choice(n_pairs, min(n_holdout, n_pairs), replace=False)
        current_ts = timestamps[t]

        for h in horizons:
            future_idx = t + h
            if future_idx >= T:
                continue
            future_returns = return_matrix[future_idx]
            for j in range(n_pairs):
                if abs(latest[j]) < 1e-12:
                    continue
                sym = pair_labels[j]
                base, quote = pair_labels[j][:3], pair_labels[j][3:6]
                if base not in CURRENCIES or quote not in CURRENCIES:
                    continue
                predicted_spread = state.smoothed.get(base, 0.0) - state.smoothed.get(quote, 0.0)
                actual_return = future_returns[j]
                all_preds_by_horizon[h].append(predicted_spread)
                all_actuals_by_horizon[h].append(actual_return)

            for j in holdout_indices:
                if abs(latest[j]) < 1e-12:
                    continue
                sym = pair_labels[j]
                base, quote = sym[:3], sym[3:6]
                if base not in CURRENCIES or quote not in CURRENCIES:
                    continue
                predicted_spread = state.smoothed.get(base, 0.0) - state.smoothed.get(quote, 0.0)
                actual_return = future_returns[j]
                predictions_by_horizon[h].append(predicted_spread)
                actuals_by_horizon[h].append(actual_return)

    results = {}
    for h in horizons:
        preds = np.array(predictions_by_horizon[h])
        actuals = np.array(actuals_by_horizon[h])
        all_preds = np.array(all_preds_by_horizon[h])
        all_actuals = np.array(all_actuals_by_horizon[h])
        if len(preds) < 10:
            results[f"{h}_bar"] = {"error": f"Only {len(preds)} holdout samples"}
            continue

        skill = predictive_skill(actuals, preds)
        all_skill = predictive_skill(all_actuals, all_preds)
        hit = hit_rate(all_actuals, all_preds, top_pct=0.20)
        ic = information_coefficient(all_actuals, all_preds)

        results[f"{h}_bar"] = {
            "n_holdout": len(preds),
            "n_all_pairs": len(all_preds),
            "holdout_mse_skill": skill["mse_skill"],
            "holdout_dir_acc": skill["direction_accuracy"],
            "holdout_model_mse": skill["model_mse"],
            "holdout_naive_mse": skill["naive_mse"],
            "holdout_mae_skill": skill["mae_skill"],
            "all_mse_skill": all_skill["mse_skill"],
            "all_dir_acc": all_skill["direction_accuracy"],
            "all_mae_skill": all_skill["mae_skill"],
            "top_mean_return": hit["top_mean_return"],
            "bottom_mean_return": hit["bottom_mean_return"],
            "spread_return": hit["spread"],
            "top_hit_rate": hit["top_hit_rate"],
            "bottom_hit_rate": hit["bottom_hit_rate"],
            "information_coefficient": ic,
        }

    results["_meta"] = {
        "n_walk_forward_steps": state.smooth_counter,
        "n_skipped_low_pairs": n_skipped_low_pairs,
        "n_skipped_solve_error": n_skipped_solve_error,
        "lam": lam,
        "window": window,
        "smoothing_alpha": smoothing_alpha,
        "prior_shrink": prior_shrink,
        "n_holdout": n_holdout,
        "n_pairs": n_pairs,
        "total_T": T,
    }
    return results


def run_derivative_validation(
    return_matrix: np.ndarray,
    timestamps: np.ndarray,
    pair_labels: list[str],
    design_matrix: np.ndarray,
    horizons: list[int] = None,
    lam: float = 0.05,
    smoothing_alpha: float = 0.1,
    prior_shrink: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Test if currency strength derivatives predict future returns better than levels.

    Predictors tested:
      - level(t)         = current smoothed strength spread
      - change(t)        = level(t) - level(t-1)  (velocity)
      - accel(t)         = change(t) - change(t-1)  (acceleration)
      - force(t)         = change(t) + accel(t) * persistence_factor
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]
    A = design_matrix
    n_pairs = A.shape[0]
    T = return_matrix.shape[0]
    min_viable = 48 + max(horizons) + 2
    if T < min_viable:
        return {"error": f"Need at least {min_viable} bars, got {T}"}

    state = WLSState()
    strength_deque = []
    predictions = {method: {h: [] for h in horizons} for method in
                   ["level", "change", "accel", "force"]}
    actuals = {h: [] for h in horizons}

    for t in range(48, T - max(horizons)):
        latest = return_matrix[t]
        active_mask = np.abs(latest) > 1e-12
        active_count = int(np.sum(active_mask))
        if active_count < 8:
            continue

        weight_vec = np.where(active_mask, 1.0, 1e-6)
        prior_vec = np.array([state.prior.get(c, 0.0) for c in CURRENCIES])
        try:
            strengths = solve_wls(A, latest, prior_vec, lam=lam, weight_vec=weight_vec)
        except np.linalg.LinAlgError:
            continue

        strength_dict = {c: float(strengths[i]) for i, c in enumerate(CURRENCIES)}
        state.strengths = strength_dict
        shrunk = {c: (1 - prior_shrink) * strength_dict[c] for c in CURRENCIES}
        state.prior = shrunk
        if state.smooth_counter == 0:
            state.smoothed = dict(shrunk)
        else:
            for c in CURRENCIES:
                state.smoothed[c] = smoothing_alpha * shrunk[c] + (1 - smoothing_alpha) * state.smoothed[c]
        state.smooth_counter += 1
        strength_deque.append(dict(state.smoothed))
        if len(strength_deque) > 3:
            strength_deque.pop(0)

        if len(strength_deque) < 3:
            continue

        prev = strength_deque[-2]
        prev2 = strength_deque[-3]
        current = strength_deque[-1]

        for h in horizons:
            future_idx = t + h
            if future_idx >= T:
                continue
            future_returns = return_matrix[future_idx]

            for j in range(n_pairs):
                if abs(latest[j]) < 1e-12:
                    continue
                sym = pair_labels[j]
                base, quote = sym[:3], sym[3:6]
                if base not in CURRENCIES or quote not in CURRENCIES:
                    continue

                level_base = current.get(base, 0.0)
                level_quote = current.get(quote, 0.0)
                prev_base = prev.get(base, 0.0)
                prev_quote = prev.get(quote, 0.0)
                prev2_base = prev2.get(base, 0.0)
                prev2_quote = prev2.get(quote, 0.0)

                spread_level = level_base - level_quote
                spread_prev = prev_base - prev_quote
                spread_prev2 = prev2_base - prev2_quote

                spread_change = spread_level - spread_prev
                spread_accel = spread_change - (spread_prev - spread_prev2)

                direction = 1 if spread_change > 0 else -1
                persistence = 0.0
                if spread_change * (spread_prev - spread_prev2) > 0:
                    c = 0
                    for k in range(len(strength_deque) - 1):
                        if (strength_deque[-1 - k].get(base, 0) - strength_deque[-1 - k].get(quote, 0)) * direction > 0:
                            c += 1
                        else:
                            break
                    persistence = min(c / 10.0, 1.0)
                spread_force = spread_change + spread_accel * persistence

                future_ret = future_returns[j]

                for method, val in [
                    ("level", spread_level),
                    ("change", spread_change),
                    ("accel", spread_accel),
                    ("force", spread_force),
                ]:
                    predictions[method][h].append(val)
                actuals[h].append(future_ret)

    results = {}
    for method in ["level", "change", "accel", "force"]:
        for h in horizons:
            preds = np.array(predictions[method][h])
            act = np.array(actuals[h])
            if len(preds) < 20:
                continue
            skill = predictive_skill(act, preds)
            ic = information_coefficient(act, preds)
            hit = hit_rate(act, preds, top_pct=0.20)
            if f"derivative_{method}" not in results:
                results[f"derivative_{method}"] = {}
            results[f"derivative_{method}"][f"{h}_bar"] = {
                "n": len(preds),
                "mse_skill": skill["mse_skill"],
                "dir_acc": skill["direction_accuracy"],
                "ic": ic,
                "spread_return": hit["spread"],
            }

    results["_meta"] = {
        "lam": lam,
        "smoothing_alpha": smoothing_alpha,
        "prior_shrink": prior_shrink,
        "n_steps": state.smooth_counter,
        "total_T": T,
    }
    return results


PARAM_GRID = {
    "lam": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    "smoothing_alpha": [0.05, 0.1, 0.2, 0.5, 1.0],
    "prior_shrink": [0.0, 0.3, 0.5, 0.7, 0.9],
}


def grid_search(
    return_matrix: np.ndarray,
    timestamps: np.ndarray,
    pair_labels: list[str],
    design_matrix: np.ndarray,
    horizons: list[int] = None,
    window: int = 24,
    param_grid: dict = None,
) -> dict:
    if param_grid is None:
        param_grid = PARAM_GRID
    if horizons is None:
        horizons = [1, 3, 6, 12]
    eval_horizon = horizons[0]
    results = []
    total = (
        len(param_grid["lam"])
        * len(param_grid["smoothing_alpha"])
        * len(param_grid["prior_shrink"])
    )
    count = 0
    best_skill = -1e9
    best_params = None
    for lam in param_grid["lam"]:
        for alpha in param_grid["smoothing_alpha"]:
            for shrink in param_grid["prior_shrink"]:
                count += 1
                t0 = time.time()
                result = run_walk_forward(
                    return_matrix, timestamps, pair_labels, design_matrix,
                    horizons=horizons, lam=lam, window=window,
                    smoothing_alpha=alpha, prior_shrink=shrink,
                )
                elapsed = time.time() - t0
                entry = {
                    "lam": lam,
                    "smoothing_alpha": alpha,
                    "prior_shrink": shrink,
                    "elapsed_s": round(elapsed, 1),
                }
                for h in horizons:
                    key = f"{h}_bar"
                    if key in result:
                        entry[f"{h}_holdout_skill"] = result[key].get("holdout_mse_skill", None)
                        entry[f"{h}_all_skill"] = result[key].get("all_mse_skill", None)
                        entry[f"{h}_dir_acc"] = result[key].get("holdout_dir_acc", None)
                        entry[f"{h}_ic"] = result[key].get("information_coefficient", None)
                        entry[f"{h}_n"] = result[key].get("n_holdout", 0)
                results.append(entry)
                skill = entry.get(f"{eval_horizon}_holdout_skill", -999)
                if skill is not None and skill > best_skill:
                    best_skill = skill
                    best_params = (lam, alpha, shrink)
                print(f"  [{count}/{total}] lam={lam:.3f} α={alpha:.1f} s={shrink:.1f} "
                      f"→ {eval_horizon}_bar holdout_skill={skill if skill is not None else 'N/A':.4f} "
                      f"all_skill={entry.get(f'{eval_horizon}_all_skill', 'N/A'):.4f} "
                      f"IC={entry.get(f'{eval_horizon}_ic', 'N/A'):.4f} "
                      f"({elapsed:.1f}s)")
    return {
        "results": results,
        "best_params": {
            "lam": best_params[0],
            "smoothing_alpha": best_params[1],
            "prior_shrink": best_params[2],
        },
        "best_skill": best_skill,
    }
