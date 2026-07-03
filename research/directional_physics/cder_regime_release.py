"""
CDER: Context-Dependent Energy Release
Investigates whether identical ES states behave differently because they belong to different hidden regimes.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import entropy as sp_entropy, pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------
# Helper: conditional probability up given mask
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Task 1: Regime Definition Analysis
# ------------------------------------------------------------
def analyze_regime_definitions(d: DPLData, sym: str) -> dict:
    states = d.states.copy().astype(int)
    es = d.es.copy()
    fut_ret = d.fut_ret
    n = len(states)

    valid_mask = ~np.isnan(es) & (states >= 0)
    s = states[valid_mask]

    unique_states = sorted(np.unique(s))
    n_states = len(unique_states)
    state_counts = {int(k): int(np.sum(s == k)) for k in unique_states}
    state_pct = {int(k): round(float(np.sum(s == k)) / max(1, len(s)) * 100, 2) for k in unique_states}

    # Transition frequency
    transitions = s[1:] - s[:-1]
    n_transitions = int(np.sum(transitions != 0))
    transition_rate = round(n_transitions / max(1, len(s)), 4)

    # Regime persistence: avg consecutive same-state run length
    runs = []
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    avg_persistence = round(float(np.mean(runs)), 2)
    max_persistence = int(np.max(runs))
    persistence_by_state = {}
    for k in unique_states:
        k_mask = s == k
        k_runs = []
        kr = 0
        for i in range(len(s)):
            if s[i] == k:
                kr += 1
            else:
                if kr > 0:
                    k_runs.append(kr)
                kr = 0
        if kr > 0:
            k_runs.append(kr)
        persistence_by_state[f"S{int(k)}"] = {
            "avg_stay": round(float(np.mean(k_runs)), 2) if k_runs else 0,
            "max_stay": int(np.max(k_runs)) if k_runs else 0,
        }

    # Regime entropy (diversity of states over time)
    state_dist = np.bincount(s.astype(int), minlength=max(unique_states) + 1) / len(s)
    regime_entropy = float(sp_entropy(state_dist + 1e-12))
    max_entropy = float(np.log(n_states)) if n_states > 1 else 1
    entropy_ratio = round(regime_entropy / max_entropy, 4) if max_entropy > 0 else 0

    # Directional behavior per regime
    dir_by_regime = {}
    for ki, k in enumerate(unique_states):
        mask = (states == k) & ~np.isnan(es)
        regime_info = {"n": int(np.sum(mask)), "state_label": f"S{int(k)}"}
        for hi, h in enumerate(HORIZONS):
            fwd = fut_ret[:, hi]
            es_high_mask = mask & (es > np.nanpercentile(es[mask], 80)) if np.sum(mask) > 10 else mask
            regime_info[HORIZON_LABELS[h]] = {
                "p_up_all": round(p_up(fwd, mask), 4),
                "p_up_es_high": round(p_up(fwd, es_high_mask), 4),
                "mean_ret_es_high": round(mean_ret(fwd, es_high_mask), 6),
            }
        dir_by_regime[f"S{int(k)}"] = regime_info

    # Test: do identical ES values belong to opposite directional regimes?
    # For each ES percentile bin, check if different regimes disagree
    es_bins = np.linspace(0, 100, 11)  # deciles
    es_pct_vals = np.nanpercentile(es, es_bins)
    es_disagreement = []
    for bi in range(len(es_bins) - 1):
        lo, hi = es_pct_vals[bi], es_pct_vals[bi + 1]
        bin_mask = (es >= lo) & (es < hi) & ~np.isnan(es) & (states >= 0)
        if np.sum(bin_mask) < 20:
            continue
        bin_states = states[bin_mask]
        for hi_idx, h in enumerate(HORIZONS):
            fwd = fut_ret[:, hi_idx]
            regime_p_up = {}
            for k in unique_states:
                rm = bin_mask & (states == k)
                regime_p_up[f"S{int(k)}"] = round(p_up(fwd, rm), 4)
            p_ups = list(regime_p_up.values())
            if len(p_ups) >= 2 and max(p_ups) - min(p_ups) > 0.2:
                es_disagreement.append({
                    "es_bin": f"P{int(es_bins[bi])}-P{int(es_bins[bi+1])}",
                    "horizon": HORIZON_LABELS[h],
                    "regime_p_up": regime_p_up,
                    "spread": round(max(p_ups) - min(p_ups), 4),
                })

    # Regime change probability stats
    regime_cp = d.regime_change.copy()
    regime_cp_stats = {
        "mean": round(float(np.nanmean(regime_cp)), 4),
        "std": round(float(np.nanstd(regime_cp)), 4),
        "p95": round(float(np.nanpercentile(regime_cp, 95)), 4),
    }

    return {
        "n": int(n),
        "n_states": n_states,
        "unique_states": [int(x) for x in unique_states],
        "state_distribution_pct": state_pct,
        "transition_rate": transition_rate,
        "n_transitions": n_transitions,
        "avg_persistence_bars": avg_persistence,
        "max_persistence_bars": max_persistence,
        "persistence_by_state": persistence_by_state,
        "regime_entropy": round(regime_entropy, 4),
        "max_possible_entropy": round(max_entropy, 4),
        "entropy_ratio": entropy_ratio,
        "directional_behavior_by_regime": dir_by_regime,
        "es_disagreement_across_regimes": es_disagreement[:10],
        "regime_change_probability_stats": regime_cp_stats,
    }


# ------------------------------------------------------------
# Task 2: Regime Transition Directionality
# ------------------------------------------------------------
def analyze_regime_transitions(d: DPLData, sym: str) -> dict:
    states = d.states.copy().astype(int)
    es = d.es.copy()
    fut_ret = d.fut_ret
    valid = (states >= 0) & ~np.isnan(es)
    s = states[valid]
    es_v = es[valid]

    n_states = int(np.max(s) + 1) if np.max(s) >= 0 else 1

    result = {}
    for hi, h in enumerate(HORIZONS):
        fwd = fut_ret[:, hi]
        f = fwd[valid]

        # Transition matrices
        trans_count = np.zeros((n_states, n_states), dtype=np.int64)
        trans_up = np.zeros((n_states, n_states), dtype=np.int64)
        trans_up_es_high = np.zeros((n_states, n_states), dtype=np.int64)
        trans_count_es_high = np.zeros((n_states, n_states), dtype=np.int64)
        trans_mean_ret = np.zeros((n_states, n_states), dtype=np.float64)

        for i in range(1, len(s)):
            si, sj = s[i - 1], s[i]
            if si < n_states and sj < n_states:
                trans_count[si, sj] += 1
                trans_mean_ret[si, sj] += f[i]
                if f[i] > 0:
                    trans_up[si, sj] += 1
                # ES high condition: check if ES was high *before* transition
                es_high = es_v[i - 1] > np.nanpercentile(es_v, 80) if np.sum(~np.isnan(es_v)) > 10 else True
                if es_high and not np.isnan(es_v[i - 1]):
                    trans_count_es_high[si, sj] += 1
                    if f[i] > 0:
                        trans_up_es_high[si, sj] += 1

        # Normalize counts
        trans_prob = np.where(trans_count > 0, trans_count / np.maximum(trans_count.sum(axis=1, keepdims=True), 1), 0.0)
        up_prob = np.where(trans_count > 0, trans_up / np.maximum(trans_count, 1), 0.5)
        up_prob_es_high = np.where(trans_count_es_high > 0, trans_up_es_high / np.maximum(trans_count_es_high, 1), 0.5)
        mean_ret_norm = np.where(trans_count > 0, trans_mean_ret / np.maximum(trans_count, 1), 0.0)

        # Find directional flips
        transitions_list = []
        for si in range(n_states):
            for sj in range(n_states):
                if trans_count[si, sj] >= 5:
                    transitions_list.append({
                        "from": f"S{si}",
                        "to": f"S{sj}",
                        "n": int(trans_count[si, sj]),
                        "prob": round(float(trans_prob[si, sj]), 4),
                        "p_up": round(float(up_prob[si, sj]), 4),
                        "p_up_es_high": round(float(up_prob_es_high[si, sj]), 4),
                        "mean_ret": round(float(mean_ret_norm[si, sj]), 6),
                    })

        # Identify flips: transitions where p_up switches from >0.55 to <0.45
        flip_candidates = []
        for si in range(n_states):
            for sj in range(n_states):
                for sk in range(n_states):
                    if sj == sk:
                        continue
                    if trans_count[si, sj] >= 5 and trans_count[si, sk] >= 5:
                        p1 = up_prob[si, sj]
                        p2 = up_prob[si, sk]
                        if abs(p1 - p2) > 0.2:
                            flip_candidates.append({
                                "from": f"S{si}",
                                "to_up": f"S{sj}",
                                "to_down": f"S{sk}",
                                "p_up_via_S{sj}": round(float(p1), 4),
                                "p_up_via_S{sk}": round(float(p2), 4),
                                "spread": round(float(abs(p1 - p2)), 4),
                            })

        # Rank transitions by directional bias
        ranked = sorted(transitions_list, key=lambda x: abs(x["p_up"] - 0.5), reverse=True)

        result[HORIZON_LABELS[h]] = {
            "n_states": n_states,
            "n_valid": int(len(s)),
            "transition_count_matrix": trans_count.tolist(),
            "transition_prob_matrix": np.round(trans_prob, 4).tolist(),
            "up_prob_matrix": np.round(up_prob, 4).tolist(),
            "up_prob_es_high_matrix": np.round(up_prob_es_high, 4).tolist(),
            "mean_return_matrix": np.round(mean_ret_norm, 6).tolist(),
            "all_transitions": transitions_list,
            "top_directional_transitions": ranked[:5],
            "flip_candidates": flip_candidates[:5],
            "max_p_up_spread": round(float(np.max(up_prob) - np.min(up_prob[up_prob != 0.5])), 4) if np.any(up_prob != 0.5) else 0,
        }

    return result


# ------------------------------------------------------------
# Task 3: Regime Classification Methods
# ------------------------------------------------------------
def analyze_regime_classification(d: DPLData, sym: str) -> dict:
    states = d.states.copy().astype(int)
    es = d.es.copy()
    n = len(states)

    valid = (states >= 0) & ~np.isnan(es)
    s = states[valid]
    n_states = int(np.max(s) + 1) if np.max(s) >= 0 else 1

    # Feature candidates for classifying regime
    features = {
        "energy_storage": es.copy(),
        "memory_density": d.memory_density.copy(),
        "adaptive_time": d.adaptive_time.copy(),
        "state_mutation": d.state_mutation.copy(),
        "regime_change_prob": d.regime_change.copy(),
    }
    # Vol metrics
    for k, v in d.vol_metrics.items():
        features[f"vol_{k}"] = v.copy()

    # Feature-regime mutual information (using ANOVA F-test proxy: correlation with state)
    feature_importance = {}
    for fname, fvals in features.items():
        fv = fvals[valid]
        if np.all(np.isnan(fv)) or np.nanstd(fv) < 1e-12:
            continue
        # Correlation with state
        try:
            c, _ = pearsonr(fv[~np.isnan(fv)], s[~np.isnan(fv)].astype(float))
            feature_importance[fname] = round(abs(float(c)), 4)
        except Exception:
            feature_importance[fname] = 0.0

    # Within-regime feature stats
    regime_feature_profiles = {}
    for k in range(n_states):
        mask = valid & (states == k)
        profile = {"n": int(np.sum(mask))}
        for fname in ["energy_storage", "memory_density", "adaptive_time", "state_mutation"]:
            fv = features[fname][mask]
            profile[f"{fname}_mean"] = round(float(np.nanmean(fv)), 4)
            profile[f"{fname}_std"] = round(float(np.nanstd(fv)), 4)
        for vk in ["realized_vol", "entropy", "atr"]:
            if vk in d.vol_metrics:
                fv = d.vol_metrics[vk][mask]
                profile[f"vol_{vk}_mean"] = round(float(np.nanmean(fv)), 4)
        regime_feature_profiles[f"S{int(k)}"] = profile

    # Simple decision tree proxy: what single variable best separates regimes?
    best_separator = max(feature_importance, key=feature_importance.get) if feature_importance else None
    best_sep_score = feature_importance.get(best_separator, 0)

    # Are regimes primarily vol-based?
    vol_features = [k for k in feature_importance if k.startswith("vol_")]
    nonvol_features = [k for k in feature_importance if not k.startswith("vol_")]
    vol_importance = np.mean([feature_importance[k] for k in vol_features]) if vol_features else 0
    nonvol_importance = np.mean([feature_importance[k] for k in nonvol_features]) if nonvol_features else 0

    return {
        "n_states": n_states,
        "n_valid": int(np.sum(valid)),
        "feature_regime_correlation": {k: v for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])},
        "best_regime_predictor": best_separator,
        "best_predictor_score": best_sep_score,
        "vol_features_avg_importance": round(float(vol_importance), 4),
        "nonvol_features_avg_importance": round(float(nonvol_importance), 4),
        "regime_feature_profiles": regime_feature_profiles,
    }


# ------------------------------------------------------------
# Task 4: Cross-Asset Regime Analysis
# ------------------------------------------------------------
def analyze_cross_asset(all_data: dict) -> dict:
    """Check if regimes align across assets."""
    # Regime distribution per asset
    regime_distributions = {}
    for sym, d in all_data.items():
        s = d.states.copy().astype(int)
        valid = s >= 0
        s_v = s[valid]
        unique = sorted(np.unique(s_v))
        dist = {f"S{int(k)}": int(np.sum(s_v == k)) for k in unique}
        total = len(s_v)
        regime_distributions[sym] = {
            "unique_states": [int(x) for x in unique],
            "n_states": len(unique),
            "counts": dist,
            "pcts": {k: round(v / total * 100, 2) for k, v in dist.items()},
        }

    # State space overlap
    all_state_sets = {sym: set(info["unique_states"]) for sym, info in regime_distributions.items()}
    common_states = set.intersection(*all_state_sets.values()) if all_state_sets else set()
    max_states = max(info["n_states"] for info in regime_distributions.values())

    # Are states synchronous? (do all assets change regime at same time?)
    # Find regime change timestamps per asset
    min_len = min(len(d.states) for d in all_data.values())
    regime_sync = {}
    for offset in range(1, 11):
        agreements = []
        for i in range(offset, min_len):
            state_slice = []
            for d in all_data.values():
                s_arr = d.states.copy().astype(int)
                if i < len(s_arr) and i - offset >= 0 and s_arr[i] >= 0 and s_arr[i - offset] >= 0:
                    state_slice.append((s_arr[i - offset], s_arr[i]))
            if len(state_slice) == len(all_data):
                changes = [1 if a != b else 0 for a, b in state_slice]
                agreements.append(1 if len(set(changes)) == 1 else 0)
        regime_sync[f"sync_{offset}bar"] = {
            "n_samples": len(agreements),
            "sync_rate": round(float(np.mean(agreements)), 4) if agreements else 0,
        }

    # Cross-asset regime correlation (truncate to min length)
    n_assets = len(all_data)
    assets = list(all_data.keys())
    cross_corr = {}
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            si = all_data[assets[i]].states.copy().astype(float)
            sj = all_data[assets[j]].states.copy().astype(float)
            min_l = min(len(si), len(sj))
            si, sj = si[:min_l], sj[:min_l]
            valid = (si >= 0) & (sj >= 0)
            if np.sum(valid) < 10:
                continue
            c = float(np.corrcoef(si[valid], sj[valid])[0, 1])
            cross_corr[f"{assets[i]}_{assets[j]}"] = round(c, 4)

    # Global regime: cluster all states together
    # Build feature matrix: state at each time across assets (truncate to min len)
    stacked = []
    for d in all_data.values():
        s_arr = d.states.copy().astype(int)
        stacked.append(s_arr[:min_len])
    state_matrix = np.column_stack(stacked)
    # Unique global regime vectors
    global_regimes = {}
    for row in state_matrix:
        key = ",".join(str(int(x)) for x in row)
        global_regimes[key] = global_regimes.get(key, 0) + 1
    total_rows = len(state_matrix)
    global_regime_pcts = {k: round(v / total_rows * 100, 2) for k, v in
                          sorted(global_regimes.items(), key=lambda x: -x[1])}

    return {
        "per_asset_regime_distribution": regime_distributions,
        "common_state_set": sorted(common_states),
        "max_states_any_asset": max_states,
        "cross_asset_state_correlation": cross_corr,
        "regime_change_sync": regime_sync,
        "global_regime_states": {
            "n_unique_global_states": len(global_regimes),
            "top_global_regimes": dict(list(global_regime_pcts.items())[:10]),
            "n_total_samples": int(total_rows),
        },
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    all_data = {}
    per_symbol = {}

    for sym in SYMBOLS:
        print(f"Loading {sym}...")
        d = DPLData(sym)
        all_data[sym] = d

        print(f"  Regime Definition Analysis...")
        r1 = analyze_regime_definitions(d, sym)

        print(f"  Regime Transition Directionality...")
        r2 = analyze_regime_transitions(d, sym)

        print(f"  Regime Classification Methods...")
        r3 = analyze_regime_classification(d, sym)

        per_symbol[sym] = {
            "regime_definition": r1,
            "regime_transitions": r2,
            "regime_classification": r3,
        }

    print("Cross-Asset Regime Analysis...")
    cross_asset = analyze_cross_asset(all_data)

    output = {
        "experiment": "CDER",
        "title": "Context-Dependent Energy Release: Regime Analysis",
        "per_symbol": per_symbol,
        "cross_asset": cross_asset,
        "summary": _build_summary(per_symbol, cross_asset, all_data),
    }

    out_path = Path(__file__).parent / "reports" / "cder_regime_release.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nCDER complete -> {out_path}")


def _build_summary(per_symbol: dict, cross_asset: dict, all_data: dict) -> dict:
    """Extract key cross-cutting findings."""
    # Regime continuity: are states always [0,1,2] or sometimes more?
    state_info = {}
    for sym, res in per_symbol.items():
        rd = res["regime_definition"]
        state_info[sym] = {
            "n_states": rd["n_states"],
            "unique_states": rd["unique_states"],
            "entropy_ratio": rd["entropy_ratio"],
            "avg_persistence": rd["avg_persistence_bars"],
            "transition_rate": rd["transition_rate"],
            "best_predictor": res["regime_classification"]["best_regime_predictor"],
        }

    # Directional disagreement: do regimes flip sign?
    disagreement_count = 0
    total_bins = 0
    for sym, res in per_symbol.items():
        disagreements = res["regime_definition"]["es_disagreement_across_regimes"]
        disagreement_count += len(disagreements)
        total_bins += 10  # approx

    # Best predictor across all assets
    all_predictors = {}
    for sym, res in per_symbol.items():
        for fname, score in res["regime_classification"]["feature_regime_correlation"].items():
            all_predictors.setdefault(fname, []).append(score)
    avg_predictor = {k: round(float(np.mean(v)), 4) for k, v in all_predictors.items()}
    top_predictor = max(avg_predictor, key=avg_predictor.get) if avg_predictor else None

    # Global alignment
    ca = cross_asset
    avg_sync = np.mean([v["sync_rate"] for k, v in ca.get("regime_change_sync", {}).items()]) if ca.get("regime_change_sync") else 0

    return {
        "n_symbols": len(per_symbol),
        "regime_structure": "discrete (3 tiers: 0=low, 1=medium, 2=high combined density)" if all(
            i["n_states"] <= 3 for i in state_info.values()) else "variable",
        "regime_continuity": "discrete (tertile-based)" if all(
            i["n_states"] == 3 for i in state_info.values()) else "mixed",
        "per_symbol_state_info": state_info,
        "top_feature_predictors_overall": dict(sorted(avg_predictor.items(), key=lambda x: -x[1])[:8]),
        "best_overall_regime_predictor": top_predictor,
        "average_cross_asset_sync_rate": round(float(avg_sync), 4),
        "any_sign_flip_detected": disagreement_count > 0,
        "es_disagreement_bins_found": int(disagreement_count),
    }


if __name__ == "__main__":
    main()
