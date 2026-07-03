"""
DPL-8: Directional Survivorship Tournament
All candidate directional layers compete head-to-head.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS, compute_gradient
import warnings
warnings.filterwarnings("ignore")

CANDIDATES = ["residual_sign", "memory_distance", "energy_gradient",
              "state_transition", "regime_interaction", "information_pressure"]


def evaluate_candidate(d: DPLData, candidate: str, hi: int) -> dict:
    """Evaluate one candidate's directional accuracy for horizon hi."""
    es = d.es.copy()
    fut = d.fut_ret[:, hi]
    md = d.memory_density.copy()
    resid = d.residuals.get("xgboost", np.zeros_like(es))
    states = d.states.copy().astype(int)
    n = min(len(es), len(fut), len(md), len(resid), len(states))

    es, fut, md, resid, states = es[:n], fut[:n], md[:n], resid[:n], states[:n]

    valid = ~np.isnan(es) & ~np.isnan(fut) & ~np.isnan(md) & ~np.isnan(resid)
    if np.sum(valid) < 30:
        return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}

    es_v, fut_v, md_v, resid_v, states_v = es[valid], fut[valid], md[valid], resid[valid], states[valid]
    dir_v = (fut_v > 0).astype(float)
    n_valid = len(es_v)

    # ES baseline directional accuracy
    es_thr = np.nanpercentile(es_v, 80)
    high_es = es_v > es_thr
    es_baseline_dir = float(np.mean(dir_v[high_es])) if np.sum(high_es) > 5 else 0.5

    if candidate == "residual_sign":
        # Predict up if residual > 0 (conditioned on high ES)
        mask = high_es & ~np.isnan(resid_v)
        if np.sum(mask) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        pred = (resid_v[mask] > 0).astype(float)
        actual = dir_v[mask]
        acc = float(np.mean(pred == actual))

    elif candidate == "memory_distance":
        # Distance from memory center
        window = 252
        preds = np.full_like(dir_v, np.nan)
        for i in range(window, len(es_v)):
            chunk_px = np.full(window, np.nan)
            chunk_md = np.full(window, np.nan)
            # Simplified: price above/below rolling median
            if i < window:
                continue
            roll_px = es_v[i - window:i]  # proxy, not real price
            roll_md = md_v[i - window:i]
            above = np.mean(roll_md[roll_px > np.median(roll_px)]) if np.sum(roll_px > np.median(roll_px)) > 0 else 0
            below = np.mean(roll_md[roll_px <= np.median(roll_px)]) if np.sum(roll_px <= np.median(roll_px)) > 0 else 0
            memory_skew = above - below
            preds[i] = 1.0 if memory_skew > 0 else 0.0
        m = ~np.isnan(preds)
        if np.sum(m) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        acc = float(np.mean(preds[m] == dir_v[m]))

    elif candidate == "energy_gradient":
        g1 = compute_gradient(es_v, window=5)
        mask = high_es & ~np.isnan(g1)
        if np.sum(mask) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        pred = (g1[mask] > 0).astype(float)
        actual = dir_v[mask]
        acc = float(np.mean(pred == actual))

    elif candidate == "state_transition":
        preds = np.full_like(dir_v, np.nan)
        for i in range(1, len(states_v)):
            si, sj = int(states_v[i - 1]), int(states_v[i])
            # Predict based on historical up-prob for this transition
            pass  # Simplified: random for now
        acc = 0.5

    elif candidate == "regime_interaction":
        n_states = int(np.max(states_v) + 1) if np.max(states_v) >= 0 else 1
        state_up_prob = {}
        for si in range(n_states):
            m = states_v == si
            if np.sum(m) > 5:
                state_up_prob[si] = float(np.mean(dir_v[m] > 0))
        preds = np.array([state_up_prob.get(int(s), 0.5) > 0.5 for s in states_v])
        preds = preds.astype(float)
        mask = high_es
        if np.sum(mask) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        acc = float(np.mean(preds[mask] == dir_v[mask]))

    elif candidate == "information_pressure":
        # Simplified: use state_mutation_rate as proxy for information flow
        smr = d.state_mutation.copy()[:n][valid]
        if np.sum(~np.isnan(smr)) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        thr = np.nanpercentile(smr, 80)
        high_flow = smr > thr
        mask = high_es & ~np.isnan(smr)
        if np.sum(mask) < 10:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}
        pred = (smr[mask] > thr).astype(float)
        actual = dir_v[mask]
        acc = float(np.mean(pred == actual))
    else:
        return {"n": 0, "accuracy": 0.5, "info_gain": 0.0}

    # Information gain over ES baseline
    info_gain = acc - es_baseline_dir

    return {"n": int(np.sum(mask)) if "mask" in dir() else n_valid,
            "accuracy": round(acc, 4),
            "info_gain": round(info_gain, 4),
            "es_baseline": round(es_baseline_dir, 4)}


# Run tournament
scores = {c: {"accuracies": [], "info_gains": []} for c in CANDIDATES}
for sym in SYMBOLS:
    d = DPLData(sym)
    for hi, h in enumerate([5, 20, 50]):
        for c in CANDIDATES:
            result = evaluate_candidate(d, c, hi)
            if result["n"] >= 10:
                scores[c]["accuracies"].append(result["accuracy"])
                scores[c]["info_gains"].append(result["info_gain"])

# Final scoring
final = {}
for c in CANDIDATES:
    accs = scores[c]["accuracies"]
    gains = scores[c]["info_gains"]
    if not accs:
        continue
    mean_acc = float(np.mean(accs))
    mean_gain = float(np.mean(gains))
    cross_time = float(np.std(accs))  # lower = more robust
    cross_asset = float(np.std([scores[c]["accuracies"][i::3] for i in range(min(3, len(accs)))])) if len(accs) >= 3 else 1.0

    # Score: 40% accuracy + 20% info gain + 20% cross-time (-std) + 10% cross-asset (-std) + 10% simplicity
    simplicity = {"residual_sign": 1.0, "memory_distance": 0.7, "energy_gradient": 0.8,
                  "state_transition": 0.5, "regime_interaction": 0.6, "information_pressure": 0.4}
    sim = simplicity.get(c, 0.5)

    score = (0.40 * mean_acc + 0.20 * mean_gain +
             0.20 * (1.0 - min(cross_time, 1.0)) +
             0.10 * (1.0 - min(cross_asset, 1.0)) +
             0.10 * sim)
    final[c] = {"directional_accuracy": round(mean_acc, 4),
                "info_gain": round(mean_gain, 4),
                "cross_time_robustness": round(1.0 - min(cross_time, 1.0), 4),
                "cross_asset_robustness": round(1.0 - min(cross_asset, 1.0), 4),
                "simplicity": sim,
                "composite_score": round(score, 4),
                "n_evaluations": len(accs)}

# Rank
ranked = sorted(final.items(), key=lambda x: x[1]["composite_score"], reverse=True)

output = {"experiment": "DPL-8", "title": "Directional Survivorship Tournament",
           "rankings": [{"rank": i + 1, "candidate": c, **s} for i, (c, s) in enumerate(ranked)],
           "winner": ranked[0][0] if ranked else None}
out_path = Path(__file__).parent / "reports" / "dpl8_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-8 complete -> {out_path}")
