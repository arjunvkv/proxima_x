"""
DPL-5: State Transition Directionality
Build transition matrices: which state transitions lead to up/down moves?
"""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

results = {}
for sym in SYMBOLS:
    d = DPLData(sym)
    states = d.states.copy().astype(int)
    fut_ret = d.fut_ret
    es = d.es.copy()

    n_unique = len(np.unique(states))
    sym_res = {}

    for hi, h in enumerate(HORIZONS):
        fwd = fut_ret[:, hi]
        valid = ~np.isnan(fwd) & (states >= 0)
        if np.sum(valid) < 50:
            continue

        s = states[valid]
        f = fwd[valid]

        # Transition matrix: P(S_{t+1} | S_t)
        n_states = int(np.max(s) + 1) if np.max(s) >= 0 else 1
        trans_count = np.zeros((n_states, n_states), dtype=np.int64)
        trans_up = np.zeros((n_states, n_states), dtype=np.int64)
        trans_down = np.zeros((n_states, n_states), dtype=np.int64)
        trans_mean_ret = np.zeros((n_states, n_states), dtype=np.float64)
        trans_n = np.zeros((n_states, n_states), dtype=np.int64)

        for i in range(1, len(s)):
            si, sj = s[i - 1], s[i]
            if si < n_states and sj < n_states:
                trans_count[si, sj] += 1
                trans_n[si, sj] += 1
                if f[i] > 0:
                    trans_up[si, sj] += 1
                else:
                    trans_down[si, sj] += 1
                trans_mean_ret[si, sj] += f[i]

        # Normalize
        for i in range(n_states):
            row_sum = trans_count[i].sum()
            if row_sum > 0:
                trans_count[i] = trans_count[i] / row_sum
            for j in range(n_states):
                if trans_n[i, j] > 0:
                    trans_mean_ret[i, j] /= trans_n[i, j]

        # Transition asymmetry: which transitions are most directional?
        up_prob = np.where(trans_n > 0, trans_up / np.maximum(trans_n, 1), 0.5)
        asymmetry = np.abs(up_prob - 0.5)

        # State entropy (unpredictability)
        from scipy.stats import entropy as sp_entropy
        state_dist = np.bincount(s.astype(int), minlength=n_states) / len(s)
        trans_entropy = float(sp_entropy(state_dist + 1e-12))

        # Conditional return given state
        state_returns = {}
        for si in range(n_states):
            mask = s == si
            if np.sum(mask) > 5:
                state_returns[f"S{si}"] = {
                    "n": int(np.sum(mask)),
                    "p_up": float(np.mean(f[mask] > 0)),
                    "mean_return": float(np.mean(f[mask])),
                }

        # Best directional transition
        max_asym_idx = np.unravel_index(np.argmax(asymmetry), asymmetry.shape)
        best_s_from, best_s_to = int(max_asym_idx[0]), int(max_asym_idx[1])
        best_up_prob = float(up_prob[best_s_from, best_s_to]) if trans_n[best_s_from, best_s_to] > 0 else 0.5
        best_n = int(trans_n[best_s_from, best_s_to])

        sym_res[HORIZON_LABELS[h]] = {
            "n_states": n_states,
            "n_transitions": int(np.sum(valid)),
            "transition_entropy": round(trans_entropy, 4),
            "best_directional_transition": f"S{best_s_from}→S{best_s_to}",
            "best_up_probability": round(best_up_prob, 4),
            "best_n": best_n,
            "state_conditional_returns": state_returns,
        }

        # Transition matrix (condensed)
        sym_res[HORIZON_LABELS[h]]["up_prob_matrix"] = np.round(up_prob, 3).tolist()

    results[sym] = sym_res

output = {"experiment": "DPL-5", "title": "State Transition Directionality",
           "per_symbol": results}
out_path = Path(__file__).parent / "reports" / "dpl5_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-5 complete -> {out_path}")
