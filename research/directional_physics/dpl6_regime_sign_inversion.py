"""
DPL-6: Regime Sign Inversion
Does the same ES state produce opposite directional outcomes in different regimes?
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
    es = d.es.copy()
    states = d.states.copy().astype(int)
    fut_ret = d.fut_ret

    sym_res = {}
    for hi, h in enumerate(HORIZONS):
        fwd = fut_ret[:, hi]
        valid = ~np.isnan(es) & ~np.isnan(fwd) & (states >= 0)
        if np.sum(valid) < 50:
            continue

        es_v = es[valid]
        s = states[valid]
        f = fwd[valid]

        # For each regime, compute E[return | ES high]
        n_states = int(np.max(s) + 1) if np.max(s) >= 0 else 1
        regime_matrix = {}
        for state in range(n_states):
            mask = s == state
            if np.sum(mask) < 10:
                continue
            es_state = es_v[mask]
            fwd_state = f[mask]

            # ES percentiles within this regime
            pcts = [50, 70, 80, 90, 95]
            for pct in pcts:
                thr = np.nanpercentile(es_state, pct)
                high_es = es_state > thr
                if np.sum(high_es) < 3:
                    continue
                ret_high = fwd_state[high_es]
                regime_matrix[f"S{state}_P{pct}"] = {
                    "n": int(np.sum(high_es)),
                    "p_up": float(np.mean(ret_high > 0)),
                    "mean_return": float(np.mean(ret_high)),
                    "std_return": float(np.std(ret_high)),
                }

        # Sign inversion detection
        pcts_all = [50, 70, 80, 90, 95]
        sign_flips = []
        for pct in pcts_all:
            p_ups = []
            for state in range(n_states):
                key = f"S{state}_P{pct}"
                if key in regime_matrix:
                    p_ups.append(regime_matrix[key]["p_up"])
            if len(p_ups) >= 2:
                if max(p_ups) > 0.6 and min(p_ups) < 0.4:
                    sign_flips.append(f"P{pct}")

        # Regime interaction: does regime determine direction while ES determines opportunity?
        # Correlation of ES with direction within each regime
        regime_es_dir_corr = {}
        for state in range(n_states):
            mask = s == state
            if np.sum(mask) < 10:
                continue
            try:
                from scipy.stats import pearsonr
                c, _ = pearsonr(es_v[mask], (f[mask] > 0).astype(float))
                regime_es_dir_corr[f"S{state}"] = round(float(c), 4)
            except Exception:
                regime_es_dir_corr[f"S{state}"] = None

        sym_res[HORIZON_LABELS[h]] = {
            "n_states": n_states,
            "n_valid": int(np.sum(valid)),
            "regime_matrix": regime_matrix,
            "sign_flip_horizons": sign_flips,
            "has_sign_inversion": len(sign_flips) > 0,
            "regime_es_direction_corr": regime_es_dir_corr,
        }

    results[sym] = sym_res

output = {"experiment": "DPL-6", "title": "Regime Sign Inversion",
           "per_symbol": results}
out_path = Path(__file__).parent / "reports" / "dpl6_results.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"DPL-6 complete -> {out_path}")
