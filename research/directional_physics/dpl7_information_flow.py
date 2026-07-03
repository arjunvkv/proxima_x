"""
DPL-7: Information Flow Layer
Does directional resolution originate externally (cross-asset pressure)?
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZONS, HORIZON_LABELS
import warnings
warnings.filterwarnings("ignore")

# Load all symbols
all_data = {sym: DPLData(sym) for sym in SYMBOLS}

results = {}
for target in SYMBOLS:
    dt = all_data[target]
    es_t = dt.es.copy()
    fut_t = dt.fut_ret

    sym_res = {}
    for hi, h in enumerate(HORIZONS):
        fwd_t = fut_t[:, hi]
        valid_t = ~np.isnan(es_t) & ~np.isnan(fwd_t)
        if np.sum(valid_t) < 30:
            continue

        fwd_t_v = fwd_t[valid_t]
        dir_t = (fwd_t_v > 0).astype(float)
        es_t_v = es_t[valid_t]

        info_flow = {"target_es_self_corr": {}}
        for hi2, h2 in enumerate(HORIZONS):
            fwd_t2 = fut_t[:, hi2]
            if np.sum(~np.isnan(fwd_t2[valid_t])) < 30:
                continue
            fwd_t2_v = fwd_t2[valid_t]
            min_n2 = min(len(es_t_v), len(fwd_t2_v))
            c, _ = pearsonr(es_t_v[:min_n2], fwd_t2_v[:min_n2])
            info_flow["target_es_self_corr"][HORIZON_LABELS[h2]] = round(float(c), 4)

        # Cross-asset pressure
        for source in SYMBOLS:
            if source == target:
                continue
            ds = all_data[source]
            es_s = ds.es.copy()

            n_common = min(len(es_t), len(es_s))
            valid_common = valid_t[:n_common]
            valid_idx = np.where(valid_common)[0]
            if len(valid_idx) < 30:
                continue
            es_s_v = es_s[valid_idx]
            es_t_common = es_t[valid_idx]
            fwd_common = fut_t[valid_idx, hi]
            dir_common = (fwd_common > 0).astype(float)

            # Does source ES correlate with target direction?
            c_es_dir, _ = pearsonr(es_s_v, dir_common)
            c_es_ret, _ = pearsonr(es_s_v, fwd_common)

            # Lead-lag: source ES → target future return
            max_lag = 5
            lead_corrs = {}
            for lag in range(1, max_lag + 1):
                if lag >= len(es_s_v):
                    continue
                c, _ = pearsonr(es_s_v[:-lag], fwd_common[lag:])
                lead_corrs[f"lag_{lag}"] = round(float(c), 4)

            info_flow[f"from_{source}"] = {
                "corr_es_source_target_dir": round(float(c_es_dir), 4),
                "corr_es_source_target_ret": round(float(c_es_ret), 4),
                "lead_lag": lead_corrs,
            }

        sym_res[HORIZON_LABELS[h]] = info_flow

    results[target] = sym_res

# Build information propagation graph
propagation = {}
for target in SYMBOLS:
    for hl in HORIZON_LABELS.values():
        if hl not in results.get(target, {}):
            continue
        for source in SYMBOLS:
            if source == target:
                continue
            key = f"from_{source}"
            if key in results[target][hl]:
                c = results[target][hl][key]["corr_es_source_target_dir"]
                propagation[f"{source}→{target}@{hl}"] = c

output = {"experiment": "DPL-7", "title": "Information Flow Layer",
           "per_symbol": results,
           "propagation_graph": propagation,
           "strongest_edges": dict(sorted(propagation.items(), key=lambda x: abs(x[1]), reverse=True)[:10])}
out_path = Path(__file__).parent / "reports" / "dpl7_results.json"
out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
print(f"DPL-7 complete -> {out_path}")
