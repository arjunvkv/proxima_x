"""DSR Phases 1 & 2 — Directional State Reconstruction and Regime × Residual Surface."""
import sys, ast
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, STATE_HORIZON_KEYS, save_report

HORIZON_IDX = {"H5": 1, "H20": 2, "H50": 3}


def compute_cell_ids(d):
    regime = d["regime"]
    residual = d["residual"]
    n = len(regime)

    q = np.full(n, -1, dtype=np.int64)
    valid = ~np.isnan(residual)
    if np.sum(valid) >= 10:
        bins = np.nanpercentile(residual[valid], [20, 40, 60, 80])
        q[valid & (residual <= bins[0])] = 0
        q[valid & (residual > bins[0]) & (residual <= bins[1])] = 1
        q[valid & (residual > bins[1]) & (residual <= bins[2])] = 2
        q[valid & (residual > bins[2]) & (residual <= bins[3])] = 3
        q[valid & (residual > bins[3])] = 4

    cell_ids = np.full(n, -1, dtype=np.int64)
    cell_map = {}
    counter = 0
    for i in range(n):
        r, qt = int(regime[i]), int(q[i])
        if r < 0 or qt < 0:
            continue
        key = (r, qt)
        if key not in cell_map:
            cell_map[key] = counter
            counter += 1
        cell_ids[i] = cell_map[key]

    return cell_ids, cell_map


def phase1(dsr):
    report = {"title": "DSR Phase 1 — Directional State Reconstruction", "symbols": {}}

    for sym in SYMBOLS:
        d = dsr._data[sym]
        state_ids, unique_map = dsr.state_array(sym)
        fut_ret = d["fut_ret"]

        sr = {"n_observations": len(state_ids), "n_unique_states": len(unique_map),
              "horizons": {}}

        for hk in STATE_HORIZON_KEYS:
            hi = HORIZON_IDX[hk]
            metrics = dsr.directional_metrics(sym, state_ids, fut_ret[:, hi], hi)
            sr["horizons"][hk] = metrics

            p_ups = [m["p_up"] for m in metrics.values()]
            sems = [m["p_up_sem"] for m in metrics.values()]
            zs = [m["z_score"] for m in metrics.values()]

            stable_up = sum(1 for z in zs if z > 2)
            stable_down = sum(1 for z in zs if z < -2)

            sr[hk] = {
                "n_states": len(metrics),
                "mean_p_up": round(float(np.mean(p_ups)), 4) if p_ups else 0,
                "std_p_up": round(float(np.std(p_ups)), 4) if p_ups else 0,
                "mean_sem": round(float(np.mean(sems)), 4) if sems else 0,
                "median_sem": round(float(np.median(sems)), 4) if sems else 0,
                "stable_up": stable_up,
                "stable_down": stable_down,
                "pct_stable": round(100 * (stable_up + stable_down) / max(len(metrics), 1), 1),
                "p_up_quartiles": {
                    "q25": round(float(np.percentile(p_ups, 25)), 4),
                    "q50": round(float(np.percentile(p_ups, 50)), 4),
                    "q75": round(float(np.percentile(p_ups, 75)), 4),
                } if p_ups else {},
            }

        for i, hk1 in enumerate(STATE_HORIZON_KEYS):
            for hk2 in STATE_HORIZON_KEYS[i + 1:]:
                m1, m2 = sr["horizons"][hk1], sr["horizons"][hk2]
                common = sorted(set(m1.keys()) & set(m2.keys()))
                if len(common) > 5:
                    p1, p2 = [m1[s]["p_up"] for s in common], [m2[s]["p_up"] for s in common]
                    corr = float(np.corrcoef(p1, p2)[0, 1])
                else:
                    corr = 0.0
                sr.setdefault("persistence", {})[f"{hk1}_vs_{hk2}"] = {
                    "correlation": round(corr, 4), "n_common": len(common)}

        report["symbols"][sym] = sr

    return report


def phase2(dsr):
    report = {"title": "DSR Phase 2 — Regime × Residual Surface", "symbols": {}}

    for sym in SYMBOLS:
        d = dsr._data[sym]
        cell_ids, cell_map = compute_cell_ids(d)
        fut_ret = d["fut_ret"]

        sr = {"n_observations": len(cell_ids), "n_cells": len(cell_map),
              "cell_map": {str(k): v for k, v in cell_map.items()}, "horizons": {}}

        for hk in STATE_HORIZON_KEYS:
            hi = HORIZON_IDX[hk]
            metrics = dsr.directional_metrics(sym, cell_ids, fut_ret[:, hi], hi)
            sr["horizons"][hk] = metrics

            p_ups = [m["p_up"] for m in metrics.values()]
            c_up = sum(1 for p in p_ups if p > 0.75)
            c_down = sum(1 for p in p_ups if p < 0.25)

            sr[hk] = {
                "n_cells": len(metrics),
                "mean_p_up": round(float(np.mean(p_ups)), 4) if p_ups else 0,
                "directional_up_pockets": c_up,
                "directional_down_pockets": c_down,
                "pct_directional": round(100 * (c_up + c_down) / max(len(metrics), 1), 1),
            }

        reg_analysis = {}
        for reg in range(3):
            cells_in_reg = {k: v for k, v in cell_map.items() if k[0] == reg}
            quints = sorted(set(k[1] for k in cells_in_reg))
            ra = {"quintiles_present": list(quints)}
            for hk in STATE_HORIZON_KEYS:
                cell_data = {}
                for (r, q), cid in cells_in_reg.items():
                    cid_str = str(cid)
                    if cid_str in sr["horizons"][hk]:
                        cell_data[f"q{q}"] = sr["horizons"][hk][cid_str]
                if cell_data:
                    ra[hk] = cell_data
            reg_analysis[f"regime_{reg}"] = ra

        sr["regime_analysis"] = reg_analysis
        report["symbols"][sym] = sr

    return report


def print_report(p1, p2):
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for sym in SYMBOLS:
        s1 = p1["symbols"][sym]
        print(f"\n{sym}")
        print(f"  Phase 1: {s1['n_unique_states']} unique states / {s1['n_observations']} obs")
        for hk in STATE_HORIZON_KEYS:
            sm = s1[hk]
            print(f"    {hk}: {sm['n_states']} states, mean P(up)={sm['mean_p_up']:.3f}, "
                  f"stable={sm['pct_stable']}% ({sm['stable_up']}up/{sm['stable_down']}dn)")

        if "persistence" in s1:
            for k, v in s1["persistence"].items():
                print(f"    Persistence {k}: r={v['correlation']:.3f} ({v['n_common']} states)")

        s2 = p2["symbols"][sym]
        print(f"  Phase 2: {s2['n_cells']} Regime×Residual cells")
        for hk in STATE_HORIZON_KEYS:
            sm = s2[hk]
            print(f"    {hk}: {sm['n_cells']} cells, mean P(up)={sm['mean_p_up']:.3f}, "
                  f"directional={sm['directional_up_pockets']}up/{sm['directional_down_pockets']}dn "
                  f"({sm['pct_directional']}%)")

    # Cross-asset consistency for Phase 2
    print("\n  Cross-Asset Directional Cell Consistency:")
    all_cell_p_ups = {hk: [] for hk in STATE_HORIZON_KEYS}
    for sym in SYMBOLS:
        s2 = p2["symbols"][sym]
        for hk in STATE_HORIZON_KEYS:
            for cid, m in s2["horizons"][hk].items():
                if isinstance(m, dict) and "p_up" in m:
                    rev = {v: ast.literal_eval(k) for k, v in s2["cell_map"].items()}
                    cid_int = int(cid)
                    if cid_int in rev:
                        reg, q = rev[cid_int]
                        all_cell_p_ups[hk].append({
                            "symbol": sym, "regime": reg, "quintile": q, "p_up": m["p_up"]})

    for hk in STATE_HORIZON_KEYS:
        cells = all_cell_p_ups[hk]
        if cells:
            reg_consistency = {}
            for c in cells:
                key = (c["regime"], c["quintile"])
                reg_consistency.setdefault(key, []).append(c["p_up"])
            consistent = sum(1 for v in reg_consistency.values()
                             if len(v) >= 3 and (all(p > 0.75 for p in v) or all(p < 0.25 for p in v)))
            print(f"    {hk}: {consistent}/{len(reg_consistency)} (regime,quintile) cells "
                  f"consistent across {len(SYMBOLS)} assets")


if __name__ == "__main__":
    print("=" * 60)
    print("DSR Phases 1 & 2")
    print("=" * 60)

    dsr = DSRCore()
    dsr.run_all_symbols()

    p1 = phase1(dsr)
    save_report(p1, "dsr_phase1_state_reconstruction")

    p2 = phase2(dsr)
    save_report(p2, "dsr_phase2_regime_residual_surface")

    print_report(p1, p2)

    print("\nDone.")
