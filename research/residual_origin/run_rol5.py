"""ROL-5: Residual Memory Coupling  --  OUT OF SAMPLE ONLY.

Tests whether memory imbalance adds predictive value beyond residual sign alone,
using strict walk-forward (no in-sample optimization).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, HORIZON_KEYS, WalkForwardValidator, save_report
from research.residual_origin.rol_core import ROLCore, save_rol_report


def _quintile_labels(x, n_quantiles=5):
    """Discretize continuous variable into quintile labels (1..n_quantiles)."""
    labels = np.full(len(x), -1, dtype=np.int64)
    valid = ~np.isnan(x)
    if np.sum(valid) < n_quantiles:
        return labels
    edges = np.nanpercentile(x[valid], np.linspace(0, 100, n_quantiles + 1)[1:-1])
    labels[valid] = np.searchsorted(edges, x[valid], side="right").astype(np.int64) + 1
    return labels


def _binary_imb(x):
    """Discretize memory_imbalance to binary: 1 for positive, 0 for negative (or zero)."""
    labels = np.full(len(x), -1, dtype=np.int64)
    valid = ~np.isnan(x)
    labels[valid & (x > 0)] = 1
    labels[valid & (x <= 0)] = 0
    return labels


def _p_up_given_state(up_flags, state_ids, n_states, min_samples=5):
    """Compute P(up) for each state ID from training data."""
    p_up = np.full(n_states, np.nan)
    for sid in range(n_states):
        mask = state_ids == sid
        cnt = int(np.sum(mask))
        if cnt < min_samples:
            continue
        p_up[sid] = float(np.mean(up_flags[mask]))
    return p_up


def _predict_from_p_up(state_ids, p_up):
    """Predict direction (1=up, 0=down) from state-conditional P(up)."""
    preds = np.full(len(state_ids), np.nan)
    n_states = len(p_up)
    for sid in range(n_states):
        mask = state_ids == sid
        if np.sum(mask) == 0:
            continue
        if np.isnan(p_up[sid]):
            continue
        preds[mask] = 1.0 if p_up[sid] > 0.5 else 0.0
    return preds


def _accuracy(preds, actual):
    valid = ~np.isnan(preds) & ~np.isnan(actual)
    if np.sum(valid) == 0:
        return np.nan
    return float(np.mean(preds[valid] == actual[valid]))


def _up_flag(fut_ret_col):
    return (fut_ret_col > 0).astype(float)


def run_rol5():
    rol = ROLCore()
    dsr = rol.dsr
    dsr.run_all_symbols()
    wfv = WalkForwardValidator(dsr)

    # Pre-registered models
    # Model 1: residual_sign only (baseline)  --  2 states
    # Model 2: residual_sign + memory_imbalance_quintile  --  10 states (2x5)
    # Model 3: residual_sign + memory_imbalance_binary  --  4 states (2x2)

    horizons_of_interest = [1, 2, 3]  # indices for H5, H20, H50 in fut_ret columns
    horizon_labels = ["H5", "H20", "H50"]

    results = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {sym}")
        print(f"{'='*60}")

        d = dsr._data[sym]
        n = len(d["es"])
        residual_sign = d["residual_sign"].copy()
        memory_imbalance = d["memory_imbalance"].copy()

        mem_imb_quintile = _quintile_labels(memory_imbalance, 5)
        mem_imb_binary = _binary_imb(memory_imbalance)

        # Build year labels for splitting
        years = wfv.prepare(sym)

        sym_results = {}

        for split_idx, (train_name, test_name) in enumerate(WalkForwardValidator.SPLITS):
            train_mask, test_mask = wfv.split(sym, train_name, test_name)

            # Check we have data in both splits
            if np.sum(train_mask) < 50 or np.sum(test_mask) < 10:
                print(f"  Skip split {train_name}->{test_name}: insufficient data "
                      f"(train={np.sum(train_mask)}, test={np.sum(test_mask)})")
                continue

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            split_key = f"{train_name}_to_{test_name}"
            split_results = {}

            for h_idx, h_label in zip(horizons_of_interest, horizon_labels):
                fut_col = d["fut_ret"][:, h_idx]
                train_up = _up_flag(fut_col[train_idx])
                test_up = _up_flag(fut_col[test_idx])

                # --- Model 1: residual_sign only (BASELINE) ---
                train_sgn = residual_sign[train_idx]
                test_sgn = residual_sign[test_idx]

                # 2 states: sign can be -1, 0, 1  --  map to {0: -1, 1: 0? no}
                # Actually residual_sign values are -1, 0, 1
                # We only use -1 and 1 (skip sign=0)
                m1_state = np.full(len(train_sgn), -1, dtype=np.int64)
                m1_state[train_sgn == -1] = 0
                m1_state[train_sgn == 1] = 1
                # Remove sign==0 from consideration
                m1_train_valid = m1_state >= 0
                m1_state_test = np.full(len(test_sgn), -1, dtype=np.int64)
                m1_state_test[test_sgn == -1] = 0
                m1_state_test[test_sgn == 1] = 1

                m1_p_up = _p_up_given_state(train_up[m1_train_valid], m1_state[m1_train_valid], 2)
                m1_pred = _predict_from_p_up(m1_state_test, m1_p_up)
                m1_acc = _accuracy(m1_pred, test_up)

                # --- Model 2: residual_sign + memory_imbalance_quintile ---
                # State ID = sign_map * 5 + quintile (where quintile is 1..5)
                m2_state = np.full(n, -1, dtype=np.int64)
                for i in range(n):
                    s = residual_sign[i]
                    q = mem_imb_quintile[i]
                    if s == -1 and q >= 1:
                        m2_state[i] = q - 1  # 0..4
                    elif s == 1 and q >= 1:
                        m2_state[i] = 5 + (q - 1)  # 5..9

                m2_state_train = m2_state[train_idx]
                m2_state_test = m2_state[test_idx]
                m2_train_valid = m2_state_train >= 0
                m2_test_valid = m2_state_test >= 0

                m2_p_up = _p_up_given_state(train_up[m2_train_valid], m2_state_train[m2_train_valid], 10)
                m2_pred = _predict_from_p_up(m2_state_test, m2_p_up)
                m2_acc = _accuracy(m2_pred, test_up)

                # --- Model 3: residual_sign + memory_imbalance binary ---
                # State ID = sign_map * 2 + imb_binary
                m3_state = np.full(n, -1, dtype=np.int64)
                for i in range(n):
                    s = residual_sign[i]
                    b = mem_imb_binary[i]
                    if s == -1 and b >= 0:
                        m3_state[i] = b  # 0 or 1
                    elif s == 1 and b >= 0:
                        m3_state[i] = 2 + b  # 2 or 3

                m3_state_train = m3_state[train_idx]
                m3_state_test = m3_state[test_idx]
                m3_train_valid = m3_state_train >= 0
                m3_test_valid = m3_state_test >= 0

                m3_p_up = _p_up_given_state(train_up[m3_train_valid], m3_state_train[m3_train_valid], 4)
                m3_pred = _predict_from_p_up(m3_state_test, m3_p_up)
                m3_acc = _accuracy(m3_pred, test_up)

                split_results[h_label] = {
                    "model1_residual_only": round(m1_acc, 4),
                    "model2_sign_plus_quintile": round(m2_acc, 4),
                    "model3_sign_plus_binary": round(m3_acc, 4),
                    "improvement_m2_over_m1": round(m2_acc - m1_acc, 4),
                    "improvement_m3_over_m1": round(m3_acc - m1_acc, 4),
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(test_mask)),
                }

                # Regime-conditioned analysis
                regime = d["regime"]
                regime_keys = {0: "low_density", 1: "mid_density", 2: "high_density"}
                regime_breakdown = {}
                for r_val, r_name in regime_keys.items():
                    r_test_mask = test_mask & (regime == r_val)
                    r_test_idx = np.where(r_test_mask)[0]
                    if len(r_test_idx) < 5:
                        continue
                    r_up = _up_flag(fut_col[r_test_idx])

                    # M1 in this regime
                    r_m1_state = np.full(len(r_test_idx), -1, dtype=np.int64)
                    r_m1_state[residual_sign[r_test_idx] == -1] = 0
                    r_m1_state[residual_sign[r_test_idx] == 1] = 1
                    r_m1_pred = _predict_from_p_up(r_m1_state, m1_p_up)
                    r_m1_acc = _accuracy(r_m1_pred, r_up)

                    # M2 in this regime
                    r_m2_state = m2_state[r_test_idx]
                    r_m2_pred = _predict_from_p_up(r_m2_state, m2_p_up)
                    r_m2_acc = _accuracy(r_m2_pred, r_up)

                    # M3 in this regime
                    r_m3_state = m3_state[r_test_idx]
                    r_m3_pred = _predict_from_p_up(r_m3_state, m3_p_up)
                    r_m3_acc = _accuracy(r_m3_pred, r_up)

                    regime_breakdown[r_name] = {
                        "n": int(len(r_test_idx)),
                        "model1_acc": round(r_m1_acc, 4) if not np.isnan(r_m1_acc) else None,
                        "model2_acc": round(r_m2_acc, 4) if not np.isnan(r_m2_acc) else None,
                        "model3_acc": round(r_m3_acc, 4) if not np.isnan(r_m3_acc) else None,
                        "m2_improvement": round(r_m2_acc - r_m1_acc, 4) if not (np.isnan(r_m2_acc) or np.isnan(r_m1_acc)) else None,
                        "m3_improvement": round(r_m3_acc - r_m3_acc, 4) if not (np.isnan(r_m3_acc) or np.isnan(r_m1_acc)) else None,
                    }

                split_results[h_label]["regime_breakdown"] = regime_breakdown

                print(f"  {h_label}: M1={m1_acc:.4f} M2={m2_acc:.4f} M3={m3_acc:.4f} | "
                       f"dM2={m2_acc - m1_acc:+.4f} dM3={m3_acc - m1_acc:+.4f} "
                      f"(train={np.sum(train_mask)}, test={np.sum(test_mask)})")

            sym_results[split_key] = split_results

        # --- Cross-split consistency analysis ---
        for h_label in horizon_labels:
            m2_improvs = []
            m3_improvs = []
            for split_key in sym_results:
                if h_label in sym_results[split_key]:
                    m2_improvs.append(sym_results[split_key][h_label]["improvement_m2_over_m1"])
                    m3_improvs.append(sym_results[split_key][h_label]["improvement_m3_over_m1"])

            if len(m2_improvs) > 1:
                sym_results["consistency"] = sym_results.get("consistency", {})
                sym_results["consistency"][h_label] = {
                    "m2_improvements": [round(v, 4) for v in m2_improvs],
                    "m3_improvements": [round(v, 4) for v in m3_improvs],
                    "m2_mean": round(float(np.mean(m2_improvs)), 4),
                    "m3_mean": round(float(np.mean(m3_improvs)), 4),
                    "m2_std": round(float(np.std(m2_improvs)), 4),
                    "m3_std": round(float(np.std(m3_improvs)), 4),
                    "m2_consistently_positive": all(v > 0 for v in m2_improvs),
                    "m3_consistently_positive": all(v > 0 for v in m3_improvs),
                }

        results[sym] = sym_results

    # --- Aggregate across symbols ---
    aggregate = {}
    for h_label in horizon_labels:
        all_m2_improvs = []
        all_m3_improvs = []
        for sym in SYMBOLS:
            if sym in results and "consistency" in results[sym] and h_label in results[sym]["consistency"]:
                all_m2_improvs.extend(results[sym]["consistency"][h_label]["m2_improvements"])
                all_m3_improvs.extend(results[sym]["consistency"][h_label]["m3_improvements"])

        aggregate[h_label] = {
            "m2_improvements": [round(v, 4) for v in all_m2_improvs],
            "m3_improvements": [round(v, 4) for v in all_m3_improvs],
            "m2_mean": round(float(np.mean(all_m2_improvs)), 4) if all_m2_improvs else None,
            "m3_mean": round(float(np.mean(all_m3_improvs)), 4) if all_m3_improvs else None,
            "m2_std": round(float(np.std(all_m2_improvs)), 4) if all_m2_improvs else None,
            "m3_std": round(float(np.std(all_m3_improvs)), 4) if all_m3_improvs else None,
            "m2_positive_fraction": round(np.mean([v > 0 for v in all_m2_improvs]), 4) if all_m2_improvs else None,
            "m3_positive_fraction": round(np.mean([v > 0 for v in all_m3_improvs]), 4) if all_m3_improvs else None,
        }

    report = {
        "experiment": "ROL-5: Residual Memory Coupling (OOS)",
        "models": {
            "model1": "residual_sign only (baseline)",
            "model2": "residual_sign + memory_imbalance_quintile (10 states)",
            "model3": "residual_sign + memory_imbalance_binary (4 states)",
        },
        "splits": WalkForwardValidator.SPLITS,
        "symbols": SYMBOLS,
        "results": results,
        "aggregate": aggregate,
        "verdict": _generate_verdict(aggregate),
    }

    save_rol_report(report, "rol5_memory_coupling")
    _print_markdown_report(report)
    return report


def _generate_verdict(aggregate):
    """Generate a structured verdict based on results."""
    parts = []
    for h_label in ["H5", "H20", "H50"]:
        agg = aggregate.get(h_label, {})
        parts.append(f"**{h_label}**: M2 mean d={agg.get('m2_mean', 'N/A')} "
                     f"(std={agg.get('m2_std', 'N/A')}, "
                     f"pos_frac={agg.get('m2_positive_fraction', 'N/A')}); "
                     f"M3 mean d={agg.get('m3_mean', 'N/A')} "
                     f"(std={agg.get('m3_std', 'N/A')}, "
                     f"pos_frac={agg.get('m3_positive_fraction', 'N/A')})")

    all_m2 = []
    all_m3 = []
    for h_label in ["H5", "H20", "H50"]:
        agg = aggregate.get(h_label, {})
        if agg.get("m2_improvements"):
            all_m2.extend(agg["m2_improvements"])
        if agg.get("m3_improvements"):
            all_m3.extend(agg["m3_improvements"])

    overall_m2 = round(float(np.mean(all_m2)), 4) if all_m2 else None
    overall_m3 = round(float(np.mean(all_m3)), 4) if all_m3 else None
    overall_m2_pos = round(np.mean([v > 0 for v in all_m2]), 4) if all_m2 else None
    overall_m3_pos = round(np.mean([v > 0 for v in all_m3]), 4) if all_m3 else None

    verdict = {
        "overall_m2_mean_improvement": overall_m2,
        "overall_m3_mean_improvement": overall_m3,
        "overall_m2_positive_fraction": overall_m2_pos,
        "overall_m3_positive_fraction": overall_m3_pos,
        "detail": parts,
        "conclusion": (
            f"Memory imbalance adds value OOS: {overall_m2_pos > 0.5 if overall_m2_pos else 'INCONCLUSIVE'} (M2). "
            f"Memory imbalance (binary) adds value OOS: {overall_m3_pos > 0.5 if overall_m3_pos else 'INCONCLUSIVE'} (M3)."
        ),
    }
    return verdict


def _print_markdown_report(report):
    """Print markdown report to stdout and save to file."""
    lines = []
    lines.append("# ROL-5: Residual Memory Coupling  --  OOS Validation")
    lines.append("")
    lines.append("## Models")
    for k, v in report["models"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Splits")
    for s in report["splits"]:
        lines.append(f"- Train `{s[0]}` -> Test `{s[1]}`")
    lines.append("")

    agg = report.get("aggregate", {})
    lines.append("## Aggregate Results (all symbols, all splits)")
    lines.append("")
    lines.append("| Horizon | M1 (residual) | M2 (+quintile) | M3 (+binary) | dM2 mean | dM2 std | dM2 pos_frac | dM3 mean | dM3 std | dM3 pos_frac |")
    lines.append("|---------|--------------|----------------|--------------|----------|---------|-------------|----------|---------|-------------|")

    for sym in report["symbols"]:
        sym_data = report["results"].get(sym, {})
        for split_key in ["2018-2022_to_2023", "2019-2023_to_2024", "2020-2024_to_2025"]:
            if split_key not in sym_data:
                continue
            for h_label in ["H5", "H20", "H50"]:
                if h_label not in sym_data[split_key]:
                    continue
                r = sym_data[split_key][h_label]
                lines.append(f"| {sym}/{split_key}/{h_label} | {r['model1_residual_only']} | "
                             f"{r['model2_sign_plus_quintile']} | {r['model3_sign_plus_binary']} | "
                             f"{r['improvement_m2_over_m1']} | ... | ... | {r['improvement_m3_over_m1']} | ... | ... |")

    lines.append("")
    lines.append("### Mean Across All")
    for h_label in ["H5", "H20", "H50"]:
        a = agg.get(h_label, {})
        if a.get("m2_mean") is not None:
            lines.append(f"- **{h_label}**: dM2={a['m2_mean']} (std={a['m2_std']}) "
                         f"(pos={a['m2_positive_fraction']}), "
                         f"dM3={a['m3_mean']} (std={a['m3_std']}) "
                         f"(pos={a['m3_positive_fraction']})")

    lines.append("")
    lines.append("## Verdict")
    v = report.get("verdict", {})
    lines.append(f"- Overall dM2: {v.get('overall_m2_mean_improvement')} (pos frac: {v.get('overall_m2_positive_fraction')})")
    lines.append(f"- Overall dM3: {v.get('overall_m3_mean_improvement')} (pos frac: {v.get('overall_m3_positive_fraction')})")
    lines.append(f"- Conclusion: {v.get('conclusion')}")
    lines.append("")

    md = "\n".join(lines)

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "ROL5_MEMORY_COUPLING.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved {md_path}")
    print("\n" + md)


if __name__ == "__main__":
    report = run_rol5()
