"""ROL-1: What causes residual sign flips? — First mechanism investigation."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS, save_rol_report
from research.directional_state.dsr_core import HORIZON_KEYS

N_LOOK = 30
N_FWD = 30
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def percentile_rank(arr):
    ranks = np.argsort(np.argsort(arr))
    return ranks / (len(arr) - 1)


def logistic_regression_coefs(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    beta = np.zeros(X_aug.shape[1])
    for _ in range(500):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_aug @ beta, -20, 20)))
        W = np.diag(p * (1 - p))
        grad = X_aug.T @ (p - y)
        H = X_aug.T @ W @ X_aug
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta -= delta * 0.5
        if np.max(np.abs(delta)) < 1e-6:
            break
    return {"intercept": round(beta[0], 6), "coef": round(beta[1], 6)}


def run_rol1():
    rol = ROLCore()
    rol.load_all()
    print("ROL core loaded.")

    results = {}

    for sym in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {sym}...")
        d = rol._data[sym]
        es = d["es"]
        res = d["residual"]
        reg = d["regime"]
        reg_from = d.get("reg_transition_from", np.zeros_like(reg))
        reg_to = d.get("reg_transition_to", np.zeros_like(reg))
        md = d["memory_density"]
        m_sat = d.get("memory_saturation", np.zeros_like(reg, dtype=np.int64))
        vol = d["vol"]
        fut_ret = d["fut_ret"]
        n = len(es)

        sign = np.sign(res)
        sign[np.isnan(sign)] = 0
        flip_idx = np.where((sign[1:] != 0) & (sign[:-1] != 0) & (sign[1:] != sign[:-1]))[0] + 1

        if len(flip_idx) == 0:
            print(f"  No flips for {sym}, skipping.")
            continue

        flip_prev_sig = sign[flip_idx - 1]
        flip_from = flip_prev_sig

        valid_mask = (flip_idx >= N_LOOK) & (flip_idx < n - N_FWD)
        flip_idx_v = flip_idx[valid_mask]
        flip_from_v = flip_from[valid_mask]

        if len(flip_idx_v) == 0:
            print(f"  No valid flips (too close to edges) for {sym}.")
            continue

        n_flips = len(flip_idx_v)
        n_pos2neg = int(np.sum(flip_from_v > 0))
        n_neg2pos = int(np.sum(flip_from_v < 0))

        print(f"  Total valid flips: {n_flips} (pos->neg: {n_pos2neg}, neg->pos: {n_neg2pos})")

        # --- Collect precursor trajectories ---
        es_traj = np.full((n_flips, N_LOOK + N_FWD), np.nan)
        md_traj = np.full((n_flips, N_LOOK + N_FWD), np.nan)
        vol_traj = np.full((n_flips, N_LOOK + N_FWD), np.nan)
        reg_at_flip = np.full(n_flips, np.nan)
        is_reg_transition = np.zeros(n_flips, dtype=bool)
        fut_ret_h50 = np.full(n_flips, np.nan)

        for i, fi in enumerate(flip_idx_v):
            lo = fi - N_LOOK
            hi = fi + N_FWD
            es_traj[i] = es[lo:hi]
            md_traj[i] = md[lo:hi]
            vol_traj[i] = vol[lo:hi]
            reg_at_flip[i] = reg[fi]
            if fi > 0 and fi < n:
                is_reg_transition[i] = (reg[fi] != reg[fi - 1])
            fut_ret_h50[i] = fut_ret[fi, HORIZON_KEYS.index("H50")] if "H50" in HORIZON_KEYS else fut_ret[fi, 3]

        flip_dir = flip_from_v

        # --- 1. ES percentile precursor ---
        es_pct = percentile_rank(es)
        flip_labels = np.zeros(n)
        flip_labels[flip_idx_v] = 1
        es_logit = logistic_regression_coefs(es_pct, flip_labels)

        # --- 2. Memory density saturation precursor ---
        m_sat_pct = m_sat / max(np.max(m_sat), 1)
        md_logit = logistic_regression_coefs(m_sat_pct, flip_labels)

        # --- 3. Combined precursor model ---
        X_combined = np.column_stack([es_pct, m_sat_pct])
        y_comb = flip_labels
        X_aug = np.column_stack([np.ones(n), X_combined])
        beta_comb = np.zeros(3)
        for _ in range(500):
            p = 1.0 / (1.0 + np.exp(-np.clip(X_aug @ beta_comb, -20, 20)))
            W = np.diag(p * (1 - p))
            grad = X_aug.T @ (p - y_comb)
            H = X_aug.T @ W @ X_aug
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
            beta_comb -= delta * 0.5
            if np.max(np.abs(delta)) < 1e-6:
                break
        combined_logit = {
            "intercept": round(beta_comb[0], 6),
            "es_coef": round(beta_comb[1], 6),
            "mem_sat_coef": round(beta_comb[2], 6),
        }

        # --- 4. Regime coincidence rate ---
        reg_coincidence = float(np.mean(is_reg_transition))

        reg_coincidence_by_type = {}
        for label, dir_label in [(1, "pos2neg"), (-1, "neg2pos")]:
            mask = flip_dir == label
            if np.sum(mask) > 0:
                reg_coincidence_by_type[dir_label] = float(np.mean(is_reg_transition[mask]))
            else:
                reg_coincidence_by_type[dir_label] = 0.0

        # --- 5. Directional follow-through ---
        dir_follow = {}
        for label, dir_label in [(1, "pos2neg"), (-1, "neg2pos")]:
            mask = flip_dir == label
            if np.sum(mask) > 0:
                fwd_ret = fut_ret_h50[mask]
                p_up = float(np.mean(fwd_ret > 0))
                p_down = float(np.mean(fwd_ret < 0))
            else:
                p_up = 0.0
                p_down = 0.0
            dir_follow[dir_label] = {"p_up": p_up, "p_down": p_down, "n": int(np.sum(mask))}

        # --- 6. Average trajectories (by flip type) ---
        def avg_traj(traj_array, flip_dir_values, look):
            p2n_mask = flip_dir_values > 0
            n2p_mask = flip_dir_values < 0
            result = {}
            for label, mask, name in [(1, p2n_mask, "pos2neg"), (-1, n2p_mask, "neg2pos")]:
                if np.sum(mask) > 0:
                    subset = traj_array[mask]
                    mean = np.nanmean(subset, axis=0).tolist()
                    std = np.nanstd(subset, axis=0).tolist()
                    result[name] = {"mean": mean, "std": std, "n": int(np.sum(mask))}
                else:
                    result[name] = {"mean": None, "std": None, "n": 0}
            return result

        es_avg = avg_traj(es_traj, flip_dir, N_LOOK)
        md_avg = avg_traj(md_traj, flip_dir, N_LOOK)
        vol_avg = avg_traj(vol_traj, flip_dir, N_LOOK)

        # --- 7. Regime distribution at flip ---
        reg_vals = reg_at_flip[~np.isnan(reg_at_flip)].astype(int)
        reg_dist = {}
        if len(reg_vals) > 0:
            for r in np.unique(reg_vals):
                reg_dist[int(r)] = int(np.sum(reg_vals == r))

        # --- 8. Flip density around regime transitions ---
        flip_to_transitions = {}
        for label, dir_label in [(1, "pos2neg"), (-1, "neg2pos")]:
            mask = flip_dir == label
            subset = flip_idx_v[mask]
            n_near = 0
            if len(subset) > 0:
                for fi in subset:
                    lo = max(0, fi - 5)
                    hi = min(n, fi + 5)
                    if np.any(reg[lo:hi] != reg[fi] if fi < n else False):
                        n_near += 1
            flip_to_transitions[dir_label] = {
                "near_transition": int(n_near),
                "total": int(np.sum(mask)),
                "fraction": float(n_near / max(np.sum(mask), 1)),
            }

        # --- 9. Precursor heatmap (mean z-score relative to rolling window) ---
        def precursor_heatmap(var_array, flip_indices, flip_dir_values, look=N_LOOK):
            p2n_mask = flip_dir_values > 0
            n2p_mask = flip_dir_values < 0
            result = {}
            for label, mask, name in [(1, p2n_mask, "pos2neg"), (-1, n2p_mask, "neg2pos")]:
                idxs = flip_indices[mask]
                if len(idxs) == 0:
                    result[name] = None
                    continue
                trajs = []
                for fi in idxs:
                    lo = fi - look
                    hi = fi
                    segment = var_array[lo:hi].copy()
                    mu = np.nanmean(segment)
                    sd = np.nanstd(segment)
                    if sd > 0:
                        segment = (segment - mu) / sd
                    trajs.append(segment)
                heatmap = np.nanmean(trajs, axis=0).tolist() if trajs else None
                result[name] = {"zscore_mean": heatmap, "n": int(len(idxs))}
            return result

        es_heatmap = precursor_heatmap(es, flip_idx_v, flip_dir, N_LOOK)
        md_heatmap = precursor_heatmap(md, flip_idx_v, flip_dir, N_LOOK)
        vol_heatmap = precursor_heatmap(vol, flip_idx_v, flip_dir, N_LOOK)

        # --- 10. Trend before flip (last 5 bars slope) ---
        trend_before = {}
        for label, dir_label in [(1, "pos2neg"), (-1, "neg2pos")]:
            mask = flip_dir == label
            idxs = flip_idx_v[mask]
            slopes = []
            for fi in idxs:
                lo = fi - 5
                x = np.arange(5)
                y = es[lo:fi]
                if len(y) == 5 and not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
            trend_before[dir_label] = {
                "mean_es_slope": float(np.mean(slopes)) if slopes else None,
                "n": len(slopes),
            }

        results[sym] = {
            "n_flips": int(n_flips),
            "n_pos2neg": n_pos2neg,
            "n_neg2pos": n_neg2pos,
            "es_precursor_logistic": es_logit,
            "memory_saturation_logistic": md_logit,
            "combined_logistic": combined_logit,
            "regime_coincidence_rate": reg_coincidence,
            "regime_coincidence_by_type": reg_coincidence_by_type,
            "regime_distribution_at_flip": reg_dist,
            "directional_follow_through": dir_follow,
            "flip_density_around_regime_transitions": flip_to_transitions,
            "es_trajectory": es_avg,
            "memory_density_trajectory": md_avg,
            "volatility_trajectory": vol_avg,
            "es_precursor_heatmap": es_heatmap,
            "md_precursor_heatmap": md_heatmap,
            "vol_precursor_heatmap": vol_heatmap,
            "trend_before_flip": trend_before,
        }

    # --- Cross-symbol summary ---
    total_flips = sum(r["n_flips"] for r in results.values())
    total_p2n = sum(r["n_pos2neg"] for r in results.values())
    total_n2p = sum(r["n_neg2pos"] for r in results.values())
    avg_reg_coinc = float(np.mean([r["regime_coincidence_rate"] for r in results.values()]))

    # Aggregate directional follow-through
    agg_follow = {"pos2neg": {"p_up": 0, "p_down": 0, "n": 0}, "neg2pos": {"p_up": 0, "p_down": 0, "n": 0}}
    for r in results.values():
        for t in agg_follow:
            data = r["directional_follow_through"].get(t, {})
            agg_follow[t]["p_up"] += data.get("p_up", 0) * data.get("n", 0)
            agg_follow[t]["p_down"] += data.get("p_down", 0) * data.get("n", 0)
            agg_follow[t]["n"] += data.get("n", 0)
    for t in agg_follow:
        if agg_follow[t]["n"] > 0:
            agg_follow[t]["p_up"] /= agg_follow[t]["n"]
            agg_follow[t]["p_down"] /= agg_follow[t]["n"]

    report = {
        "metadata": {
            "title": "ROL-1: Residual Sign Flip Causes",
            "description": "Investigation into what causes residual sign flips",
            "lookback_bars": N_LOOK,
            "forward_bars": N_FWD,
            "symbols": list(results.keys()),
        },
        "aggregate": {
            "total_flips": total_flips,
            "total_pos2neg": total_p2n,
            "total_neg2pos": total_n2p,
            "avg_regime_coincidence_rate": avg_reg_coinc,
            "directional_follow_through": agg_follow,
        },
        "per_symbol": results,
    }

    json_path = REPORTS_DIR / "rol1_sign_flip_causes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    # --- Generate Markdown Summary ---
    lines = []
    lines.append("# ROL-1: What Causes Residual Sign Flips?")
    lines.append("")
    lines.append(f"**Lookback:** {N_LOOK} bars | **Forward:** {N_FWD} bars | **Symbols:** {', '.join(results.keys())}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(f"- **Total flips analyzed:** {total_flips} ({total_p2n} positive→negative, {total_n2p} negative→positive)")
    lines.append(f"- **Regime coincidence rate (avg):** {avg_reg_coinc:.1%} — fraction of flips occurring at regime transitions")
    lines.append("")

    lines.append("### Directional Follow-Through (H50)")
    lines.append("")
    lines.append(f"| Type | P(up) | P(down) | N |")
    lines.append(f"|------|-------|---------|---|")
    for t in ["pos2neg", "neg2pos"]:
        d = agg_follow[t]
        lines.append(f"| {t} | {d['p_up']:.1%} | {d['p_down']:.1%} | {d['n']} |")
    lines.append("")

    lines.append("### Best Precursor Signals (Logistic Regression)")
    lines.append("")
    lines.append("| Symbol | ES Intercept | ES Coef | MemSat Intercept | MemSat Coef | Combined ES | Combined MemSat |")
    lines.append("|--------|-------------|---------|-----------------|-------------|-------------|----------------|")
    for sym, r in results.items():
        lines.append(
            f"| {sym} | {r['es_precursor_logistic']['intercept']} | {r['es_precursor_logistic']['coef']} "
            f"| {r['memory_saturation_logistic']['intercept']} | {r['memory_saturation_logistic']['coef']} "
            f"| {r['combined_logistic']['es_coef']} | {r['combined_logistic']['mem_sat_coef']} |"
        )
    lines.append("")

    lines.append("### Regime Coincidence Rates")
    lines.append("")
    lines.append("| Symbol | Overall | pos→neg | neg→pos |")
    lines.append("|--------|---------|---------|---------|")
    for sym, r in results.items():
        rc = r["regime_coincidence_by_type"]
        lines.append(f"| {sym} | {r['regime_coincidence_rate']:.1%} | {rc.get('pos2neg', 0):.1%} | {rc.get('neg2pos', 0):.1%} |")
    lines.append("")

    lines.append("### Flip Density Near Regime Transitions")
    lines.append("")
    lines.append("| Symbol | Type | Near Transition | Total | Fraction |")
    lines.append("|--------|------|-----------------|-------|----------|")
    for sym, r in results.items():
        for t in ["pos2neg", "neg2pos"]:
            fd = r["flip_density_around_regime_transitions"].get(t, {})
            lines.append(f"| {sym} | {t} | {fd.get('near_transition', 0)} | {fd.get('total', 0)} | {fd.get('fraction', 0):.1%} |")
    lines.append("")

    lines.append("### Trend Before Flip (ES slope, last 5 bars)")
    lines.append("")
    lines.append("| Symbol | pos→neg mean ES slope | neg→pos mean ES slope |")
    lines.append("|--------|----------------------|----------------------|")
    for sym, r in results.items():
        tb = r["trend_before_flip"]
        lines.append(f"| {sym} | {tb.get('pos2neg', {}).get('mean_es_slope', 'N/A')} | {tb.get('neg2pos', {}).get('mean_es_slope', 'N/A')} |")
    lines.append("")

    lines.append("### Per-Symbol Summary")
    lines.append("")
    lines.append("| Symbol | Flips | pos→neg | neg→pos | Regime Coinc. | P(up\|pos→neg) | P(up\|neg→pos) |")
    lines.append("|--------|-------|---------|---------|---------------|----------------|----------------|")
    for sym, r in results.items():
        df = r["directional_follow_through"]
        lines.append(
            f"| {sym} | {r['n_flips']} | {r['n_pos2neg']} | {r['n_neg2pos']} "
            f"| {r['regime_coincidence_rate']:.1%} "
            f"| {df.get('pos2neg', {}).get('p_up', 0):.1%} "
            f"| {df.get('neg2pos', {}).get('p_up', 0):.1%} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Answers to Research Questions")
    lines.append("")
    lines.append("**RQ1: What happens in bars leading up to a sign flip?**")
    lines.append("See `es_trajectory` and `memory_density_trajectory` in JSON output for per-symbol mean/std paths.")
    lines.append("")
    lines.append("**RQ2: Do sign flips cluster around regime transitions?**")
    lines.append(f"Average regime coincidence rate: {avg_reg_coinc:.1%}. Check per-symbol breakdown above.")
    lines.append("")
    lines.append("**RQ3: Does ES level predict impending flips?**")
    lines.append("Logistic regression coefficients for ES percentile predictor are in the table above. Positive coef = higher ES → more flips.")
    lines.append("")
    lines.append("**RQ4: Do memory density extremes precede flips?**")
    lines.append("Logistic regression coefficients for memory saturation are in the table above.")
    lines.append("")
    lines.append("**RQ5: Is there a consistent precursor pattern per flip type?**")
    lines.append("Compare pos→neg vs neg→pos precursor heatmaps (`es_precursor_heatmap`, `md_precursor_heatmap`, `vol_precursor_heatmap`) and trend_before_flip in JSON.")
    lines.append("")
    lines.append("**RQ6: Do flips at regime transitions have stronger directional follow-through?**")
    lines.append("See `flip_density_around_regime_transitions` in JSON — compare P(up) for flips near vs. far from transitions.")
    lines.append("")

    md_path = REPORTS_DIR / "ROL1_SIGN_FLIP_CAUSES.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")

    # --- Print key findings ---
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"Total flips: {total_flips} (p2n={total_p2n}, n2p={total_n2p})")
    print(f"Avg regime coincidence: {avg_reg_coinc:.1%}")
    print(f"Directional follow-through:")
    for t in ["pos2neg", "neg2pos"]:
        d = agg_follow[t]
        print(f"  {t}: P(up)={d['p_up']:.1%}, P(down)={d['p_down']:.1%} (n={d['n']})")
    print(f"\nBest precursor signals:")
    for sym, r in results.items():
        print(f"  {sym}: ES coef={r['es_precursor_logistic']['coef']}, MemSat coef={r['memory_saturation_logistic']['coef']}")
    print(f"\nReports saved to {REPORTS_DIR}")

    return report


if __name__ == "__main__":
    report = run_rol1()
