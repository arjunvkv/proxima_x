"""OMS-1: Drift Interaction Test — does the residual marker amplify existing upward drift?"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.observable_market_state.oms_core import OMSCore, save_oms_report
from research.directional_state.dsr_core import WalkForwardValidator, SYMBOLS

HORIZONS = {"H5": 1, "H20": 2, "H50": 3}
HORIZON_LABELS = ["H5", "H20", "H50"]
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def p_up(ret):
    return float(np.mean(ret > 0)) if len(ret) > 0 else 0.0


def run_oms1():
    oms = OMSCore()
    oms.load_all()
    print("OMS core loaded.\n")

    results = {}

    for sym in SYMBOLS:
        print(f"{'='*60}")
        print(f"Processing {sym}...")
        d = oms._data[sym]
        marker = oms.marker_present(sym)
        regime = oms.get_regime(sym)
        fut_ret = oms.get_future_returns(sym)
        es = oms.get_es(sym)
        n = len(marker)

        sym_res = {}

        # --- 1. Baseline drift vs marker drift per horizon ---
        horizons_data = {}
        for hlabel, hidx in HORIZONS.items():
            ret = fut_ret[:, hidx]
            valid = ~np.isnan(ret)
            m_valid = marker.astype(bool) & valid
            nm_valid = ~marker.astype(bool) & valid

            base_drift = p_up(ret[nm_valid])
            mark_drift = p_up(ret[m_valid])
            ampl = mark_drift - base_drift
            ampl_ratio = mark_drift / base_drift if base_drift > 0 else float("inf")

            horizons_data[hlabel] = {
                "baseline_drift": base_drift,
                "marker_drift": mark_drift,
                "amplification_abs": ampl,
                "amplification_ratio": ampl_ratio,
                "n_baseline": int(np.sum(nm_valid)),
                "n_marker": int(np.sum(m_valid)),
            }
        sym_res["horizons"] = horizons_data

        # --- 2. Regime-conditioned amplification ---
        regime_data = {}
        for hlabel, hidx in HORIZONS.items():
            ret = fut_ret[:, hidx]
            valid = ~np.isnan(ret) & (regime >= 0)
            reg_data = {}
            for r in sorted(np.unique(regime[valid])):
                r_mask = regime == r
                m_r = marker.astype(bool) & r_mask & valid
                nm_r = ~marker.astype(bool) & r_mask & valid
                base_d = p_up(ret[nm_r])
                mark_d = p_up(ret[m_r])
                reg_data[int(r)] = {
                    "baseline_drift": base_d,
                    "marker_drift": mark_d,
                    "amplification_abs": mark_d - base_d,
                    "amplification_ratio": mark_d / base_d if base_d > 0 else float("inf"),
                    "n_baseline": int(np.sum(nm_r)),
                    "n_marker": int(np.sum(m_r)),
                }
            regime_data[hlabel] = reg_data

            # --- Additive vs multiplicative test ---
            vals = list(reg_data.values())
            if len(vals) >= 2:
                deltas = [v["amplification_abs"] for v in vals]
                ratios = [v["amplification_ratio"] for v in vals if v["amplification_ratio"] != float("inf")]
                delta_consistency = float(np.std(deltas)) if len(deltas) > 0 else 0
                ratio_consistency = float(np.std(ratios)) if len(ratios) > 0 else 0
                regime_data[hlabel]["_additive_consistency"] = delta_consistency
                regime_data[hlabel]["_multiplicative_consistency"] = ratio_consistency

        sym_res["regime_conditioned"] = regime_data

        # --- 3. Drift ceiling test: ES quintile bucketing within marker=1 ---
        ceiling_data = {}
        for hlabel, hidx in HORIZONS.items():
            ret = fut_ret[:, hidx]
            m_mask = marker.astype(bool) & ~np.isnan(ret) & ~np.isnan(es)
            m_es = es[m_mask]
            m_ret = ret[m_mask]
            if len(m_es) < 50:
                ceiling_data[hlabel] = {"error": "insufficient data"}
                continue
            quintiles = np.linspace(0, 100, 6)
            thresholds = np.nanpercentile(m_es, quintiles)
            quint_data = {}
            for qi in range(5):
                lo = thresholds[qi]
                hi = thresholds[qi + 1]
                if qi == 0:
                    q_mask = m_es <= hi
                elif qi == 4:
                    q_mask = m_es >= lo
                else:
                    q_mask = (m_es > lo) & (m_es <= hi)
                q_ret = m_ret[q_mask]
                quint_data[f"Q{qi+1}"] = {
                    "es_range": [float(lo), float(hi)],
                    "p_up": p_up(q_ret),
                    "n": int(np.sum(q_mask)),
                }
            # check plateau: returns of P(up) across quintiles
            p_vals = [quint_data[f"Q{qi+1}"]["p_up"] for qi in range(5)]
            p_range = max(p_vals) - min(p_vals)
            p_trend = float(np.polyfit(range(5), p_vals, 1)[0]) if len(p_vals) == 5 else 0
            ceiling_data[hlabel] = {
                "quintiles": quint_data,
                "p_up_range_across_quintiles": p_range,
                "p_up_trend_slope": p_trend,
                "plateau_verdict": "likely plateau" if p_range < 0.05 and abs(p_trend) < 0.01 else "no clear plateau",
            }
        sym_res["drift_ceiling"] = ceiling_data

        # --- 4. Walk-forward validation ---
        wfv = WalkForwardValidator(oms.dsr)
        wfv.prepare(sym)
        wf_data = {}
        for train_name, test_name in WalkForwardValidator.SPLITS:
            train_mask, test_mask = wfv.split(sym, train_name, test_name)
            split_key = f"{train_name}_to_{test_name}"
            split_res = {}
            for hlabel, hidx in HORIZONS.items():
                ret = fut_ret[:, hidx]
                valid = ~np.isnan(ret)

                # Train
                t_mask = train_mask & valid
                t_m = marker.astype(bool) & t_mask
                t_nm = ~marker.astype(bool) & t_mask
                train_base = p_up(ret[t_nm])
                train_mark = p_up(ret[t_m])

                # Test
                te_mask = test_mask & valid
                te_m = marker.astype(bool) & te_mask
                te_nm = ~marker.astype(bool) & te_mask
                te_ret = ret[te_mask]
                te_marker = marker[te_mask]

                preds = np.where(te_marker == 1, train_mark > 0.5, train_base > 0.5)
                actual = te_ret > 0
                oos_acc = float(np.mean(preds == actual)) if len(preds) > 0 else 0
                oos_n = int(len(preds))

                split_res[hlabel] = {
                    "train_baseline_drift": train_base,
                    "train_marker_drift": train_mark,
                    "test_oos_accuracy": oos_acc,
                    "test_n": oos_n,
                    "test_n_marker": int(np.sum(te_m)),
                    "test_n_baseline": int(np.sum(te_nm)),
                }
            wf_data[split_key] = split_res
        sym_res["walk_forward"] = wf_data

        results[sym] = sym_res

        # --- Print key findings ---
        print(f"\n  --- HORIZON DRIFT ---")
        for h, hd in horizons_data.items():
            print(f"  {h}: baseline={hd['baseline_drift']:.3f}, marker={hd['marker_drift']:.3f}, "
                  f"ampl={hd['amplification_abs']:+.3f}, ratio={hd['amplification_ratio']:.2f}x")
            print(f"       n_base={hd['n_baseline']}, n_marker={hd['n_marker']}")

        print(f"  --- REGIME DRILLDOWN ---")
        for h, rd in regime_data.items():
            print(f"  {h}:")
            for r, rv in rd.items():
                if r in (0, 1, 2):
                    print(f"    regime={int(r)}: baseline={rv['baseline_drift']:.3f}, marker={rv['marker_drift']:.3f}, "
                          f"ampl={rv['amplification_abs']:+.3f}, ratio={rv['amplification_ratio']:.2f}x")
            if "_additive_consistency" in rd:
                print(f"    additive_consistency(std_delta)={rd['_additive_consistency']:.4f}, "
                      f"multiplicative_consistency(std_ratio)={rd['_multiplicative_consistency']:.4f}")

        print(f"  --- DRIFT CEILING ---")
        for h, cd in ceiling_data.items():
            if "error" in cd:
                print(f"  {h}: {cd['error']}")
                continue
            q_str = " | ".join([f"Q{qi+1}: {cd['quintiles'][f'Q{qi+1}']['p_up']:.3f}" for qi in range(5)])
            print(f"  {h}: {q_str}")
            print(f"       range={cd['p_up_range_across_quintiles']:.4f}, trend={cd['p_up_trend_slope']:.4f}, "
                  f"verdict={cd['plateau_verdict']}")

        print(f"  --- WALK-FORWARD OOS ACCURACY ---")
        for sk, sr in wf_data.items():
            acc_str = " | ".join([f"{h}: {sr[h]['test_oos_accuracy']:.3f}" for h in HORIZON_LABELS])
            print(f"  {sk}: {acc_str}")

    # --- Cross-symbol aggregate ---
    agg_baseline = {h: [] for h in HORIZON_LABELS}
    agg_marker = {h: [] for h in HORIZON_LABELS}
    agg_ampl = {h: [] for h in HORIZON_LABELS}

    for sym, r in results.items():
        for h in HORIZON_LABELS:
            hd = r["horizons"][h]
            agg_baseline[h].append(hd["baseline_drift"])
            agg_marker[h].append(hd["marker_drift"])
            agg_ampl[h].append(hd["amplification_abs"])

    aggregate = {
        "mean_baseline_drift": {h: float(np.mean(agg_baseline[h])) for h in HORIZON_LABELS},
        "mean_marker_drift": {h: float(np.mean(agg_marker[h])) for h in HORIZON_LABELS},
        "mean_amplification_abs": {h: float(np.mean(agg_ampl[h])) for h in HORIZON_LABELS},
        "std_amplification": {h: float(np.std(agg_ampl[h])) for h in HORIZON_LABELS},
    }

    report = {
        "metadata": {"title": "OMS-1: Drift Interaction Test", "symbols": list(results.keys())},
        "aggregate": aggregate,
        "per_symbol": results,
    }

    # --- Save JSON ---
    json_path = REPORTS_DIR / "oms1_drift_interaction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    # --- Generate MD Report ---
    lines = []
    lines.append("# OMS-1: Drift Interaction Test")
    lines.append("")
    lines.append(f"**Symbols:** {', '.join(results.keys())}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Research Questions")
    lines.append("")
    lines.append("1. What is the baseline drift P(up) when NO marker is present?")
    lines.append("2. What is P(up) when marker IS present?")
    lines.append("3. Does the marker amplify drift consistently across all regimes, or only in specific regimes?")
    lines.append("4. Is the amplification linear (additive) or multiplicative?")
    lines.append("5. Does drift amplification vary by horizon? (H5, H20, H50)")
    lines.append("6. Is there a \"drift ceiling\" beyond which the marker adds nothing?")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Cross-Symbol Aggregate")
    lines.append("")
    lines.append(f"| Horizon | Mean Baseline Drift | Mean Marker Drift | Mean Amplification (abs) | Std Amplification |")
    lines.append(f"|---------|--------------------|-------------------|-------------------------|-------------------|")
    for h in HORIZON_LABELS:
        a = aggregate
        lines.append(f"| {h} | {a['mean_baseline_drift'][h]:.4f} | {a['mean_marker_drift'][h]:.4f} | "
                     f"{a['mean_amplification_abs'][h]:+.4f} | {a['std_amplification'][h]:.4f} |")
    lines.append("")

    for sym in SYMBOLS:
        r = results[sym]
        lines.append(f"## {sym}")
        lines.append("")

        lines.append("### Baseline vs Marker Drift")
        lines.append("")
        lines.append("| Horizon | P(up | marker=0) | P(up | marker=1) | Amplification (abs) | Amplification (ratio) | N(0) | N(1) |")
        lines.append("|---------|-----------------|-----------------|--------------------|--------------------|------|------|")
        for h in HORIZON_LABELS:
            hd = r["horizons"][h]
            lines.append(
                f"| {h} | {hd['baseline_drift']:.4f} | {hd['marker_drift']:.4f} | "
                f"{hd['amplification_abs']:+.4f} | {hd['amplification_ratio']:.2f}x | "
                f"{hd['n_baseline']} | {hd['n_marker']} |")
        lines.append("")

        lines.append("### Regime-Conditioned Drift")
        lines.append("")
        lines.append("| Horizon | Regime | P(up | marker=0) | P(up | marker=1) | Amplification (abs) | Ratio |")
        lines.append("|---------|--------|-----------------|-----------------|--------------------|-------|")
        for h in HORIZON_LABELS:
            rd = r["regime_conditioned"][h]
            for reg in sorted([k for k in rd if isinstance(k, int)]):
                rv = rd[reg]
                lines.append(
                    f"| {h} | {reg} | {rv['baseline_drift']:.4f} | {rv['marker_drift']:.4f} | "
                    f"{rv['amplification_abs']:+.4f} | {rv['amplification_ratio']:.2f}x |")
        lines.append("")

        # Additive vs multiplicative summary
        lines.append("#### Additive vs Multiplicative Test")
        lines.append("")
        lines.append("| Horizon | Additive Consistency (std Δ) | Multiplicative Consistency (std ratio) | Verdict |")
        lines.append("|---------|---------------------------|-------------------------------------|---------|")
        for h in HORIZON_LABELS:
            rd = r["regime_conditioned"][h]
            if "_additive_consistency" in rd:
                ac = rd["_additive_consistency"]
                mc = rd["_multiplicative_consistency"]
                if ac < 0.02:
                    verdict = "likely additive (consistent deltas)"
                elif mc < 0.3:
                    verdict = "likely multiplicative (consistent ratios)"
                else:
                    verdict = "neither additive nor multiplicative"
                lines.append(f"| {h} | {ac:.4f} | {mc:.4f} | {verdict} |")
        lines.append("")

        lines.append("### Drift Ceiling Analysis")
        lines.append("")
        lines.append("| Horizon | Q1 | Q2 | Q3 | Q4 | Q5 | Range | Trend | Verdict |")
        lines.append("|---------|----|----|----|----|----|-------|-------|---------|")
        for h in HORIZON_LABELS:
            cd = r["drift_ceiling"].get(h, {})
            if "error" in cd:
                lines.append(f"| {h} | {cd['error']} | | | | | | |")
                continue
            q = cd["quintiles"]
            lines.append(
                f"| {h} | {q['Q1']['p_up']:.3f} | {q['Q2']['p_up']:.3f} | {q['Q3']['p_up']:.3f} | "
                f"{q['Q4']['p_up']:.3f} | {q['Q5']['p_up']:.3f} | {cd['p_up_range_across_quintiles']:.4f} | "
                f"{cd['p_up_trend_slope']:.4f} | {cd['plateau_verdict']} |")
        lines.append("")

        lines.append("### Walk-Forward OOS Accuracy")
        lines.append("")
        lines.append("| Split | H5 | H20 | H50 |")
        lines.append("|-------|----|-----|-----|")
        for sk, sr in r["walk_forward"].items():
            lines.append(f"| {sk} | {sr['H5']['test_oos_accuracy']:.4f} | {sr['H20']['test_oos_accuracy']:.4f} | "
                         f"{sr['H50']['test_oos_accuracy']:.4f} |")
        lines.append("")

    # --- Verdict ---
    lines.append("---")
    lines.append("")
    lines.append("## Verdict: Does the Marker Amplify Existing Drift?")
    lines.append("")

    avg_ampl = {h: aggregate["mean_amplification_abs"][h] for h in HORIZON_LABELS}
    any_positive = all(v > 0.01 for v in avg_ampl.values())
    consistent_across_horizons = max(avg_ampl.values()) - min(avg_ampl.values()) < 0.02

    lines.append(f"**Average amplification across symbols:** H5={avg_ampl['H5']:+.4f}, "
                 f"H20={avg_ampl['H20']:+.4f}, H50={avg_ampl['H50']:+.4f}")
    lines.append("")
    if any_positive:
        lines.append("**YES — The residual marker amplifies upward drift.** P(up | marker=1) > P(up | marker=0) "
                     "consistently across symbols and horizons.")
    else:
        lines.append("**INCONCLUSIVE — Amplification is not consistently positive across all horizons/symbols.**")
    if consistent_across_horizons:
        lines.append("Amplification is consistent across horizons (H5, H20, H50).")
    else:
        lines.append("Amplification varies across horizons — see horizon breakdown.")

    lines.append("")
    lines.append("### Per-Regime Behavior")
    lines.append("")
    for sym in SYMBOLS:
        r = results[sym]
        for h in HORIZON_LABELS:
            rd = r["regime_conditioned"][h]
            regs = sorted([k for k in rd if isinstance(k, int)])
            if len(regs) >= 2:
                deltas = [rd[reg]["amplification_abs"] for reg in regs]
                delta_range = max(deltas) - min(deltas)
                lines.append(f"- **{sym} {h}**: regime amplification deltas range={delta_range:.4f} "
                             f"({'consistent across regimes' if delta_range < 0.03 else 'varies by regime'})")

    lines.append("")
    lines.append("### Ceiling Conclusion")
    lines.append("")
    plateau_count = 0
    for sym in SYMBOLS:
        for h in HORIZON_LABELS:
            cd = results[sym]["drift_ceiling"].get(h, {})
            if cd.get("plateau_verdict") == "likely plateau":
                plateau_count += 1
    total_horizons = len(SYMBOLS) * len(HORIZON_LABELS)
    lines.append(f"**{plateau_count}/{total_horizons} symbol-horizon combinations show a drift plateau.**")
    if plateau_count / total_horizons > 0.5:
        lines.append("A drift ceiling exists — beyond a certain ES level, the marker adds no additional directional lift.")
    else:
        lines.append("No strong evidence of a drift ceiling — P(up) may continue to increase with ES.")

    md_path = REPORTS_DIR / "OMS1_DRIFT_INTERACTION.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")

    # --- stdout summary ---
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    for h in HORIZON_LABELS:
        print(f"{h}: avg baseline={aggregate['mean_baseline_drift'][h]:.3f}, "
              f"avg marker={aggregate['mean_marker_drift'][h]:.3f}, "
              f"avg ampl={aggregate['mean_amplification_abs'][h]:+.3f}")
    print(f"\nWalk-forward OOS accuracy (avg across splits):")
    for sym in SYMBOLS:
        wf = results[sym]["walk_forward"]
        accs = []
        for sk in wf:
            for h in HORIZON_LABELS:
                accs.append(wf[sk][h]["test_oos_accuracy"])
        print(f"  {sym}: {np.mean(accs):.3f} (n_splits={len(wf)}, n_horizons={len(HORIZON_LABELS)})")
    print(f"\nReports saved to {REPORTS_DIR}")

    return report


if __name__ == "__main__":
    report = run_oms1()
