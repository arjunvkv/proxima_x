import json, os
import numpy as np


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

SYNTHETIC_BASELINE = {
    "execution_stress": {"alignment": 0.756, "pnl_corr": 0.740, "gate_pass": 0.92},
    "regime_shock": {"alignment": 0.585, "pnl_corr": 0.636, "gate_pass": 0.94},
    "correlation_break": {"alignment": 0.701, "pnl_corr": 0.941, "gate_pass": 0.92},
    "position_feedback": {"alignment": 0.513, "pnl_corr": 0.403, "gate_pass": 0.90},
}

ACCEPTANCE_THRESHOLDS = {
    "execution_stress": {"alignment_min": 0.50, "pnl_corr_min": 0.40, "gate_pass_min": 0.70},
    "regime_shock": {"alignment_min": 0.45, "pnl_corr_min": 0.35, "gate_pass_min": 0.65},
    "correlation_break": {"alignment_min": 0.30, "pnl_corr_min": 0.10, "gate_pass_min": 0.60},
    "position_feedback": {"alignment_min": 0.45, "pnl_corr_min": 0.35, "gate_pass_min": 0.70},
}


def compute_verdict(mt5_results: dict[str, dict]) -> dict:
    comparisons: dict[str, dict] = {}
    all_pass = True
    any_degraded = False

    for test, baseline in SYNTHETIC_BASELINE.items():
        mt5 = mt5_results.get(test, {})
        if "error" in mt5:
            comparisons[test] = {"verdict": "ERROR", "error": mt5["error"]}
            all_pass = False
            continue

        align = mt5.get("alignment", 0)
        pnl = mt5.get("pnl_correlation", 0)
        gate = mt5.get("gate_pass_rate", 0)
        align_delta = align - baseline["alignment"]
        pnl_delta = pnl - baseline["pnl_corr"]
        gate_delta = gate - baseline["gate_pass"]

        thresholds = ACCEPTANCE_THRESHOLDS[test]
        align_pass = align >= thresholds["alignment_min"]
        pnl_pass = pnl >= thresholds["pnl_corr_min"]
        gate_pass = gate >= thresholds["gate_pass_min"]
        test_pass = align_pass and pnl_pass and gate_pass

        if not test_pass:
            all_pass = False
        if align_delta < -0.2 or pnl_delta < -0.3:
            any_degraded = True

        comparisons[test] = {
            "mt5_alignment": align,
            "synthetic_alignment": baseline["alignment"],
            "alignment_delta": round(align_delta, 4),
            "mt5_pnl_corr": pnl,
            "synthetic_pnl_corr": baseline["pnl_corr"],
            "pnl_delta": round(pnl_delta, 4),
            "mt5_gate_pass": gate,
            "synthetic_gate_pass": baseline["gate_pass"],
            "gate_delta": round(gate_delta, 4),
            "veto_rate": mt5.get("veto_rate", 0),
            "structural_rate": mt5.get("structural_rate", 0),
            "dampened_rate": mt5.get("dampened_rate", 0),
            "alignment_pass": align_pass,
            "pnl_pass": pnl_pass,
            "gate_pass": gate_pass,
            "test_pass": test_pass,
        }

    if all_pass and not any_degraded:
        final_verdict = "PASS"
    elif all_pass and any_degraded:
        final_verdict = "DEGRADED"
    else:
        final_verdict = "FAIL"

    report = {
        "final_verdict": final_verdict,
        "all_tests_pass": all_pass,
        "any_degraded": any_degraded,
        "comparisons": comparisons,
        "data_source": mt5_results.get("execution_stress", {}).get("total_ticks", 0),
    }
    return report


def save_report(report: dict, path: str | None = None) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "mt5_acceptance_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, cls=SafeEncoder)
    print(f"[MT5 Report] saved to {path}")
    return path


def print_report(report: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"MT5 ACCEPTANCE REPORT — VERDICT: {report['final_verdict']}")
    print(f"{'=' * 60}")
    for test, comp in report.get("comparisons", {}).items():
        if "error" in comp:
            print(f"  {test}: ERROR — {comp['error']}")
            continue
        status = "PASS" if comp["test_pass"] else "FAIL"
        print(f"\n  {test}: {status}")
        print(f"    Alignment:   {comp['mt5_alignment']:.4f} (synth={comp['synthetic_alignment']:.4f}, Δ={comp['alignment_delta']:+.4f})")
        print(f"    PnL Corr:    {comp['mt5_pnl_corr']:.4f} (synth={comp['synthetic_pnl_corr']:.4f}, Δ={comp['pnl_delta']:+.4f})")
        print(f"    Gate Pass:   {comp['mt5_gate_pass']:.2%} (synth={comp['synthetic_gate_pass']:.2%}, Δ={comp['gate_delta']:+.2%})")
        print(f"    Veto Rate:   {comp.get('veto_rate', 0):.2%}")
    print(f"\n{'=' * 60}")
    print(f"FINAL: {report['final_verdict']}")
    print(f"{'=' * 60}")
