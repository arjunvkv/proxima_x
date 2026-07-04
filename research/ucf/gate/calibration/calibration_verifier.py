import json, os


THRESHOLDS = {
    "alignment_min": 0.60,
    "rc_veto_min": 0.05,
    "rc_veto_max": 0.15,
    "mra_high_saturation_max": 0.10,
    "mra_low_saturation_max": 0.10,
    "emd_dominance_max": 0.40,
}


def verify_calibration(report_path: str | None = None) -> dict:
    if report_path is None:
        report_path = os.path.join(os.path.dirname(__file__), "calibration_results.json")
    if not os.path.exists(report_path):
        return {"pass": False, "error": f"Report not found: {report_path}"}
    with open(report_path) as f:
        report = json.load(f)
    results: dict[str, bool | str] = {}
    regimes = report.get("regimes", {})
    for regime, data in regimes.items():
        align = data.get("alignment", 0)
        results[f"{regime}_alignment"] = align >= THRESHOLDS["alignment_min"]
        results[f"{regime}_gate_pass"] = data.get("gate_pass_rate", 0) > 0.5
    summary = report.get("summary", {})
    mean_align = summary.get("mean_alignment", 0)
    total_vetoes = summary.get("total_vetoes", 0)
    results["mean_alignment"] = mean_align
    results["mean_alignment_pass"] = mean_align >= THRESHOLDS["alignment_min"]
    if isinstance(total_vetoes, (int, float)):
        total_cycles_pool = sum(
            r.get("total_cycles", 0) for r in regimes.values()
        )
        veto_rate = total_vetoes / max(1, total_cycles_pool)
        results["veto_rate"] = veto_rate
        results["veto_rate_pass"] = THRESHOLDS["rc_veto_min"] <= veto_rate <= THRESHOLDS["rc_veto_max"]
    all_pass = all(
        v is True for k, v in results.items() if k.endswith("_pass")
    )
    results["overall_pass"] = all_pass
    return results


if __name__ == "__main__":
    result = verify_calibration()
    for k, v in result.items():
        status = "✓" if v is True else ("✗" if v is False else str(v))
        print(f"  {k}: {status}")
    print(f"\n  OVERALL: {'PASS' if result.get('overall_pass') else 'FAIL'}")
