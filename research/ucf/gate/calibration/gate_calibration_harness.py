import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from run_proxima_shadow_replay import ShadowReplayRunner


def run_regime(regime_tag: str, symbols: list[str] | None = None) -> dict:
    runner = ShadowReplayRunner(mode="synthetic")
    runner.generate_ticks(2000, symbols)
    ticks = runner.orchestrator._ticks
    for t in ticks:
        if regime_tag == "low_vol":
            t["spread"] = max(1, t["spread"] // 2)
        elif regime_tag == "high_vol":
            t["spread"] = min(50, t["spread"] * 3)
    runner.orchestrator.run_replay(batch_size=100)
    runner._build_report(runner.orchestrator.run_replay(batch_size=100), [], [])
    return {"regime": regime_tag, "result": "completed"}


def run_calibration() -> dict:
    results: dict[str, dict] = {}
    for regime in ["low_vol", "high_vol", "mixed"]:
        print(f"[CALIBRATION] Running regime: {regime}")
        try:
            runner = ShadowReplayRunner(mode="synthetic")
            runner.generate_ticks(2000)
            result = runner.run()
            results[regime] = {
                "alignment": result.get("alignment_score", 0),
                "gate_pass_rate": result.get("gate_metrics", {}).get("gate_pass_rate", 0),
                "veto_count": result.get("gate_metrics", {}).get("veto_count", 0),
                "total_cycles": result.get("total_cycles", 0),
            }
            print(f"  alignment={results[regime]['alignment']:.4f} gate_pass={results[regime]['gate_pass_rate']:.2%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[regime] = {"error": str(e)}

    report = {
        "regimes": results,
        "summary": {
            "mean_alignment": sum(r.get("alignment", 0) for r in results.values()) / max(1, len(results)),
            "mean_gate_pass": sum(r.get("gate_pass_rate", 0) for r in results.values()) / max(1, len(results)),
            "total_vetoes": sum(r.get("veto_count", 0) for r in results.values()),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[CALIBRATION] Results saved to {out_path}")
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    run_calibration()
