import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from proxima_ops.decision.gate.phase6_rollout_controller import Phase6RolloutController
from proxima_ops.decision.gate.phase6_kill_switch import Phase6KillSwitch
from proxima_ops.decision.gate.phase6_scaling_engine import Phase6ScalingEngine
from proxima_ops.decision.gate.phase6_recovery_protocol import Phase6RecoveryProtocol
from proxima_ops.decision.gate.phase6_audit_logger import Phase6AuditLogger


def run_stage2_validation(num_cycles: int = 30) -> dict:
    rollout = Phase6RolloutController()
    killswitch = Phase6KillSwitch()
    scaling = Phase6ScalingEngine()
    recovery = Phase6RecoveryProtocol()
    audit = Phase6AuditLogger()

    results = []

    for cycle in range(num_cycles):
        stable_metrics = {
            "alignment": 0.65,
            "rc_veto_rate": 0.08,
            "mra_score": 0.55,
            "emd_score": 0.15,
        }
        degraded_metrics = {
            "alignment": 0.30,
            "rc_veto_rate": 0.35,
            "mra_score": 0.15,
            "emd_score": 0.55,
        }
        metrics = stable_metrics if cycle < 25 else degraded_metrics

        ks = killswitch.evaluate(metrics)
        if ks["triggered"]:
            recovery.trigger(cycle)
            rollout.force_state("SHADOW")
            audit.log_kill_switch(metrics, "; ".join(ks.get("failures", [])))

        roll = rollout.evaluate(metrics)
        if roll.get("transition"):
            audit.log_transition(
                roll.get("from_state", "SHADOW"), roll["state"], metrics,
                roll.get("reason", "state_change"),
            )

        sc = scaling.evaluate(metrics["alignment"], metrics["rc_veto_rate"], metrics["emd_score"])
        current_mult = sc["position_size_multiplier"]

        pv = recovery.evaluate(cycle, metrics["alignment"], metrics["rc_veto_rate"])
        if pv.get("active"):
            current_mult = min(current_mult, pv.get("max_exposure", 1.0))

        audit.log(f"STAGE2_CYCLE", roll["state"], metrics, {
            "multiplier": current_mult,
            "stability_score": sc["stability_score"],
            "stability_tier": sc["stability_tier"],
        })

        results.append({
            "cycle": cycle,
            "state": roll["state"],
            "ks_triggered": ks["triggered"],
            "multiplier": current_mult,
            "stability_score": sc["stability_score"],
            "tier": sc["stability_tier"],
            "recovery_phase": pv.get("phase", "NORMAL"),
            "transition": roll.get("transition", False),
        })

    transitions = [r for r in results if r["transition"]]
    return {
        "total_cycles": num_cycles,
        "final_state": results[-1]["state"],
        "transitions": transitions,
        "timeline": results,
        "audit_summary": audit.get_summary(),
    }


if __name__ == "__main__":
    print("Stage 2 Validation — 30 cycles")
    print("=" * 60)
    report = run_stage2_validation(30)
    print(f"Final state: {report['final_state']}")
    print(f"Transitions: {len(report['transitions'])}")
    for t in report["transitions"]:
        print(f"  Cycle {t['cycle']}: -> {t['state']}")
    print(f"\nAudit: {report['audit_summary']}")
    print(f"\nTimeline:")
    for r in report["timeline"]:
        ks_mark = " KS!" if r["ks_triggered"] else ""
        tr_mark = " <-TRANSITION" if r.get("transition") else ""
        print(f"  Cycle {r['cycle']:2d}: state={r['state']:15s} mult={r['multiplier']:.2f} score={r['stability_score']:.3f}{ks_mark}{tr_mark}")

    if report["final_state"] == "SHADOW" and report["timeline"][-1]["multiplier"] == 0.0:
        print(f"\n{'=' * 60}")
        print("STAGE 2 VALIDATION PASSED")
        print("  Rollback to SHADOW + multiplier=0.0 after degradation confirmed")
    else:
        print(f"\n{'=' * 60}")
        print(f"STAGE 2 VALIDATION — State={report['final_state']} Mult={report['timeline'][-1]['multiplier']}")
