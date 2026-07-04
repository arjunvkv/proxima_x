import sys, os, json, random, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from proxima_ops.decision.gate.phase6_rollout_controller import Phase6RolloutController
from proxima_ops.decision.gate.phase6_kill_switch import Phase6KillSwitch
from proxima_ops.decision.gate.phase6_scaling_engine import Phase6ScalingEngine
from proxima_ops.decision.gate.phase6_recovery_protocol import Phase6RecoveryProtocol
from proxima_ops.decision.gate.phase6_audit_logger import Phase6AuditLogger


def simulate_phase(cycles: int, label: str, base_metrics: dict, noise: float = 0.05,
                   degrade_at: int | None = None) -> tuple[list[dict], dict]:
    rollout = Phase6RolloutController()
    killswitch = Phase6KillSwitch()
    scaling = Phase6ScalingEngine()
    recovery = Phase6RecoveryProtocol()
    audit = Phase6AuditLogger()

    results = []
    for cycle in range(cycles):
        metrics = {k: v + random.uniform(-noise, noise) for k, v in base_metrics.items()}
        metrics["alignment"] = max(0.1, min(1.0, metrics["alignment"]))
        metrics["rc_veto_rate"] = max(0.0, min(1.0, metrics["rc_veto_rate"]))
        metrics["mra_score"] = max(0.0, min(1.0, metrics["mra_score"]))
        metrics["emd_score"] = max(0.0, min(1.0, metrics["emd_score"]))

        if degrade_at is not None and cycle >= degrade_at:
            metrics["alignment"] = max(0.1, metrics["alignment"] - 0.4 * (1 - math.exp(-(cycle - degrade_at) / 10)))
            metrics["rc_veto_rate"] = min(1.0, metrics["rc_veto_rate"] + 0.3 * (1 - math.exp(-(cycle - degrade_at) / 10)))

        ks = killswitch.evaluate(metrics)
        if ks["triggered"]:
            recovery.trigger(cycle)
            rollout.force_state("SHADOW")
            audit.log_kill_switch(metrics, "; ".join(ks.get("failures", [])))
            scaling = Phase6ScalingEngine()

        roll = rollout.evaluate(metrics)
        if roll.get("transition"):
            audit.log_transition(roll.get("from_state", "SHADOW"), roll["state"], metrics, roll.get("reason", "transition"))

        sc = scaling.evaluate(metrics["alignment"], metrics["rc_veto_rate"], metrics["emd_score"])
        mult = sc["position_size_multiplier"]
        pv = recovery.evaluate(cycle, metrics["alignment"], metrics["rc_veto_rate"])
        if pv.get("active"):
            mult = min(mult, pv.get("max_exposure", 1.0))

        results.append({
            "cycle": cycle, "state": roll["state"],
            "multiplier": mult, "ks": ks["triggered"],
            "stability": sc["stability_score"],
            "tier": sc["stability_tier"],
            "recovery_phase": pv.get("phase", "NORMAL"),
            "alignment": metrics["alignment"],
            "rc_veto": metrics["rc_veto_rate"],
        })

    return results, audit.get_summary()


def generate_report(results: list[dict], label: str) -> dict:
    states = {}
    for r in results:
        states.setdefault(r["state"], []).append(r["cycle"])

    state_time = {s: len(c) for s, c in states.items()}
    total = len(results)
    state_pct = {s: c / total for s, c in state_time.items()}

    transitions = sum(1 for i in range(1, len(results)) if results[i]["state"] != results[i-1]["state"])
    ks_events = sum(1 for r in results if r["ks"])
    avg_mulitplier = sum(r["multiplier"] for r in results) / total
    avg_alignment = sum(r["alignment"] for r in results) / total
    avg_rc = sum(r["rc_veto"] for r in results) / total

    return {
        "label": label,
        "total_cycles": total,
        "transitions": transitions,
        "kill_switches": ks_events,
        "ks_rate": ks_events / max(1, total),
        "state_distribution_pct": state_pct,
        "avg_multiplier": round(avg_mulitplier, 4),
        "avg_alignment": round(avg_alignment, 4),
        "avg_rc_veto": round(avg_rc, 4),
        "final_state": results[-1]["state"],
        "final_multiplier": results[-1]["multiplier"],
    }


def run_long_horizon() -> dict:
    scenarios = {
        "stable": {
            "cycles": 300, "label": "Stable Market (300 cycles)",
            "base": {"alignment": 0.65, "rc_veto_rate": 0.08, "mra_score": 0.55, "emd_score": 0.15},
            "noise": 0.05, "degrade_at": None,
        },
        "oscillating": {
            "cycles": 200, "label": "Oscillating Quality (200 cycles)",
            "base": {"alignment": 0.55, "rc_veto_rate": 0.12, "mra_score": 0.45, "emd_score": 0.20},
            "noise": 0.12, "degrade_at": None,
        },
        "late_degradation": {
            "cycles": 150, "label": "Late Degradation at 100 (150 cycles)",
            "base": {"alignment": 0.60, "rc_veto_rate": 0.10, "mra_score": 0.50, "emd_score": 0.18},
            "noise": 0.05, "degrade_at": 100,
        },
    }

    reports = {}
    for name, params in scenarios.items():
        print(f"\n  Running: {params['label']}")
        t0 = time.time()
        results, audit = simulate_phase(
            params["cycles"], params["label"], params["base"],
            params["noise"], params["degrade_at"],
        )
        elapsed = time.time() - t0
        report = generate_report(results, params["label"])
        reports[name] = report
        print(f"  Done in {elapsed:.1f}s — final state={report['final_state']} "
              f"trans={report['transitions']} ks={report['kill_switches']} "
              f"mult={report['final_multiplier']:.2f} avg_align={report['avg_alignment']:.3f}")

    return reports


if __name__ == "__main__":
    print("=" * 60)
    print("LONG-HORIZON STABILITY VALIDATION")
    print("=" * 60)
    reports = run_long_horizon()
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, r in reports.items():
        print(f"\n  {r['label']}:")
        print(f"    Cycles: {r['total_cycles']}")
        print(f"    State dist: {r['state_distribution_pct']}")
        print(f"    Avg Multiplier: {r['avg_multiplier']:.4f}")
        print(f"    Avg Alignment:  {r['avg_alignment']:.3f}")
        print(f"    Avg RC Veto:    {r['avg_rc_veto']:.3f}")
        print(f"    Transitions:    {r['transitions']}")
        print(f"    Kill Switches:  {r['kill_switches']} (rate={r['ks_rate']:.4f})")
        print(f"    Final:          state={r['final_state']} mult={r['final_multiplier']}")
