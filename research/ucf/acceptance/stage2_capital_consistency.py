import sys, os, json, random, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from proxima_ops.decision.gate.phase6_rollout_controller import Phase6RolloutController
from proxima_ops.decision.gate.phase6_kill_switch import Phase6KillSwitch
from proxima_ops.decision.gate.phase6_scaling_engine import Phase6ScalingEngine
from proxima_ops.decision.gate.phase6_recovery_protocol import Phase6RecoveryProtocol
from proxima_ops.decision.gate.phase6_audit_logger import Phase6AuditLogger

REGIME_PROFILES = {
    "calm":     {"alignment": 0.70, "rc_veto": 0.06, "mra": 0.60, "emd": 0.10},
    "normal":   {"alignment": 0.60, "rc_veto": 0.10, "mra": 0.50, "emd": 0.18},
    "volatile": {"alignment": 0.50, "rc_veto": 0.15, "mra": 0.40, "emd": 0.28},
    "crisis":   {"alignment": 0.35, "rc_veto": 0.22, "mra": 0.25, "emd": 0.40},
}


def generate_regime_sequence(total_cycles: int) -> list[str]:
    seq = []
    i = 0
    while i < total_cycles:
        regime = random.choices(
            ["calm", "normal", "volatile", "crisis"],
            weights=[0.25, 0.40, 0.25, 0.10],
        )[0]
        duration = random.randint(15, 60)
        for _ in range(min(duration, total_cycles - i)):
            seq.append(regime)
            i += 1
    return seq


def run_capital_consistency(cycles: int, label: str) -> dict:
    rollout = Phase6RolloutController()
    killswitch = Phase6KillSwitch()
    scaling = Phase6ScalingEngine()
    recovery = Phase6RecoveryProtocol()
    audit = Phase6AuditLogger()

    regimes = generate_regime_sequence(cycles)
    results = []

    for cycle in range(cycles):
        reg = regimes[cycle]
        profile = REGIME_PROFILES[reg]
        noise = 0.04 if reg in ("calm", "normal") else 0.08
        metrics = {
            "alignment": max(0.1, min(1.0, profile["alignment"] + random.uniform(-noise, noise))),
            "rc_veto_rate": max(0.0, min(1.0, profile["rc_veto"] + random.uniform(-noise * 0.5, noise * 0.5))),
            "mra_score": max(0.0, min(1.0, profile["mra"] + random.uniform(-noise, noise))),
            "emd_score": max(0.0, min(1.0, profile["emd"] + random.uniform(-noise, noise))),
        }

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
            "cycle": cycle, "state": roll["state"], "regime": reg,
            "multiplier": mult, "ks": ks["triggered"],
            "stability": sc["stability_score"],
            "alignment": metrics["alignment"],
            "rc_veto": metrics["rc_veto_rate"],
            "recovery_phase": pv.get("phase", "NORMAL"),
        })

    states = {}
    regime_map = {}
    for r in results:
        states.setdefault(r["state"], []).append(r["cycle"])
        regime_map.setdefault(r["regime"], {}).setdefault(r["state"], 0)
        regime_map[r["regime"]][r["state"]] += 1

    total = len(results)
    state_pct = {s: len(c) / total for s, c in states.items()}
    transitions = sum(1 for i in range(1, len(results)) if results[i]["state"] != results[i-1]["state"])
    ks_events = sum(1 for r in results if r["ks"])
    avg_mult = sum(r["multiplier"] for r in results) / total
    avg_align = sum(r["alignment"] for r in results) / total

    recovery_events = sum(1 for r in results if r["recovery_phase"] != "NORMAL")
    full_live_cycles = len(states.get("FULL_LIVE", []))
    micro_capital_cycles = len(states.get("MICRO_CAPITAL", []))
    shadow_cycles = len(states.get("SHADOW", []))

    regime_state_pct = {}
    for reg, state_counts in regime_map.items():
        total_reg = sum(state_counts.values())
        regime_state_pct[reg] = {s: c / total_reg for s, c in state_counts.items()}

    return {
        "label": label,
        "total_cycles": total,
        "transitions": transitions,
        "transitions_per_100": round(transitions / max(1, total) * 100, 2),
        "kill_switches": ks_events,
        "ks_rate": round(ks_events / max(1, total), 4),
        "recovery_events": recovery_events,
        "state_distribution_pct": state_pct,
        "avg_multiplier": round(avg_mult, 4),
        "avg_alignment": round(avg_align, 4),
        "regime_state_distribution": regime_state_pct,
        "final_state": results[-1]["state"],
        "final_multiplier": results[-1]["multiplier"],
        "time_in_full_live_pct": round(full_live_cycles / total, 4),
        "time_in_micro_pct": round(micro_capital_cycles / total, 4),
        "time_in_shadow_pct": round(shadow_cycles / total, 4),
    }


def main() -> None:
    print("=" * 60)
    print("PHASE 6 — CAPITAL CONSISTENCY CERTIFICATION")
    print("=" * 60)

    scenarios = [
        (500, "2-week equivalent (500 cycles)"),
        (1000, "1-month equivalent (1000 cycles)"),
        (2000, "2-month equivalent (2000 cycles)"),
    ]

    final_report = {}
    for cycles, label in scenarios:
        print(f"\n  Running: {label}")
        t0 = time.time()
        report = run_capital_consistency(cycles, label)
        elapsed = time.time() - t0
        final_report[label] = report
        print(f"  Done in {elapsed:.1f}s")
        print(f"    State dist:  SHADOW={report['state_distribution_pct'].get('SHADOW',0):.1%} "
              f"MICRO={report['state_distribution_pct'].get('MICRO_CAPITAL',0):.1%} "
              f"FULL={report['state_distribution_pct'].get('FULL_LIVE',0):.1%}")
        print(f"    Transitions: {report['transitions']} ({report['transitions_per_100']}/100)")
        print(f"    KS events:   {report['kill_switches']} (rate={report['ks_rate']:.4f})")
        print(f"    Avg Mult:    {report['avg_multiplier']:.4f}")
        print(f"    Avg Align:   {report['avg_alignment']:.3f}")
        print(f"    Final:       {report['final_state']} mult={report['final_multiplier']}")

    print(f"\n{'=' * 60}")
    print("FINAL READINESS ASSESSMENT")
    print(f"{'=' * 60}")
    for label, r in final_report.items():
        ks_ok = r["ks_rate"] < 0.05
        full_pct = r["time_in_full_live_pct"]
        trans_ok = r["transitions_per_100"] < 5
        recovery_ok = r["recovery_events"] < r["total_cycles"] * 0.1
        score = sum([ks_ok, full_pct > 0.5, trans_ok, recovery_ok, r["avg_alignment"] > 0.55]) / 5
        print(f"  {label}: readiness={score:.0%} {'PASS' if score > 0.6 else 'DEGRADED' if score > 0.3 else 'FAIL'}")
        print(f"    KS stable:      {'PASS' if ks_ok else 'FAIL'} ({r['ks_rate']:.4f})")
        print(f"    FULL_LIVE >50%: {'PASS' if full_pct > 0.5 else 'FAIL'} ({full_pct:.1%})")
        print(f"    Trans stable:   {'PASS' if trans_ok else 'FAIL'} ({r['transitions_per_100']}/100)")
        print(f"    Recovery low:   {'PASS' if recovery_ok else 'FAIL'} ({r['recovery_events']})")
        print(f"    Alignment >.55: {'PASS' if r['avg_alignment'] > 0.55 else 'FAIL'} ({r['avg_alignment']:.3f})")


if __name__ == "__main__":
    main()
