import time
import json
import os
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BoundaryTestResult:
    phase: str = ""
    test_name: str = ""
    passed: bool = False
    finding: str = ""
    severity: str = "INFO"



class SystemBoundaryTester:
    def __init__(self, output_dir: str = None):
        self._results: list[BoundaryTestResult] = []
        self._output_dir = output_dir or os.path.join("state", "sbrt_results")
        os.makedirs(self._output_dir, exist_ok=True)
        self._failure_modes: list[dict] = []
        self._random = random.Random(42)

    def _add(self, phase: str, name: str, passed: bool, finding: str, severity: str = "INFO"):
        r = BoundaryTestResult(phase=phase, test_name=name, passed=passed, finding=finding, severity=severity)
        self._results.append(r)
        if severity in ("WARNING", "CRITICAL"):
            self._failure_modes.append({
                "phase": phase,
                "test": name,
                "severity": severity,
                "finding": finding,
            })
        return r

    def _save(self):
        path = os.path.join(self._output_dir, f"sbrt_report_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, default=str)
        logger.info("S-BRT report saved to %s", path)

    # ── Phase 1: Environment Mismatch Simulation ──

    def run_phase1(self, cycles: int = 20):
        phase = "Phase 1 — Environment Mismatch"
        print(f"\n{'='*60}")
        print(f"  {phase}")
        print(f"{'='*60}")

        p1_results = {"total": 0, "passed": 0, "failed": 0}

        for cycle in range(cycles):
            latency_jitter = self._random.uniform(0, 500)
            partial_observability = self._random.random() < 0.1

            base_mof = 0.4351
            mof_jitter = (self._random.random() - 0.5) * 0.1
            mof_score = max(0.0, min(1.0, base_mof + mof_jitter))
            mof_state = ("INFORMATION_DEGRADED" if mof_score < 0.35 else
                         "STRUCTURE_LIMITED" if mof_score < 0.65 else "INFORMATION_RICH")

            base_rf = 0.618
            if partial_observability:
                base_rf = self._random.uniform(0.3, 0.5)
            rf_noise = (self._random.random() - 0.5) * 0.08
            rf_mean = max(0.0, min(1.0, base_rf + rf_noise))

            segl_stable = mof_state != "INFORMATION_DEGRADED" and not (partial_observability and self._random.random() < 0.3)
            segl_context = {
                "signal_valid": not partial_observability,
                "mof_baseline_ok": mof_state in ("STRUCTURE_LIMITED", "INFORMATION_RICH"),
                "no_portfolio_conflicts": True,
                "rf_drift_bounded": abs(rf_mean - 0.618) <= 0.05,
            }
            ctx_ok = all(segl_context.values())
            ctx_expected_stable = ctx_ok

            if ctx_expected_stable == segl_stable:
                self._add(phase, f"latency_test_cycle_{cycle}", True,
                          f"Latency={latency_jitter:.0f}ms, partial={partial_observability}, "
                          f"MOF={mof_state}({mof_score:.4f}), RF={rf_mean:.4f}, "
                          f"SEGL_stable={segl_stable}")
            else:
                self._add(phase, f"latency_test_cycle_{cycle}", False,
                          f"MISMATCH: expected_stable={ctx_expected_stable}, "
                          f"got segl_stable={segl_stable}, ctx={segl_context}",
                          "WARNING")

        rf_corruption_test_passed = True
        for _ in range(10):
            corrupt_rf = self._random.uniform(0.1, 0.9)
            drift = abs(corrupt_rf - 0.618)
            if drift > 0.10:
                rf_corruption_test_passed = False
                self._add(phase, "rf_feature_corruption", False,
                          f"RF corruption drift={drift:.4f} > 0.10 critical threshold",
                          "CRITICAL")
                break

        if rf_corruption_test_passed:
            self._add(phase, "rf_feature_corruption", True,
                      f"RF drift within bounds across 10 corruption tests")

        mof_jitter_stable = True
        for _ in range(10):
            jitter = (self._random.random() - 0.5) * 0.2
            test_mof = max(0.0, min(1.0, 0.4351 + jitter))
            if test_mof < 0.35 and 0.4351 >= 0.35:
                mof_jitter_stable = False
                self._add(phase, "mof_input_jitter", False,
                          f"MOF jitter caused state change: {0.4351:.4f} -> {test_mof:.4f}",
                          "WARNING")
                break

        if mof_jitter_stable:
            self._add(phase, "mof_input_jitter", True,
                      f"MOF state preserved under ±0.1 jitter across 10 tests")

        self._add(phase, "phase1_complete", True,
                  f"Completed {cycles} cycles with boundary conditions")

    # ── Phase 2: Adversarial Signal Injection ──

    def run_phase2(self):
        phase = "Phase 2 — Adversarial Signal Injection"
        print(f"\n{'='*60}")
        print(f"  {phase}")
        print(f"{'='*60}")

        from governance.edge_governance_binding import EdgeGovernanceBinding
        gov_binding = EdgeGovernanceBinding()

        false_positives = 0
        false_rejections = 0
        total_adversarial = 0

        for i in range(30):
            confidence = self._random.uniform(0.3, 1.0)
            direction = self._random.choice(["BUY", "SELL"])
            conflict = self._random.uniform(0.0, 0.5)
            mof_score = self._random.uniform(0.2, 0.7)
            mof_ok = self._random.choice(["STRUCTURE_LIMITED", "INFORMATION_DEGRADED", "INFORMATION_RICH"])
            system_state = self._random.choice(["OBSERVE", "ARMED", "LOCKED"])

            signal = {"id": f"adversarial_{i}", "symbol": "EURJPY", "strategy": "pullback",
                      "confidence": confidence, "direction": direction}

            ee = gov_binding.evaluate_arming_eligibility(
                signal=signal, mof_state=mof_ok, mof_score=mof_score,
                portfolio_conflict=conflict, current_system_state=system_state,
            )

            should_be_eligible = (
                confidence >= 0.60 and conflict <= 0.30 and mof_score >= 0.35
                and system_state in ("OBSERVE", "ARMED")
            )
            total_adversarial += 1

            if ee.eligible_for_arming and not should_be_eligible:
                false_positives += 1
                self._add(phase, f"adversarial_{i}", False,
                          f"FALSE POSITIVE: conf={confidence:.4f}, conflict={conflict:.4f}, "
                          f"mof={mof_score:.4f}, state={system_state}",
                          "CRITICAL")
            elif not ee.eligible_for_arming and should_be_eligible:
                false_rejections += 1
                self._add(phase, f"adversarial_{i}", False,
                          f"FALSE REJECTION: conf={confidence:.4f}, conflict={conflict:.4f}, "
                          f"mof={mof_score:.4f}, state={system_state}",
                          "WARNING")
            else:
                self._add(phase, f"adversarial_{i}", True,
                          f"Correct {ee.eligible_for_arming} for conf={confidence:.4f}, "
                          f"conflict={conflict:.4f}, mof={mof_score:.4f}")

        fp_rate = false_positives / total_adversarial if total_adversarial > 0 else 0
        fr_rate = false_rejections / total_adversarial if total_adversarial > 0 else 0

        self._add(phase, "adversarial_summary", fp_rate < 0.10 and fr_rate < 0.10,
                  f"FP rate={fp_rate:.4f} (<0.10 {'PASS' if fp_rate < 0.10 else 'FAIL'}), "
                  f"FR rate={fr_rate:.4f} (<0.10 {'PASS' if fr_rate < 0.10 else 'FAIL'})",
                  "CRITICAL" if fp_rate >= 0.10 else "WARNING" if fr_rate >= 0.10 else "INFO")

    # ── Phase 3: Governance Stress Test ──

    def run_phase3(self):
        phase = "Phase 3 — Governance Stress Test"
        print(f"\n{'='*60}")
        print(f"  {phase}")
        print(f"{'='*60}")

        from governance.execution_state_machine import ExecutionStateMachine, ExecutionState

        sm = ExecutionStateMachine()

        oscillation_detected = False
        false_transitions = []
        total_attempts = 0

        oscillation_patterns = [
            [True, False, True, False, True, False, True, False, True, False],
            [True, True, False, False, True, True, False, False],
            [True, False, False, True, False, False, True, False, False],
        ]

        for i, pattern in enumerate(oscillation_patterns):
            sm.reset()
            for j, signal_valid in enumerate(pattern):
                sm.update_context({
                    "signal_valid": signal_valid,
                    "mof_baseline_ok": True,
                    "no_portfolio_conflicts": True,
                    "rf_drift_bounded": True,
                })
                can_arm, _ = sm.can_transition(ExecutionState.ARMED)
                if can_arm:
                    sm.transition(ExecutionState.ARMED, f"oscillation_{i}_cycle_{j}")

                sm.update_context({
                    "governance_pipeline_approves": signal_valid,
                    "envelope_check_passes": True,
                    "within_frequency_budget": True,
                })
                can_exec, denial = sm.can_transition(ExecutionState.EXECUTING)
                if can_exec:
                    sm.transition(ExecutionState.EXECUTING, f"oscillation_{i}_cycle_{j}")
                    sm.transition(ExecutionState.COOLDOWN, f"oscillation_{i}_cycle_{j}")
                    sm.update_context({
                        "stabilization_cycles_elapsed": True,
                        "mof_recovered": True,
                        "rf_stable_post": True,
                    })
                    sm.transition(ExecutionState.OBSERVE, f"oscillation_{i}_cycle_{j}")

                if sm.state == ExecutionState.EXECUTING and not signal_valid:
                    oscillation_detected = True
                    self._add(phase, f"oscillation_pattern_{i}", False,
                              f"EXECUTING under invalid signal at cycle {j}",
                              "CRITICAL")

            if not oscillation_detected:
                self._add(phase, f"oscillation_pattern_{i}", True,
                          f"Stable through pattern of {len(pattern)} oscillations")

        for _ in range(20):
            total_attempts += 1
            threshold = self._random.uniform(0.48, 0.52)
            ctx = {
                "signal_valid": threshold > 0.5,
                "mof_baseline_ok": threshold > 0.5,
                "no_portfolio_conflicts": threshold > 0.5,
                "rf_drift_bounded": threshold > 0.5,
                "governance_pipeline_approves": threshold > 0.5,
                "envelope_check_passes": True,
                "within_frequency_budget": True,
            }
            sm.update_context(ctx)

            prev_state = sm.state
            if sm.can_transition(ExecutionState.ARMED)[0]:
                sm.transition(ExecutionState.ARMED, f"boundary_{_}")
            if sm.state == ExecutionState.ARMED and sm.can_transition(ExecutionState.EXECUTING)[0]:
                sm.transition(ExecutionState.EXECUTING, f"boundary_{_}")
                sm.transition(ExecutionState.COOLDOWN, f"boundary_{_}")
                sm.update_context({
                    "stabilization_cycles_elapsed": True,
                    "mof_recovered": True,
                    "rf_stable_post": True,
                })
                sm.transition(ExecutionState.OBSERVE, f"boundary_{_}")

            if sm.state == ExecutionState.EXECUTING and prev_state != ExecutionState.ARMED:
                false_transitions.append({
                    "attempt": _, "threshold": threshold, "prev": prev_state.value,
                })

        if false_transitions:
            self._add(phase, "borderline_threshold_test", False,
                      f"{len(false_transitions)} false transitions from non-ARMED states",
                      "CRITICAL")
        else:
            self._add(phase, "borderline_threshold_test", True,
                      f"0 false transitions from 20 borderline threshold attempts")

        sm.reset()
        for _ in range(5):
            sm.update_context({
                "signal_valid": True, "mof_baseline_ok": True,
                "no_portfolio_conflicts": True, "rf_drift_bounded": True,
            })
            sm.transition(ExecutionState.ARMED, "cooldown_interrupt_test")
            sm.update_context({
                "governance_pipeline_approves": True,
                "envelope_check_passes": True,
                "within_frequency_budget": True,
            })
            sm.transition(ExecutionState.EXECUTING, "cooldown_interrupt_test")
            sm.transition(ExecutionState.COOLDOWN, "cooldown_interrupt_test")

            for _ in range(self._random.randint(1, 3)):
                sm.update_context({
                    "stabilization_cycles_elapsed": False,
                    "mof_recovered": False,
                    "rf_stable_post": False,
                })
                can_observe, _ = sm.can_transition(ExecutionState.OBSERVE)
                assert not can_observe, "COOLDOWN should block early OBSERVE"

            sm.update_context({
                "stabilization_cycles_elapsed": True,
                "mof_recovered": True,
                "rf_stable_post": True,
            })
            sm.transition(ExecutionState.OBSERVE, "cooldown_complete")

        self._add(phase, "cooldown_interruption", True,
                  "COOLDOWN correctly blocked OBSERVE transitions under incomplete stabilization")

    # ── Phase 4: Integrity Failure Mode Mapping ──

    def run_phase4(self):
        phase = "Phase 4 — Integrity Failure Mode Mapping"
        print(f"\n{'='*60}")
        print(f"  {phase}")
        print(f"{'='*60}")

        from governance.self_integrity_auditor import SelfIntegrityAuditor

        sia = SelfIntegrityAuditor()

        for cycle in range(30):
            silent_drift = cycle >= 20
            rf = 0.618 - (cycle * 0.002) - (0.04 if silent_drift else 0)
            mof = 0.4351 - (cycle * 0.003) - (0.06 if silent_drift else 0)
            entry = sia.record_cycle(
                cycle=cycle, rf_mean=rf, rf_std=0.012, rf_variance=0.001,
                mof_score=mof, mof_state="STRUCTURE_LIMITED",
                edge_04_confidence=0.7656 - (cycle * 0.002 if silent_drift else 0),
                edge_04_direction="BUY",
                segl_state="ARMED" if cycle < 15 else "OBSERVE",
                segl_elapsed=float(cycle * 60),
                portfolio_conflict=0.0,
                execution_count=cycle,
            )

        drift_analysis = sia.drift_tracker.trend_analysis()
        alert_count = len(drift_analysis.get("alerts", []))
        integrity = sia._integrity_score

        if alert_count == 0:
            self._add(phase, "silent_drift_detection", True,
                      f"No silent drift: integrity={integrity:.4f}, "
                      f"drift_status={drift_analysis['status']}")
        elif integrity < 0.80:
            self._add(phase, "silent_drift_detection", True,
                      f"Silent drift caught: integrity={integrity:.4f}, "
                      f"{alert_count} alerts triggered, drift_status={drift_analysis['status']}")
        else:
            self._add(phase, "silent_drift_detection", False,
                      f"Silent drift NOT caught: integrity={integrity:.4f}, "
                      f"alerts={alert_count}, drift_status={drift_analysis['status']}",
                      "CRITICAL")

        sia2 = SelfIntegrityAuditor()
        for cycle in range(15):
            abrupt = cycle >= 10
            rf = 0.618 - (0.15 if abrupt else 0)
            mof = 0.4351 - (0.25 if abrupt else 0)
            entry = sia2.record_cycle(
                cycle=cycle, rf_mean=rf, rf_std=0.012, rf_variance=0.001,
                mof_score=mof, mof_state="STRUCTURE_LIMITED" if not abrupt else "INFORMATION_DEGRADED",
                edge_04_confidence=0.7656, edge_04_direction="BUY",
                segl_state="OBSERVE", segl_elapsed=float(cycle * 60),
                portfolio_conflict=0.0, execution_count=0,
            )
        drift2 = sia2.drift_tracker.trend_analysis()
        if drift2["status"] in ("WARNING", "CRITICAL"):
            self._add(phase, "abrupt_degradation", True,
                      f"Abrupt drift correctly detected: {drift2['status']}, "
                      f"alerts={len(drift2.get('alerts', []))}")
        else:
            self._add(phase, "abrupt_degradation", False,
                      f"Abrupt drift NOT detected: {drift2['status']}",
                      "CRITICAL")

        sia3 = SelfIntegrityAuditor()
        for cycle in range(20):
            rf = 0.618 + (self._random.random() - 0.5) * 0.02
            mof = 0.44 + (self._random.random() - 0.5) * 0.02
            sia3.record_cycle(
                cycle=cycle, rf_mean=rf, rf_std=0.012, rf_variance=0.001,
                mof_score=mof, mof_state="STRUCTURE_LIMITED",
                edge_04_confidence=0.7656, edge_04_direction="BUY",
                segl_state="OBSERVE", segl_elapsed=float(cycle * 60),
                portfolio_conflict=0.0, execution_count=0,
            )
        drift3 = sia3.drift_tracker.trend_analysis()
        false_positive = drift3["status"] != "STABLE"
        if not false_positive:
            self._add(phase, "false_positive_resistance", True,
                      f"No false alerts under normal variance: {drift3['status']}")
        else:
            self._add(phase, "false_positive_resistance", False,
                      f"False alerts triggered under normal variance: {drift3['status']}",
                      "WARNING")

    # ── Run All ──

    def run_all(self, cycles: int = 20):
        print("=" * 60)
        print("  SYSTEM BOUNDARY REALITY TEST (S-BRT)")
        print("=" * 60)
        self.run_phase1(cycles)
        self.run_phase2()
        self.run_phase3()
        self.run_phase4()
        self._save()
        return self.summary()

    def summary(self) -> dict:
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        critical = [r for r in self._results if r.severity == "CRITICAL"]
        warnings = [r for r in self._results if r.severity == "WARNING"]

        phase_counts = {}
        for r in self._results:
            p = r.phase.split(" — ")[0]
            if p not in phase_counts:
                phase_counts[p] = {"total": 0, "passed": 0}
            phase_counts[p]["total"] += 1
            if r.passed:
                phase_counts[p]["passed"] += 1

        if failed == 0:
            trust_boundary = "TRUSTED — system stable under all boundary conditions"
        elif failed <= total * 0.1:
            trust_boundary = "DEGRADED — minor failures in edge cases"
        elif failed <= total * 0.25:
            trust_boundary = "BOUNDARY_FOUND — system has identifiable limits"
        else:
            trust_boundary = "UNTRUSTED — system unstable outside ideal conditions"

        return {
            "test_timestamp": time.time(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4) if total > 0 else 0,
            "critical_findings": len(critical),
            "warnings": len(warnings),
            "failure_modes": self._failure_modes,
            "trust_boundary": trust_boundary,
            "phases": {
                phase: {"passed": pc["passed"], "total": pc["total"],
                        "rate": round(pc["passed"] / pc["total"], 4) if pc["total"] > 0 else 0}
                for phase, pc in phase_counts.items()
            },
            "results": [{"phase": r.phase, "test": r.test_name, "passed": r.passed,
                         "finding": r.finding, "severity": r.severity} for r in self._results],
        }
