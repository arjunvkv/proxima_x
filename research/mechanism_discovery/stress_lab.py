from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator
from research.mechanism_discovery.base import BaseMechanism, MechanismScore
from research.mechanism_discovery.attacks import AttackSuite
from research.mechanism_discovery.coupling_engine import CouplingEngine


class StressLab:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.attacks = AttackSuite(mi_estimator=self.mi)
        self.coupling = CouplingEngine(mi_estimator=self.mi)
        self.results: dict[str, Any] = {}

    def compute_msi(self, attack_results: dict[str, Any]) -> float:
        total = 0
        passed = 0
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and "passed" in result:
                total += 1
                if result["passed"]:
                    passed += 1
        if total == 0:
            return 0.0
        return passed / total

    def compute_attack_summary(self, attack_results: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {"n_attacks": 0, "n_passed": 0, "n_failed": 0, "msi": 0.0, "per_attack": {}}
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and "passed" in result:
                summary["n_attacks"] += 1
                label = "PASS" if result["passed"] else "FAIL"
                score_key = next((k for k in result if k.endswith("_score") and k != "per_asset" and k != "per_regime" and k != "per_noise_level" and k != "per_perturbation" and k != "per_rate" and k != "per_n" and k != "per_delay" and k != "per_shuffle"), None)
                score = result.get(score_key, 0.0) if score_key else 0.0
                if isinstance(score, dict):
                    score = 0.0
                summary["per_attack"][attack_name] = {"passed": result["passed"], "score": float(score) if isinstance(score, (int, float)) else 0.0}
                if result["passed"]:
                    summary["n_passed"] += 1
                else:
                    summary["n_failed"] += 1
        summary["msi"] = self.compute_msi(attack_results)
        return summary

    def test_mechanism(self, mechanism: BaseMechanism, data: dict, states: NDArray, target: NDArray,
                       all_mechanisms: dict[str, BaseMechanism], test_assets: dict[str, dict] | None = None,
                       test_data: dict | None = None, regime_masks: dict[str, NDArray] | None = None) -> dict[str, Any]:
        attack_results = self.attacks.run_all_attacks(mechanism, data, states, target, all_mechanisms, test_assets, test_data, regime_masks)
        summary = self.compute_attack_summary(attack_results)
        return {"attack_results": attack_results, "summary": summary}

    def run_stress_lab(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray, target: NDArray,
                       test_assets: dict[str, dict] | None = None, test_data: dict | None = None,
                       regime_masks: dict[str, NDArray] | None = None) -> dict[str, Any]:
        mech_results: dict[str, Any] = {}
        for name, mech in mechanisms.items():
            mech.reset()
            test_result = self.test_mechanism(mech, data, states, target, mechanisms, test_assets, test_data, regime_masks)
            mech_results[name] = test_result
        couplings = self.coupling.compute_all_couplings(mechanisms, data, states)
        coupling_info = self.coupling.compute_coupling_information(mechanisms, data, states, target)
        emergence = self.coupling.detect_emergence(mechanisms, data, states, target)
        combined_contrib = self.coupling.compute_combined_contribution(mechanisms, data, states)
        hierarchy = self.coupling.compute_hierarchy(mechanisms, data, states, states, target)
        self.results = {
            "mechanism_results": mech_results,
            "couplings": {(str(k[0]), str(k[1])): v for k, v in couplings.items()},
            "coupling_information": coupling_info,
            "emergence": emergence,
            "combined_contribution": combined_contrib,
            "hierarchy": hierarchy,
        }
        return self.results

    def get_surviving_mechanisms(self, mechanisms: dict[str, BaseMechanism], msi_threshold: float = 0.5) -> dict[str, BaseMechanism]:
        survivors: dict[str, BaseMechanism] = {}
        for name, mech in mechanisms.items():
            mech_result = self.results.get("mechanism_results", {}).get(name, {})
            summary = mech_result.get("summary", {})
            msi = summary.get("msi", 0.0)
            if msi >= msi_threshold:
                survivors[name] = mech
        return survivors

    def get_eliminated_mechanisms(self, mechanisms: dict[str, BaseMechanism], msi_threshold: float = 0.5) -> dict[str, BaseMechanism]:
        eliminated: dict[str, BaseMechanism] = {}
        for name, mech in mechanisms.items():
            mech_result = self.results.get("mechanism_results", {}).get(name, {})
            summary = mech_result.get("summary", {})
            msi = summary.get("msi", 0.0)
            if msi < msi_threshold:
                eliminated[name] = mech
        return eliminated

    def generate_report(self, mechanisms: dict[str, BaseMechanism]) -> str:
        lines: list[str] = []
        lines.append("# PROXIMA X Phase 5 - Mechanism Stress Laboratory Report")
        lines.append("")
        lines.append("## 1. Survivability (MSI)")
        lines.append("")
        lines.append("| Mechanism | MSI | Attacks Passed | Attacks Failed | Status |")
        lines.append("|-----------|-----|----------------|----------------|--------|")
        mech_results = self.results.get("mechanism_results", {})
        survivors: list[str] = []
        eliminated: list[str] = []
        for name in mechanisms:
            r = mech_results.get(name, {})
            s = r.get("summary", {})
            msi = s.get("msi", 0.0)
            n_p = s.get("n_passed", 0)
            n_f = s.get("n_failed", 0)
            status = "SURVIVES" if msi >= 0.5 else "ELIMINATED"
            if status == "SURVIVES":
                survivors.append(name)
            else:
                eliminated.append(name)
            lines.append(f"| {name} | {msi:.3f} | {n_p} | {n_f} | {status} |")
        lines.append("")
        lines.append(f"**Surviving ({len(survivors)})**: {', '.join(survivors)}")
        lines.append(f"**Eliminated ({len(eliminated)})**: {', '.join(eliminated)}")
        lines.append("")
        lines.append("## 2. Attack Details")
        lines.append("")
        for name in mechanisms:
            r = mech_results.get(name, {})
            attacks = r.get("attack_results", {})
            s = r.get("summary", {})
            lines.append(f"### {name}")
            lines.append(f"- **MSI**: {s.get('msi', 0):.3f}")
            lines.append(f"- **Passed**: {s.get('n_passed', 0)}/{s.get('n_attacks', 0)}")
            lines.append("")
            lines.append("| Attack | Result | Score |")
            lines.append("|--------|--------|-------|")
            for aname, aresult in sorted(attacks.items()):
                if isinstance(aresult, dict) and "passed" in aresult:
                    label = "PASS" if aresult["passed"] else "FAIL"
                    score_key = next((k for k in aresult if k.endswith("_score") and isinstance(aresult[k], (int, float))), None)
                    score = aresult.get(score_key, 0.0) if score_key else ""
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else ""
                    lines.append(f"| {aname} | {label} | {score_str} |")
            lines.append("")
        lines.append("## 3. Mechanism Coupling")
        lines.append("")
        couplings = self.results.get("couplings", {})
        if couplings:
            lines.append("| Pair | Correlation | MI |")
            lines.append("|------|-------------|-----|")
            for pair, cinfo in couplings.items():
                lines.append(f"| {pair[0]} + {pair[1]} | {cinfo.get('coupling_corr', 0):.4f} | {cinfo.get('coupling_mi', 0):.4f} |")
        lines.append("")
        ci = self.results.get("coupling_information", {})
        lines.append("### Combined Coupling Information")
        lines.append(f"- **Information Gain**: {ci.get('coupling_information_gain', 0):.4f}")
        lines.append(f"- **Coupling SID**: {ci.get('coupling_sid', 0):.4f}")
        lines.append(f"- **Coupling SIR**: {ci.get('coupling_sir', 0):.4f}")
        lines.append("")
        lines.append("## 4. Emergence Detection")
        lines.append("")
        em = self.results.get("emergence", {})
        lines.append(f"- **Combined MI**: {em.get('combined_mi', 0):.4f}")
        lines.append(f"- **Max Individual MI**: {em.get('max_individual_mi', 0):.4f}")
        lines.append(f"- **Emergent Information Gain**: {em.get('emergent_information_gain', 0):.4f}")
        lines.append(f"- **Emergence Detected**: {str(em.get('emergence_detected', False))}")
        lines.append("")
        lines.append("## 5. Hierarchy Analysis")
        lines.append("")
        hi = self.results.get("hierarchy", {})
        m2s = hi.get("mechanism_to_state", {})
        if m2s:
            lines.append("| Mechanism -> State MI |")
            lines.append("|----------------------|")
            for mname, mi_val in sorted(m2s.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {mname}: {mi_val:.4f} |")
        lines.append("")
        lines.append("## 6. Elimination Summary")
        lines.append("")
        lines.append(f"**Total mechanisms**: {len(mechanisms)}")
        lines.append(f"**Surviving**: {len(survivors)}")
        lines.append(f"**Eliminated**: {len(eliminated)}")
        if eliminated:
            lines.append("### Eliminated Mechanisms")
            for ename in eliminated:
                r = mech_results.get(ename, {})
                s = r.get("summary", {})
                per_attack = s.get("per_attack", {})
                failures = [an for an, av in per_attack.items() if not av.get("passed", False)]
                lines.append(f"- **{ename}**: MSI={s.get('msi', 0):.3f}, failed: {', '.join(failures)}")
        lines.append("")
        lines.append(f"**Objective achieved**: {'YES' if len(survivors) < len(mechanisms) else 'NO  -  no mechanisms eliminated'}  -  wait for real data")
        lines.append("")
        return "\n".join(lines)
