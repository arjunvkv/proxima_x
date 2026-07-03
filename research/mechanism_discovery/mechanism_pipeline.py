from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore
from research.mechanism_discovery.propagation_network import PropagationNetwork
from research.mechanism_discovery.participant_ecology import ParticipantEcology
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.state_lifecycle import StateLifecycle
from research.mechanism_discovery.memory_landscape import MemoryLandscape
from research.mechanism_discovery.liquidity_migration_v2 import LiquidityMigrationSystem
from research.mechanism_discovery.mechanism_validator import MechanismValidator
from research.mechanism_discovery.mechanism_scorer import MechanismScorer
from research.information_discovery.mi_estimator import MIEstimator
from research.information_discovery.sid_sir import SIDCalculator, SIRCalculator


class MechanismPipeline:

    def __init__(self) -> None:
        self.mechanisms: dict[str, BaseMechanism] = {
            "propagation_network": PropagationNetwork(),
            "participant_ecology": ParticipantEcology(),
            "temporal_topology": TemporalTopology(),
            "energy_dynamics": EnergyDynamics(),
            "state_lifecycle": StateLifecycle(),
            "memory_landscape": MemoryLandscape(),
            "liquidity_migration": LiquidityMigrationSystem(),
        }
        self.validator = MechanismValidator()
        self.scorer = MechanismScorer()
        self.mi_estimator = MIEstimator()
        self.sid_calc = SIDCalculator(mi_estimator=self.mi_estimator)
        self.sir_calc = SIRCalculator(sid_calculator=self.sid_calc)
        self.results: dict[str, Any] = {}

    def _is_multi_asset(self, data: dict) -> bool:
        return "assets" in data and isinstance(data["assets"], dict)

    def _extract_single_asset(self, data: dict) -> dict[str, NDArray]:
        if self._is_multi_asset(data):
            asset_ids = list(data["assets"].keys())
            if not asset_ids:
                return {}
            first = data["assets"][asset_ids[0]]
            return {k: np.asarray(v, dtype=np.float64) for k, v in first.items() if isinstance(v, (list, np.ndarray))}
        return {k: np.asarray(v, dtype=np.float64) for k, v in data.items() if isinstance(v, (list, np.ndarray))}

    def _extract_states(self, data: dict, asset_id: Optional[str] = None) -> Optional[NDArray]:
        if self._is_multi_asset(data):
            states_dict = data.get("states", {})
            if asset_id is not None and asset_id in states_dict:
                arr = states_dict[asset_id]
                return np.asarray(arr, dtype=np.float64) if isinstance(arr, (list, np.ndarray)) else None
            if states_dict:
                first_id = next(iter(states_dict))
                arr = states_dict[first_id]
                return np.asarray(arr, dtype=np.float64) if isinstance(arr, (list, np.ndarray)) else None
            return None
        raw = data.get("states", None)
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.float64) if isinstance(raw, (list, np.ndarray)) else None

    def discover_mechanisms(self, data: dict, states: Optional[NDArray] = None) -> dict[str, dict]:
        if states is None:
            states = self._extract_states(data)
        single_data = self._extract_single_asset(data)

        raw_results: dict[str, dict] = {}
        for name, mechanism in self.mechanisms.items():
            result = mechanism.compute(single_data, states)
            contrib = mechanism.get_state_contribution()
            result["state_contribution"] = contrib
            raw_results[name] = result

        self.results["discovered_mechanisms"] = raw_results
        return raw_results

    def validate_mechanisms(
        self,
        mechanism_results: dict,
        data: dict,
        states: Optional[NDArray] = None,
        forward_returns: Optional[NDArray] = None,
        compressed_dim: int = 5,
    ) -> dict:
        if states is None:
            states = self._extract_states(data)
        if forward_returns is None:
            single_data = self._extract_single_asset(data)
            forward_returns = np.asarray(single_data.get("forward_returns", single_data.get("returns", [])), dtype=np.float64)

        validation_results: dict[str, dict] = {}
        for name, mechanism in self.mechanisms.items():
            mech_result = mechanism_results.get(name, mechanism.compute(self._extract_single_asset(data), states))
            if "state_contribution" not in mech_result:
                mech_result["state_contribution"] = mechanism.get_state_contribution()

            info_r = self.validator.information_test(mech_result, forward_returns, states)
            sid_r = self.validator.sid_test(mech_result, states, forward_returns)
            sir_r = self.validator.sir_test(mech_result, states, forward_returns, compressed_dim)
            persist_r = self.validator.persistence_test(mech_result)

            all_passed = all(
                r.get("passed", False)
                for r in [info_r, sid_r, sir_r, persist_r]
            )

            validation_results[name] = {
                "mechanism_name": name,
                "mechanism_category": mechanism.category,
                "information_test": info_r,
                "sid_test": sid_r,
                "sir_test": sir_r,
                "persistence_test": persist_r,
                "overall": {
                    "all_passed": all_passed,
                    "n_passed": int(sum(
                        r.get("passed", False)
                        for r in [info_r, sid_r, sir_r, persist_r]
                    )),
                    "n_total": 4,
                },
            }

        self.results["validation_results"] = validation_results
        return validation_results

    def score_mechanisms(
        self,
        mechanism_results: dict,
        validation_results: dict,
    ) -> list[MechanismScore]:
        scores: list[MechanismScore] = []
        for name, mechanism in self.mechanisms.items():
            mech_result = mechanism_results.get(name, {})
            val_result = validation_results.get(name, {})
            score = self.scorer.score_mechanism(mechanism, mech_result, val_result)
            scores.append(score)

        ranked = self.scorer.rank_mechanisms(scores)
        self.results["scores"] = ranked
        self.results["score_summary"] = self.scorer.score_summary(scores)
        return ranked

    def get_state_contributions(self, mechanism_results: dict[str, dict]) -> dict[str, NDArray]:
        contributions: dict[str, NDArray] = {}
        for name, mechanism in self.mechanisms.items():
            mech_result = mechanism_results.get(name, {})
            contrib = mech_result.get("state_contribution", None)
            if contrib is None:
                contrib = mechanism.get_state_contribution()
            contributions[name] = np.asarray(contrib, dtype=np.float64)

        if contributions:
            all_arrays = [arr for arr in contributions.values() if len(arr) > 0]
            if all_arrays:
                min_len = min(len(arr) for arr in all_arrays)
                aligned = np.stack([arr[:min_len] for arr in all_arrays], axis=0)
                combined = np.mean(aligned, axis=0)
                contributions["combined_contribution"] = combined

        self.results["state_contributions"] = contributions
        return contributions

    def run_full_pipeline(
        self,
        data: dict,
        states: Optional[NDArray] = None,
        forward_returns: Optional[NDArray] = None,
        compressed_dim: int = 5,
    ) -> dict:
        mechanism_results = self.discover_mechanisms(data, states)
        if states is None:
            states = self._extract_states(data)
        if forward_returns is None:
            single_data = self._extract_single_asset(data)
            forward_returns = np.asarray(single_data.get("forward_returns", single_data.get("returns", [])), dtype=np.float64)

        validation_results = self.validate_mechanisms(
            mechanism_results, data, states, forward_returns, compressed_dim
        )
        scores = self.score_mechanisms(mechanism_results, validation_results)
        contributions = self.get_state_contributions(mechanism_results)

        self.results["pipeline_complete"] = True
        self.results["config"] = {
            "compressed_dim": compressed_dim,
            "n_mechanisms": len(self.mechanisms),
            "multi_asset": self._is_multi_asset(data),
        }

        return {
            "mechanism_results": mechanism_results,
            "validation_results": validation_results,
            "scores": scores,
            "score_summary": self.results.get("score_summary", {}),
            "state_contributions": contributions,
            "config": self.results["config"],
        }

    def generate_report(self, results: dict) -> str:
        lines: list[str] = []
        lines.append("# PROXIMA X Phase 4 — Mechanism Discovery Report")
        lines.append("")

        mech_results = results.get("mechanism_results", {})
        val_results = results.get("validation_results", {})
        scores = results.get("scores", [])
        score_summary = results.get("score_summary", {})
        contributions = results.get("state_contributions", {})
        config = results.get("config", {})

        lines.append("## 1. Mechanisms Discovered")
        lines.append("")

        for name in self.mechanisms:
            mech_result = mech_results.get(name, {})
            val_result = val_results.get(name, {})
            mechanism = self.mechanisms[name]
            contrib = contributions.get(name, np.array([], dtype=np.float64))

            lines.append(f"### {name} ({mechanism.category})")
            lines.append("")

            n_keys = len(mech_result)
            lines.append(f"- **Output Keys**: {n_keys}")
            lines.append(f"- **Contribution Length**: {len(contrib)}")

            if len(contrib) > 0:
                lines.append(f"- **Contribution Mean**: {float(np.mean(np.abs(contrib))):.6f}")
                lines.append(f"- **Contribution Std**: {float(np.std(contrib)):.6f}")
                lines.append(f"- **Contribution Min**: {float(np.min(contrib)):.6f}")
                lines.append(f"- **Contribution Max**: {float(np.max(contrib)):.6f}")

            overall = val_result.get("overall", {})
            lines.append(f"- **Validation Passed**: {overall.get('n_passed', 0)}/{overall.get('n_total', 4)}")
            lines.append("")

        lines.append("## 2. Mechanism Rankings")
        lines.append("")

        if score_summary:
            lines.append(f"- **Total Mechanisms**: {score_summary.get('n_total', 0)}")
            lines.append(f"- **Surviving**: {score_summary.get('n_surviving', 0)}")
            lines.append(f"- **Failed**: {score_summary.get('n_failed', 0)}")
            lines.append(f"- **Mean Composite Score**: {score_summary.get('mean_score', 0.0):.4f}")
            lines.append(f"- **Top Mechanism**: {score_summary.get('top_mechanism', 'N/A')}")
            lines.append(f"- **Worst Mechanism**: {score_summary.get('worst_mechanism', 'N/A')}")
            lines.append("")

        if scores:
            lines.append("| Rank | Name | Category | Composite | Info Gain | SID | SIR | Persist | Robust | Simple | Novelty | Survives |")
            lines.append("|------|------|----------|-----------|-----------|-----|-----|---------|--------|--------|---------|----------|")
            for rank, s in enumerate(scores, start=1):
                survives_mark = "Y" if s.survives else "N"
                lines.append(
                    f"| {rank} | {s.name} | {s.category} "
                    f"| {s.composite_score:.4f} | {s.information_gain:.4f} "
                    f"| {s.sid:.4f} | {s.sir:.4f} "
                    f"| {s.persistence:.4f} | {s.robustness:.4f} "
                    f"| {s.simplicity:.4f} | {s.novelty:.4f} "
                    f"| {survives_mark} |"
                )
            lines.append("")

        lines.append("## 3. Validation Summary")
        lines.append("")

        for name in self.mechanisms:
            val_result = val_results.get(name, {})
            overall = val_result.get("overall", {})
            info_r = val_result.get("information_test", {})
            sid_r = val_result.get("sid_test", {})
            sir_r = val_result.get("sir_test", {})
            persist_r = val_result.get("persistence_test", {})

            lines.append(f"### {name}")
            lines.append(f"- **Overall**: {overall.get('n_passed', 0)}/{overall.get('n_total', 4)} passed")
            lines.append(f"- **Information Test**: {'PASS' if info_r.get('passed') else 'FAIL'} (IG={info_r.get('information_gain', 0.0):.4f})")
            lines.append(f"- **SID Test**: {'PASS' if sid_r.get('passed') else 'FAIL'} (SID={sid_r.get('sid', 0.0):.4f})")
            lines.append(f"- **SIR Test**: {'PASS' if sir_r.get('passed') else 'FAIL'} (SIR={sir_r.get('sir', 0.0):.4f})")
            lines.append(f"- **Persistence Test**: {'PASS' if persist_r.get('passed') else 'FAIL'} (score={persist_r.get('persistence_score', 0.0):.4f})")
            lines.append("")

        lines.append("## 4. Information Contribution")
        lines.append("")

        lines.append("| Mechanism | Information Gain | SID | SIR | Persistence |")
        lines.append("|-----------|-----------------|-----|-----|-------------|")

        for name in self.mechanisms:
            val_result = val_results.get(name, {})
            info_r = val_result.get("information_test", {})
            sid_r = val_result.get("sid_test", {})
            sir_r = val_result.get("sir_test", {})
            persist_r = val_result.get("persistence_test", {})

            lines.append(
                f"| {name} | {info_r.get('information_gain', 0.0):.4f} "
                f"| {sid_r.get('sid', 0.0):.4f} "
                f"| {sir_r.get('sir', 0.0):.4f} "
                f"| {persist_r.get('persistence_score', 0.0):.4f} |"
            )
        lines.append("")

        lines.append("## 5. State Generation Analysis")
        lines.append("")

        if contributions:
            combined = contributions.get("combined_contribution", np.array([], dtype=np.float64))
            lines.append(f"- **Combined Contribution (mean of all)**: length={len(combined)}")
            if len(combined) > 0:
                lines.append(f"  - Mean: {float(np.mean(combined)):.6f}")
                lines.append(f"  - Std: {float(np.std(combined)):.6f}")
                lines.append(f"  - Abs Mean: {float(np.mean(np.abs(combined))):.6f}")
            lines.append("")

            lines.append("| Mechanism | Contribution Mean | Contribution Std | Abs Mean | Pattern |")
            lines.append("|-----------|------------------|-----------------|----------|---------|")
            for name in self.mechanisms:
                contrib = contributions.get(name, np.array([], dtype=np.float64))
                if len(contrib) > 0:
                    c_mean = float(np.mean(contrib))
                    c_std = float(np.std(contrib))
                    c_abs = float(np.mean(np.abs(contrib)))
                    c_min = float(np.min(contrib))
                    c_max = float(np.max(contrib))
                    if abs(c_mean) < 1e-12 and c_std < 1e-12:
                        pattern = "flat/dead"
                    elif c_std > 2.0 * abs(c_mean):
                        pattern = "highly_variable"
                    elif c_std > abs(c_mean):
                        pattern = "variable"
                    elif c_min >= 0:
                        pattern = "non_negative"
                    elif c_max <= 0:
                        pattern = "non_positive"
                    elif c_mean > 0:
                        pattern = "positive_bias"
                    else:
                        pattern = "negative_bias"
                else:
                    c_mean, c_std, c_abs, pattern = 0.0, 0.0, 0.0, "empty"

                lines.append(
                    f"| {name} | {c_mean:.6f} | {c_std:.6f} | {c_abs:.6f} | {pattern} |"
                )
            lines.append("")

        lines.append("## 6. Failure Analysis")
        lines.append("")

        failures_found = False
        for name in self.mechanisms:
            val_result = val_results.get(name, {})
            failed_tests: list[str] = []

            info_r = val_result.get("information_test", {})
            if not info_r.get("passed", False):
                failed_tests.append(f"information_test (IG={info_r.get('information_gain', 0.0):.4f})")

            sid_r = val_result.get("sid_test", {})
            if not sid_r.get("passed", False):
                failed_tests.append(f"sid_test (SID={sid_r.get('sid', 0.0):.4f})")

            sir_r = val_result.get("sir_test", {})
            if not sir_r.get("passed", False):
                failed_tests.append(f"sir_test (SIR={sir_r.get('sir', 0.0):.4f})")

            persist_r = val_result.get("persistence_test", {})
            if not persist_r.get("passed", False):
                failed_tests.append(f"persistence_test (score={persist_r.get('persistence_score', 0.0):.4f})")

            if failed_tests:
                failures_found = True
                lines.append(f"### {name}")
                for ft in failed_tests:
                    lines.append(f"- FAILED: {ft}")

                state_contrib = contributions.get(name, np.array([], dtype=np.float64))
                if len(state_contrib) > 0 and float(np.mean(np.abs(state_contrib))) < 1e-12:
                    lines.append(f"  - Root cause: state_contribution is near-zero (mean_abs={float(np.mean(np.abs(state_contrib))):.2e})")
                lines.append("")

        if not failures_found:
            lines.append("All mechanisms passed all validation tests.")
            lines.append("")

        lines.append("## 7. Recommended Next Experiments")
        lines.append("")

        if scores:
            top = scores[0] if scores else None
            if top:
                lines.append(f"- **Primary candidate**: `{top.name}` (composite={top.composite_score:.4f})")
                if top.survives:
                    lines.append(f"  - Survives all filters — proceed to forward testing and portfolio integration")
                else:
                    weak = []
                    if top.information_gain <= 1e-8:
                        weak.append("information_gain")
                    if top.sid <= 1e-8:
                        weak.append("SID")
                    if top.sir <= 1e-8:
                        weak.append("SIR")
                    if top.persistence <= 0.5:
                        weak.append("persistence")
                    lines.append(f"  - Weak areas: {', '.join(weak)} — needs refinement before deployment")

            survivors = [s for s in scores if s.survives]
            if len(survivors) > 1:
                lines.append(f"- **Ensemble candidate**: combine top {min(3, len(survivors))} surviving mechanisms")
            elif len(survivors) == 0:
                lines.append("- **No survivors**: investigate root causes — low signal-to-noise, inadequate feature space, or regime mismatch")
                lines.append("- Consider: longer training window, different discretization strategy, or higher-resolution data")

            n_improving = sum(1 for s in scores if s.information_gain > 0)
            lines.append(f"- **Information-theoretic signals**: {n_improving}/{len(scores)} mechanisms show positive information gain")

            top_sid = max(s.sid for s in scores) if scores else 0.0
            top_sir = max(s.sir for s in scores) if scores else 0.0
            if top_sid > 0:
                lines.append(f"- **Highest SID**: {top_sid:.4f} — indicates state-dependent predictive structure")
            if top_sir > 0:
                lines.append(f"- **Highest SIR**: {top_sir:.4f} — indicates efficient state representation relative to complexity")

            pos_persistence = sum(1 for s in scores if s.persistence > 0.5)
            lines.append(f"- **Persistence**: {pos_persistence}/{len(scores)} mechanisms show persistent structure")

            lines.append("- **Cross-validation**: run cross_asset_test and cross_regime_test to verify generalization")
            lines.append("- **OOS testing**: split data temporally and run oos_test for out-of-sample robustness")

        return "\n".join(lines)
