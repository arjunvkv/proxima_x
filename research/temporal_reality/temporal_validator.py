import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import numpy as np

from research.temporal_reality.conditional_information import ConditionalInformationAnalyzer
from research.temporal_reality.evolution_clock import EvolutionClockAnalyzer, EvolutionClockReport
from research.temporal_reality.regime_mutation import RegimeMutationAnalyzer, RegimeMutationReport
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
from research.temporal_reality.universality import UniversalityAnalyzer, UniversalityReport
from research.temporal_reality.null_models import NullModelGenerator, NullModelComparison
from research.temporal_reality.dependency_graph import DependencyGraphBuilder, DependencyGraph


@dataclass
class TemporalValidationReport:
    asset: str
    conditional_info: dict
    evolution_clock: EvolutionClockReport
    regime_mutation: RegimeMutationReport
    causality: dict
    universality: Any  # UniversalityReport or None for single-asset
    null_models: NullModelComparison
    dependency_graph: DependencyGraph
    timing: dict
    final_verdict: str


class TemporalRealityValidator:
    """
    Orchestrates all Reality Phase 4 analyses and produces a final verdict.
    """
    
    VERDICTS = [
        "ARTIFACT",
        "VOLATILITY_PROXY",
        "ENTROPY_PROXY",
        "ACTIVITY_PROXY",
        "STATE_TRANSITION_PROXY",
        "WEAK_EVOLUTION_CLOCK",
        "MODERATE_EVOLUTION_CLOCK",
        "STRONG_EVOLUTION_CLOCK",
        "FUNDAMENTAL_EVOLUTION_CLOCK",
    ]
    
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.conditional_analyzer = ConditionalInformationAnalyzer()
        self.evolution_analyzer = EvolutionClockAnalyzer()
        self.regime_analyzer = RegimeMutationAnalyzer()
        self.causality_analyzer = AdaptiveTimeCausality()
        self.null_generator = NullModelGenerator()
        self.dependency_builder = DependencyGraphBuilder()
    
    def validate(self, data: dict) -> TemporalValidationReport:
        """
        Run all analyses and generate report.
        
        data dict must contain:
        - adaptive_time: np.ndarray
        - states: np.ndarray
        - returns: np.ndarray
        - volume: np.ndarray
        - energy_creation: np.ndarray
        - energy_storage: np.ndarray
        - energy_dissipation: np.ndarray
        - volatility: np.ndarray
        - entropy: np.ndarray
        - event_density: np.ndarray
        - state_mutation_rate: np.ndarray (optional, computed if not present)
        - regime_change_events: np.ndarray (optional, computed if not present)
        """
        wall = time.time()
        timing = {}
        
        # 1. Conditional Information
        t0 = time.time()
        cond_info = self._run_conditional_analysis(data)
        timing["conditional_info"] = time.time() - t0
        
        # 2. Evolution Clock
        t0 = time.time()
        evo_report = self.evolution_analyzer.compute(data["adaptive_time"], data["states"])
        timing["evolution_clock"] = time.time() - t0
        
        # 3. Regime Mutation
        t0 = time.time()
        regime_report = self.regime_analyzer.compute(
            data["adaptive_time"],
            data.get("energy_creation"),
            data.get("energy_storage"),
            data.get("energy_dissipation"),
        )
        timing["regime_mutation"] = time.time() - t0
        
        # 4. Causality
        t0 = time.time()
        mutation_rate = data.get("state_mutation_rate", 
            self._compute_mutation_rate(data["states"]))
        regime_events = data.get("regime_change_events", 
            self._compute_regime_changes(data["states"]))
        causality_result = self.causality_analyzer.compute(data["adaptive_time"], mutation_rate)
        timing["causality"] = time.time() - t0
        
        # 5. Null Models
        t0 = time.time()
        null_result = self.null_generator.generate(
            data["adaptive_time"],
            data["returns"],
            data.get("volume", np.ones_like(data["returns"])),
            data.get("volatility"),
            data.get("entropy"),
        )
        timing["null_models"] = time.time() - t0
        
        # 6. Dependency Graph
        t0 = time.time()
        dep_graph = self.dependency_builder.build(
            data["adaptive_time"],
            mutation_rate,
            regime_events,
            data["returns"],
            data.get("volatility"),
        )
        timing["dependency_graph"] = time.time() - t0
        
        timing["total"] = time.time() - wall
        
        # Compute final verdict
        final_verdict = self._compute_verdict(
            cond_info, evo_report, regime_report, 
            causality_result, null_result, dep_graph
        )
        
        return TemporalValidationReport(
            asset=self.asset,
            conditional_info=cond_info,
            evolution_clock=evo_report,
            regime_mutation=regime_report,
            causality=causality_result,
            universality=None,
            null_models=null_result,
            dependency_graph=dep_graph,
            timing=timing,
            final_verdict=final_verdict,
        )
    
    def _run_conditional_analysis(self, data: dict) -> dict:
        """Run conditional MI analysis and return results dict."""
        at = data["adaptive_time"]
        future_states = np.roll(data["states"], -1)  # future state mutation
        
        raw_mi = self.conditional_analyzer.conditional_mi_multiple(at, future_states, [])
        
        conditions = {}
        for name in ["volatility", "entropy", "event_density"]:
            if name in data and data[name] is not None:
                cond_mi = self.conditional_analyzer.conditional_mi(at, future_states, data[name])
                conditions[name] = {
                    "conditioned_mi": float(cond_mi),
                    "survival_ratio": float(cond_mi / raw_mi) if raw_mi > 1e-12 else 0.0,
                }
        
        # Combined conditioning
        all_conditions = [data[k] for k in ["volatility", "entropy", "event_density"] 
                         if k in data and data[k] is not None]
        combined_mi = self.conditional_analyzer.conditional_mi_multiple(at, future_states, all_conditions)
        
        return {
            "raw_mi": float(raw_mi),
            "conditioned_mi": float(combined_mi),
            "information_survival_ratio": float(combined_mi / raw_mi) if raw_mi > 1e-12 else 0.0,
            "per_condition": conditions,
        }
    
    def _compute_mutation_rate(self, states: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute rolling state mutation rate."""
        from research.temporal_reality.evolution_clock import EvolutionClockAnalyzer
        rates, _, _, _ = EvolutionClockAnalyzer._compute_state_rates(states.astype(np.float64), window)
        return rates
    
    def _compute_regime_changes(self, states: np.ndarray) -> np.ndarray:
        """Detect regime change points."""
        changes = np.zeros(len(states), dtype=np.float64)
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                changes[i] = 1.0
        return changes
    
    def _compute_verdict(self, cond_info: dict, evo: EvolutionClockReport,
                        regime: RegimeMutationReport, causality: dict,
                        null: NullModelComparison, dep: DependencyGraph) -> str:
        # Decision logic (will be refined)
        scores = {v: 0.0 for v in self.VERDICTS}
        
        # Conditional info survival ratio
        survival = cond_info.get("information_survival_ratio", 0.0)
        if survival > 0.8:
            scores["STRONG_EVOLUTION_CLOCK"] += 2.0
        elif survival > 0.5:
            scores["MODERATE_EVOLUTION_CLOCK"] += 1.0
        else:
            scores["ARTIFACT"] += 1.0
        
        # Evolution clock verdict
        evo_str = evo.verdict if hasattr(evo, 'verdict') else ""
        if "STRONG" in evo_str:
            scores["STRONG_EVOLUTION_CLOCK"] += 2.0
        elif "MODERATE" in evo_str:
            scores["MODERATE_EVOLUTION_CLOCK"] += 1.0
        
        # Null model comparison
        if null.verdict == "adaptive_time_is_unique":
            scores["FUNDAMENTAL_EVOLUTION_CLOCK"] += 2.0
        elif null.verdict == "simpler_explanations_exist":
            scores["VOLATILITY_PROXY"] += 1.0
            scores["ENTROPY_PROXY"] += 1.0
        
        # Causality
        lead = causality.get("lead_or_follow", "")
        if "leads" in lead:
            scores["STRONG_EVOLUTION_CLOCK"] += 1.5
        
        # Dependency graph
        dep_verdict = dep.verdict if hasattr(dep, 'verdict') else ""
        if "drives_evolution" in dep_verdict:
            scores["FUNDAMENTAL_EVOLUTION_CLOCK"] += 1.5
        elif "pathway" in dep_verdict:
            scores["STRONG_EVOLUTION_CLOCK"] += 1.0
        
        # Return highest scoring verdict
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "ARTIFACT"
