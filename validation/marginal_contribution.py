"""
Marginal Contribution Analysis — E2.

Measures each D1-D7 module's incremental value through
leave-one-out and leave-one-in ablation studies.

Each experiment feeds a batch of decision states through a subset of
observer modules and records the aggregated D7 arbitration metrics.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from signals.state_transition_graph import StateTransitionGraph
from signals.causal_fingerprint import CausalFingerprintEngine
from signals.counterfactual_branch import CounterfactualBranchEngine
from signals.path_memory import PathMemoryEngine
from signals.regime_attractor import RegimeAttractorEngine
from signals.pre_rupture_forecast import PreRuptureForecastEngine
from signals.reflexive_arbitration import ReflexiveArbitrationEngine

logger = logging.getLogger("proxima_demo")

# All observer module identifiers in the D1-D7 pipeline.
ALL_MODULES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
# Modules that consume state (D7 is the evaluator / producer of metrics).
FEEDER_MODULES = ["D1", "D2", "D3", "D4", "D5", "D6"]

# Default values for decision-state fields consumed by D7.
_DEFAULT_D7_INPUTS = {
    "thesis_rf_probability": 0.5,
    "memory_weight": 0.5,
    "counterfactual_score": 0.0,
    "rupture_probability": 0.0,
    "path_probability": 0.5,
    "causal_confidence": 0.5,
}


def _compact_state(state: dict) -> tuple:
    """Build the 4-tuple state used by D1, D4, D5."""
    return (
        state.get("topology_state", "DISTRIBUTED"),
        state.get("trust_band", "HIGH"),
        state.get("pressure_band", "LOW"),
        state.get("rupture_flag", 0),
    )


def _fingerprint(state: dict) -> tuple:
    """Build the 5-tuple fingerprint used by D2."""
    fracture = state.get("fracture", 0.0)
    rf_prob = state.get("thesis_rf_probability", 0.5)
    return (
        state.get("topology_state", "DISTRIBUTED"),
        state.get("trust_band", "HIGH"),
        state.get("pressure_band", "LOW"),
        "HIGH" if fracture > 0.5 else "LOW",
        "HIGH" if rf_prob > 0.5 else "LOW",
    )


def _organism(state: dict) -> dict:
    """Build the organism-state dict used by D6."""
    return {
        "pressure": state.get("pressure", 0.0),
        "fracture": state.get("fracture", 0.0),
        "cohort_instability": state.get("cohort_instability", 0.0),
        "trust": state.get("trust", 0.5),
        "attractor_strength": state.get("attractor_strength", 0.0),
        "causal_confidence": state.get("causal_confidence", 0.5),
        "escape_time": state.get("escape_time", 10.0),
        "return_time": state.get("return_time", 5.0),
        "path_probability": state.get("path_probability", 0.5),
    }


class MarginalContributionAnalyzer:
    """Ablation-based marginal contribution analyser for D1-D7 observers."""

    def __init__(self):
        self._experiments: List[dict] = []
        self._results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API: experiment runners
    # ------------------------------------------------------------------

    def leave_one_out(self, decision_states: List[dict],
                      module_name: str) -> dict:
        """Run the full observer stack *minus* one module.

        Returns aggregated metrics for the ablated stack.
        """
        self._validate_module(module_name)
        if not decision_states:
            return self._empty_result(module_name, "leave_one_out")

        # Full stack (all modules active).
        full_result = self._run_experiment(decision_states,
                                           set(FEEDER_MODULES),
                                           "full")

        # Ablated stack (one module removed from feed).
        active = set(FEEDER_MODULES) - {module_name}
        ablated_result = self._run_experiment(decision_states,
                                              active,
                                              "leave_one_out")

        ablated_result["module"] = module_name
        self._experiments.append(ablated_result)
        self._results[module_name] = ablated_result
        return ablated_result

    def leave_one_in(self, decision_states: List[dict],
                     module_name: str) -> dict:
        """Run *only* one module alone (plus D7 for evaluation).

        Returns aggregated metrics for that single-module stack.
        """
        self._validate_module(module_name)
        if not decision_states:
            return self._empty_result(module_name, "leave_one_in")

        if module_name == "D7":
            # D7 alone — no feeder modules, only the D7 evaluator.
            active: Set[str] = set()
        else:
            active = {module_name}

        result = self._run_experiment(decision_states,
                                      active,
                                      "leave_one_in")
        result["module"] = module_name
        self._experiments.append(result)
        self._results[module_name] = result
        return result

    # ------------------------------------------------------------------
    # Comparison & ranking
    # ------------------------------------------------------------------

    def compare(self, full_stack_results: dict,
                ablation_results: dict) -> dict:
        """Compute the delta (full minus ablated) for each metric.

        Positive delta means the module added value.
        """
        keys = [
            "decisions", "mean_quality", "mean_confidence",
            "mean_consensus", "override_rate", "disagreement_rate",
        ]
        diffs = {}
        for k in keys:
            full_val = full_stack_results.get(k, 0.0)
            abl_val = ablation_results.get(k, 0.0)
            diffs[k] = round(full_val - abl_val, 6)

        # Recommendation distribution diff (per-class count delta).
        full_recs = full_stack_results.get("recommendation_distribution", {})
        abl_recs = ablation_results.get("recommendation_distribution", {})
        all_classes = set(full_recs.keys()) | set(abl_recs.keys())
        rec_delta = {}
        for rc in all_classes:
            delta = full_recs.get(rc, 0) - abl_recs.get(rc, 0)
            if delta != 0:
                rec_delta[rc] = delta
        diffs["recommendation_distribution"] = rec_delta

        return diffs

    def rank_modules(self, results_dict: dict) -> List[str]:
        """Sort modules by aggregate delta descending.

        *results_dict* maps module_name -> compare() output dict.
        Modules not present are omitted.
        """
        scored = []
        for mod, deltas in results_dict.items():
            # Aggregate delta: sum of absolute changes in quality,
            # confidence, consensus (positive means module adds value).
            agg = (deltas.get("mean_quality", 0.0)
                   + deltas.get("mean_confidence", 0.0)
                   + deltas.get("mean_consensus", 0.0))
            scored.append((agg, mod))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mod for _, mod in scored]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of all experiments run so far."""
        return {
            "experiments": len(self._experiments),
            "results": {r["module"]: r["mode"]
                        for r in self._experiments},
            "aggregate": self._experiments,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_module(self, name: str):
        if name not in ALL_MODULES:
            raise ValueError(
                f"Unknown module '{name}'. Must be one of {ALL_MODULES}"
            )

    @staticmethod
    def _empty_result(module: str, mode: str) -> dict:
        return {
            "module": module,
            "mode": mode,
            "decisions": 0,
            "mean_quality": 0.0,
            "mean_confidence": 0.0,
            "mean_consensus": 0.0,
            "override_rate": 0.0,
            "recommendation_distribution": {},
            "disagreement_rate": 0.0,
        }

    def _new_stack(self) -> dict:
        """Create fresh, independent instances of all 7 modules."""
        return {
            "D1": StateTransitionGraph(),
            "D2": CausalFingerprintEngine(),
            "D3": CounterfactualBranchEngine(),
            "D4": PathMemoryEngine(),
            "D5": RegimeAttractorEngine(),
            "D6": PreRuptureForecastEngine(),
            "D7": ReflexiveArbitrationEngine(),
        }

    def _feed(self, mod, name: str, state: dict, symbol: str,
              counter: int):
        """Feed a single decision state into the named module."""
        if name == "D1":
            mod.update(symbol, _compact_state(state))
        elif name == "D2":
            mod.record(symbol, _fingerprint(state))
        elif name == "D3":
            direction = state.get("direction", 0)
            entry_price = state.get("entry_price", 0.0)
            if direction != 0 and entry_price > 0:
                mod.register(counter, direction, entry_price)
        elif name == "D4":
            mod.update(symbol, _compact_state(state))
        elif name == "D5":
            mod.update(symbol, _compact_state(state))
        elif name == "D6":
            mod.update(symbol, _organism(state))
        # D7 is never "fed" — it receives the decision state via evaluate().

    def _build_d7_input(self, state: dict, stack: dict,
                        active_feeder: Set[str],
                        counter: int) -> dict:
        """Build the :meth:`D7.evaluate` input dict.

        Starts from state-dict defaults and enriches with module-derived
        values when the corresponding module is active.
        """
        ds = {
            "thesis_rf_probability": state.get(
                "thesis_rf_probability",
                _DEFAULT_D7_INPUTS["thesis_rf_probability"]),
            "memory_weight": state.get(
                "memory_weight",
                _DEFAULT_D7_INPUTS["memory_weight"]),
            "counterfactual_score": state.get(
                "counterfactual_score",
                _DEFAULT_D7_INPUTS["counterfactual_score"]),
            "rupture_probability": state.get(
                "rupture_probability",
                _DEFAULT_D7_INPUTS["rupture_probability"]),
            "path_probability": state.get(
                "path_probability",
                _DEFAULT_D7_INPUTS["path_probability"]),
            "causal_confidence": state.get(
                "causal_confidence",
                _DEFAULT_D7_INPUTS["causal_confidence"]),
        }

        symbol = state.get("symbol", "UNKNOWN")

        # -- D2: fingerprint observation count → causal confidence --
        if "D2" in active_feeder:
            stats_d2 = stack["D2"].stats()
            n_obs = stats_d2.get("total_observations", 0)
            if n_obs > 0:
                # More observations => higher confidence in causal patterns.
                ds["causal_confidence"] = min(1.0, n_obs / 30.0 + 0.2)

        # -- D3: active thesis tracking → counterfactual awareness --
        if "D3" in active_feeder:
            direction = state.get("direction", 0)
            entry_price = state.get("entry_price", 0.0)
            if direction != 0 and entry_price > 0:
                # Module is engaged with an active thesis.
                ds["counterfactual_score"] = 0.3

        # -- D4: path probability from observed path history --
        if "D4" in active_feeder:
            path = stack["D4"].current_path(symbol)
            if path:
                pp = stack["D4"].path_probability(path)
                if pp > 0:
                    ds["path_probability"] = pp

        # -- D6: rupture forecast probability --
        if "D6" in active_feeder:
            fcast = stack["D6"].forecast(symbol)
            if fcast is not None:
                ds["rupture_probability"] = fcast["rupture_probability"]

        return ds

    def _run_experiment(self, decision_states: List[dict],
                        active_feeder: Set[str],
                        mode: str) -> dict:
        """Feed states through *active_feeder* modules and collect D7
        metrics.  Returns the aggregated result dict."""
        stack = self._new_stack()
        d7 = stack["D7"]

        # Accumulators for aggregated metrics.
        n = len(decision_states)
        quality_sum = 0.0
        confidence_sum = 0.0
        consensus_sum = 0.0
        override_count = 0
        disagreement_count = 0
        rec_counts: Dict[str, int] = defaultdict(int)

        for i, state in enumerate(decision_states):
            symbol = state.get("symbol", "UNKNOWN")

            # Feed state into each active feeder module.
            for mod_name in FEEDER_MODULES:
                if mod_name in active_feeder:
                    self._feed(stack[mod_name], mod_name, state,
                               symbol, i)

            # Build D7 decision state (enriched by active modules).
            ds = self._build_d7_input(state, stack, active_feeder, i)
            d7_result = d7.evaluate(ds)

            # Record D7 outputs.
            quality_sum += d7_result["decision_quality"]
            confidence_sum += d7_result["confidence"]
            consensus_sum += d7_result["consensus"]
            if d7_result["override"]:
                override_count += 1
            rec_counts[d7_result["recommendation"]] += 1

            # Disagreement between D7 and production action.
            prod_action = state.get("production_action", "FLAT")
            disc = d7.disagreement(prod_action)
            if disc["disagreement"]:
                disagreement_count += 1

        return {
            "module": None,  # filled by caller
            "mode": mode,
            "decisions": n,
            "mean_quality": round(quality_sum / n, 4) if n > 0 else 0.0,
            "mean_confidence": round(confidence_sum / n, 4) if n > 0 else 0.0,
            "mean_consensus": round(consensus_sum / n, 4) if n > 0 else 0.0,
            "override_rate": round(override_count / n, 4) if n > 0 else 0.0,
            "recommendation_distribution": dict(rec_counts),
            "disagreement_rate": round(disagreement_count / n, 4) if n > 0 else 0.0,
        }
