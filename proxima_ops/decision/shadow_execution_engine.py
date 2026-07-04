"""Shadow Execution Engine — Counterfactual parallel decision observer.

Instruments the decision pipeline parallel to live execution without modifying logic,
reconstructing pre-governance intent and tracking layer-by-layer suppression.
"""

import copy
import math
import logging
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger("proxima_ops.decision.shadow_execution_engine")


class GateInterceptorRegistry:
    """Registry to tap and cache signal states at different boundary layers."""
    def __init__(self) -> None:
        self._layers: Dict[str, Dict[str, Any]] = {}

    def intercept(self, layer_name: str, symbol: str, state: Dict[str, Any]) -> None:
        if layer_name not in self._layers:
            self._layers[layer_name] = {}
        self._layers[layer_name][symbol] = copy.deepcopy(state)

    def get_layer_state(self, layer_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        return self._layers.get(layer_name, {}).get(symbol)

    def clear(self) -> None:
        self._layers.clear()


class SuppressionGraphBuilder:
    """Builds a directed, weighted suppression causality graph between gates."""
    def __init__(self) -> None:
        self.nodes: List[str] = ["L0_Raw", "L1_DecisionGate", "L2_Governor", "L3_Intent", "L4_CB", "L5_VEL"]
        self.suppression_flow: Dict[Tuple[str, str], float] = {}

    def record_suppression_flow(self, from_gate: str, to_gate: str, magnitude: float) -> None:
        self.suppression_flow[(from_gate, to_gate)] = magnitude

    def get_graph_data(self) -> Dict[str, Any]:
        edges = []
        for (u, v), w in self.suppression_flow.items():
            edges.append({"source": u, "target": v, "suppression_magnitude": w})
        return {
            "nodes": self.nodes,
            "edges": edges,
            "cascade_type": "multiplicative" if len(edges) > 1 else "linear"
        }


class CounterfactualConvictionEngine:
    """Computes unsuppressed conviction vs governed conviction and decision loss."""
    def __init__(self) -> None:
        pass

    def compute_suppression_delta(self, raw_conviction: float, final_conviction: float) -> float:
        return max(0.0, raw_conviction - final_conviction)

    def reconstruct_intent(self, raw_states: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct pre-governance decision intent."""
        return {
            "unsuppressed_conviction": raw_states.get("conviction", 0.5),
            "unsuppressed_direction": raw_states.get("direction", 0),
            "unsuppressed_stability": raw_states.get("stability", 0.8),
        }


class LKGComparator:
    """Compares current pipeline decisions against reconstructed LKG / minimal engine states."""
    def __init__(self) -> None:
        pass

    def compute_similarity(self, current_vector: List[float], lkg_vector: List[float]) -> float:
        """Calculate cosine similarity between decision vectors."""
        if not current_vector or not lkg_vector or len(current_vector) != len(lkg_vector):
            return 0.0
        dot_product = sum(c * l for c, l in zip(current_vector, lkg_vector))
        norm_c = math.sqrt(sum(c * c for c in current_vector))
        norm_l = math.sqrt(sum(l * l for l in lkg_vector))
        if norm_c == 0.0 or norm_l == 0.0:
            return 0.0
        return dot_product / (norm_c * norm_l)


class ShadowExecutionOrchestrator:
    """Orchestrates parallel shadow state mirroring and counterfactual tracking."""
    def __init__(self) -> None:
        self.registry = GateInterceptorRegistry()
        self.graph_builder = SuppressionGraphBuilder()
        self.conviction_engine = CounterfactualConvictionEngine()
        self.comparator = LKGComparator()
        self.shadow_history: List[Dict[str, Any]] = []

    def process_cycle(self, cycle_id: int, symbols: List[str]) -> Dict[str, Any]:
        cycle_report = {"cycle": cycle_id, "symbols": {}}

        for symbol in symbols:
            # Extract tapped layers
            l0 = self.registry.get_layer_state("L0_Raw", symbol) or {"conviction": 0.5, "direction": 0}
            l1 = self.registry.get_layer_state("L1_DecisionGate", symbol) or l0
            l2 = self.registry.get_layer_state("L2_Governor", symbol) or l1
            l3 = self.registry.get_layer_state("L3_Intent", symbol) or l2
            l4 = self.registry.get_layer_state("L4_CB", symbol) or l3
            l5 = self.registry.get_layer_state("L5_VEL", symbol) or l4

            raw_c = l0.get("conviction", 0.5)
            final_c = l5.get("conviction", 0.5)
            supp_delta = self.conviction_engine.compute_suppression_delta(raw_c, final_c)

            # Record suppression cascade
            self.graph_builder.record_suppression_flow("L0_Raw", "L1_DecisionGate", raw_c - l1.get("conviction", raw_c))
            self.graph_builder.record_suppression_flow("L1_DecisionGate", "L2_Governor", l1.get("conviction", raw_c) - l2.get("conviction", raw_c))
            self.graph_builder.record_suppression_flow("L2_Governor", "L3_Intent", l2.get("conviction", raw_c) - l3.get("conviction", raw_c))
            self.graph_builder.record_suppression_flow("L3_Intent", "L4_CB", l3.get("conviction", raw_c) - l4.get("conviction", raw_c))
            self.graph_builder.record_suppression_flow("L4_CB", "L5_VEL", l4.get("conviction", raw_c) - final_c)

            # LKG comparison
            current_v = [raw_c, l1.get("conviction", 0.5), l2.get("conviction", 0.5), final_c]
            lkg_v = [raw_c, raw_c * 0.9, raw_c * 0.85, raw_c * 0.85] # LKG model
            similarity = self.comparator.compute_similarity(current_v, lkg_v)

            cycle_report["symbols"][symbol] = {
                "raw_signal_state": l0,
                "post_governance_state": l5,
                "suppression_delta": supp_delta,
                "lkg_similarity_score": similarity,
                "gate_trace_chain": [l0, l1, l2, l3, l4, l5]
            }

        cycle_report["suppression_graph"] = self.graph_builder.get_graph_data()
        self.shadow_history.append(cycle_report)
        return cycle_report

    def clear(self) -> None:
        self.registry.clear()
