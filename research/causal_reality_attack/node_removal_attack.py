from __future__ import annotations

from typing import Any

import numpy as np

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


REMOVAL_EXPERIMENTS = [
    {
        "name": "adaptive_time_removal",
        "removed": {"adaptive_time"},
        "target_check": "state_mutation_rate",
        "secondary_check": "regime_change_probability",
        "question": "Is adaptive_time structurally necessary?",
    },
    {
        "name": "memory_density_removal",
        "removed": {"memory_density"},
        "target_check": "adaptive_time",
        "secondary_check": "state_mutation_rate",
        "question": "Can adaptive_time still emerge?",
    },
    {
        "name": "energy_storage_removal",
        "removed": {"energy_storage"},
        "target_check": "memory_density",
        "secondary_check": "adaptive_time",
        "question": "Does memory_density still form?",
    },
]


class NodeRemovalAttack:
    """Attacks 3-6: Node removal tests.

    Remove each node in the causal chain and measure information loss.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 50

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        full_graph, full_cands, full_ordering = self.validator.build_causal_graph(signals)
        full_info = self.validator.graph_information_score(full_graph)
        full_chain = full_graph.get_market_physics_chain()

        print(f"  Full graph: {full_info:.4f}, chain: {full_chain}")

        experiment_results = []
        metrics: dict[str, Any] = {
            "full_graph_info_score": full_info,
            "full_chain": full_chain,
            "full_edge_count": len(full_graph.edges),
            "full_node_count": len(full_graph.nodes),
        }

        for exp in REMOVAL_EXPERIMENTS:
            exp_name = exp["name"]
            removed = exp["removed"]
            target = exp["target_check"]
            secondary = exp["secondary_check"]

            try:
                graph, cands = self.validator.build_graph_with_removed_vars(signals, removed)
                removed_info = self.validator.graph_information_score(graph)
                chain = graph.get_market_physics_chain()

                # Compare full vs removed edge strengths for shared edges
                full_edge_map = {(e.source, e.target): e.causal_strength for e in full_graph.edges}
                removed_edge_map = {(e.source, e.target): e.causal_strength for e in graph.edges}
                shared_edges = set(full_edge_map.keys()) & set(removed_edge_map.keys())
                if shared_edges:
                    full_vals = [abs(full_edge_map[e]) for e in shared_edges]
                    removed_vals = [abs(removed_edge_map[e]) for e in shared_edges]
                    avg_strength_change = float(np.mean(removed_vals)) / max(float(np.mean(full_vals)), 1e-12) - 1.0
                else:
                    avg_strength_change = 0.0

                # Max info flow to target from any source (BEFORE removal includes ALL sources)
                all_sources = [v for v in TARGET_VARIABLES if v != target and v in signals]
                target_flow_before = max(self.validator.information_flow_between(signals, s, target) for s in all_sources)
                # AFTER removal excludes removed sources
                remaining = [s for s in all_sources if s not in removed]
                target_flow_after = max(self.validator.information_flow_between(signals, s, target) for s in remaining) if remaining else 0.0

                target_info_loss = (target_flow_before - target_flow_after) / max(target_flow_before, 1e-12)

                # Secondary check
                sec_sources = [v for v in TARGET_VARIABLES if v != (secondary or target) and v in signals]
                secondary_flow_before = max(self.validator.information_flow_between(signals, s, secondary or target) for s in sec_sources if s not in removed)
                sec_remaining = [s for s in sec_sources if s not in removed]
                secondary_flow_after = max(self.validator.information_flow_between(signals, s, secondary or target) for s in sec_remaining) if sec_remaining else 0.0
                secondary_loss = (secondary_flow_before - secondary_flow_after) / max(secondary_flow_before, 1e-12)

                info_loss = (full_info - removed_info) / max(full_info, 1e-12)

                experiment_results.append({
                    "name": exp_name,
                    "removed": list(removed),
                    "question": exp["question"],
                    "graph_info_score": removed_info,
                    "info_loss": info_loss,
                    "avg_strength_change_shared_edges": avg_strength_change,
                    "target_information_loss": target_info_loss,
                    "secondary_information_loss": secondary_loss,
                    "target_flow_before": target_flow_before,
                    "target_flow_after": target_flow_after,
                    "edge_count": len(graph.edges),
                    "chain": chain,
                })
                print(f"  [{exp_name}] removed={removed}, target_info_loss={target_info_loss:.4f}, target_flow: {target_flow_before:.4f}->{target_flow_after:.4f}, chain={chain}")
            except Exception as e:
                print(f"  [{exp_name}] FAILED: {e}")
                experiment_results.append({"name": exp_name, "removed": list(removed), "error": str(e)})

        metrics["experiments"] = experiment_results

        n_critical = sum(1 for r in experiment_results if isinstance(r.get("target_information_loss"), (int, float)) and r["target_information_loss"] > 0.3)
        n_weak = sum(1 for r in experiment_results if isinstance(r.get("target_information_loss"), (int, float)) and r["target_information_loss"] < 0.1)
        metrics["critical_removals_by_info_loss"] = n_critical
        metrics["weak_removals_by_info_loss"] = n_weak

        # Also check if any removal SHORTENED the chain materially
        full_chain_set = set(full_chain)
        chain_changes = []
        for r in experiment_results:
            r_chain = r.get("chain", [])
            chain_change = len(full_chain_set - set(r_chain)) + len(set(r_chain) - full_chain_set)
            chain_changes.append(chain_change)
        metrics["chain_disruptions"] = chain_changes

        n_critical_total = n_critical + sum(1 for c in chain_changes if c >= 2)

        if n_critical_total >= 2:
            status = "PASSED"
        elif n_critical_total >= 1:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return AttackResult(
            attack_name="node_removal_attacks",
            status=status,
            metrics=metrics,
        )
