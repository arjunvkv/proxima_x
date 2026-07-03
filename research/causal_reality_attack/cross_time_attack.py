from __future__ import annotations

from typing import Any

import numpy as np

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_ASSETS, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph


TIME_WINDOWS = [
    ("2018-01-01", "2020-01-01", "2018-2020"),
    ("2020-01-01", "2022-01-01", "2020-2022"),
    ("2022-01-01", "2024-01-01", "2022-2024"),
    ("2024-01-01", "2027-01-01", "2024-2026"),
]


class CrossTimeAttack:
    """Attack 2: Cross-Time Causal Invariance.

    Split data into 4 time windows (2018-2020, 2020-2022, 2022-2024, 2024-2026)
    and rebuild the causal graph for each window to measure temporal stability.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AttackResult:
        window_results: dict[str, dict] = {}
        all_graphs: dict[str, GeneratorGraph] = {}
        metrics: dict[str, Any] = {}

        for start, end, label in TIME_WINDOWS:
            try:
                data = self.validator.load_data_window(self.asset, start, end)
                signals = self.validator.compute_signals(data)
                graph, cands, ordering = self.validator.build_causal_graph(signals)
                all_graphs[label] = graph
                chain = graph.get_market_physics_chain()
                window_results[label] = {
                    "n_points": len(data["price"]),
                    "nodes": graph.nodes,
                    "edges": [(e.source, e.target) for e in graph.edges],
                    "chain": chain,
                    "n_candidates": len(cands),
                }
                metrics[f"{label}_chain"] = chain
                print(f"  [{label}] {len(data['price'])} pts, chain: {chain}")
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")
                window_results[label] = {"error": str(e)}

        similarities = {}
        labels = [l for _, _, l in TIME_WINDOWS if l in all_graphs]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                sim = self.validator.graph_similarity(all_graphs[l1], all_graphs[l2])
                similarities[f"{l1}_vs_{l2}"] = sim
                print(f"  {l1} vs {l2}: node_jaccard={sim['node_jaccard']:.3f}, edge_jaccard={sim['edge_jaccard']:.3f}")

        metrics["pairwise_similarities"] = similarities

        avg_metrics = {}
        if similarities:
            vals = list(similarities.values())
            avg_metrics["avg_node_jaccard"] = float(np.mean([v["node_jaccard"] for v in vals]))
            avg_metrics["avg_edge_jaccard"] = float(np.mean([v["edge_jaccard"] for v in vals]))
            avg_metrics["avg_order_similarity"] = float(np.mean([v["order_similarity"] for v in vals]))
            avg_metrics["avg_strength_similarity"] = float(np.mean([v["strength_similarity"] for v in vals]))
        metrics["averages"] = avg_metrics

        chain_set = set(tuple(w.get("chain", [])) for w in window_results.values() if isinstance(w.get("chain"), list))
        metrics["unique_chains"] = len(chain_set)

        status = "FAILED"
        if avg_metrics:
            if avg_metrics.get("avg_node_jaccard", 0) > 0.35 and avg_metrics.get("avg_edge_jaccard", 0) > 0.15:
                status = "PASSED"
            elif avg_metrics.get("avg_node_jaccard", 0) > 0.15:
                status = "INCONCLUSIVE"

        return AttackResult(
            attack_name="cross_time_causal_invariance",
            status=status,
            metrics=metrics,
            details={"windows": window_results},
        )
