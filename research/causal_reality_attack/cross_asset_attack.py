from __future__ import annotations

from typing import Any

import numpy as np

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_ASSETS, TARGET_VARIABLES, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph


class CrossAssetAttack:
    """Attack 1: Cross-Asset Causal Invariance.

    Rebuild the complete causal graph for EURJPY, USDJPY, GBPJPY, XAUUSD
    and measure graph stability across assets.
    """

    def __init__(self, validator: AttackValidator):
        self.validator = validator

    def run(self) -> AttackResult:
        graphs: dict[str, GeneratorGraph] = {}
        candidates_map: dict[str, list[dict]] = {}
        chains: dict[str, list[str]] = {}
        metrics: dict[str, Any] = {}

        for asset in TARGET_ASSETS:
            try:
                data = self.validator.load_asset_data(asset)
                signals = self.validator.compute_signals(data)
                graph, cands, ordering = self.validator.build_causal_graph(signals)
                graphs[asset] = graph
                candidates_map[asset] = cands
                chain = graph.get_market_physics_chain()
                chains[asset] = chain
                metrics[f"{asset}_nodes"] = graph.nodes
                metrics[f"{asset}_edges"] = [(e.source, e.target) for e in graph.edges]
                metrics[f"{asset}_chain"] = chain
                metrics[f"{asset}_candidates"] = len(cands)
                print(f"  [{asset}] Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges, chain: {chain}")
            except Exception as e:
                print(f"  [{asset}] FAILED: {e}")
                metrics[f"{asset}_error"] = str(e)

        similarities = {}
        ref_asset = "EURJPY"
        if ref_asset in graphs:
            for asset in TARGET_ASSETS:
                if asset != ref_asset and asset in graphs:
                    sim = self.validator.graph_similarity(graphs[ref_asset], graphs[asset])
                    similarities[f"{ref_asset}_vs_{asset}"] = sim
                    print(f"  {ref_asset} vs {asset}: node_jaccard={sim['node_jaccard']:.3f}, edge_jaccard={sim['edge_jaccard']:.3f}, order_sim={sim['order_similarity']:.3f}")

        metrics["pairwise_similarities"] = similarities

        all_pairs = []
        assets_list = [a for a in TARGET_ASSETS if a in graphs]
        for i in range(len(assets_list)):
            for j in range(i + 1, len(assets_list)):
                a1, a2 = assets_list[i], assets_list[j]
                if a1 in graphs and a2 in graphs:
                    all_pairs.append(self.validator.graph_similarity(graphs[a1], graphs[a2]))

        avg_metrics = {}
        if all_pairs:
            avg_metrics["avg_node_jaccard"] = float(np.mean([p["node_jaccard"] for p in all_pairs]))
            avg_metrics["avg_edge_jaccard"] = float(np.mean([p["edge_jaccard"] for p in all_pairs]))
            avg_metrics["avg_order_similarity"] = float(np.mean([p["order_similarity"] for p in all_pairs]))
            avg_metrics["avg_strength_similarity"] = float(np.mean([p["strength_similarity"] for p in all_pairs]))
        metrics["averages"] = avg_metrics

        chain_scores = []
        chain_list = [c for c in chains.values() if c]
        if chain_list:
            for asset, chain in chains.items():
                canonical = ["energy_storage", "memory_density", "adaptive_time", "state_mutation_rate", "regime_change_probability"]
                overlap = len([v for v in chain if v in canonical])
                chain_scores.append(overlap / max(len(canonical), 1))
            metrics["chain_scores"] = {a: s for a, s in zip(chains.keys(), chain_scores)}
            avg_chain = float(np.mean(chain_scores)) if chain_scores else 0.0
            metrics["avg_chain_similarity"] = avg_chain
        else:
            avg_chain = 0.0

        status = "FAILED"
        if avg_metrics and avg_metrics.get("avg_node_jaccard", 0) > 0.4 and avg_metrics.get("avg_edge_jaccard", 0) > 0.2:
            status = "PASSED"
        elif avg_metrics and avg_metrics.get("avg_node_jaccard", 0) > 0.2:
            status = "INCONCLUSIVE"

        details = {
            "graphs_per_asset": {a: {"nodes": g.nodes, "edge_count": len(g.edges), "chain": chains.get(a, [])}
                                 for a, g in graphs.items()},
        }

        return AttackResult(
            attack_name="cross_asset_causal_invariance",
            status=status,
            metrics=metrics,
            details=details,
        )
