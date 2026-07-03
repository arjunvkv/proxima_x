from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph, GraphEdge


class RandomGraphAttack:
    """Attack 8: Random Graph Benchmark.

    Generate 1000 random causal graphs with identical node count and edge density,
    then compare the discovered graph's score against the random distribution.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY", n_random: int = 1000):
        self.validator = validator
        self.asset = asset
        self.n_random = n_random

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        graph, cands, ordering = self.validator.build_causal_graph(signals)

        real_score = self.validator.graph_information_score(graph)
        real_chain = graph.get_market_physics_chain()

        n_nodes = len(graph.nodes)
        n_edges = len(graph.edges)

        node_names = list(graph.nodes) if graph.nodes else TARGET_VARIABLES
        if not node_names:
            node_names = TARGET_VARIABLES

        random_scores = np.zeros(self.n_random, dtype=np.float64)
        for i in range(self.n_random):
            rg = self._random_graph(node_names, n_edges)
            random_scores[i] = self.validator.graph_information_score(rg)

        mean_random = float(np.mean(random_scores))
        std_random = float(np.std(random_scores))
        if std_random > 0:
            z_score = float((real_score - mean_random) / std_random)
        else:
            z_score = 0.0

        n_above = int(np.sum(random_scores >= real_score))
        p_value = float((n_above + 1) / (self.n_random + 1))

        percentile = float(np.sum(random_scores < real_score)) / self.n_random * 100.0

        metrics = {
            "real_graph_score": real_score,
            "random_mean": mean_random,
            "random_std": std_random,
            "z_score": z_score,
            "p_value": p_value,
            "percentile": percentile,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "real_chain": real_chain,
            "random_scores_summary": {
                "min": float(np.min(random_scores)),
                "max": float(np.max(random_scores)),
                "p5": float(np.percentile(random_scores, 5)),
                "p25": float(np.percentile(random_scores, 25)),
                "p50": float(np.percentile(random_scores, 50)),
                "p75": float(np.percentile(random_scores, 75)),
                "p95": float(np.percentile(random_scores, 95)),
            },
        }

        if p_value < 0.05 and z_score > 2.0:
            status = "PASSED"
            print(f"  Graph is SPECIAL (z={z_score:.2f}, p={p_value:.4f})")
        elif p_value < 0.10 and z_score > 1.0:
            status = "INCONCLUSIVE"
            print(f"  Graph is somewhat special (z={z_score:.2f}, p={p_value:.4f})")
        else:
            status = "FAILED"
            print(f"  Graph is NOT special (z={z_score:.2f}, p={p_value:.4f})")

        return AttackResult("random_graph_benchmark", status, metrics=metrics)

    def _random_graph(self, node_names: list[str], n_edges: int) -> GeneratorGraph:
        n_nodes = len(node_names)
        max_edges = n_nodes * (n_nodes - 1)
        actual_edges = min(n_edges, max(1, max_edges))

        edges = []
        candidates = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        if actual_edges < len(candidates):
            chosen = np.random.choice(len(candidates), size=actual_edges, replace=False)
            selected = [candidates[idx] for idx in chosen]
        else:
            selected = candidates

        for i, j in selected:
            edges.append(GraphEdge(
                source=node_names[i],
                target=node_names[j],
                causal_strength=float(abs(np.random.normal(0.3, 0.15))),
                information_flow=float(abs(np.random.normal(0.05, 0.03))),
            ))

        topo = list(node_names)
        np.random.shuffle(topo)

        return GeneratorGraph(nodes=node_names, edges=edges, topological_order=topo)
