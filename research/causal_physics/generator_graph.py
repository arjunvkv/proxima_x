from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class GraphEdge:
    source: str
    target: str
    causal_strength: float
    information_flow: float
    survival_probability: float = 1.0


@dataclass
class GeneratorGraph:
    nodes: list[str]
    edges: list[GraphEdge]
    topological_order: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "causal_strength": e.causal_strength,
                    "information_flow": e.information_flow,
                    "survival_probability": e.survival_probability,
                }
                for e in self.edges
            ],
            "topological_order": self.topological_order,
        }

    def get_strongest_path(self, source: str, target: str) -> list[str]:
        edges_by_source: dict[str, list[GraphEdge]] = {}
        for e in self.edges:
            edges_by_source.setdefault(e.source, []).append(e)

        best_path: list[str] = []
        best_min_strength = -1.0
        visited: set[str] = set()

        def _dfs(current: str, target: str, path: list[str], min_strength: float) -> None:
            nonlocal best_path, best_min_strength
            if current == target:
                if not best_path or min_strength > best_min_strength:
                    best_path = list(path)
                    best_min_strength = min_strength
                return
            for e in edges_by_source.get(current, []):
                if e.target not in visited:
                    visited.add(e.target)
                    new_min = min(min_strength, e.causal_strength)
                    _dfs(e.target, target, path + [e.target], new_min)
                    visited.remove(e.target)

        visited.add(source)
        _dfs(source, target, [source], float("inf"))
        return best_path

    def get_market_physics_chain(self) -> list[str]:
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for e in self.edges:
            in_degree[e.target] = in_degree.get(e.target, 0) + 1

        sources = [n for n, d in in_degree.items() if d == 0]
        if not sources:
            sources = [self.topological_order[0]] if self.topological_order else []

        chain: list[str] = []
        visited: set[str] = set()

        def _longest_dfs(current: str, path: list[str]) -> None:
            nonlocal chain
            outgoing = [e for e in self.edges if e.source == current]
            if not outgoing:
                if len(path) > len(chain):
                    chain = list(path)
                return
            for e in sorted(outgoing, key=lambda x: x.causal_strength, reverse=True):
                if e.target not in visited:
                    visited.add(e.target)
                    _longest_dfs(e.target, path + [e.target])
                    visited.remove(e.target)

        for s in sources:
            visited.add(s)
            _longest_dfs(s, [s])
            visited.remove(s)

        return chain if chain else self.topological_order

    def to_adjacency_matrix(self) -> NDArray[np.float64]:
        n = len(self.nodes)
        idx_map = {name: i for i, name in enumerate(self.nodes)}
        mat = np.zeros((n, n), dtype=np.float64)
        for e in self.edges:
            i = idx_map.get(e.source)
            j = idx_map.get(e.target)
            if i is not None and j is not None:
                mat[i, j] = e.causal_strength
        return mat


class GeneratorGraphBuilder:
    TARGET_VARIABLES = [
        "adaptive_time",
        "energy_storage",
        "memory_density",
        "state_mutation_rate",
        "regime_volatility",
        "information_flow",
    ]

    def build(self, candidates: list[dict[str, Any]], ordering: dict[str, int]) -> GeneratorGraph:
        edges: list[GraphEdge] = []
        for c in candidates:
            source = str(c.get("source", ""))
            target = str(c.get("target", ""))
            if not source or not target:
                continue
            edge = GraphEdge(
                source=source,
                target=target,
                causal_strength=float(c.get("causal_strength", 0.0)),
                information_flow=float(c.get("information_flow", 0.0)),
                survival_probability=float(c.get("survival_probability", 1.0)),
            )
            edges.append(edge)

        nodes = sorted(set(e.source for e in edges) | set(e.target for e in edges))
        if not nodes:
            nodes = list(self.TARGET_VARIABLES)

        ordered = sorted(nodes, key=lambda x: ordering.get(x, len(nodes)))
        for n in nodes:
            if n not in ordering:
                ordered.remove(n)
                ordered.append(n)

        return GeneratorGraph(nodes=nodes, edges=edges, topological_order=ordered)
