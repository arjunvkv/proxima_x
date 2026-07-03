from __future__ import annotations

from typing import Any

from research.causal_physics.generator_discovery import GeneratorCandidate

_TARGET_VARS = [
    "adaptive_time",
    "energy_storage",
    "memory_density",
    "memory_gradient",
    "state_mutation_rate",
    "regime_change_probability",
]


class CausalOrderingEngine:
    def compute(self, candidates: list[GeneratorCandidate]) -> dict[str, Any]:
        te_lookup: dict[tuple[str, str], float] = {}
        for c in candidates:
            te_lookup[(c.source_variable, c.target_variable)] = c.transfer_entropy

        adj: dict[tuple[str, str], float] = {}
        flow: dict[tuple[str, str], float] = {}

        for c in candidates:
            if c.peak_lag < 0:
                causer, caused = c.source_variable, c.target_variable
            elif c.peak_lag > 0:
                causer, caused = c.target_variable, c.source_variable
            else:
                continue

            key = (causer, caused)
            if key not in adj or c.causal_strength > adj[key]:
                adj[key] = c.causal_strength
                flow[key] = te_lookup.get((causer, caused), c.transfer_entropy)

        edges = list(adj.keys())
        for a, b in edges:
            if (b, a) in adj:
                if adj[(a, b)] >= adj[(b, a)]:
                    del adj[(b, a)]
                    del flow[(b, a)]
                else:
                    del adj[(a, b)]
                    del flow[(a, b)]

        all_vars: list[str] = []
        seen: set[str] = set()
        for src, tgt in adj:
            if src not in seen:
                all_vars.append(src)
                seen.add(src)
            if tgt not in seen:
                all_vars.append(tgt)
                seen.add(tgt)

        out_degree: dict[str, int] = {}
        for src, _ in adj:
            out_degree[src] = out_degree.get(src, 0) + 1

        topological_order = sorted(all_vars, key=lambda v: out_degree.get(v, 0), reverse=True)

        for var in _TARGET_VARS:
            if var not in seen:
                topological_order.append(var)

        return {
            "adjacency_matrix": dict(adj),
            "topological_order": topological_order,
            "information_flow_matrix": dict(flow),
        }
