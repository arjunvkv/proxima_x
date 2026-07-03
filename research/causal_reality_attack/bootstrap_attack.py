from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph
from research.causal_physics.generator_discovery import GeneratorDiscoveryEngine, GeneratorCandidate
from research.causal_physics.causal_ordering import CausalOrderingEngine
from research.causal_physics.generator_graph import GeneratorGraphBuilder


@numba.jit(nopython=True, cache=True)
def _bootstrap_sample_indices(n: int) -> NDArray[np.int32]:
    idx = np.empty(n, dtype=np.int32)
    for i in range(n):
        idx[i] = np.random.randint(0, n)
    return idx


class BootstrapAttack:
    """Attack 9: Bootstrap Stability.

    Run 1000 bootstrap samples, measuring edge survival and ordering stability.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY",
                 n_bootstrap: int = 1000, max_edges: int = 200):
        self.validator = validator
        self.asset = asset
        self.n_bootstrap = n_bootstrap
        self.max_edges = max_edges

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        full_graph, full_cands, full_ordering = self.validator.build_causal_graph(signals)
        full_edges = {(e.source, e.target) for e in full_graph.edges}

        if not full_edges:
            return AttackResult("bootstrap_stability", "FAILED",
                                metrics={"error": "no edges in full graph"})

        signal_names = [k for k, v in signals.items() if isinstance(v, np.ndarray) and len(v) > 0]
        n = min(len(signals.get("adaptive_time", [0])), len(signals.get("energy_storage", [0])), len(signals.get("memory_density", [0])))

        if n < 50:
            return AttackResult("bootstrap_stability", "FAILED",
                                metrics={"error": f"insufficient data length: {n}"})

        edge_counts: dict[tuple[str, str], int] = {}
        chain_lengths = []
        n_valid = 0

        for boot_i in range(self.n_bootstrap):
            if boot_i % 200 == 0:
                print(f"    bootstrap {boot_i}/{self.n_bootstrap}")

            try:
                idx = _bootstrap_sample_indices(n)
                boot_signals = {}
                for k in signal_names:
                    arr = np.asarray(signals.get(k, np.zeros(n)), dtype=np.float64)
                    boot_signals[k] = arr[idx]

                engine = GeneratorDiscoveryEngine()
                candidates = engine.compute(boot_signals)

                cand_dicts = [{
                    "source": c.source_variable, "target": c.target_variable,
                    "causal_strength": c.causal_strength,
                    "information_flow": c.transfer_entropy,
                    "peak_lag": c.peak_lag, "peak_corr": c.peak_corr,
                } for c in candidates]

                order_engine = CausalOrderingEngine()
                ordering_result = order_engine.compute(candidates)
                ordering = ordering_result.get("adjacency_matrix", {})

                builder = GeneratorGraphBuilder()
                graph = builder.build(cand_dicts, ordering)

                for e in graph.edges:
                    key = (e.source, e.target)
                    edge_counts[key] = edge_counts.get(key, 0) + 1

                chain = graph.get_market_physics_chain()
                chain_lengths.append(len(chain))
                n_valid += 1

            except Exception:
                pass

        if n_valid == 0:
            return AttackResult("bootstrap_stability", "FAILED",
                                metrics={"error": "all bootstrap samples failed"})

        survival_rates = {}
        for src, tgt in full_edges:
            key = (src, tgt)
            count = edge_counts.get(key, 0)
            survival_rates[f"{src}->{tgt}"] = count / n_valid

        all_rates = list(survival_rates.values())
        avg_survival = float(np.mean(all_rates)) if all_rates else 0.0

        n_surviving_half = sum(1 for r in all_rates if r >= 0.5)
        n_surviving_quarter = sum(1 for r in all_rates if r >= 0.25)

        metrics = {
            "full_edges": [(e.source, e.target) for e in full_graph.edges],
            "n_bootstrap_valid": n_valid,
            "edge_survival_rates": survival_rates,
            "avg_survival_rate": float(avg_survival),
            "n_edges_surviving_50pct": n_surviving_half,
            "n_edges_surviving_25pct": n_surviving_quarter,
            "avg_chain_length": float(np.mean(chain_lengths)) if chain_lengths else 0.0,
        }

        if avg_survival > 0.5:
            status = "PASSED"
            print(f"  Bootstrap PASSED: avg_survival={avg_survival:.3f}, {n_surviving_half}/{len(all_rates)} edges > 50%")
        elif avg_survival > 0.25:
            status = "INCONCLUSIVE"
            print(f"  Bootstrap INCONCLUSIVE: avg_survival={avg_survival:.3f}")
        else:
            status = "FAILED"
            print(f"  Bootstrap FAILED: avg_survival={avg_survival:.3f}")

        return AttackResult("bootstrap_stability", status, metrics=metrics)
