from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)


NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]
NOISE_TARGETS = ["memory_density", "adaptive_time", "energy_storage"]


class NoiseAttack:
    """Attack 10: Noise Injection.

    Inject 1%, 5%, 10%, 20% noise into memory_density, adaptive_time, energy_storage
    and measure graph degradation.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 50

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        full_graph, full_cands, full_ordering = self.validator.build_causal_graph(signals)
        full_chain = full_graph.get_market_physics_chain()
        baseline_edges = {(e.source, e.target): abs(e.causal_strength) for e in full_graph.edges}

        experiments = []
        all_metrics: dict[str, Any] = {
            "full_chain": full_chain,
        }

        for target_var in NOISE_TARGETS:
            orig_signal = np.asarray(signals.get(target_var, np.zeros(1)), dtype=np.float64)
            if len(orig_signal) < 10:
                continue

            sigma = float(np.std(orig_signal))
            if sigma < 1e-12:
                continue

            for noise_level in NOISE_LEVELS:
                try:
                    noise_std = sigma * noise_level
                    noise = np.random.normal(0.0, noise_std, len(orig_signal)).astype(np.float64)
                    noisy_signals = dict(signals)
                    noisy_signals[target_var] = orig_signal + noise

                    graph, cands, ordering = self.validator.build_causal_graph(noisy_signals)
                    noisy_edges = {(e.source, e.target): abs(e.causal_strength) for e in graph.edges}
                    chain = graph.get_market_physics_chain()

                    # Edge rank correlation
                    common = list(set(baseline_edges.keys()) & set(noisy_edges.keys()))
                    if len(common) > 5:
                        base_vals = np.array([baseline_edges[e] for e in common])
                        noisy_vals = np.array([noisy_edges[e] for e in common])
                        rank_corr = float(np.corrcoef(
                            np.argsort(np.argsort(base_vals)),
                            np.argsort(np.argsort(noisy_vals))
                        )[0, 1])
                    else:
                        rank_corr = 0.0

                    # Chain Jaccard
                    base_chain_set = set(full_chain)
                    noisy_chain_set = set(chain)
                    chain_jaccard = len(base_chain_set & noisy_chain_set) / max(len(base_chain_set | noisy_chain_set), 1)

                    # Composite degradation
                    degradation_rate = 0.6 * (1.0 - max(0.0, rank_corr)) + 0.4 * (1.0 - chain_jaccard)

                    experiments.append({
                        "target_variable": target_var,
                        "noise_level": noise_level,
                        "rank_correlation": rank_corr,
                        "chain_jaccard": chain_jaccard,
                        "degradation_rate": degradation_rate,
                        "edge_count": len(graph.edges),
                        "chain": chain,
                    })
                    print(f"  [{target_var}] noise={noise_level:.0%}: rank_corr={rank_corr:.4f}, chain_jaccard={chain_jaccard:.4f}, degradation={degradation_rate:.4f}")
                except Exception as e:
                    print(f"  [{target_var}] noise={noise_level:.0%}: FAILED - {e}")
                    experiments.append({
                        "target_variable": target_var,
                        "noise_level": noise_level,
                        "error": str(e),
                    })

        all_metrics["experiments"] = experiments

        if experiments:
            degradations = [e["degradation_rate"] for e in experiments if isinstance(e.get("degradation_rate"), (int, float))]
            avg_degradation = float(np.mean(degradations)) if degradations else 0.0
            max_degradation = float(np.max(degradations)) if degradations else 0.0
            all_metrics["avg_degradation"] = avg_degradation
            all_metrics["max_degradation"] = max_degradation

        status = "PASSED"
        for e in experiments:
            if isinstance(e.get("degradation_rate"), (int, float)) and e["degradation_rate"] > 0.5:
                if e.get("noise_level", 0) <= 0.05:
                    status = "FAILED"
                    break
                elif e.get("noise_level", 0) <= 0.10:
                    status = "INCONCLUSIVE"

        print(f"  Noise attack: {status} (avg_degradation={all_metrics.get('avg_degradation', 0):.4f}, max_degradation={all_metrics.get('max_degradation', 0):.4f})")

        return AttackResult("noise_injection", status, metrics=all_metrics)
