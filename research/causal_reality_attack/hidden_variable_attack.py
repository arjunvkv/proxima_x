from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
from research.information_discovery.mi_estimator import _fast_conditional_mutual_info


ALTERNATIVE_CANDIDATES = [
    "entropy",
    "compression",
    "behavior_density",
    "cohort_alignment",
    "memory_conflict",
    "energy_dissipation",
]


class HiddenVariableAttack:
    """Attack 11: Hidden Variable Challenge.

    Test whether any alternative Proxima feature can replace nodes in the
    discovered causal chain.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 50

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        graph, cands, ordering = self.validator.build_causal_graph(signals)
        chain = graph.get_market_physics_chain()

        full_info = self.validator.graph_information_score(graph)

        n = len(signals.get("adaptive_time", np.zeros(1)))

        returns = np.asarray(signals.get("returns", np.zeros(n)), dtype=np.float64)
        vols = np.asarray(signals.get("volatility", np.zeros(n)), dtype=np.float64)

        entropy = self._compute_entropy(returns, 20)
        compression = self._rolling_std(returns, 20)
        behavior_density = self._behavior_density(signals.get("states", np.zeros(n, dtype=np.int64)), n)
        cohort_alignment = self._cohort_alignment(signals.get("memory_density", np.zeros(n)), signals.get("time_regime", np.zeros(n, dtype=np.int64)), n)
        memory_conflict = self._memory_conflict(signals.get("memory_density", np.zeros(n)), signals.get("memory_gradient", np.zeros(n)))
        energy_dissipation = np.asarray(signals.get("energy_dissipation", np.zeros(n)), dtype=np.float64)

        alt_signals = {
            "entropy": entropy,
            "compression": compression,
            "behavior_density": behavior_density,
            "cohort_alignment": cohort_alignment,
            "memory_conflict": memory_conflict,
            "energy_dissipation": energy_dissipation,
        }

        chain_targets = ["energy_storage", "memory_density", "adaptive_time", "state_mutation_rate", "regime_change_probability"]

        replacement_results = []

        for alt_name, alt_sig in alt_signals.items():
            alt_info = 0.0
            replacements_found = 0
            for target in chain_targets:
                if target not in signals:
                    continue
                orig_sig = np.asarray(signals[target], dtype=np.float64)
                common = min(len(alt_sig), len(orig_sig))

                if common < self._max_lag * 2 + 1:
                    continue

                corr = AdaptiveTimeCausality._cross_correlate(
                    alt_sig[:common], orig_sig[:common], self._max_lag
                )
                peak_r = float(np.max(np.abs(corr)))

                if peak_r > 0.3:
                    replacements_found += 1
                    alt_info += peak_r

            replacement_results.append({
                "alternative": alt_name,
                "peaks_above_03": replacements_found,
                "cumulative_info": alt_info,
            })
            print(f"  [{alt_name}] replacements_above_0.3={replacements_found}")

        replacement_results.sort(key=lambda x: x["cumulative_info"], reverse=True)

        best_replacement = replacement_results[0] if replacement_results else None

        chain_replacement_test = {}
        for alt_name, alt_sig in alt_signals.items():
            flow_scores = {}
            for target in chain_targets:
                if target not in signals:
                    continue
                orig_sig = np.asarray(signals[target], dtype=np.float64)
                common = min(len(alt_sig), len(orig_sig))
                if common < 3:
                    continue
                te = _fast_conditional_mutual_info(
                    alt_sig[:common - 1], orig_sig[1:common], orig_sig[:common - 1], 20
                )
                flow_scores[target] = float(te)
            chain_replacement_test[alt_name] = flow_scores

        best_alt_name = best_replacement["alternative"] if best_replacement else "none"
        best_alt_info = best_replacement["cumulative_info"] if best_replacement else 0.0

        metrics = {
            "chain_targets": chain_targets,
            "replacement_results": replacement_results,
            "chain_replacement_info_flow": chain_replacement_test,
            "best_alternative": best_alt_name,
            "best_alternative_info": best_alt_info,
            "full_chain_info": float(full_info),
        }

        if best_alt_info > full_info * 0.8 and best_replacement and best_replacement["peaks_above_03"] >= 3:
            status = "FAILED"
            print(f"  Hidden variable '{best_alt_name}' can replace chain (info={best_alt_info:.4f} vs full={full_info:.4f})")
        elif best_alt_info > full_info * 0.5:
            status = "INCONCLUSIVE"
            print(f"  Hidden variable '{best_alt_name}' partially replaces chain")
        else:
            status = "PASSED"
            print(f"  No hidden variable can replace chain (best={best_alt_name}, info={best_alt_info:.4f})")

        return AttackResult("hidden_variable_challenge", status, metrics=metrics)

    def _compute_entropy(self, returns: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        n = len(returns)
        result = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            chunk = returns[i - window:i]
            if len(chunk) < 2:
                continue
            uq = np.unique(np.floor(chunk * 100) / 100)
            p = np.ones(len(uq), dtype=np.float64) / len(uq)
            ent = -np.sum(p * np.log(p))
            result[i] = ent
        return result

    def _rolling_std(self, arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        n = len(arr)
        result = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            chunk = arr[i - window:i]
            result[i] = float(np.std(chunk))
        return result

    def _behavior_density(self, states: NDArray[np.int64], n: int) -> NDArray[np.float64]:
        result = np.zeros(n, dtype=np.float64)
        window = 20
        for i in range(window, min(n, len(states))):
            chunk = states[i - window:i]
            uq = len(np.unique(chunk))
            result[i] = uq / window
        return result

    def _cohort_alignment(self, memory_density: NDArray[np.float64],
                          time_regime: NDArray[np.int64], n: int) -> NDArray[np.float64]:
        result = np.zeros(n, dtype=np.float64)
        window = 20
        for i in range(window, min(n, len(memory_density), len(time_regime))):
            md = memory_density[i - window:i]
            tr = time_regime[i - window:i]
            if np.std(md) > 0 and np.std(tr) > 0:
                result[i] = float(np.corrcoef(md, tr)[0, 1])
        return result

    def _memory_conflict(self, density: NDArray[np.float64], gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        n = min(len(density), len(gradient))
        return np.abs(density[:n] - gradient[:n])
