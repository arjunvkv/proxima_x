from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, CANONICAL_CHAIN, _clean_serializable,
)
from research.causal_physics.generator_graph import GeneratorGraph
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


REMOVAL_SEQUENCE = ["compression", "energy_storage", "memory_density", "adaptive_time"]


class ChainCollapse:
    """Attack 12: Full Chain Collapse Test.

    Sequentially remove compression, energy_storage, memory_density, adaptive_time
    and measure prediction/information loss for mutation_rate and regime_change.
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

        n = len(signals.get("adaptive_time", np.zeros(1)))
        mutation = np.asarray(signals.get("state_mutation_rate", np.zeros(n)), dtype=np.float64)
        regime = np.asarray(signals.get("regime_change_probability", np.zeros(n)), dtype=np.float64)

        phases = []
        removed_set: set[str] = set()

        phase0_info = full_info
        phase0_mutation_pred = self._prediction_score(signals, mutation, "state_mutation_rate", removed_set)
        phase0_regime_pred = self._prediction_score(signals, regime, "regime_change_probability", removed_set)

        phases.append({
            "phase": 0,
            "removed": list(removed_set),
            "graph_info": full_info,
            "mutation_prediction": phase0_mutation_pred,
            "regime_prediction": phase0_regime_pred,
            "chain": full_chain,
            "cumulative_info_loss": 0.0,
        })

        print(f"  Phase 0 (none removed): info={full_info:.4f}, mutation_pred={phase0_mutation_pred:.4f}")

        for removal_step in REMOVAL_SEQUENCE:
            removed_set.add(removal_step)

            available = [v for v in TARGET_VARIABLES if v not in removed_set]
            if len(available) < 2:
                phases.append({
                    "phase": len(removed_set),
                    "removed": list(removed_set),
                    "graph_info": 0.0,
                    "mutation_prediction": 0.0,
                    "regime_prediction": 0.0,
                    "chain": [],
                    "cumulative_info_loss": 1.0,
                    "note": "too few variables remaining",
                })
                print(f"  Phase {len(removed_set)} (removed={removal_step}): too few variables")
                break

            try:
                graph, cands = self.validator.build_graph_with_removed_vars(signals, removed_set)
                phase_info = self.validator.graph_information_score(graph)
                chain = graph.get_market_physics_chain()

                mutation_pred = self._prediction_score(signals, mutation, "state_mutation_rate", removed_set)
                regime_pred = self._prediction_score(signals, regime, "regime_change_probability", removed_set)

                info_loss = (phase0_info - phase_info) / max(phase0_info, 1e-12)
                mutation_loss = (phase0_mutation_pred - mutation_pred) / max(phase0_mutation_pred, 1e-12)
                regime_loss = (phase0_regime_pred - regime_pred) / max(phase0_regime_pred, 1e-12)

                phases.append({
                    "phase": len(removed_set),
                    "removed": list(removed_set),
                    "removed_step": removal_step,
                    "graph_info": phase_info,
                    "mutation_prediction": mutation_pred,
                    "regime_prediction": regime_pred,
                    "chain": chain,
                    "info_loss_since_phase0": info_loss,
                    "mutation_pred_loss_since_phase0": mutation_loss,
                    "regime_pred_loss_since_phase0": regime_loss,
                    "cumulative_info_loss": max(0.0, info_loss),
                })
                print(f"  Phase {len(removed_set)} (removed={removal_step}): info={phase_info:.4f}, info_loss={info_loss:.4f}, chain={chain}")
            except Exception as e:
                print(f"  Phase {len(removed_set)} (removed={removal_step}): FAILED - {e}")
                phases.append({
                    "phase": len(removed_set),
                    "removed": list(removed_set),
                    "removed_step": removal_step,
                    "error": str(e),
                })

        critical_nodes = []
        for phase in phases[1:]:
            if phase.get("cumulative_info_loss", 0) > 0.3:
                critical_nodes.append(phase.get("removed_step", phase.get("removed", [])))

        final_mutation_loss = phases[-1].get("mutation_pred_loss_since_phase0", 0) if len(phases) > 1 else 0
        final_regime_loss = phases[-1].get("regime_pred_loss_since_phase0", 0) if len(phases) > 1 else 0

        metrics = {
            "full_chain": full_chain,
            "removal_sequence": REMOVAL_SEQUENCE,
            "phases": phases,
            "critical_nodes": [n for n in critical_nodes if isinstance(n, str)],
            "final_mutation_prediction_loss": final_mutation_loss,
            "final_regime_prediction_loss": final_regime_loss,
        }

        if final_mutation_loss > 0.5 or final_regime_loss > 0.5:
            status = "PASSED"
            print(f"  Chain collapse PASSED: mutation_loss={final_mutation_loss:.3f}, regime_loss={final_regime_loss:.3f}")
        elif final_mutation_loss > 0.2 or final_regime_loss > 0.2:
            status = "INCONCLUSIVE"
            print(f"  Chain collapse INCONCLUSIVE: mutation_loss={final_mutation_loss:.3f}")
        else:
            status = "FAILED"
            print(f"  Chain collapse FAILED: chain not essential (mutation_loss={final_mutation_loss:.3f})")

        return AttackResult("chain_collapse_test", status, metrics=metrics)

    def _prediction_score(self, signals: dict, target: NDArray[np.float64],
                          target_name: str, removed_set: set[str]) -> float:
        predictors = [v for v in TARGET_VARIABLES if v != target_name and v not in removed_set]
        if not predictors:
            return 0.0

        n = len(target)
        scores = []
        for p in predictors:
            if p not in signals:
                continue
            sig = np.asarray(signals[p], dtype=np.float64)
            common = min(len(sig), n)
            if common < self._max_lag * 2 + 1:
                continue
            corr = AdaptiveTimeCausality._cross_correlate(
                sig[:common], target[:common], self._max_lag
            )
            scores.append(float(np.max(np.abs(corr))))

        # Use max correlation (best predictor), not mean
        # This avoids dilution from weakly-predictive variables
        return float(max(scores)) if scores else 0.0
