from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator
from research.information_discovery.feature_scorer import FeatureScorer, FeatureSurvivalEngine, FeatureScore
from research.information_discovery.information_stability import InformationStability
from research.information_discovery.state_constructor import StateConstructor
from research.information_discovery.sid_sir import SIDCalculator, SIRCalculator
from research.information_discovery.sequence_discovery import SequenceDiscovery
from research.information_discovery.behavioral_genome import BehavioralGenomeEngine
from research.information_discovery.transition_intelligence import TransitionIntelligence
from research.information_discovery.information_compression import InformationCompression
from research.information_discovery.validation_framework import ValidationFramework
from research.forward_analyzer import ForwardAnalyzer


class DiscoveryPipeline:
    def __init__(self):
        self.mi = MIEstimator(n_bins=20)
        self.feature_scorer = FeatureScorer(mi_estimator=self.mi)
        self.survival_engine = FeatureSurvivalEngine(mi_threshold=0.0)
        self.stability = InformationStability(mi_estimator=self.mi)
        self.state_constructor = StateConstructor(mi_estimator=self.mi)
        self.sid_calc = SIDCalculator(mi_estimator=self.mi)
        self.sir_calc = SIRCalculator(sid_calculator=self.sid_calc)
        self.sequence_discovery = SequenceDiscovery(mi_estimator=self.mi)
        self.behavioral_genome = BehavioralGenomeEngine(mi_estimator=self.mi)
        self.transition_intel = TransitionIntelligence(mi_estimator=self.mi)
        self.compression = InformationCompression(mi_estimator=self.mi)
        self.validator = ValidationFramework(mi_estimator=self.mi)
        self.forward_analyzer = ForwardAnalyzer(horizons=[1, 5, 20, 50])

    def build_targets(self, price: NDArray, returns: NDArray, volume: NDArray) -> dict[str, NDArray]:
        metrics = self.forward_analyzer.compute_all_forward_metrics(price, returns)
        targets: dict[str, NDArray] = {}
        for key, arr in metrics.items():
            targets[key] = arr.astype(np.float64)
        rolling_vol = np.zeros_like(returns)
        for i in range(20, len(returns)):
            rolling_vol[i] = float(np.std(returns[i - 20 : i]))
        targets["regime"] = (rolling_vol > np.percentile(rolling_vol[rolling_vol > 0], 70)).astype(np.int32)
        return targets

    def build_features(self, all_features: dict[str, NDArray]) -> dict[str, NDArray]:
        features: dict[str, NDArray] = {}
        for name, arr in all_features.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                features[name] = arr.astype(np.float64)
        return features

    def run_information_discovery(self, features: dict[str, NDArray], targets: dict[str, NDArray]) -> dict:
        scores = self.feature_scorer.score_all_features(features, targets)
        survivors = self.survival_engine.filter_survivors(scores)
        ranked = self.survival_engine.rank_survivors(survivors)
        stability_results = self.stability.compute_all_stability(features, targets)
        mi_ranking = self.compression.rank_by_information_density(features, list(targets.values())[0])
        return {
            "scores": {s.name: s for s in scores},
            "survivors": [s.name for s in survivors],
            "ranked_survivors": [(s.name, s.composite_score) for s in ranked],
            "stability": stability_results,
            "mi_ranking": mi_ranking,
            "n_total_features": len(features),
            "n_survivors": len(survivors),
            "n_discarded": len(features) - len(survivors),
        }

    def run_state_construction(self, features: dict[str, NDArray], survivor_names: list[str], targets: dict[str, NDArray]) -> dict:
        forward_return = targets.get("forward_return_1", list(targets.values())[0])
        state_result = self.state_constructor.iterative_state_construction(features, survivor_names, forward_return)
        if state_result["n_clusters"] < 2:
            state_result = self.state_constructor.construct_from_survivors(features, survivor_names[:max(5, len(survivor_names) // 4)])
        compressed_dim = state_result.get("compressed", np.zeros((1, 1))).shape[1]
        sid_results = self.sid_calc.compute_sid_horizons(state_result["labels"], targets)
        sir_results = self.sir_calc.compute_sir_all(state_result["labels"], targets, compressed_dim)
        return {
            "state_result": state_result,
            "sid": sid_results,
            "sir": sir_results,
        }

    def run_sequence_analysis(self, states: NDArray[np.int32], targets: dict[str, NDArray]) -> dict:
        forward_return = targets.get("forward_return_1", list(targets.values())[0])
        seq_results = self.sequence_discovery.find_informative_sequences(states, forward_return)
        best_seq = self.sequence_discovery.find_best_sequence_length(states, forward_return)
        sequences_2 = self.sequence_discovery.extract_sequences(states, 2)
        seq_sid = 0.0
        if len(sequences_2) >= 2:
            seq_sid = self.sequence_discovery.compute_sequence_sid(sequences_2, forward_return)
        return {
            "informative_sequences": seq_results,
            "best_sequence": best_seq,
            "sequence_sid": seq_sid,
        }

    def run_transition_analysis(self, states: NDArray[np.int32], targets: dict[str, NDArray]) -> dict:
        forward_return = targets.get("forward_return_1", list(targets.values())[0])
        return self.transition_intel.compute_all_transition_metrics(states, forward_return)

    def run_genome_analysis(self, states: NDArray[np.int32]) -> dict:
        genome = self.behavioral_genome.build_genome(states)
        return {
            "genome": genome,
            "genome_length": len(genome),
        }

    def run_full_pipeline(self, features: dict[str, NDArray], price: NDArray, returns: NDArray, volume: NDArray) -> dict:
        targets = self.build_targets(price, returns, volume)
        info_discovery = self.run_information_discovery(features, targets)
        survivor_names = info_discovery["survivors"]
        if not survivor_names:
            survivor_names = list(features.keys())[:max(5, len(features) // 4)]
        state_construction = self.run_state_construction(features, survivor_names, targets)
        states = state_construction["state_result"]["labels"]
        sequence = self.run_sequence_analysis(states, targets)
        transition = self.run_transition_analysis(states, targets)
        genome = self.run_genome_analysis(states)
        multi_target_selected = self.compression.multi_target_selection(features, targets, top_k_per_target=10)
        ib_selected = self.compression.information_bottleneck(features, list(targets.values())[0], compression_ratio=0.5)
        return {
            "info_discovery": info_discovery,
            "state_construction": state_construction,
            "sequence": sequence,
            "transition": transition,
            "genome": genome,
            "compression": {
                "multi_target_selected": multi_target_selected,
                "information_bottleneck_selected": ib_selected,
            },
            "targets": {k: v.shape for k, v in targets.items()},
            "n_features_total": len(features),
            "n_features_surviving": len(survivor_names),
            "n_states": state_construction["state_result"]["n_clusters"],
        }

    def generate_report(self, results: dict) -> str:
        lines: list[str] = []
        lines.append("# PROXIMA X — Phase 3 Research Report")
        lines.append("")
        id_ = results.get("info_discovery", {})
        sc = results.get("state_construction", {})
        seq = results.get("sequence", {})
        trans = results.get("transition", {})
        comp = results.get("compression", {})
        lines.append("## Information Discovery")
        lines.append(f"- **Total Features**: {id_.get('n_total_features', 0)}")
        lines.append(f"- **Surviving Features**: {id_.get('n_survivors', 0)}")
        lines.append(f"- **Discarded**: {id_.get('n_discarded', 0)}")
        survivors = id_.get("ranked_survivors", [])
        lines.append("### Top Surviving Features")
        for name, score in survivors[:10]:
            lines.append(f"  - {name}: composite={score:.4f}")
        mi_rank = id_.get("mi_ranking", [])
        lines.append("### MI Ranking")
        for name, mi in mi_rank[:10]:
            lines.append(f"  - {name}: MI={mi:.4f}")
        lines.append("")
        lines.append("## State Construction")
        sr = sc.get("state_result", {})
        lines.append(f"- **States Discovered**: {sr.get('n_clusters', 0)}")
        lines.append(f"- **Compressed Dim**: {sr.get('compressed', np.zeros((1, 1))).shape[1]}")
        sid = sc.get("sid", {})
        lines.append("### State Information Density")
        for key, val in sid.items():
            lines.append(f"  - {key}: avg_sid={val.get('avg_sid', 0):.4f}, n_states={val.get('n_states', 0)}")
        sir = sc.get("sir", {})
        lines.append("### State Information Ratio")
        for key, val in sir.items():
            lines.append(f"  - {key}: SIR={val:.6f}")
        lines.append("")
        lines.append("## Sequence Analysis")
        best = seq.get("best_sequence", {})
        lines.append(f"- **Best Sequence Length**: {best.get('length', 0)}")
        lines.append(f"- **Best Sequence MI**: {best.get('mi', 0):.4f}")
        lines.append(f"- **Best Sequence SIR**: {best.get('sir', 0):.4f}")
        lines.append(f"- **Sequence SID (len=2)**: {seq.get('sequence_sid', 0):.4f}")
        lines.append("")
        lines.append("## Transition Intelligence")
        lines.append(f"- **Transition Information Gain**: {trans.get('transition_information_gain', 0):.4f}")
        lines.append(f"- **Transition Stability**: {trans.get('transition_stability', 0):.4f}")
        lines.append(f"- **Transition MI with Future**: {trans.get('transition_mi', 0):.4f}")
        lines.append(f"- **Transition Survival Rate**: {trans.get('transition_survival_rate', 0):.4f}")
        lines.append("")
        lines.append("## Compression")
        lines.append(f"- **Multi-Target Selected**: {len(comp.get('multi_target_selected', []))} features")
        lines.append(f"- **Information Bottleneck Selected**: {len(comp.get('information_bottleneck_selected', []))} features")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- **Features**: {results.get('n_features_total', 0)} → {results.get('n_features_surviving', 0)} survivors")
        lines.append(f"- **States**: {results.get('n_states', 0)}")
        return "\n".join(lines)
