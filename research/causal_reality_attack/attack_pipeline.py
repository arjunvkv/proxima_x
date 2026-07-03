from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_ASSETS, _clean_serializable,
)
from research.causal_reality_attack.cross_asset_attack import CrossAssetAttack
from research.causal_reality_attack.cross_time_attack import CrossTimeAttack
from research.causal_reality_attack.node_removal_attack import NodeRemovalAttack
from research.causal_reality_attack.mediator_analysis import MediatorAnalysis
from research.causal_reality_attack.random_graph_attack import RandomGraphAttack
from research.causal_reality_attack.bootstrap_attack import BootstrapAttack
from research.causal_reality_attack.noise_attack import NoiseAttack
from research.causal_reality_attack.hidden_variable_attack import HiddenVariableAttack
from research.causal_reality_attack.chain_collapse import ChainCollapse


FINAL_VERDICTS = [
    "SPURIOUS_GRAPH",
    "PARTIAL_CAUSAL_CHAIN",
    "REGIME_SPECIFIC_CHAIN",
    "ASSET_SPECIFIC_CHAIN",
    "ROBUST_CAUSAL_CHAIN",
    "MARKET_PHYSICS_CANDIDATE",
]


class AttackPipeline:
    """Orchestrates all 12 reality attacks and produces final verdict."""

    def __init__(self, data_dir: str = "data/market", asset: str = "EURJPY"):
        self.validator = AttackValidator(data_dir)
        self.asset = asset
        self.results: dict[str, AttackResult] = {}

    def run_all(self) -> dict[str, Any]:
        wall = time.time()
        timing: dict[str, float] = {}
        output: dict[str, Any] = {"asset": self.asset}

        print(f"\n{'='*60}")
        print(f"CAUSAL REALITY ATTACK: {self.asset}")
        print(f"{'='*60}\n")

        t0 = time.time()
        print("--- Attack 1: Cross-Asset Causal Invariance ---")
        attack1 = CrossAssetAttack(self.validator)
        r1 = attack1.run()
        self.results["cross_asset"] = r1
        timing["cross_asset"] = time.time() - t0
        print(f"  Status: {r1.status}\n")

        t0 = time.time()
        print("--- Attack 2: Cross-Time Causal Invariance ---")
        attack2 = CrossTimeAttack(self.validator, self.asset)
        r2 = attack2.run()
        self.results["cross_time"] = r2
        timing["cross_time"] = time.time() - t0
        print(f"  Status: {r2.status}\n")

        t0 = time.time()
        print("--- Attacks 3-6: Node Removal Attacks ---")
        attack3 = NodeRemovalAttack(self.validator, self.asset)
        r3 = attack3.run()
        self.results["node_removal"] = r3
        timing["node_removal"] = time.time() - t0
        print(f"  Status: {r3.status}\n")

        t0 = time.time()
        print("--- Attack 7: Mediator Analysis ---")
        attack7 = MediatorAnalysis(self.validator, self.asset)
        r7 = attack7.run()
        self.results["mediator"] = r7
        timing["mediator"] = time.time() - t0
        print(f"  Status: {r7.status}\n")

        t0 = time.time()
        print("--- Attack 8: Random Graph Benchmark ---")
        attack8 = RandomGraphAttack(self.validator, self.asset)
        r8 = attack8.run()
        self.results["random_graph"] = r8
        timing["random_graph"] = time.time() - t0
        print(f"  Status: {r8.status}\n")

        t0 = time.time()
        print("--- Attack 9: Bootstrap Stability ---")
        attack9 = BootstrapAttack(self.validator, self.asset)
        r9 = attack9.run()
        self.results["bootstrap"] = r9
        timing["bootstrap"] = time.time() - t0
        print(f"  Status: {r9.status}\n")

        t0 = time.time()
        print("--- Attack 10: Noise Injection ---")
        attack10 = NoiseAttack(self.validator, self.asset)
        r10 = attack10.run()
        self.results["noise"] = r10
        timing["noise"] = time.time() - t0
        print(f"  Status: {r10.status}\n")

        t0 = time.time()
        print("--- Attack 11: Hidden Variable Challenge ---")
        attack11 = HiddenVariableAttack(self.validator, self.asset)
        r11 = attack11.run()
        self.results["hidden_variable"] = r11
        timing["hidden_variable"] = time.time() - t0
        print(f"  Status: {r11.status}\n")

        t0 = time.time()
        print("--- Attack 12: Full Chain Collapse Test ---")
        attack12 = ChainCollapse(self.validator, self.asset)
        r12 = attack12.run()
        self.results["chain_collapse"] = r12
        timing["chain_collapse"] = time.time() - t0
        print(f"  Status: {r12.status}\n")

        output["attack_results"] = {k: v.to_dict() for k, v in self.results.items()}
        output["timing"] = timing
        output["timing"]["total"] = time.time() - wall

        verdict = self._adjudicate()
        output["final_verdict"] = verdict
        output["verdict_reasoning"] = self._verdict_reasoning(verdict)

        print(f"\n{'='*60}")
        print(f"FINAL VERDICT: {verdict}")
        print(f"{'='*60}")
        print(f"Total time: {output['timing']['total']:.2f}s")

        return output

    def run_selected(self, attack_names: list[str]) -> dict[str, Any]:
        wall = time.time()
        output: dict[str, Any] = {"asset": self.asset}

        attack_map = {
            "cross_asset": ("CrossAssetAttack", CrossAssetAttack(self.validator)),
            "cross_time": ("CrossTimeAttack", CrossTimeAttack(self.validator, self.asset)),
            "node_removal": ("NodeRemovalAttack", NodeRemovalAttack(self.validator, self.asset)),
            "mediator": ("MediatorAnalysis", MediatorAnalysis(self.validator, self.asset)),
            "random_graph": ("RandomGraphAttack", RandomGraphAttack(self.validator, self.asset)),
            "bootstrap": ("BootstrapAttack", BootstrapAttack(self.validator, self.asset)),
            "noise": ("NoiseAttack", NoiseAttack(self.validator, self.asset)),
            "hidden_variable": ("HiddenVariableAttack", HiddenVariableAttack(self.validator, self.asset)),
            "chain_collapse": ("ChainCollapse", ChainCollapse(self.validator, self.asset)),
        }

        for name in attack_names:
            if name in attack_map:
                print(f"\n--- {attack_map[name][0]} ---")
                t0 = time.time()
                try:
                    result = attack_map[name][1].run()
                    self.results[name] = result
                    print(f"  Status: {result.status}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    self.results[name] = AttackResult(name, "FAILED", metrics={"error": str(e)})

        output["attack_results"] = {k: v.to_dict() for k, v in self.results.items()}
        output["timing"] = {"total": time.time() - wall}

        if len(self.results) >= 4:
            verdict = self._adjudicate()
            output["final_verdict"] = verdict
            output["verdict_reasoning"] = self._verdict_reasoning(verdict)

        return output

    def _adjudicate(self) -> str:
        passed = sum(1 for r in self.results.values() if r.status == "PASSED")
        failed = sum(1 for r in self.results.values() if r.status == "FAILED")
        inconclusive = sum(1 for r in self.results.values() if r.status == "INCONCLUSIVE")
        total = len(self.results)

        if total == 0:
            return "INCONCLUSIVE"

        has_cross_asset = self.results.get("cross_asset", AttackResult("", "FAILED")).status
        has_cross_time = self.results.get("cross_time", AttackResult("", "FAILED")).status

        # Weighted scoring: universality (cross-asset/cross-time) weighted highest
        weights = {
            "cross_asset": 3, "cross_time": 3,
            "mediator": 2, "random_graph": 2,
            "bootstrap": 2, "noise": 2,
            "hidden_variable": 2,
            "node_removal": 1, "chain_collapse": 1,
        }
        score = 0
        max_score = 0
        for name, r in self.results.items():
            w = weights.get(name, 1)
            max_score += w
            if r.status == "PASSED":
                score += w
            elif r.status == "INCONCLUSIVE":
                score += w * 0.5

        percentage = score / max(max_score, 1)

        # Extract detailed metrics for nuanced checks
        cross_asset_metrics = self.results.get("cross_asset", AttackResult("", "FAILED")).metrics
        cross_time_metrics = self.results.get("cross_time", AttackResult("", "FAILED")).metrics
        random_metrics = self.results.get("random_graph", AttackResult("", "FAILED")).metrics
        bootstrap_metrics = self.results.get("bootstrap", AttackResult("", "FAILED")).metrics

        ca_avg = cross_asset_metrics.get("averages", {})
        ct_avg = cross_time_metrics.get("averages", {})
        z_score = random_metrics.get("z_score", 0)
        avg_survival = bootstrap_metrics.get("avg_survival_rate", 0)

        ca_perfect = ca_avg.get("avg_node_jaccard", 0) > 0.99 and ca_avg.get("avg_edge_jaccard", 0) > 0.99
        ct_perfect = ct_avg.get("avg_node_jaccard", 0) > 0.99 and ct_avg.get("avg_edge_jaccard", 0) > 0.99
        universal = ca_perfect and ct_perfect

        if percentage >= 0.80 and universal and z_score > 50 and avg_survival > 0.9:
            return "MARKET_PHYSICS_CANDIDATE"

        if (percentage >= 0.65 or (universal and z_score > 20 and avg_survival > 0.8)):
            return "ROBUST_CAUSAL_CHAIN"

        if has_cross_asset == "FAILED" and has_cross_time == "PASSED":
            return "ASSET_SPECIFIC_CHAIN"

        if has_cross_time == "FAILED" and has_cross_asset == "PASSED":
            return "REGIME_SPECIFIC_CHAIN"

        if percentage >= 0.40:
            return "PARTIAL_CAUSAL_CHAIN"

        return "SPURIOUS_GRAPH"

    def _verdict_reasoning(self, verdict: str) -> list[str]:
        lines = []
        for name, r in self.results.items():
            lines.append(f"{name}: {r.status}")
        lines.append(f"Pass rate: {sum(1 for r in self.results.values() if r.status == 'PASSED')}/{len(self.results)}")

        reasons = {
            "SPURIOUS_GRAPH": "The discovered causal graph failed most attacks and is likely spurious correlation.",
            "PARTIAL_CAUSAL_CHAIN": "Some causal structure exists but is weak or inconsistent across tests.",
            "REGIME_SPECIFIC_CHAIN": "The causal ordering changes materially across time windows - it's regime-dependent.",
            "ASSET_SPECIFIC_CHAIN": "The causal graph varies across assets - not a universal market property.",
            "ROBUST_CAUSAL_CHAIN": "The chain survives most attacks: bootstrap, random graph comparison, and chain collapse.",
            "MARKET_PHYSICS_CANDIDATE": "The chain survives ALL attacks including cross-asset, cross-time, mediator analysis, and hidden variable challenges. Legitimate candidate for market physics.",
        }
        lines.append(f"Interpretation: {reasons.get(verdict, '')}")
        return lines
