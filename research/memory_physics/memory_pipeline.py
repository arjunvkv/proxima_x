from __future__ import annotations

import time
from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult

from research.memory_physics.conflict_density import ConflictDensityAnalysis
from research.memory_physics.conflict_adaptive_time import ConflictAdaptiveTimeAnalysis
from research.memory_physics.conflict_mutation import ConflictMutationAnalysis
from research.memory_physics.mediator_resolution import MediatorResolution
from research.memory_physics.memory_birth_model import MemoryBirthModel
from research.memory_physics.memory_collapse import MemoryCollapse
from research.memory_physics.asset_invariance import AssetInvariance
from research.memory_physics.time_invariance import TimeInvariance
from research.memory_physics.generator_tournament import GeneratorTournament
from research.memory_physics.minimal_chain import MinimalChain


VERDICTS = [
    "NOISE",
    "REDUNDANT",
    "CORRELATED_PROXY",
    "PARALLEL_GENERATOR",
    "UPSTREAM_GENERATOR",
    "PRIMARY_GENERATOR",
    "FOUNDATIONAL_MARKET_DRIVER",
]

VERDICT_THRESHOLDS = {
    "NOISE": {"min_confidence": 0.0},
    "REDUNDANT": {"min_confidence": 0.1},
    "CORRELATED_PROXY": {"min_confidence": 0.2},
    "PARALLEL_GENERATOR": {"min_confidence": 0.4},
    "UPSTREAM_GENERATOR": {"min_confidence": 0.55},
    "PRIMARY_GENERATOR": {"min_confidence": 0.7},
    "FOUNDATIONAL_MARKET_DRIVER": {"min_confidence": 0.85},
}


class MemoryPipeline:
    def __init__(self, data_dir: str = "data/market", asset: str = "EURJPY"):
        self.validator = MemoryValidator(data_dir)
        self.asset = asset
        self.results: dict[str, MPRResult] = {}

    def run_all(self) -> dict[str, Any]:
        wall = time.time()
        timing: dict[str, float] = {}
        output: dict[str, Any] = {"asset": self.asset}

        print(f"\n{'='*60}")
        print(f"MEMORY PHYSICS RESOLUTION: {self.asset}")
        print(f"{'='*60}\n")

        t0 = time.time()
        print("--- RQ1: Conflict -> Density ---")
        r1 = ConflictDensityAnalysis(self.validator, self.asset).run()
        self.results["conflict_density"] = r1
        timing["conflict_density"] = time.time() - t0
        print(f"  Status: {r1.status}\n")

        t0 = time.time()
        print("--- RQ2: Conflict -> Adaptive Time (vs Density) ---")
        r2 = ConflictAdaptiveTimeAnalysis(self.validator, self.asset).run()
        self.results["conflict_adaptive_time"] = r2
        timing["conflict_adaptive_time"] = time.time() - t0
        print(f"  Status: {r2.status}\n")

        t0 = time.time()
        print("--- RQ3: Conflict -> Mutation (vs Density) ---")
        r3 = ConflictMutationAnalysis(self.validator, self.asset).run()
        self.results["conflict_mutation"] = r3
        timing["conflict_mutation"] = time.time() - t0
        print(f"  Status: {r3.status}\n")

        t0 = time.time()
        print("--- RQ4: Mediator Resolution ---")
        r4 = MediatorResolution(self.validator, self.asset).run()
        self.results["mediator_resolution"] = r4
        timing["mediator_resolution"] = time.time() - t0
        print(f"  Status: {r4.status}\n")

        t0 = time.time()
        print("--- RQ5: Memory Birth Model ---")
        r5 = MemoryBirthModel(self.validator, self.asset).run()
        self.results["memory_birth_model"] = r5
        timing["memory_birth_model"] = time.time() - t0
        print(f"  Status: {r5.status}\n")

        t0 = time.time()
        print("--- RQ6: Memory Collapse ---")
        r6 = MemoryCollapse(self.validator, self.asset).run()
        self.results["memory_collapse"] = r6
        timing["memory_collapse"] = time.time() - t0
        print(f"  Status: {r6.status}\n")

        t0 = time.time()
        print("--- RQ7: Asset Invariance ---")
        r7 = AssetInvariance(self.validator).run()
        self.results["asset_invariance"] = r7
        timing["asset_invariance"] = time.time() - t0
        print(f"  Status: {r7.status}\n")

        t0 = time.time()
        print("--- RQ8: Time Invariance ---")
        r8 = TimeInvariance(self.validator, self.asset).run()
        self.results["time_invariance"] = r8
        timing["time_invariance"] = time.time() - t0
        print(f"  Status: {r8.status}\n")

        t0 = time.time()
        print("--- RQ9: Generator Tournament ---")
        r9 = GeneratorTournament(self.validator, self.asset).run()
        self.results["generator_tournament"] = r9
        timing["generator_tournament"] = time.time() - t0
        print(f"  Status: {r9.status}\n")

        t0 = time.time()
        print("--- RQ10: Minimal Chain ---")
        r10 = MinimalChain(self.validator, self.asset).run()
        self.results["minimal_chain"] = r10
        timing["minimal_chain"] = time.time() - t0
        print(f"  Status: {r10.status}\n")

        output["rq_results"] = {k: v.to_dict() for k, v in self.results.items()}
        output["timing"] = timing
        output["timing"]["total"] = time.time() - wall

        verdict, confidence = self._adjudicate()
        output["final_verdict"] = verdict
        output["confidence"] = confidence
        output["verdict_reasoning"] = self._verdict_reasoning(verdict, confidence)

        print(f"\n{'='*60}")
        print(f"FINAL VERDICT: {verdict} (confidence={confidence:.2f})")
        print(f"{'='*60}")
        print(f"Total time: {output['timing']['total']:.2f}s")

        return output

    def _adjudicate(self) -> tuple[str, float]:
        if not self.results:
            return "NOISE", 0.0

        confidence = 0.0
        max_confidence = 0.0

        weights = {
            "conflict_density": 2,
            "conflict_adaptive_time": 2,
            "conflict_mutation": 1,
            "mediator_resolution": 2,
            "memory_birth_model": 1,
            "memory_collapse": 2,
            "asset_invariance": 3,
            "time_invariance": 3,
            "generator_tournament": 2,
            "minimal_chain": 2,
        }

        total_weight = sum(weights.values())

        passed_checks = []
        for name, r in self.results.items():
            w = weights.get(name, 1)
            if r.status == "PASSED":
                confidence += w
                passed_checks.append(name)
            elif r.status == "INCONCLUSIVE":
                confidence += w * 0.5

        confidence = confidence / max(total_weight, 1)

        r1 = self.results.get("conflict_density", MPRResult("", "FAILED"))
        r7 = self.results.get("asset_invariance", MPRResult("", "FAILED"))
        r8 = self.results.get("time_invariance", MPRResult("", "FAILED"))

        conflict_leads_density = r1.status == "PASSED"
        asset_stable = r7.status == "PASSED"
        time_stable = r8.status == "PASSED"

        verdict = "NOISE"
        for v in reversed(VERDICTS):
            thresh = VERDICT_THRESHOLDS[v]["min_confidence"]
            if confidence >= thresh:
                verdict = v
                break

        if verdict == "CORRELATED_PROXY" and conflict_leads_density and asset_stable:
            verdict = "PARALLEL_GENERATOR"
        if verdict == "PARALLEL_GENERATOR" and conflict_leads_density:
            verdict = "UPSTREAM_GENERATOR"
        if verdict in ("UPSTREAM_GENERATOR",) and confidence > 0.7:
            r4 = self.results.get("mediator_resolution", MPRResult("", "FAILED"))
            r9 = self.results.get("generator_tournament", MPRResult("", "FAILED"))
            gen_ranking = r9.metrics.get("rank_order", [])
            if r4.status == "PASSED" and gen_ranking and gen_ranking[0] == "memory_conflict":
                verdict = "PRIMARY_GENERATOR"
            if confidence > 0.85 and r4.status == "PASSED":
                verdict = "FOUNDATIONAL_MARKET_DRIVER"

        return verdict, confidence

    def _verdict_reasoning(self, verdict: str, confidence: float) -> list[str]:
        lines = [f"Confidence: {confidence:.2f}"]
        for name, r in self.results.items():
            lines.append(f"  {name}: {r.status}")
        lines.append(f"Interpretation: {VERDICTS[VERDICTS.index(verdict)] if verdict in VERDICTS else 'Unknown'}")
        return lines
