import time
import json
import os
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RegimeSimulation:
    name: str = ""
    volatility: float = 0.0
    cycles: int = 0
    mof_scores: list = field(default_factory=list)
    mof_transitions: list = field(default_factory=list)
    segl_transitions: list = field(default_factory=list)
    rf_drift_values: list = field(default_factory=list)
    portfolio_conflict_values: list = field(default_factory=list)
    edge_confidence_values: list = field(default_factory=list)
    final_stable: bool = False
    decision_quality_degraded: bool = False
    collapse_detected: bool = False


class RegimeSurvivalSimulator:
    REGIMES = {
        "HIGH_VOLATILITY": {"volatility": 0.05, "shock_probability": 0.3, "spread_multiplier": 3.0},
        "LOW_VOLATILITY": {"volatility": 0.005, "shock_probability": 0.05, "spread_multiplier": 1.0},
        "MIXED_SHOCK": {"volatility": 0.02, "shock_probability": 0.15, "spread_multiplier": 2.0},
        "STRUCTURAL_STABILITY": {"volatility": 0.01, "shock_probability": 0.08, "spread_multiplier": 1.2},
    }
    DEFAULT_CYCLES = 20

    def __init__(self, state_dir: str = None):
        self._results: list[RegimeSimulation] = []
        self._state_dir = state_dir or os.path.join("state", "regime_simulation_logs")
        os.makedirs(self._state_dir, exist_ok=True)

    def simulate(self, regime: str = "HIGH_VOLATILITY", cycles: int = None) -> RegimeSimulation:
        cfg = self.REGIMES.get(regime)
        if not cfg:
            raise ValueError(f"Unknown regime: {regime}. Options: {list(self.REGIMES.keys())}")
        cycles = cycles or self.DEFAULT_CYCLES
        sim = RegimeSimulation(name=regime, volatility=cfg["volatility"], cycles=cycles)
        base_mof = 0.45 if regime == "HIGH_VOLATILITY" else 0.60 if regime == "LOW_VOLATILITY" else 0.50
        base_rf = 0.62
        base_edge = 0.65
        base_conflict = 0.10

        for i in range(cycles):
            shock = 1.0
            if random.random() < cfg["shock_probability"]:
                shock = 1.0 + (random.random() - 0.5) * cfg["spread_multiplier"] * 0.3

            mof_score = base_mof * shock
            mof_noise = (random.random() - 0.5) * cfg["volatility"]
            mof_score = max(0.0, min(1.0, mof_score + mof_noise))
            sim.mof_scores.append(round(mof_score, 4))

            if mof_score < 0.35:
                sim.mof_transitions.append("INFORMATION_DEGRADED")
            elif mof_score < 0.65:
                sim.mof_transitions.append("STRUCTURE_LIMITED")
            else:
                sim.mof_transitions.append("INFORMATION_RICH")

            rf_noise = (random.random() - 0.5) * cfg["volatility"] * 0.5
            rf_drift = max(0, abs(rf_noise))
            sim.rf_drift_values.append(round(rf_drift, 4))

            edge_noise = (random.random() - 0.5) * cfg["volatility"] * 0.3
            edge_conf = max(0.0, min(1.0, base_edge + edge_noise))
            sim.edge_confidence_values.append(round(edge_conf, 4))

            conflict_noise = (random.random() - 0.5) * cfg["volatility"] * 0.5
            conflict = max(0.0, min(1.0, base_conflict + conflict_noise))
            sim.portfolio_conflict_values.append(round(conflict, 4))

            mof_drop = min(sim.mof_scores[-5:]) if len(sim.mof_scores) >= 5 else min(sim.mof_scores)
            mof_degraded_count = sum(1 for s in sim.mof_transitions[-10:] if s == "INFORMATION_DEGRADED")
            if mof_degraded_count >= 6:
                sim.segl_transitions.append("LOCKED")
            elif mof_score < 0.35:
                sim.segl_transitions.append("COOLDOWN")
            elif mof_score >= 0.35 and mof_score < 0.65:
                sim.segl_transitions.append("ARMED")
            else:
                sim.segl_transitions.append("OBSERVE")

        final_5 = sim.mof_scores[-5:] if len(sim.mof_scores) >= 5 else sim.mof_scores
        sim.final_stable = len(set(sim.mof_transitions[-5:])) <= 2 if len(sim.mof_transitions) >= 5 else False

        mof_variance = sum((s - sum(final_5)/len(final_5))**2 for s in final_5) / len(final_5)
        sim.decision_quality_degraded = mof_variance > 0.02

        if sim.decision_quality_degraded:
            sim.collapse_detected = True

        self._results.append(sim)
        self._save_simulation(sim)
        return sim

    def _save_simulation(self, sim: RegimeSimulation):
        path = os.path.join(self._state_dir, f"regime_sim_{sim.name}_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump({
                "name": sim.name,
                "volatility": sim.volatility,
                "cycles": sim.cycles,
                "final_stable": sim.final_stable,
                "decision_quality_degraded": sim.decision_quality_degraded,
                "collapse_detected": sim.collapse_detected,
                "mof_scores": sim.mof_scores,
                "mof_transitions": sim.mof_transitions,
                "segl_transitions": sim.segl_transitions,
                "rf_drift_values": sim.rf_drift_values,
                "portfolio_conflict_values": sim.portfolio_conflict_values,
                "edge_confidence_values": sim.edge_confidence_values,
            }, f, indent=2, default=str)

    def summary(self) -> dict:
        if not self._results:
            return {"status": "NO_SIMULATIONS", "total": 0}
        return {
            "total_simulations": len(self._results),
            "regimes": [r.name for r in self._results],
            "results": [
                {
                    "regime": r.name,
                    "cycles": r.cycles,
                    "final_stable": r.final_stable,
                    "decision_quality_degraded": r.decision_quality_degraded,
                    "collapse_detected": r.collapse_detected,
                    "mof_final": r.mof_scores[-1] if r.mof_scores else 0,
                    "mof_variance": round(sum((s - sum(r.mof_scores[-5:])/len(r.mof_scores[-5:]))**2 / len(r.mof_scores[-5:]) for s in r.mof_scores[-5:]), 6) if len(r.mof_scores) >= 5 else 0,
                }
                for r in self._results
            ],
        }
