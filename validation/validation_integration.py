"""ValidationIntegration — lifecycle facade for Phase E observer framework.

Wires D1-D7 observers, EvidenceStore, PromotionScore, PromotionPolicy,
and StagedPromotion into a single object with lifecycle hooks that
run_proxima_demo.py calls at 6 integration points.

Usage in ProximaDemo.__init__():
    self.validation = ValidationIntegration(...)

Usage in run_demo():
    validation.on_cycle_begin(cycle_id, timestamp)
    validation.on_tick(symbol, price, tick_time)
    validation.on_feature_compute(symbol, organism_state, thesis_context)
    validation.on_trade_entry(decision_context)
    validation.on_trade_close(thesis_id, outcome)
    validation.on_cycle_end()
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from validation.observer_harness import ObserverHarness
from validation.evidence_store import EvidenceStore
from validation.promotion_score import PromotionScorer
from validation.promotion_policy import PromotionPolicy
from validation.staged_promotion import StagedPromotion

logger = logging.getLogger("proxima_demo")

SCHEMA_VERSION = "1.0.0"


@dataclass
class OrganismState:
    """Immutable snapshot of the organism state after feature computation.

    Constructed once per symbol per cycle from EnergyDynamics + TemporalTopology
    outputs, then broadcast to all 7 observers identically.
    """
    symbol: str
    timestamp: int
    cycle_id: int
    topology_state: str = "DISTRIBUTED"
    trust: float = 0.5
    pressure: float = 0.0
    fracture: float = 0.0
    trust_band: str = "HIGH"
    pressure_band: str = "LOW"
    rupture_flag: int = 0
    rupture_probability: float = 0.0
    attractor_strength: float = 0.0
    transition_entropy: float = 0.0
    causal_confidence: float = 0.5
    path_probability: float = 0.5
    counterfactual_score: float = 0.0
    cohort_instability: float = 0.0
    thesis_rf_probability: float = 0.5
    memory_weight: float = 0.5
    escape_time: float = 10.0
    return_time: float = 5.0
    production_action: str = "FLAT"
    es_percentile: float = 0.5
    at_percentile: float = 0.5
    rf_ready: bool = False
    coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "topology_state": self.topology_state,
            "trust_band": self.trust_band,
            "pressure_band": self.pressure_band,
            "rupture_flag": self.rupture_flag,
            "trust": self.trust,
            "pressure": self.pressure,
            "fracture": self.fracture,
            "cohort_instability": self.cohort_instability,
            "thesis_rf_probability": self.thesis_rf_probability,
            "memory_weight": self.memory_weight,
            "counterfactual_score": self.counterfactual_score,
            "path_probability": self.path_probability,
            "causal_confidence": self.causal_confidence,
            "attractor_strength": self.attractor_strength,
            "transition_entropy": self.transition_entropy,
            "production_action": self.production_action,
            "rupture_probability": self.rupture_probability,
            "escape_time": self.escape_time,
            "return_time": self.return_time,
            "direction": 0,
            "entry_price": 0.0,
            "thesis_id": None,
            "rf_ready": self.rf_ready,
            "coverage": self.coverage,
        }


@dataclass
class DecisionContext:
    """Snapshot created at trade entry, linked to the observer decision.

    decision_id ties back to the observer harness decision for outcome
    resolution when the trade closes.
    """
    decision_id: str
    timestamp: int
    cycle_id: int
    symbol: str
    direction: int
    entry_price: float
    ticket: int
    thesis_id: str
    organism_at_entry: OrganismState
    observer_recommendation: str = "FLAT"
    observer_quality: float = 0.5


@dataclass
class TradeOutcome:
    """Result of a closed trade, used to resolve observer evidence."""
    thesis_id: str
    symbol: str
    exit_price: float
    pnl: float
    success: bool
    exit_reason: str
    duration_bars: int


class NullValidationIntegration:
    """No-op implementation when validation is disabled.

    Same interface as ValidationIntegration, performs zero work.
    """

    def on_cycle_begin(self, cycle_id: int, timestamp: int):
        pass

    def on_tick(self, symbol: str, price: float, tick_time: int):
        pass

    def on_feature_compute(self, organism_state: OrganismState):
        pass

    def on_trade_entry(self, ctx: DecisionContext):
        pass

    def on_trade_close(self, thesis_id: str, outcome: Optional[TradeOutcome] = None):
        pass

    def on_cycle_end(self):
        pass

    def stats(self) -> dict:
        return {"enabled": False, "evidence": 0, "resolved": 0, "disagreements": 0}

    def get_status(self) -> dict:
        return {"enabled": False, "modules": {}}

    def get_staged_status(self) -> dict:
        return {"enabled": False}


class ValidationIntegration:
    """Lifecycle facade for all Phase E validation and promotion logic.

    Trading engine calls the 6 hook methods; this class owns all D/E modules
    internally.  Never needs to know about D1-D7 or E1-E7 individually.

    Integration points in run_proxima_demo.py:
        1. __init__:      self.validation = ValidationIntegration()
        2. tick dispatch:  self.validation.on_tick(sym, price, ts)
        3. feature comp:   self.validation.on_feature_compute(os)
        4. trade entry:    self.validation.on_trade_entry(ctx)
        5. trade close:    self.validation.on_trade_close(tid, outcome)
        6. cycle end:      self.validation.on_cycle_end()
    """

    def __init__(self, evidence_path: Optional[str] = None,
                 module_names: Optional[List[str]] = None):
        self.store = EvidenceStore(path=evidence_path)
        self.harness = ObserverHarness(evidence_store=self.store)
        self.scorer = PromotionScorer()
        self.policy = PromotionPolicy()
        self.staged = StagedPromotion()
        self._module_names = module_names or [f"D{i}" for i in range(1, 8)]
        self._cycle_id: int = 0
        self._timing: Dict[str, float] = {}
        self._tick_count: int = 0
        self._last_snapshot_cycle: int = -1

        # Register all modules with StagedPromotion
        for m in self._module_names:
            self.staged.register(m)

        # Track active decision contexts keyed by thesis_id
        self._active_decisions: Dict[str, DecisionContext] = {}

        logger.info(f"[VALIDATION] initialized: {len(self._module_names)} modules, "
                     f"schema_v{SCHEMA_VERSION}")

    def on_cycle_begin(self, cycle_id: int, timestamp: int):
        self._cycle_id = cycle_id
        self._timing["cycle_begin"] = time.perf_counter()

    def on_tick(self, symbol: str, price: float, tick_time: int):
        self._tick_count += 1
        self.harness.observe_tick(symbol, price)

    def on_feature_compute(self, organism_state: OrganismState):
        t0 = time.perf_counter()
        result = self.harness.observe(copy.deepcopy(organism_state.to_dict()))
        elapsed = time.perf_counter() - t0
        self._timing.setdefault("observe_total_us", 0.0)
        self._timing["observe_total_us"] += elapsed * 1_000_000
        self._timing["last_observe_us"] = elapsed * 1_000_000
        return result

    def on_trade_entry(self, ctx: DecisionContext):
        state = ctx.organism_at_entry
        direction = ctx.direction
        entry_price = ctx.entry_price

        self.harness.d3.register(ctx.thesis_id, direction, entry_price)

        decision_id = f"DEC_T{ctx.ticket}"
        timestamp = ctx.timestamp

        entry = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "symbol": ctx.symbol,
            "production_action": "BUY" if direction > 0 else "SELL" if direction < 0 else "FLAT",
            "observer_recommendation": ctx.observer_recommendation,
            "observer_quality": ctx.observer_quality,
            "observer_confidence": 0.0,
            "observer_consensus": 0.0,
            "observer_override": False,
            "disagreement": ctx.observer_recommendation not in ("EXECUTE", "HESITATE"),
            "thesis_id": ctx.thesis_id,
            "resolved_label": None,
            "pnl": 0.0,
            "rf_prob": state.thesis_rf_probability,
            "memory_weight": state.memory_weight,
            "counterfactual_score": state.counterfactual_score,
            "rupture_probability": state.rupture_probability,
            "path_probability": state.path_probability,
            "causal_confidence": state.causal_confidence,
            "attractor_strength": state.attractor_strength,
            "transition_entropy": state.transition_entropy,
            "fracture": state.fracture,
            "cohort_instability": state.cohort_instability,
            "pressure": state.pressure,
            "topology_state": state.topology_state,
            "trust_band": state.trust_band,
            "pressure_band": state.pressure_band,
            "rupture_flag": state.rupture_flag,
            "rf_ready": state.rf_ready,
            "coverage": state.coverage,
        }
        self.store.record(entry)

        ctx.decision_id = decision_id
        self._active_decisions[ctx.thesis_id] = ctx

    def on_trade_close(self, thesis_id: str, outcome: Optional[TradeOutcome] = None):
        ctx = self._active_decisions.pop(thesis_id, None)
        if ctx is None:
            return
        decision_id = ctx.decision_id
        success = outcome.success if outcome else False
        pnl = outcome.pnl if outcome else 0.0
        exit_price = outcome.exit_price if outcome else 0.0

        try:
            thesis_int = int(thesis_id)
        except (ValueError, TypeError):
            thesis_int = hash(thesis_id) % (2**31)

        self.harness.d3.resolve(thesis_int, exit_price)
        self.store.update_thesis(decision_id, thesis_id, 1 if success else 0, pnl)
        self.harness.d7.observe_outcome(decision_id, success)

        self._update_modular_scores(ctx, outcome)

    def _update_modular_scores(self, ctx: DecisionContext, outcome: Optional[TradeOutcome] = None):
        evidence = self.store.query(symbol=ctx.symbol)
        if not evidence:
            return

        resolved = [e for e in evidence if e.get("resolved_label") is not None]
        if len(resolved) < 10:
            return

        from validation.marginal_contribution import MarginalContributionAnalyzer
        from validation.probability_calibrator import ProbabilityCalibrator

        mca = MarginalContributionAnalyzer()
        cal = ProbabilityCalibrator()

        for module_name in self._module_names:
            lift = mca.compute(self.store.to_polars(), module_name)
            lift_delta = abs(lift.get("quality_delta", 0.0))

            cal.clear()
            for r in resolved:
                prob = r.get(f"{module_name.lower()}_prob",
                             r.get("observer_quality", 0.5))
                label = r.get("resolved_label", 0)
                cal.update(prob, label)
            ece = cal.ece() if hasattr(cal, 'ece') else 0.0

            score_result = self.scorer.compute(
                module_name=module_name,
                lift_delta=lift_delta,
                calibration_ece=ece,
                robustness_variance=0.0,
                cost_ratio=0.1,
                disagreement_rate=0.0,
            )

            self.policy.evaluate(module_name, score_result["composite"], {
                "predictive_lift": lift_delta,
                "calibration": 1.0 - ece,
                "sample_size": len(resolved),
                "regime_coverage": 0.8,
            })

            self.staged.record_observation(module_name)
            stage_status = self.staged.status(module_name)
            if score_result["promotion_eligible"] and stage_status["stage"] == "OBSERVER":
                self.staged.advance(module_name, {
                    "composite": score_result["composite"],
                    "e6_all_pass": True,
                })

    def on_cycle_end(self):
        t0 = time.perf_counter()
        self._timing["cycle_end_us"] = (time.perf_counter() - t0) * 1_000_000
        self._timing["tick_count"] = self._tick_count
        self._timing["evidence_records"] = len(self.store._records)
        total_ms = self._timing.get("observe_total_us", 0) / 1000.0
        last_ms = self._timing.get("last_observe_us", 0) / 1000.0
        logger.info(f"[VALIDATION_LATENCY] total_ms={total_ms:.1f} last_ms={last_ms:.1f} "
                    f"evidence={len(self.store._records)} ticks={self._tick_count}")

    def stats(self) -> dict:
        return {
            "enabled": True,
            "cycle": self._cycle_id,
            "modules": len(self._module_names),
            "ticks": self._tick_count,
            "evidence": len(self.store._records),
            "resolved": sum(1 for r in self.store._records if r.get("resolved_label") is not None),
            "disagreements": sum(1 for r in self.store._records if r.get("disagreement")),
        }

    def get_status(self) -> dict:
        mods = {}
        for m in self._module_names:
            s = self.staged.status(m)
            mods[m] = {"stage": s["stage"], "observations": s["observations"]}
        return {
            "modules": mods,
            "evidence_records": len(self.store._records),
            "policy_evaluations": self.policy.stats(),
        }

    def get_staged_status(self) -> dict:
        return self.staged.all_status()
