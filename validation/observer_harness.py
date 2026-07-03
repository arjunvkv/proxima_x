import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from signals.state_transition_graph import StateTransitionGraph
from signals.causal_fingerprint import CausalFingerprintEngine
from signals.counterfactual_branch import CounterfactualBranchEngine
from signals.path_memory import PathMemoryEngine
from signals.regime_attractor import RegimeAttractorEngine
from signals.pre_rupture_forecast import PreRuptureForecastEngine
from signals.reflexive_arbitration import ReflexiveArbitrationEngine
from validation.evidence_store import EvidenceStore

logger = logging.getLogger("proxima_demo")


class ObserverHarness:
    def __init__(self, evidence_store: Optional[EvidenceStore] = None):
        self.d1 = StateTransitionGraph()
        self.d2 = CausalFingerprintEngine()
        self.d3 = CounterfactualBranchEngine()
        self.d4 = PathMemoryEngine()
        self.d5 = RegimeAttractorEngine()
        self.d6 = PreRuptureForecastEngine()
        self.d7 = ReflexiveArbitrationEngine()
        self.store = evidence_store or EvidenceStore()
        self._decision_count = 0

    def observe(self, state: dict) -> dict:
        symbol = state.get("symbol", "UNKNOWN")
        topology_state = state.get("topology_state", "DISTRIBUTED")
        trust_band = state.get("trust_band", "HIGH")
        pressure_band = state.get("pressure_band", "LOW")
        rupture_flag = state.get("rupture_flag", 0)
        trust = state.get("trust", 0.5)
        pressure = state.get("pressure", 0.0)
        fracture = state.get("fracture", 0.0)
        cohort_instability = state.get("cohort_instability", 0.0)
        rf_prob = state.get("thesis_rf_probability", 0.5)
        memory_weight = state.get("memory_weight", 0.5)
        counterfactual_score = state.get("counterfactual_score", 0.0)
        path_prob = state.get("path_probability", 0.5)
        causal_conf = state.get("causal_confidence", 0.5)
        attractor_strength = state.get("attractor_strength", 0.0)
        transition_entropy = state.get("transition_entropy", 0.0)
        production_action = state.get("production_action", "FLAT")

        compact_state = (topology_state, trust_band, pressure_band, rupture_flag)

        # Feed observer engines
        self.d1.update(symbol, compact_state)
        fingerprint = (topology_state, trust_band, pressure_band,
                       "HIGH" if fracture > 0.5 else "LOW",
                       "HIGH" if rf_prob > 0.5 else "LOW")
        self.d2.record(symbol, fingerprint)
        self.d4.update(symbol, compact_state)
        self.d5.update(symbol, compact_state)

        # D3: register thesis if direction is provided
        direction = state.get("direction", 0)
        entry_price = state.get("entry_price", 0.0)
        thesis_id = state.get("thesis_id", self._decision_count)
        if direction != 0 and entry_price > 0:
            self.d3.register(thesis_id, direction, entry_price)

        # Build organism state for D6
        organism = {
            "pressure": pressure,
            "fracture": fracture,
            "cohort_instability": cohort_instability,
            "trust": trust,
            "attractor_strength": attractor_strength,
            "causal_confidence": causal_conf,
            "escape_time": state.get("escape_time", 10.0),
            "return_time": state.get("return_time", 5.0),
            "path_probability": path_prob,
        }
        self.d6.update(symbol, organism)

        # Build decision state for D7
        decision_state = {
            "thesis_rf_probability": rf_prob,
            "memory_weight": memory_weight,
            "counterfactual_score": counterfactual_score,
            "rupture_probability": state.get("rupture_probability", 0.0),
            "path_probability": path_prob,
            "causal_confidence": causal_conf,
        }
        d7_result = self.d7.evaluate(decision_state)

        # Check for disagreement
        disc = self.d7.disagreement(production_action)

        # Record in evidence store
        decision_id = f"DEC_{self._decision_count}"
        self._decision_count += 1

        entry = {
            "decision_id": decision_id,
            "timestamp": state.get("timestamp", 0),
            "symbol": symbol,
            "production_action": production_action,
            "observer_recommendation": d7_result["recommendation"],
            "observer_quality": d7_result["decision_quality"],
            "observer_confidence": d7_result["confidence"],
            "observer_consensus": d7_result["consensus"],
            "observer_override": d7_result["override"],
            "disagreement": disc["disagreement"],
            "rf_prob": rf_prob,
            "memory_weight": memory_weight,
            "counterfactual_score": counterfactual_score,
            "rupture_probability": state.get("rupture_probability", 0.0),
            "path_probability": path_prob,
            "causal_confidence": causal_conf,
            "attractor_strength": attractor_strength,
            "transition_entropy": transition_entropy,
            "fracture": fracture,
            "cohort_instability": cohort_instability,
            "pressure": pressure,
            "topology_state": topology_state,
            "trust_band": trust_band,
            "pressure_band": pressure_band,
            "rupture_flag": rupture_flag,
        }
        self.store.record(entry)

        logger.info(f"[OBSERVER_HARNESS] {decision_id} {symbol} "
                    f"production={production_action} "
                    f"observer={d7_result['recommendation']} "
                    f"disagree={disc['disagreement']}")

        return {
            "decision_id": decision_id,
            "d7_result": d7_result,
            "disagreement": disc,
        }

    def observe_tick(self, symbol: str, price: float):
        self.d3.tick(symbol, price)

    def resolve_thesis(self, decision_id: str, thesis_id: int,
                       exit_price: float, pnl: float, success: bool):
        self.d3.resolve(thesis_id, exit_price)
        self.store.update_thesis(decision_id, str(thesis_id), 1 if success else 0, pnl)
        self.d7.observe_outcome(decision_id, success)
        logger.info(f"[OBSERVER_HARNESS] resolve {decision_id} "
                    f"thesis={thesis_id} pnl={pnl:.2f} success={success}")

    def stats(self) -> dict:
        return self.store.stats()
