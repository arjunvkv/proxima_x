from .input import NarrativeInput
from .detector import NarrativeDetector
from .tracker import NarrativeTracker
from .maturity import MaturityCalculator
from .serializer import serialize_narrative
from .state import NarrativeState, NarrativeMetrics
from typing import Optional


class NarrativeEngine:
    def __init__(self, strength_threshold: float = 0.00005, persistence_required: int = 5):
        self.detector = NarrativeDetector(
            strength_threshold=strength_threshold,
            persistence_required=persistence_required,
        )
        self.tracker = NarrativeTracker()
        self.maturity = MaturityCalculator()
        self._previous_strengths = {}
        self._pre_birth_persistence: dict[str, int] = {}

    def update(self, market_state: NarrativeInput) -> Optional[NarrativeState]:
        candidate = self.detector.detect_candidate(market_state.currency_strengths)
        if candidate is None:
            self.tracker.update(None, market_state.cycle, self.detector, market_state.graph_quality)
            self._pre_birth_persistence.clear()
            return None

        persistence = self._compute_persistence(candidate)
        participation = market_state.currency_bursts.get(candidate["leader"], 0)

        if not self.tracker.active and not self.detector.should_birth(
            candidate, persistence, market_state.graph_quality, participation
        ):
            self._increment_pre_birth(candidate)
            import sys as _sys
            print(f"[NME TRACE] pre-birth: leader={candidate['leader']} dir={'BUY' if candidate['direction']>0 else 'SELL'} "
                  f"persist={persistence + 1} part={participation:.4f} gq={market_state.graph_quality:.3f}",
                  file=_sys.stderr)
            return None

        narrative = self.tracker.update(
            candidate, market_state.cycle, self.detector, market_state.graph_quality
        )

        if narrative is None:
            return None

        self._update_metrics(narrative, market_state)
        self.maturity.calculate(narrative)
        self._update_expressions(narrative, market_state)

        return narrative

    def _compute_persistence(self, candidate: dict) -> int:
        return self._pre_birth_persistence.get(candidate['leader'], 0)

    def _increment_pre_birth(self, candidate: dict):
        key = candidate['leader']
        self._pre_birth_persistence[key] = self._pre_birth_persistence.get(key, 0) + 1

    def _update_metrics(self, narrative: NarrativeState, state: NarrativeInput):
        m = narrative.metrics
        strengths = state.currency_strengths
        leader = narrative.identity.leader

        m.conviction = min(abs(strengths.get(leader, 0)) / 0.001, 1.0)

        prev = self._previous_strengths.get(leader, 0)
        current = strengths.get(leader, 0)
        m.velocity = abs(current - prev) if prev != 0 else 0.0

        if hasattr(self, '_prev_velocity'):
            m.acceleration = abs(m.velocity - self._prev_velocity)
        self._prev_velocity = m.velocity

        stable_count = sum(
            1 for c in strengths
            if abs(strengths[c]) > 0.00005
        )
        m.leadership_stability = min(stable_count / 8.0, 1.0)

        churn = self.tracker._churn_count
        m.rank_churn = min(churn / 20.0, 1.0)

        m.propagation = state.currency_bursts.get(leader, 0)

        m.der_improvement = abs(state.currency_der.get(leader, 0))

        m.cohesion = state.graph_quality

        m.expression_score = None
        m.opportunity_density = None

        self._previous_strengths = dict(strengths)

    def _update_expressions(self, narrative: NarrativeState, state: NarrativeInput):
        leader = narrative.identity.leader
        expressions = []
        for ccy, val in sorted(state.currency_strengths.items(), key=lambda x: abs(x[1]), reverse=True):
            if ccy == leader:
                continue
            if len(expressions) >= 3:
                break
            if abs(val) > 0.00005:
                pair = f"{leader}{ccy}" if leader < ccy else f"{ccy}{leader}"
                expressions.append(pair)
        narrative.expressions = expressions if expressions else narrative.expressions

    def get_state(self) -> Optional[dict]:
        return serialize_narrative(self.tracker.active)

    def close(self):
        self.tracker.history.clear()
