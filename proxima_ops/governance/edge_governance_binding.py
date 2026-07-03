import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeEvent:
    timestamp: float = field(default_factory=time.time)
    signal_id: str = ""
    symbol: str = ""
    strategy: str = ""
    confidence: float = 0.0
    direction: str = ""
    current_state: str = ""
    eligible_for_arming: bool = False
    eligibility_reason: str = ""


class EdgeGovernanceBinding:
    ARMING_CONFIDENCE_MIN = 0.60
    ARMING_CONFLICT_MAX = 0.30
    ARMING_MOF_MIN = 0.35

    def __init__(self):
        self._events: list[EdgeEvent] = []

    def evaluate_arming_eligibility(
        self,
        signal: dict,
        mof_state: str,
        mof_score: float,
        portfolio_conflict: float,
        current_system_state: str,
    ) -> EdgeEvent:
        event = EdgeEvent(
            signal_id=signal.get("id", "unknown"),
            symbol=signal.get("symbol", ""),
            strategy=signal.get("strategy", ""),
            confidence=signal.get("confidence", 0.0),
            direction=signal.get("direction", ""),
            current_state=current_system_state,
        )

        reasons = []
        eligible = True

        if current_system_state not in ("OBSERVE", "ARMED"):
            reasons.append(f"System state {current_system_state} does not accept arming eligibility")
            eligible = False

        if event.confidence < self.ARMING_CONFIDENCE_MIN:
            reasons.append(f"Confidence {event.confidence:.4f} < min {self.ARMING_CONFIDENCE_MIN}")
            eligible = False

        if portfolio_conflict > self.ARMING_CONFLICT_MAX:
            reasons.append(f"Portfolio conflict {portfolio_conflict:.4f} > max {self.ARMING_CONFLICT_MAX}")
            eligible = False

        if mof_score < self.ARMING_MOF_MIN:
            reasons.append(f"MOF score {mof_score:.4f} < min {self.ARMING_MOF_MIN}")
            eligible = False

        event.eligible_for_arming = eligible
        event.eligibility_reason = "; ".join(reasons) if reasons else "All arming conditions satisfied"
        self._events.append(event)
        return event

    def can_trigger_execution_via_arming(self, event: EdgeEvent) -> tuple[bool, str]:
        if not event.eligible_for_arming:
            return False, f"Edge {event.signal_id} not arming-eligible: {event.eligibility_reason}"
        return True, f"Edge {event.signal_id} eligible — system may consider OBSERVE -> ARMED transition"

    @property
    def events(self) -> list[EdgeEvent]:
        return list(self._events)

    def describe(self) -> dict:
        return {
            "arming_confidence_min": self.ARMING_CONFIDENCE_MIN,
            "arming_conflict_max": self.ARMING_CONFLICT_MAX,
            "arming_mof_min": self.ARMING_MOF_MIN,
            "total_edge_events": len(self._events),
            "eligible_count": sum(1 for e in self._events if e.eligible_for_arming),
            "ineligible_count": sum(1 for e in self._events if not e.eligible_for_arming),
            "latest_event": self._events[-1] if self._events else None,
        }
