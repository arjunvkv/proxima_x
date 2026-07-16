from typing import Dict, Optional, Tuple
from .state import NarrativeIdentity


class NarrativeDetector:
    def __init__(self, strength_threshold: float = 0.00005, persistence_required: int = 5):
        self.threshold = strength_threshold
        self.persistence_required = persistence_required

    def detect_candidate(self, strengths: Dict[str, float]) -> Optional[dict]:
        if not strengths:
            return None
        leader = max(strengths, key=lambda c: abs(strengths[c]))
        value = strengths[leader]
        if abs(value) < self.threshold:
            return None
        direction = 1 if value > 0 else -1
        opponents = tuple(
            c for c, v in sorted(strengths.items(), key=lambda x: abs(x[1]), reverse=True)
            if c != leader and abs(v) > self.threshold * 0.5
        )[:3]
        return {
            "leader": leader,
            "opponents": opponents,
            "direction": direction,
            "strength": value,
        }

    def should_birth(self, candidate: dict, persistence: int, graph_quality: float, participation: float) -> bool:
        if candidate is None:
            return False
        return persistence >= self.persistence_required and graph_quality > 0.3 and abs(participation) > 0.02

    def should_die(self, narrative, strengths: Dict[str, float]) -> bool:
        leader = narrative.identity.leader
        if leader not in strengths:
            return True
        current = strengths[leader]
        return abs(current) < self.threshold

    def get_opponents(self, strengths: Dict[str, float], leader: str) -> Tuple[str, ...]:
        return tuple(
            c for c, v in sorted(strengths.items(), key=lambda x: abs(x[1]), reverse=True)
            if c != leader
        )[:3]
