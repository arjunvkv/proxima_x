from typing import Dict, Optional
from .state import (
    NarrativeState,
    NarrativeIdentity,
    NarrativePhase,
    NarrativeEvent,
    NarrativeMetrics,
)
from .history import NarrativeHistory, NarrativeSnapshot
from .detector import NarrativeDetector


class NarrativeTracker:
    def __init__(self):
        self.active: Optional[NarrativeState] = None
        self.history = NarrativeHistory()
        self._persistence_counter: Dict[str, int] = {}
        self._previous_leader: Optional[str] = None
        self._churn_count: int = 0

    def update(
        self,
        candidate: dict,
        cycle: int,
        detector: NarrativeDetector,
        graph_quality: float,
    ) -> NarrativeState:
        if candidate is None:
            if self.active is not None:
                self.active.active = False
                self.active.last_event = NarrativeEvent.DEATH
                self._record_snapshot(cycle)
                dead = self.active
                self.active = None
                return dead
            return None

        identity = NarrativeIdentity(
            leader=candidate["leader"],
            opponents=candidate.get("opponents", ()),
            direction=candidate["direction"],
        )

        if self.active is None:
            self.active = NarrativeState(
                identity=identity,
                birth_cycle=cycle,
                last_seen_cycle=cycle,
                age=0,
                phase=NarrativePhase.EMERGING,
                current_strength=candidate["strength"],
                peak_strength=candidate["strength"],
                last_event=NarrativeEvent.BIRTH,
            )
            self._reset_persistence(identity)
            self._record_snapshot(cycle)
            return self.active

        if self.active.identity.leader != identity.leader:
            self.active.active = False
            self.active.last_event = NarrativeEvent.ROTATION
            self._record_snapshot(cycle)
            self._churn_count += 1
            self.active = NarrativeState(
                identity=identity,
                birth_cycle=cycle,
                last_seen_cycle=cycle,
                age=0,
                phase=NarrativePhase.EMERGING,
                current_strength=candidate["strength"],
                peak_strength=candidate["strength"],
                last_event=NarrativeEvent.BIRTH,
            )
            self._reset_persistence(identity)
            self._record_snapshot(cycle)
            return self.active

        self.active.previous_strength = self.active.current_strength
        self.active.current_strength = candidate["strength"]
        self.active.strength_delta = self.active.current_strength - self.active.previous_strength
        self.active.age = cycle - self.active.birth_cycle
        self.active.last_seen_cycle = cycle
        self.active.last_event = NarrativeEvent.CONTINUATION

        if abs(self.active.current_strength) > abs(self.active.peak_strength):
            self.active.peak_strength = self.active.current_strength

        self._update_persistence(identity)

        if self.active.nmi > 0.9:
            self.active.phase = NarrativePhase.EXHAUSTED
            self.active.last_event = NarrativeEvent.EXHAUSTION
        elif self.active.nmi > 0.7:
            self.active.phase = NarrativePhase.DECAYING
        elif self.active.nmi > 0.5:
            self.active.phase = NarrativePhase.MATURE
        elif self.active.nmi > 0.2:
            self.active.phase = NarrativePhase.GROWING
        else:
            self.active.phase = NarrativePhase.EMERGING

        self._record_snapshot(cycle)
        return self.active

    def _reset_persistence(self, identity: NarrativeIdentity):
        key = self._identity_key(identity)
        self._persistence_counter[key] = 1

    def _update_persistence(self, identity: NarrativeIdentity):
        key = self._identity_key(identity)
        self._persistence_counter[key] = self._persistence_counter.get(key, 0) + 1

    def persistence(self, identity: NarrativeIdentity) -> int:
        return self._persistence_counter.get(self._identity_key(identity), 0)

    def _identity_key(self, identity: NarrativeIdentity) -> str:
        return f"{identity.leader}_{identity.direction}"

    def _record_snapshot(self, cycle: int):
        if self.active is None:
            return
        self.history.record(
            NarrativeSnapshot(
                cycle=cycle,
                strength=self.active.current_strength,
                nmi=self.active.nmi,
                phase=self.active.phase.value,
                leader=self.active.identity.leader,
            )
        )
