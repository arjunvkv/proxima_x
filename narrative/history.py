from dataclasses import dataclass, field
from typing import List


@dataclass
class NarrativeSnapshot:
    cycle: int
    strength: float
    nmi: float
    phase: str
    leader: str


@dataclass
class NarrativeHistory:
    snapshots: List[NarrativeSnapshot] = field(default_factory=list)

    def record(self, snapshot: NarrativeSnapshot):
        self.snapshots.append(snapshot)

    def recent(self, n: int = 10) -> List[NarrativeSnapshot]:
        return self.snapshots[-n:]

    def clear(self):
        self.snapshots.clear()
