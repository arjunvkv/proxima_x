from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ConflictType(str, Enum):
    FALSE_BELIEF = "FALSE_BELIEF"
    DELAYED_BELIEF = "DELAYED_BELIEF"
    CORRUPTED_BELIEF = "CORRUPTED_BELIEF"
    SUPPRESSED_GOOD = "SUPPRESSED_GOOD"
    ALLOWED_BAD = "ALLOWED_BAD"
    PATH_DISSONANCE = "PATH_DISSONANCE"
    TIMING_MISALIGNMENT = "TIMING_MISALIGNMENT"


@dataclass(slots=True)
class ConflictRecord:
    tick_id: int
    conflict_type: ConflictType
    severity: float
    description: str
    layer: str
    timestamp: int

    def __post_init__(self) -> None:
        self.severity = max(0.0, min(1.0, self.severity))


@dataclass(slots=True)
class ConflictResult:
    conflicts: List[ConflictRecord] = field(default_factory=list)
    aggregate_score: float = 0.0
    timestamp: int = 0

    def recompute(self) -> None:
        if not self.conflicts:
            self.aggregate_score = 0.0
            return
        self.aggregate_score = sum(x.severity for x in self.conflicts) / len(self.conflicts)
