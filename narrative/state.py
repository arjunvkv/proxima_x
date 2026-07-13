from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class NarrativePhase(Enum):
    EMERGING = "EMERGING"
    GROWING = "GROWING"
    MATURE = "MATURE"
    DECAYING = "DECAYING"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True)
class NarrativeIdentity:
    leader: str
    opponents: tuple
    direction: int


class NarrativeEvent(Enum):
    BIRTH = "BIRTH"
    CONTINUATION = "CONTINUATION"
    ROTATION = "ROTATION"
    DEATH = "DEATH"
    EXHAUSTION = "EXHAUSTION"


@dataclass
class NarrativeMetrics:
    conviction: Optional[float] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    leadership_stability: Optional[float] = None
    rank_churn: Optional[float] = None
    propagation: Optional[float] = None
    der_improvement: Optional[float] = None
    cohesion: Optional[float] = None
    expression_score: Optional[float] = None
    opportunity_density: Optional[float] = None


@dataclass
class NarrativeState:
    identity: NarrativeIdentity
    birth_cycle: int = 0
    last_seen_cycle: int = 0
    age: int = 0
    phase: NarrativePhase = NarrativePhase.EMERGING
    current_strength: float = 0.0
    peak_strength: float = 0.0
    previous_strength: float = 0.0
    nmi: float = 0.0
    rns: float = 0.0
    metrics: NarrativeMetrics = field(default_factory=NarrativeMetrics)
    expressions: List[str] = field(default_factory=list)
    active: bool = True
    last_event: Optional[NarrativeEvent] = None
    strength_delta: float = 0.0
