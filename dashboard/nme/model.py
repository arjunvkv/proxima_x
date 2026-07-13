from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NMEViewModel:
    cycle: int = 0
    phase: str = "--"
    nmi: float = 0.0
    trajectory: str = "→"
    leader: str = "--"
    leader_strength: float = 0.0
    leader_delta: float = 0.0
    direction: int = 0
    opponent_strengths: Dict[str, float] = field(default_factory=dict)
    age: int = 0
    last_event: Optional[str] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    expressions: List[dict] = field(default_factory=list)
    research_layers: Dict[str, float] = field(default_factory=dict)
    tick_quality: float = 0.0
    graph_quality: float = 0.0
    reliability: Dict[str, float] = field(default_factory=dict)
    active: bool = False
