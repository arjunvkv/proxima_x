from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class NarrativeInput:
    cycle: int
    currency_strengths: Dict[str, float]
    currency_bursts: Dict[str, float]
    currency_der: Dict[str, float]
    graph_quality: float
    tick_quality: float
    reliability: Dict[str, float]
