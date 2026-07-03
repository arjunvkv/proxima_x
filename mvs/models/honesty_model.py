from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HonestyScore:
    layer_name: str
    score: float
    directional_accuracy: float
    timing_precision: float
    path_alignment: float
    delay_penalty: float
    contradiction_penalty: float
    sample_count: int
    timestamp: int

    def __post_init__(self) -> None:
        self.score = max(0.0, min(100.0, self.score))
