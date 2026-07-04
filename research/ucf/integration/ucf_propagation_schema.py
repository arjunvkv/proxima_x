from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UCFPropagationField:
    timestamp: float
    regime: str
    ranked_symbols: list[dict[str, Any]]
    field: dict[str, dict[str, Any]]
    weights_used: dict[str, float]
    field_coherence: float
    dominant_direction: int


@dataclass
class LayerUCFFeedback:
    layer_name: str
    agreement_delta: float
    divergence: float
    confidence_tension: float
    alignment_drift: float
