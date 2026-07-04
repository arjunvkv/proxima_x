from __future__ import annotations

from typing import Any

from research.ucf.integration.ucf_propagation_schema import UCFPropagationField
from research.ucf.integration.ucf_pipeline_bridge import UCFPipelineBridge


_bridge: UCFPipelineBridge | None = None


def _get_bridge() -> UCFPipelineBridge:
    global _bridge
    if _bridge is None:
        _bridge = UCFPipelineBridge()
    return _bridge


def compute_ucf_field(
    symbols: list[str],
    technical_states: dict[str, dict[str, float]],
    fsv_states: dict[str, dict[str, float]],
    regime_state: dict[str, Any],
    timestamp: float | None = None,
) -> UCFPropagationField | None:
    bridge = _get_bridge()
    raw = bridge.process(symbols, technical_states, fsv_states, cev_state=None, regime_state=regime_state)
    if raw is None or not raw:
        return None
    import time
    return UCFPropagationField(
        timestamp=timestamp or time.time(),
        regime=regime_state.get("regime", "neutral"),
        ranked_symbols=raw.get("ranked_symbols", []),
        field=raw.get("field", {}),
        weights_used=raw.get("weights_used", {}),
        field_coherence=raw.get("field_coherence", 0.0),
        dominant_direction=raw.get("dominant_direction", 0),
    )
