from __future__ import annotations

import math
import random
from typing import Any

from ...fsv.core.fsv_engine import FSVEngine
from ...fsv.simulation.synthetic_event_generator import SyntheticMacroGenerator
from ...fsv.core.fsv_schema import NormalizedEvent


class ReplayMacroAdapter:
    def __init__(self, fsv_engine: FSVEngine, symbols: list[str] | None = None) -> None:
        self.fsv_engine: FSVEngine = fsv_engine
        self.symbols: list[str] = symbols or ["EURUSD", "GBPUSD", "USDJPY"]
        self._macro_source: SyntheticMacroGenerator = SyntheticMacroGenerator()
        self._macro_schedule: list[tuple[float, NormalizedEvent]] = []
        self._current_regime: str = "neutral"
        self._current_pressure: float = 0.2
        self._initialized: bool = False

    def initialize(self, duration_seconds: float = 7200) -> None:
        for scenario in ["crisis", "trend", "conflict"]:
            events = self._macro_source.stress_scenario(scenario)
            for e in events:
                self.fsv_engine.update_with_event(e)

        stream = self._macro_source.generate_event_stream(
            duration_seconds=duration_seconds, events_per_minute=2
        )
        for e in stream:
            self.fsv_engine.update_with_event(e)

        self._macro_schedule = self._build_macro_schedule(duration_seconds)
        self._initialized = True

    def _build_macro_schedule(
        self, duration_seconds: float
    ) -> list[tuple[float, NormalizedEvent]]:
        schedule: list[tuple[float, NormalizedEvent]] = []
        event_types: list[str] = ["NEWS", "MACRO_RELEASE", "SENTIMENT", "ALERT"]
        num_events: int = int(duration_seconds / 60) * 2
        for _ in range(num_events):
            ts = random.uniform(0, duration_seconds)
            event_type: str = random.choice(event_types)
            event = NormalizedEvent(
                event_type=event_type,
                symbol=random.choice(self.symbols),
                timestamp=ts,
                surprise_score=random.uniform(-0.8, 0.8),
                direction_bias=random.uniform(-2.0, 2.0),
                impact_weight=random.uniform(0.1, 1.0),
                source="macro_adapter",
                raw_data={"event_type": event_type, "source": "macro_adapter", "value": 0.0, "previous": 0.0, "forecast": 0.0},
            )
            schedule.append((ts, event))
        schedule.sort(key=lambda x: x[0])
        return schedule

    def get_macro_state(self, timestamp: float) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()

        recent_events: list[NormalizedEvent] = []
        for ts, event in self._macro_schedule:
            if 0 <= timestamp - ts < 900:
                recent_events.append(event)

        if recent_events:
            pressure: float = sum(abs(e.direction_bias) for e in recent_events) / len(recent_events)
            avg_magnitude: float = sum(e.direction_bias for e in recent_events) / len(recent_events)
            self._current_pressure = min(1.0, pressure / 5.0)
            if abs(avg_magnitude) > 1.5:
                self._current_regime = "risk_off"
            elif avg_magnitude > 0.5:
                self._current_regime = "risk_on"
            elif pressure > 2.0:
                self._current_regime = "transition"
            else:
                self._current_regime = "neutral"
        else:
            self._current_regime = "neutral"
            self._current_pressure = max(0.05, self._current_pressure * 0.95)

        bias_alignment: float = 0.0
        if recent_events:
            raw_bias: float = sum(e.surprise_score for e in recent_events)
            bias_alignment = math.tanh(raw_bias / max(1, len(recent_events)))

        return {
            "regime": self._current_regime,
            "bias_alignment": bias_alignment,
            "macro_pressure": self._current_pressure,
            "recent_event_count": len(recent_events),
            "avg_magnitude": avg_magnitude if recent_events else 0.0,
        }

    def apply_to_fsv(self, timestamp: float) -> None:
        macro_state = self.get_macro_state(timestamp)

        pressure: float = macro_state["macro_pressure"]
        bias_dir: float = 1.0 if macro_state["bias_alignment"] >= 0 else -1.0

        for symbol in self.symbols:
            raw_state = self.fsv_engine.get_state(symbol, timestamp)
            new_bias = pressure * 0.3 * bias_dir
            current_bias = raw_state.bias_alignment
            blended_bias = current_bias * 0.7 + new_bias * 0.3
            state = self.fsv_engine.state_map[symbol]
            state.bias_alignment = blended_bias
            state.regime_stability = max(0.1, 1.0 - pressure * 0.5)

    def get_regime_context(
        self, timestamp: float
    ) -> dict[str, bool | float | str]:
        macro_state = self.get_macro_state(timestamp)
        return {
            "regime": macro_state["regime"],
            "regime_stability": max(
                0.1, 1.0 - macro_state["macro_pressure"] * 0.5
            ),
            "fsv_entropy": macro_state["macro_pressure"],
            "technical_volatility": macro_state["macro_pressure"],
            "recent_prediction_error": 0.0,
            "exposure_concentration": 0.0,
        }
