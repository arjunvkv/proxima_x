from typing import Optional
from .fsv_schema import FundamentalStateVector, NormalizedEvent, neutral_fsv, validate_fsv


class FSVEngine:

    def __init__(self, default_decay_lambda: float = 0.01) -> None:
        self.state_map: dict[str, FundamentalStateVector] = {}
        self.event_log: list[NormalizedEvent] = []
        self.update_count: int = 0
        self.default_decay_lambda: float = default_decay_lambda

    def update_with_event(self, event: NormalizedEvent) -> FundamentalStateVector:
        if event.symbol not in self.state_map:
            self.state_map[event.symbol] = neutral_fsv(event.symbol, self.default_decay_lambda)

        effect: dict = {
            "bias_alignment": event.direction_bias * event.impact_weight,
            "macro_pressure": event.surprise_score * event.impact_weight,
            "sentiment_gradient": event.direction_bias * 0.5,
            "event_risk": event.impact_weight,
            "regime_stability": 1.0 - abs(event.surprise_score) * 0.5,
        }

        current_state = self.state_map[event.symbol]
        decayed_state = current_state.apply_decay(event.timestamp)
        merged_state = decayed_state.merge(effect)
        merged_state.last_update_ts = event.timestamp
        self.state_map[event.symbol] = merged_state
        self.update_count += 1

        self.event_log.append(event)
        if len(self.event_log) > 10000:
            self.event_log = self.event_log[-10000:]

        return merged_state

    def get_state(self, symbol: str, current_time: Optional[float] = None) -> FundamentalStateVector:
        if symbol not in self.state_map:
            self.state_map[symbol] = neutral_fsv(symbol, self.default_decay_lambda)

        state = self.state_map[symbol]
        if current_time is not None:
            return state.apply_decay(current_time)

        return state

    def decay_all(self, current_time: float) -> None:
        for symbol in list(self.state_map.keys()):
            self.state_map[symbol] = self.state_map[symbol].apply_decay(current_time)

    def bulk_update(self, events: list[NormalizedEvent]) -> None:
        for event in events:
            self.update_with_event(event)

    def get_all_states(self) -> dict[str, FundamentalStateVector]:
        return dict(self.state_map)

    def get_stats(self) -> dict:
        if not self.state_map and not self.event_log:
            return {
                "total_symbols": 0,
                "total_updates": self.update_count,
                "avg_bias_alignment_magnitude": 0.0,
                "avg_event_risk": 0.0,
                "timestamp_range": None,
            }

        total_symbols: int = len(self.state_map)
        avg_bias_mag: float = 0.0
        avg_risk: float = 0.0
        if self.state_map:
            total_bias_mag = sum(abs(s.bias_alignment) for s in self.state_map.values())
            total_risk = sum(s.event_risk for s in self.state_map.values())
            avg_bias_mag = total_bias_mag / len(self.state_map)
            avg_risk = total_risk / len(self.state_map)

        timestamp_range: Optional[dict] = None
        if self.event_log:
            timestamps = [e.timestamp for e in self.event_log]
            timestamp_range = {
                "earliest": min(timestamps),
                "latest": max(timestamps),
            }

        return {
            "total_symbols": total_symbols,
            "total_updates": self.update_count,
            "avg_bias_alignment_magnitude": avg_bias_mag,
            "avg_event_risk": avg_risk,
            "timestamp_range": timestamp_range,
        }

    def reset(self) -> None:
        self.state_map.clear()
        self.event_log.clear()
        self.update_count = 0

    def get_event_history(self, symbol: Optional[str] = None, limit: int = 100) -> list[NormalizedEvent]:
        if symbol is None:
            return self.event_log[-limit:]
        filtered = [e for e in self.event_log if e.symbol == symbol]
        return filtered[-limit:]
