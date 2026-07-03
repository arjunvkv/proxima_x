from dataclasses import dataclass, field
from typing import Final
import time

EVENT_TYPES: Final[tuple[str, ...]] = ("CPI", "NEWS", "RATE", "GDP", "SENTIMENT", "UNKNOWN")


@dataclass
class FundamentalStateVector:
    symbol: str
    bias_alignment: float = 0.0
    macro_pressure: float = 0.0
    sentiment_gradient: float = 0.0
    event_risk: float = 0.5
    regime_stability: float = 0.5
    last_update_ts: float = field(default_factory=time.time)
    decay_lambda: float = 0.01

    def apply_decay(self, current_time: float) -> "FundamentalStateVector":
        dt = current_time - self.last_update_ts
        if dt < 0.0:
            dt = 0.0
        factor = __import__("math").exp(-self.decay_lambda * dt)
        return FundamentalStateVector(
            symbol=self.symbol,
            bias_alignment=self.bias_alignment * factor,
            macro_pressure=self.macro_pressure * factor,
            sentiment_gradient=self.sentiment_gradient * factor,
            event_risk=0.5 + (self.event_risk - 0.5) * factor,
            regime_stability=0.5 + (self.regime_stability - 0.5) * factor,
            last_update_ts=current_time,
            decay_lambda=self.decay_lambda,
        )

    def merge(self, event_vector: dict) -> "FundamentalStateVector":
        merged = {}
        for attr in (
            "bias_alignment",
            "macro_pressure",
            "sentiment_gradient",
            "event_risk",
            "regime_stability",
        ):
            current = getattr(self, attr)
            incoming = event_vector.get(attr)
            if incoming is not None:
                weight = event_vector.get("impact_weight", 0.5)
                merged[attr] = current * (1.0 - weight) + incoming * weight
            else:
                merged[attr] = current
        return FundamentalStateVector(
            symbol=event_vector.get("symbol", self.symbol),
            bias_alignment=merged["bias_alignment"],
            macro_pressure=merged["macro_pressure"],
            sentiment_gradient=merged["sentiment_gradient"],
            event_risk=merged["event_risk"],
            regime_stability=merged["regime_stability"],
            last_update_ts=time.time(),
            decay_lambda=self.decay_lambda,
        )

    def to_feature_vector(self) -> list[float]:
        return [
            self.bias_alignment,
            self.macro_pressure,
            self.sentiment_gradient,
            self.event_risk,
            self.regime_stability,
        ]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bias_alignment": self.bias_alignment,
            "macro_pressure": self.macro_pressure,
            "sentiment_gradient": self.sentiment_gradient,
            "event_risk": self.event_risk,
            "regime_stability": self.regime_stability,
            "last_update_ts": self.last_update_ts,
            "decay_lambda": self.decay_lambda,
        }


@dataclass
class NormalizedEvent:
    symbol: str
    event_type: str = "UNKNOWN"
    surprise_score: float = 0.0
    direction_bias: float = 0.0
    impact_weight: float = 0.5
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    raw_data: dict = field(default_factory=dict)


def neutral_fsv(symbol: str, decay_lambda: float = 0.01) -> FundamentalStateVector:
    return FundamentalStateVector(
        symbol=symbol,
        bias_alignment=0.0,
        macro_pressure=0.0,
        sentiment_gradient=0.0,
        event_risk=0.5,
        regime_stability=0.5,
        last_update_ts=time.time(),
        decay_lambda=decay_lambda,
    )


def validate_fsv(fsv: FundamentalStateVector) -> bool:
    if not (-1.0 <= fsv.bias_alignment <= 1.0):
        return False
    if not (-1.0 <= fsv.macro_pressure <= 1.0):
        return False
    if not (-1.0 <= fsv.sentiment_gradient <= 1.0):
        return False
    if not (0.0 <= fsv.event_risk <= 1.0):
        return False
    if not (0.0 <= fsv.regime_stability <= 1.0):
        return False
    if not isinstance(fsv.symbol, str) or not fsv.symbol:
        return False
    if fsv.decay_lambda < 0.0:
        return False
    return True


def validate_event(event: NormalizedEvent) -> bool:
    if not isinstance(event.symbol, str) or not event.symbol:
        return False
    if event.event_type not in EVENT_TYPES:
        return False
    if not (-1.0 <= event.surprise_score <= 1.0):
        return False
    if not (-1.0 <= event.direction_bias <= 1.0):
        return False
    if not (0.0 <= event.impact_weight <= 1.0):
        return False
    if not isinstance(event.raw_data, dict):
        return False
    return True
