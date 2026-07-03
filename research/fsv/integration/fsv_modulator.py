from dataclasses import dataclass
from typing import Any

from ..core.fsv_schema import FundamentalStateVector


class FSVModulator:
    MAX_ADJUSTMENT: float = 0.15
    NEUTRAL_BIAS: float = 0.0

    @classmethod
    def compute_modulation(cls, fsv: FundamentalStateVector) -> float:
        alignment_factor: float = fsv.bias_alignment * 0.4
        pressure_factor: float = fsv.macro_pressure * 0.3
        sentiment_factor: float = fsv.sentiment_gradient * 0.2
        risk_penalty: float = -(fsv.event_risk - 0.5) * 0.1
        total: float = alignment_factor + pressure_factor + sentiment_factor + risk_penalty
        if total > cls.MAX_ADJUSTMENT:
            total = cls.MAX_ADJUSTMENT
        elif total < -cls.MAX_ADJUSTMENT:
            total = -cls.MAX_ADJUSTMENT
        return total

    @classmethod
    def modulate(cls, base_conviction: float, fsv: FundamentalStateVector) -> tuple[float, dict[str, Any]]:
        modulation: float = cls.compute_modulation(fsv)
        adjusted: float = base_conviction * (1.0 + modulation)
        if adjusted > 1.0:
            adjusted = 1.0
        elif adjusted < 0.0:
            adjusted = 0.0
        alignment_factor: float = fsv.bias_alignment * 0.4
        pressure_factor: float = fsv.macro_pressure * 0.3
        sentiment_factor: float = fsv.sentiment_gradient * 0.2
        risk_penalty: float = -(fsv.event_risk - 0.5) * 0.1
        explanation: dict[str, Any] = {
            "base_conviction": base_conviction,
            "modulation_factor": modulation,
            "adjusted_conviction": adjusted,
            "components": {
                "alignment_contribution": alignment_factor,
                "pressure_contribution": pressure_factor,
                "sentiment_contribution": sentiment_factor,
                "risk_penalty": risk_penalty,
            },
            "fsv_snapshot": {
                "symbol": fsv.symbol,
                "bias_alignment": fsv.bias_alignment,
                "macro_pressure": fsv.macro_pressure,
                "sentiment_gradient": fsv.sentiment_gradient,
                "event_risk": fsv.event_risk,
                "regime_stability": fsv.regime_stability,
            },
        }
        return (adjusted, explanation)

    @classmethod
    def modulate_batch(
        cls, convictions: dict[str, tuple[float, FundamentalStateVector]]
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        for symbol, (base_conviction, fsv) in convictions.items():
            result[symbol] = cls.modulate(base_conviction, fsv)
        return result

    @classmethod
    def neutral_modulation(cls) -> dict[str, Any]:
        return {
            "base_conviction": 0.0,
            "modulation_factor": 0.0,
            "adjusted_conviction": 0.0,
            "components": {
                "alignment_contribution": 0.0,
                "pressure_contribution": 0.0,
                "sentiment_contribution": 0.0,
                "risk_penalty": 0.0,
            },
            "fsv_snapshot": None,
        }


@dataclass
class FSVIntegrationPoint:
    pipeline_stage: str = "post_signal_authority_pre_uesl"
    influence_type: str = "soft_modulation"
    max_influence: float = 0.15
    is_gate: bool = False
    is_blocking: bool = False
