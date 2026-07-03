import math
from ...core.fsv_schema import FundamentalStateVector


class MacroAlignmentEngine:

    def evaluate_central_bank_bias(self, fsv: FundamentalStateVector) -> float:
        if fsv.regime_stability > 0.7:
            return max(-0.1, min(0.1, (1.0 - fsv.regime_stability) * fsv.macro_pressure * 2.0))
        if fsv.regime_stability < 0.4:
            if fsv.macro_pressure > 0.3:
                return max(0.3, min(0.7, (1.0 - fsv.regime_stability) * fsv.macro_pressure * 2.0))
            if fsv.macro_pressure < -0.3:
                return max(-0.7, min(-0.3, (1.0 - fsv.regime_stability) * fsv.macro_pressure * 2.0))
        bias: float = (1.0 - fsv.regime_stability) * fsv.macro_pressure * 2.0
        return max(-1.0, min(1.0, bias))

    def evaluate_macro_divergence(
        self, fsv_a: FundamentalStateVector, fsv_b: FundamentalStateVector
    ) -> float:
        bias_diff: float = abs(fsv_a.bias_alignment - fsv_b.bias_alignment)
        pressure_diff: float = abs(fsv_a.macro_pressure - fsv_b.macro_pressure)
        sentiment_diff: float = abs(fsv_a.sentiment_gradient - fsv_b.sentiment_gradient)
        divergence: float = (bias_diff + pressure_diff + sentiment_diff) / 6.0
        return max(0.0, min(1.0, divergence))

    def evaluate_risk_environment(
        self, fsves: dict[str, FundamentalStateVector]
    ) -> str:
        if not fsves:
            return "neutral"
        total: int = len(fsves)
        risk_on_count: int = sum(
            1 for fsv in fsves.values() if fsv.sentiment_gradient > 0.2
        )
        risk_off_count: int = sum(
            1 for fsv in fsves.values() if fsv.sentiment_gradient < -0.2
        )
        if risk_on_count / total >= 0.7:
            return "risk_on"
        if risk_off_count / total >= 0.7:
            return "risk_off"
        avg_pressure_mag: float = (
            sum(abs(fsv.macro_pressure) for fsv in fsves.values()) / total
        )
        if avg_pressure_mag < 0.15:
            return "neutral"
        return "mixed"

    def check_direction_alignment(
        self,
        fsv: FundamentalStateVector,
        predicted_direction: int,
        environment: str | None = None,
    ) -> tuple[bool, float]:
        if predicted_direction == 0:
            return (True, 0.0)
        alignment: float = fsv.bias_alignment * predicted_direction
        is_aligned: bool = alignment > 0
        alignment_strength: float = max(0.0, min(1.0, abs(alignment)))
        if environment == "risk_off" and predicted_direction == 1:
            alignment_strength *= 0.8
        if environment == "risk_on" and predicted_direction == -1:
            alignment_strength *= 0.8
        return (is_aligned, alignment_strength)

    def compute_environment_score(self, environment: str, direction: int) -> float:
        if environment == "risk_on":
            return 1.0 if direction == 1 else 0.3
        if environment == "risk_off":
            return 1.0 if direction == -1 else 0.3
        if environment == "neutral":
            return 0.6
        return 0.5

    def full_evaluation(
        self,
        symbol: str,
        fsv: FundamentalStateVector,
        direction: int,
        all_fsves: dict[str, FundamentalStateVector],
    ) -> dict:
        central_bank_bias: float = self.evaluate_central_bank_bias(fsv)
        risk_environment: str = self.evaluate_risk_environment(all_fsves)
        is_aligned: bool
        alignment_strength: float
        is_aligned, alignment_strength = self.check_direction_alignment(
            fsv, direction, risk_environment
        )
        environment_score: float = self.compute_environment_score(
            risk_environment, direction
        )
        composite_alignment: float = (alignment_strength + environment_score) / 2.0
        return {
            "symbol": symbol,
            "central_bank_bias": central_bank_bias,
            "direction_aligned": is_aligned,
            "alignment_strength": alignment_strength,
            "risk_environment": risk_environment,
            "environment_score": environment_score,
            "composite_alignment": composite_alignment,
        }
