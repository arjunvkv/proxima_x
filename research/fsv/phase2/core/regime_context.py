import math
import statistics
from ...core.fsv_schema import FundamentalStateVector


class RegimeContextClassifier:

    REGIMES = ["risk_on", "risk_off", "neutral", "transition"]
    TRANSITION_THRESHOLD = 0.2

    def classify(
        self,
        fsves: dict[str, FundamentalStateVector],
        snapshot_history: list[dict] | None = None,
    ) -> str:
        sentiment_values: list[float] = [fsv.sentiment_gradient for fsv in fsves.values()]
        pressure_values: list[float] = [fsv.macro_pressure for fsv in fsves.values()]

        avg_sentiment: float = statistics.mean(sentiment_values) if sentiment_values else 0.0
        avg_pressure: float = statistics.mean(pressure_values) if pressure_values else 0.0

        if avg_sentiment > 0.15 and avg_pressure > -0.1:
            regime: str = "risk_on"
        elif avg_sentiment < -0.15 and avg_pressure < 0.1:
            regime: str = "risk_off"
        elif abs(avg_pressure) < 0.1:
            regime: str = "neutral"
        else:
            regime: str = "neutral"

        if snapshot_history is not None and len(snapshot_history) > 0:
            previous_snapshot: dict = snapshot_history[-1]
            previous_regime: str = previous_snapshot.get("regime", "neutral")
            if self.detect_transition(regime, previous_regime):
                regime = "transition"

        return regime

    def detect_transition(self, current: str, previous: str) -> bool:
        return current != previous

    def compute_regime_stability(
        self, fsves: dict[str, FundamentalStateVector]
    ) -> float:
        if not fsves:
            return 0.5

        stability_values: list[float] = [fsv.regime_stability for fsv in fsves.values()]
        alignment_values: list[float] = [fsv.bias_alignment for fsv in fsves.values()]
        event_risk_values: list[float] = [fsv.event_risk for fsv in fsves.values()]

        avg_stability: float = statistics.mean(stability_values)
        align_stdev: float = statistics.stdev(alignment_values) if len(alignment_values) > 1 else 0.0
        inverse_align_stdev: float = 1.0 - min(align_stdev, 1.0)
        low_event_ratio: float = sum(1.0 for er in event_risk_values if er < 0.6) / len(event_risk_values)

        stability_score: float = (
            avg_stability * 0.5
            + inverse_align_stdev * 0.3
            + low_event_ratio * 0.2
        )

        return max(0.0, min(1.0, stability_score))

    def get_regime_parameters(self, regime: str) -> dict:
        parameters: dict[str, dict] = {
            "risk_on": {
                "modulation_multiplier": 1.2,
                "conviction_boost": 0.05,
                "trend_follow_bias": 0.3,
            },
            "risk_off": {
                "modulation_multiplier": 1.1,
                "conviction_boost": 0.0,
                "defensive_bias": 0.3,
            },
            "neutral": {
                "modulation_multiplier": 0.8,
                "conviction_boost": 0.0,
                "trend_follow_bias": 0.0,
            },
            "transition": {
                "modulation_multiplier": 0.5,
                "conviction_boost": -0.05,
                "caution_flag": True,
            },
        }
        return parameters.get(regime, parameters["neutral"])

    def evaluate_regime_shift_risk(
        self,
        fsves: dict[str, FundamentalStateVector],
        current_regime: str,
    ) -> float:
        if not fsves:
            return 0.0

        max_update_ts: float = max(fsv.last_update_ts for fsv in fsves.values())
        recent_count: int = sum(
            1 for fsv in fsves.values()
            if max_update_ts - fsv.last_update_ts < 3600.0
        )
        recent_risk: float = min(recent_count / max(len(fsves), 1), 1.0) if recent_count >= 3 else 0.0

        avg_event_risk: float = statistics.mean(
            [fsv.event_risk for fsv in fsves.values()]
        )

        avg_stability: float = statistics.mean(
            [fsv.regime_stability for fsv in fsves.values()]
        )
        stability_risk: float = 1.0 - avg_stability

        shift_risk: float = recent_risk * 0.3 + avg_event_risk * 0.4 + stability_risk * 0.3

        return max(0.0, min(1.0, shift_risk))

    def compute_regime_aware_weight(
        self,
        symbol: str,
        fsv: FundamentalStateVector,
        regime: str,
    ) -> float:
        if regime == "neutral":
            return 1.0
        elif regime == "risk_on":
            return max(0.0, min(1.0, 0.5 + 0.5 * fsv.bias_alignment))
        elif regime == "risk_off":
            return max(0.0, min(1.0, 0.5 - 0.5 * fsv.bias_alignment))
        elif regime == "transition":
            return max(0.0, min(1.0, 0.6))
        else:
            return 1.0
