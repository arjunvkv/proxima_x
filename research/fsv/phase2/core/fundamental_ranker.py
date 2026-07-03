import math
import statistics
from ...core.fsv_schema import FundamentalStateVector


SCORE_WEIGHTS: dict[str, float] = {
    "bias_alignment": 0.25,
    "macro_pressure": 0.25,
    "sentiment_gradient": 0.15,
    "regime_stability": 0.20,
    "event_risk_penalty": 0.15,
}


class FundamentalRanker:

    def score_symbol(
        self, symbol: str, fsv: FundamentalStateVector, direction_prediction: int
    ) -> dict:
        direction_alignment: bool = (
            (fsv.bias_alignment > 0 and direction_prediction > 0)
            or (fsv.bias_alignment < 0 and direction_prediction < 0)
        )
        sign_multiplier: float = 1.0 if direction_alignment else 0.5

        alignment_score: float = (
            abs(fsv.bias_alignment)
            * SCORE_WEIGHTS["bias_alignment"]
            * sign_multiplier
        )
        pressure_score: float = (
            abs(fsv.macro_pressure)
            * SCORE_WEIGHTS["macro_pressure"]
            * sign_multiplier
        )
        sentiment_score: float = (
            abs(fsv.sentiment_gradient) * SCORE_WEIGHTS["sentiment_gradient"]
        )
        stability_score: float = (
            fsv.regime_stability * SCORE_WEIGHTS["regime_stability"]
        )
        risk_penalty: float = (
            (1.0 - fsv.event_risk) * SCORE_WEIGHTS["event_risk_penalty"]
        )

        raw_score: float = (
            alignment_score
            + pressure_score
            + sentiment_score
            + stability_score
            + risk_penalty
        )
        fundamental_score: float = max(0.0, min(1.0, raw_score))
        confidence: float = max(
            0.0,
            min(
                1.0,
                fundamental_score * (1.0 + abs(fsv.bias_alignment) * 0.3),
            ),
        )
        adjusted_for_regime: bool = fsv.regime_stability < 0.5

        return {
            "symbol": symbol,
            "fundamental_score": fundamental_score,
            "confidence": confidence,
            "components": {
                "alignment_score": alignment_score,
                "pressure_score": pressure_score,
                "sentiment_score": sentiment_score,
                "stability_score": stability_score,
                "risk_penalty": risk_penalty,
            },
            "direction_alignment": direction_alignment,
            "adjusted_for_regime": adjusted_for_regime,
        }

    def rank_symbols(
        self, candidates: dict[str, tuple[FundamentalStateVector, int]]
    ) -> list[dict]:
        results: list[dict] = []
        for symbol, (fsv, direction_prediction) in candidates.items():
            results.append(
                self.score_symbol(symbol, fsv, direction_prediction)
            )

        results.sort(key=lambda r: r["fundamental_score"], reverse=True)

        if results:
            max_score: float = max(
                r["fundamental_score"] for r in results
            )
            if max_score > 0:
                for r in results:
                    r["fundamental_score"] = r["fundamental_score"] / max_score

        return results

    def get_ranking_vector(
        self, ranked: list[dict]
    ) -> dict[str, float]:
        scores: list[float] = [math.exp(r["fundamental_score"] * 3) for r in ranked]
        total: float = sum(scores)
        vector: dict[str, float] = {}
        for i, r in enumerate(ranked):
            vector[r["symbol"]] = scores[i] / total if total > 0 else 0.0
        return vector

    def compute_conviction_adjustment(
        self, ranked: list[dict], base_convictions: dict[str, float]
    ) -> dict[str, float]:
        adjusted: dict[str, float] = {}
        for r in ranked:
            symbol: str = r["symbol"]
            base: float = base_convictions.get(symbol, 0.0)
            normalized_score: float = r["fundamental_score"]
            factor: float = 0.8 + 0.2 * normalized_score
            adjusted[symbol] = base * factor
        return adjusted
