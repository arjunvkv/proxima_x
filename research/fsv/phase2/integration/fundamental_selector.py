from ..core.fundamental_ranker import FundamentalRanker
from ..core.macro_alignment import MacroAlignmentEngine
from ..core.symbol_comparator import FundamentalComparator
from ...core.fsv_schema import FundamentalStateVector


class FundamentalSelector:

    def __init__(self) -> None:
        self.ranker = FundamentalRanker()
        self.alignment = MacroAlignmentEngine()
        self.comparator = FundamentalComparator()

    def select_best(
        self,
        top3_symbols: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
        base_convictions: dict[str, float],
    ) -> dict:
        n = len(top3_symbols)
        uniform_weight = 1.0 / n if n > 0 else 0.0

        try:
            compare_result = self.comparator.compare_symbols(top3_symbols, fsves, directions)
        except Exception:
            return self._fallback_result(top3_symbols, base_convictions, uniform_weight, True)

        recommendation = compare_result.get("recommendation", {})
        if not isinstance(recommendation, dict):
            recommendation = {}

        best_symbol = recommendation.get("best_symbol", top3_symbols[0]) if top3_symbols else ""
        if not best_symbol or best_symbol not in top3_symbols:
            best_symbol = top3_symbols[0] if top3_symbols else ""

        ranking: list[dict] = compare_result.get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) == 0:
            ranking = [{"symbol": s, "score": 0.0, "rank": i} for i, s in enumerate(top3_symbols)]

        ranking_vector: dict[str, float] = compare_result.get("ranking_vector", {})
        if not isinstance(ranking_vector, dict) or len(ranking_vector) == 0:
            ranking_vector = {s: uniform_weight for s in top3_symbols}

        confidence = recommendation.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = recommendation.get("reason", "fundamental selection")
        if not isinstance(reason, str):
            reason = "fundamental selection"

        adjusted_convictions = self._modulate_convictions(
            base_convictions, best_symbol, confidence, ranking_vector
        )

        modulation_applied = confidence > 0.0 and best_symbol in base_convictions

        return {
            "selected_symbol": best_symbol,
            "ranking": ranking,
            "ranking_vector": ranking_vector,
            "adjusted_convictions": adjusted_convictions,
            "recommendation": {
                "best_symbol": best_symbol,
                "confidence": confidence,
                "reason": reason,
            },
            "modulation_applied": modulation_applied,
            "is_selection": True,
            "is_blocking": False,
            "fallback_used": False,
        }

    def select_with_fallback(
        self,
        top3_symbols: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
        base_convictions: dict[str, float],
    ) -> dict:
        try:
            return self.select_best(top3_symbols, fsves, directions, base_convictions)
        except Exception:
            n = len(top3_symbols)
            uniform_weight = 1.0 / n if n > 0 else 0.0
            return self._fallback_result(top3_symbols, base_convictions, uniform_weight, True)

    def get_selection_influence(self, best_symbol: str, ranking_vector: dict[str, float]) -> float:
        n = len(ranking_vector)
        if n < 2:
            return 0.0
        uniform = 1.0 / n
        values = list(ranking_vector.values())
        actual_spread = max(values) - min(values)
        max_spread = 1.0 - uniform
        if max_spread <= 0.0:
            return 0.0
        influence = (actual_spread - uniform) / max_spread
        return max(0.0, min(1.0, influence))

    def _modulate_convictions(
        self,
        base_convictions: dict[str, float],
        best_symbol: str,
        confidence: float,
        ranking_vector: dict[str, float],
    ) -> dict[str, float]:
        adjusted: dict[str, float] = {}
        for symbol, base in base_convictions.items():
            weight = ranking_vector.get(symbol, 0.0)
            modulation = 1.0 + (weight * confidence)
            if symbol == best_symbol:
                modulation = 1.0 + confidence
            adjusted[symbol] = base * modulation
        return adjusted

    def _fallback_result(
        self,
        top3_symbols: list[str],
        base_convictions: dict[str, float],
        uniform_weight: float,
        fallback_used: bool,
    ) -> dict:
        selected = top3_symbols[0] if top3_symbols else ""
        n = len(top3_symbols)
        ranking = [{"symbol": s, "score": 0.0, "rank": i} for i, s in enumerate(top3_symbols)]
        ranking_vector = {s: uniform_weight for s in top3_symbols}
        adjusted_convictions = dict(base_convictions)

        return {
            "selected_symbol": selected,
            "ranking": ranking,
            "ranking_vector": ranking_vector,
            "adjusted_convictions": adjusted_convictions,
            "recommendation": {
                "best_symbol": selected,
                "confidence": 0.0,
                "reason": "fallback",
            },
            "modulation_applied": False,
            "is_selection": True,
            "is_blocking": False,
            "fallback_used": fallback_used,
        }
