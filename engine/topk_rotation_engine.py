from typing import List, Tuple, Dict, Set


# Currency factor map: symbol -> set of currency factors it exposes
# Multi-membership prevents hidden correlation (e.g. EURJPY exposes both EUR and JPY)
_CURRENCY_FACTORS = {
    "EURUSD": {"EUR", "USD"}, "GBPUSD": {"GBP", "USD"}, "USDJPY": {"USD", "JPY"},
    "USDCAD": {"USD", "CAD"}, "USDCHF": {"USD", "CHF"}, "AUDUSD": {"AUD", "USD"},
    "NZDUSD": {"NZD", "USD"}, "EURJPY": {"EUR", "JPY"}, "EURGBP": {"EUR", "GBP"},
    "EURCHF": {"EUR", "CHF"}, "EURAUD": {"EUR", "AUD"}, "EURCAD": {"EUR", "CAD"},
    "EURNZD": {"EUR", "NZD"}, "GBPJPY": {"GBP", "JPY"}, "GBPCHF": {"GBP", "CHF"},
    "GBPAUD": {"GBP", "AUD"}, "GBPCAD": {"GBP", "CAD"}, "GBPNZD": {"GBP", "NZD"},
    "AUDJPY": {"AUD", "JPY"}, "AUDNZD": {"AUD", "NZD"}, "AUDCAD": {"AUD", "CAD"},
    "AUDCHF": {"AUD", "CHF"}, "CADJPY": {"CAD", "JPY"}, "CHFJPY": {"CHF", "JPY"},
    "NZDCAD": {"NZD", "CAD"}, "NZDCHF": {"NZD", "CHF"}, "NZDJPY": {"NZD", "JPY"},
    "XAUUSD": {"XAU", "USD"}, "XAGUSD": {"XAG", "USD"},
}

_OVERRIDE_MARGIN = 0.15  # if top cluster score exceeds next by this, allow override


def _factors_of(symbol: str) -> Set[str]:
    """Return the set of currency factors a symbol exposes."""
    return _CURRENCY_FACTORS.get(symbol, {symbol})


class TopKRotationEngine:
    """
    Stabilized Top-K selector with:
    - Hysteresis margin
    - Persistence requirement
    - Rotation cooldown
    - Multi-factor correlation cluster constraint (no overlapping currency factors)
    """

    def __init__(self,
                 top_k: int = 2,
                 min_margin: float = 0.03,
                 persistence: int = 3):
        self.top_k = top_k
        self.min_margin = min_margin
        self.persistence = persistence

        self._last_selected: List[str] = []
        self._last_scores: Dict[str, float] = {}
        self._stable_counter: int = 0

    def _enforce_cluster_constraint(self, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Ensure no two symbols share any currency factor in the selection.
        Override: if top score exceeds next candidate by min_margin × 3, allow."""
        selected = []
        covered_factors: Set[str] = set()
        if ranked:
            best_score = ranked[0][1]
        else:
            best_score = 0.0
        for sym, score in ranked:
            factors = _factors_of(sym)
            # Override: if this symbol's score dominates, allow despite overlap
            is_dominated = (best_score - score) > _OVERRIDE_MARGIN
            if factors.isdisjoint(covered_factors) or (not selected and not is_dominated):
                selected.append((sym, score))
                covered_factors.update(factors)
            if len(selected) >= self.top_k and not is_dominated:
                break
        return selected[:self.top_k]

    def select(self,
               ranked: List[Tuple[str, float]]) -> List[str]:
        if not ranked:
            return []

        # Apply correlation cluster constraint
        constrained = self._enforce_cluster_constraint(ranked)
        candidates = constrained[:self.top_k]

        new_symbols = [s for s, _ in candidates]
        new_scores = {s: sc for s, sc in candidates}

        if not self._last_selected:
            self._last_selected = new_symbols
            self._last_scores = new_scores
            return new_symbols

        same_as_last = (new_symbols == self._last_selected)

        if len(candidates) >= 2:
            margin = candidates[0][1] - candidates[1][1]
        else:
            margin = 1.0

        strong_dominance = margin >= self.min_margin

        if same_as_last:
            self._stable_counter += 1
            return self._last_selected

        # Different candidates — check if conditions allow rotation
        if self._stable_counter >= self.persistence and strong_dominance:
            self._last_selected = new_symbols
            self._last_scores = new_scores
            self._stable_counter = 0
            return new_symbols

        # Lock to previous
        self._stable_counter = 0
        return self._last_selected
