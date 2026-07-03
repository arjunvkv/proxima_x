"""
SYMBOL UNIVERSE SELECTOR — Combines SymbolDiscovery + SIL scoring to produce
the final active trading universe.

Flow:
  1. Discover candidate symbols via SymbolDiscovery
  2. Score candidates via SymbolIntelligenceLayer.score_symbols()
  3. Filter by minimum score threshold
  4. Anchor symbols are guaranteed inclusion
  5. Remaining slots filled by top-scoring non-anchor symbols
"""

import logging
from datetime import datetime
from typing import List, Optional

from proxima_x.proxima_ops.intelligence.symbol_intelligence_layer import SymbolIntelligenceLayer
from proxima_x.proxima_ops.intelligence.symbol_discovery import SymbolDiscovery

logger = logging.getLogger("proxima_ops.intelligence.universe_selector")

ANCHOR_SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]
MAX_UNIVERSE_SIZE = 15
MIN_SCORE_THRESHOLD = 0.45


class SymbolUniverseSelector:
    """Selects and maintains the active trading universe.

    Combines symbol discovery with tradability scoring to produce a
    ranked, bounded universe with guaranteed anchor symbols.
    """

    def __init__(
        self,
        sil: Optional[SymbolIntelligenceLayer] = None,
        discovery: Optional[SymbolDiscovery] = None,
        max_size: int = MAX_UNIVERSE_SIZE,
        min_threshold: float = MIN_SCORE_THRESHOLD,
    ):
        self.sil = sil or SymbolIntelligenceLayer()
        self.discovery = discovery or SymbolDiscovery()
        self.max_size = max_size
        self.min_threshold = min_threshold
        self._active_universe = list(ANCHOR_SYMBOLS)  # default fallback
        self._last_update = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_universe(self) -> List[str]:
        """Run discovery + scoring to produce the active trading universe.

        Returns:
            List of symbol names in the active universe.

        Guarantees:
            - ANCHOR_SYMBOLS are always included.
            - Non-anchor symbols are ranked by total_score descending.
            - Total symbols <= max_size.
            - Never raises.
        """
        try:
            return self._select_universe_impl()
        except Exception:
            logger.exception(
                "[UNIVERSE] Unexpected failure in select_universe — "
                "falling back to anchor symbols"
            )
            self._active_universe = list(ANCHOR_SYMBOLS)
            return self._active_universe

    def get_active_universe(self) -> List[str]:
        """Return the current active universe without re-selecting."""
        return list(self._active_universe)

    def get_full_report(self) -> dict:
        """Return a detailed report of the last universe selection."""
        now = datetime.utcnow().isoformat()
        return {
            "timestamp": now,
            "universe_size": len(self._active_universe),
            "max_size": self.max_size,
            "min_threshold": self.min_threshold,
            "active_universe": list(self._active_universe),
            "candidates_discovered": len(getattr(self, "_last_candidates", [])),
            "scored_symbols": list(getattr(self, "_last_scored", [])),
            "anchors_guaranteed": list(ANCHOR_SYMBOLS),
        }

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _select_universe_impl(self) -> List[str]:
        """Core selection logic — split out so the public wrapper can
        catch any unexpected exception."""
        import time
        start = time.time()

        # ----- Step 1: Discover candidates -----
        try:
            candidates = self.discovery.discover()
        except Exception:
            logger.warning(
                "[UNIVERSE] SymbolDiscovery.discover() failed — "
                "falling back to anchor symbols",
                exc_info=True,
            )
            self._active_universe = list(ANCHOR_SYMBOLS)
            self._last_update = time.time()
            self._last_candidates = []
            self._last_scored = []
            return self._active_universe

        if not candidates:
            logger.warning(
                "[UNIVERSE] Discovery returned no candidates — "
                "falling back to anchor symbols"
            )
            self._active_universe = list(ANCHOR_SYMBOLS)
            self._last_update = time.time()
            self._last_candidates = []
            self._last_scored = []
            return self._active_universe

        candidate_names = [c["symbol"] for c in candidates]
        logger.info(
            "[UNIVERSE] Discovered %d candidates: %s",
            len(candidate_names),
            ", ".join(candidate_names[:10]) + ("..." if len(candidate_names) > 10 else ""),
        )

        # ----- Step 2: Score candidates -----
        try:
            scores = self.sil.score_symbols(candidate_names)
        except Exception:
            logger.warning(
                "[UNIVERSE] SIL score_symbols() failed — "
                "falling back to anchor symbols",
                exc_info=True,
            )
            self._active_universe = list(ANCHOR_SYMBOLS)
            self._last_update = time.time()
            self._last_candidates = candidate_names
            self._last_scored = []
            return self._active_universe

        if not scores:
            logger.warning(
                "[UNIVERSE] SIL returned no scores — "
                "falling back to anchor symbols"
            )
            self._active_universe = list(ANCHOR_SYMBOLS)
            self._last_update = time.time()
            self._last_candidates = candidate_names
            self._last_scored = []
            return self._active_universe

        # ----- Step 3: Apply threshold filter -----
        total_before = len(scores)
        qualified = [s for s in scores if s.get("total_score", 0) >= self.min_threshold]
        filtered_out = total_before - len(qualified)
        logger.info(
            "[UNIVERSE] Threshold filtered out %d symbols (below %.2f)",
            filtered_out,
            self.min_threshold,
        )

        # Sort qualified by total_score descending (they come pre-sorted from SIL,
        # but sort again to be safe)
        qualified.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        # ----- Step 4: Build final universe -----
        anchor_set = set(ANCHOR_SYMBOLS)
        universe = list(ANCHOR_SYMBOLS)  # guaranteed anchors
        remaining_slots = self.max_size - len(universe)

        top_non_anchor = []
        for s in qualified:
            if remaining_slots <= 0:
                break
            sym = s.get("symbol", "")
            if sym not in anchor_set:
                universe.append(sym)
                top_non_anchor.append(s)
                remaining_slots -= 1

        # Deduplicate (shouldn't be necessary but guard against edge cases)
        seen = set()
        deduped = []
        for sym in universe:
            if sym not in seen:
                seen.add(sym)
                deduped.append(sym)
        universe = deduped

        # ----- Logging -----
        logger.info("[UNIVERSE] Anchor symbols: %s", ", ".join(ANCHOR_SYMBOLS))
        if top_non_anchor:
            top_str = ", ".join(
                f"{s.get('symbol', '?')}({s.get('total_score', 0):.2f})"
                for s in top_non_anchor
            )
            logger.info("[UNIVERSE] Top non-anchor: %s", top_str)
        logger.info(
            "[UNIVERSE] Selected %d symbols: %s",
            len(universe),
            ", ".join(universe),
        )

        # ----- Persist state -----
        self._active_universe = universe
        self._last_update = time.time()
        self._last_candidates = candidate_names
        self._last_scored = scores

        elapsed = time.time() - start
        logger.info("[UNIVERSE] Selection completed in %.3fs", elapsed)

        return universe


if __name__ == "__main__":
    import MetaTrader5 as mt5

    mt5.initialize()
    selector = SymbolUniverseSelector()
    universe = selector.select_universe()
    print(f"Active universe ({len(universe)}): {universe}")
    report = selector.get_full_report()
    for s in sorted(report["scored_symbols"], key=lambda x: x["total_score"], reverse=True):
        print(f"  {s['symbol']:15s} score={s['total_score']:.3f}")
    mt5.shutdown()
