import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger("proxima_demo")

ALLOC_BANDS = {
    "MICRO": (0.0, 0.40),
    "LIGHT": (0.40, 0.75),
    "NORMAL": (0.75, 1.10),
    "HEAVY": (1.10, float("inf")),
}


class CapitalReflexEngine:
    def __init__(self):
        self._alloc_history: Dict[str, List[float]] = defaultdict(list)
        self._symbol_alloc: Dict[str, float] = {}

    def allocate(self, symbol: str, base_risk: float,
                 trust: float, pressure: float, rupture: bool) -> float:
        rupture_comp = 0.0 if rupture else 0.20
        mult = (
            trust * 0.55
            + (1.0 - pressure) * 0.30
            + rupture_comp
        )
        alloc = base_risk * max(0.10, min(1.50, mult))
        self._alloc_history[symbol].append(alloc)
        self._symbol_alloc[symbol] = alloc
        band = self._band(alloc)
        logger.info(f"[CAPITAL_REFLEX] {symbol} alloc={alloc:.3f}x "
                    f"mult={mult:.3f} band={band}")
        return alloc

    def allocation_band(self, symbol: str) -> str:
        alloc = self._symbol_alloc.get(symbol, 1.0)
        return self._band(alloc)

    def stress_cut(self, symbol: str, pressure: Optional[float] = None,
                   trust: Optional[float] = None) -> bool:
        if pressure is not None and trust is not None:
            if pressure > 0.80 and trust < 0.30:
                prev = self._symbol_alloc.get(symbol, 1.0)
                self._symbol_alloc[symbol] = 0.10
                self._alloc_history[symbol].append(0.10)
                logger.info(f"[STRESS_CUT] {symbol} pressure={pressure:.2f} "
                            f"trust={trust:.2f} prev={prev:.3f}x -> 0.10x")
                return True
        return False

    def _band(self, alloc: float) -> str:
        for name, (lo, hi) in ALLOC_BANDS.items():
            if lo <= alloc < hi:
                return name
        return "HEAVY"

    def history(self, symbol: str) -> List[float]:
        return list(self._alloc_history.get(symbol, []))

    def stats(self) -> dict:
        all_allocs = [a for v in self._alloc_history.values() for a in v]
        bands = set(self._band(a) for a in all_allocs)
        expansions = sum(1 for a in all_allocs if a > 1.0)
        compressions = sum(1 for a in all_allocs if a < 0.5)
        stress_cuts = sum(1 for a in all_allocs if a <= 0.11)
        return {
            "symbols": len(self._alloc_history),
            "allocations": len(all_allocs),
            "bands_seen": len(bands),
            "bands": sorted(bands),
            "expansions": expansions,
            "compressions": compressions,
            "stress_cuts": stress_cuts,
        }
