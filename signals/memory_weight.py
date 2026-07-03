import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger("proxima_demo")


class MemoryWeightEngine:
    def __init__(self):
        self._weights: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, symbol: str, pressure: float,
               fracture: float, thesis_rf_prob: float) -> float:
        weight = (
            thesis_rf_prob * 0.50
            + (1.0 - pressure) * 0.30
            + (1.0 - fracture) * 0.20
        )
        weight = max(0.0, min(1.0, weight))
        self._weights[symbol] = weight
        self._history[symbol].append(weight)
        band = self._band(weight)
        logger.info(f"[MEMORY_WEIGHT] {symbol} W={weight:.3f} "
                    f"band={band} n={len(self._history[symbol])}")
        return weight

    def trust(self, symbol: str) -> float:
        return self._weights.get(symbol, 0.5)

    def drift(self, symbol: str) -> Optional[float]:
        history = self._history.get(symbol, [])
        if len(history) < 2:
            return 0.0
        drift_val = history[-1] - history[-2]
        logger.info(f"[MEMORY_DRIFT] {symbol} drift={drift_val:+.3f}")
        return drift_val

    def history(self, symbol: str) -> List[float]:
        return list(self._history.get(symbol, []))

    def _band(self, weight: float) -> str:
        if weight >= 0.70:
            return "high"
        elif weight >= 0.40:
            return "mid"
        return "low"

    def trusted_symbols(self, min_weight: float = 0.70) -> List[str]:
        return [s for s, w in self._weights.items() if w >= min_weight]

    def stats(self) -> dict:
        all_weights = [w for v in self._history.values() for w in v]
        bands = set(self._band(w) for w in all_weights)
        decays = 0
        recoveries = 0
        for sym, hist in self._history.items():
            for i in range(1, len(hist)):
                if hist[i] < hist[i - 1]:
                    decays += 1
                elif hist[i] > hist[i - 1]:
                    recoveries += 1
        return {
            "symbols": len(self._history),
            "total_updates": len(all_weights),
            "bands_seen": len(bands),
            "bands": sorted(bands),
            "trust_decays": decays,
            "trust_recoveries": recoveries,
        }
