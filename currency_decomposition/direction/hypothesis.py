import numpy as np
import time
from typing import Optional
from config.settings import SYMBOLS, BASE_CURRENCY_MAP, MIN_CONFIDENCE, MIN_SPREAD
from data.models import DirectionHypothesis


class HypothesisGenerator:
    def __init__(self):
        self._last_spreads: dict[str, float] = {}

    def generate(self, graph, symbol: str, timestamp: Optional[float] = None) -> Optional[DirectionHypothesis]:
        ts = timestamp or time.time()
        base, quote = BASE_CURRENCY_MAP[symbol]
        base_strength = graph.strength(base)
        quote_strength = graph.strength(quote)
        residual = graph.residual(symbol)
        spread = base_strength - quote_strength

        if abs(spread) < MIN_SPREAD:
            return None

        direction = 1.0 if spread >= 0 else -1.0

        _SIGN_EPS = 1e-8
        if direction > 0:
            if base_strength < _SIGN_EPS or quote_strength > -_SIGN_EPS:
                return None
        else:
            if base_strength > -_SIGN_EPS or quote_strength < _SIGN_EPS:
                return None

        spread_mag = abs(spread) + 1e-10
        raw_residual = abs(residual)
        residual_ratio = min(raw_residual / spread_mag, 1.0)
        signal_conf = max(0.0, 1.0 - residual_ratio)

        confidence = max(0.0, min(1.0, signal_conf))

        if confidence < MIN_CONFIDENCE:
            return None

        return DirectionHypothesis(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            base_strength=base_strength,
            quote_strength=quote_strength,
            residual=residual,
            timestamp=ts
        )

    def generate_all(self, graph, timestamp: Optional[float] = None) -> list[DirectionHypothesis]:
        hypotheses = []
        for symbol in SYMBOLS:
            h = self.generate(graph, symbol, timestamp)
            if h is not None:
                hypotheses.append(h)
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
