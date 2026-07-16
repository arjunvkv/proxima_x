import numpy as np
import time
from typing import Optional
from config.settings import SYMBOLS, BASE_CURRENCY_MAP, MIN_CONFIDENCE, MIN_SPREAD
from data.models import DirectionHypothesis


class HypothesisGenerator:
    def __init__(self):
        self._last_spreads: dict[str, float] = {}
        self._spread_history: dict[str, list[float]] = {}

    def generate(self, graph, symbol: str, timestamp: Optional[float] = None) -> Optional[DirectionHypothesis]:
        ts = timestamp or time.time()
        base, quote = BASE_CURRENCY_MAP[symbol]
        base_strength = graph.strength(base)
        quote_strength = graph.strength(quote)
        residual = graph.residual(symbol)
        graph_quality = graph.quality()
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

        raw_residual = abs(residual)
        spread_mag = abs(spread) + 1e-10
        residual_ratio = min(raw_residual / spread_mag, 1.0) if spread_mag > 1e-8 else 0.5
        signal_conf = 1.0 - residual_ratio

        stability = self._compute_stability(symbol, spread)
        recency = min(len(self._spread_history.get(symbol, [])), 60) / 60.0

        graph_obs = graph.state.observability
        base_obs = graph_obs.get(base, 0.5)
        quote_obs = graph_obs.get(quote, 0.5)
        observability_factor = (base_obs + quote_obs) / 2.0
        strength_stab = graph.strength_stability()
        base_stab = strength_stab.get(base, 0.5)
        quote_stab = strength_stab.get(quote, 0.5)
        stability_factor = (base_stab + quote_stab) / 2.0
        confidence = float(signal_conf * graph_quality * stability * observability_factor * stability_factor * (0.5 + 0.5 * recency))
        confidence = max(0.0, min(1.0, confidence))

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

    def _compute_stability(self, symbol: str, spread: float) -> float:
        if symbol not in self._spread_history:
            self._spread_history[symbol] = []
        self._spread_history[symbol].append(spread)
        if len(self._spread_history[symbol]) > 60:
            self._spread_history[symbol] = self._spread_history[symbol][-60:]

        hist = self._spread_history[symbol]
        if len(hist) < 5:
            return 0.6
        recent = hist[-5:]
        flip_count = sum(1 for i in range(1, len(recent)) if recent[i] * recent[i-1] < 0)
        consistency = 1.0 - min(flip_count / 4.0, 1.0)
        return float(max(0.6, consistency))
