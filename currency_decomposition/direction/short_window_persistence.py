from typing import Dict, List, Tuple
import math


_EPSILON = 1e-12


class ShortWindowPersistenceScanner:
    """
    Stateless burst-only signal extractor.

    Takes N consecutive WLS strength snapshots (N >= 5)
    and ranks all currency pairs by within-window directional persistence.

    No memory across runs — each burst is independent.
    """

    MIN_WINDOW = 5

    def __init__(
        self,
        currencies: List[str],
        pairs: Dict[str, Tuple[str, str]]
    ):
        self.currencies = currencies
        self.pairs = pairs

    def rank(
        self,
        snapshots: List[dict]
    ) -> List[tuple]:
        """
        Input: list of {ccy: strength} dicts, in chronological order.
        Output: [(symbol, direction, score), ...] sorted descending by score.
        """
        if len(snapshots) < self.MIN_WINDOW:
            return []

        results = []

        for symbol, (base, quote) in self.pairs.items():
            spreads = []
            for snap in snapshots:
                value = snap.get(base, 0.0) - snap.get(quote, 0.0)
                spreads.append(value)

            score = self._score(spreads)

            last = spreads[-1]
            if abs(last) < _EPSILON:
                direction = "NONE"
            else:
                direction = "BUY" if last > 0 else "SELL"

            if direction != "NONE":
                results.append((symbol, direction, round(score, 4)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def top_n(
        self,
        snapshots: List[dict],
        n: int = 3
    ) -> List[tuple]:
        ranked = self.rank(snapshots)
        return ranked[:n]

    def _score(self, spreads: List[float]) -> float:
        if not spreads:
            return 0.0

        directions = [1 if x > _EPSILON else -1 if x < -_EPSILON else 0 for x in spreads]

        active = [d for d in directions if d != 0]
        pos = active.count(1)
        neg = active.count(-1)

        # No directional information at all
        if not active:
            return 0.0

        # 1. Direction persistence (neutral excluded)
        dominant = max(pos, neg)
        persistence = dominant / len(active)

        # 2. Flip stability (neutral compressed out)
        if len(active) <= 1:
            flip_stability = 0.0
        else:
            flips = sum(
                1 for i in range(1, len(active))
                if active[i] != active[i - 1]
            )
            flip_stability = 1.0 - (flips / (len(active) - 1))

        # 3. Magnitude quality (signal-to-noise with variance floor)
        mean_abs = sum(abs(x) for x in spreads) / len(spreads)
        variance = sum((x - mean_abs) ** 2 for x in spreads) / len(spreads)
        std = math.sqrt(variance)
        noise_floor = max(std, mean_abs * 0.05)
        magnitude = min(mean_abs / (noise_floor + _EPSILON) / 10.0, 1.0)

        # 4. Recent confirmation
        recent = abs(spreads[-1])
        recent_confirmation = min(recent / (mean_abs + _EPSILON), 1.0)

        # 5. Trend slope (start-to-end delta)
        delta = spreads[-1] - spreads[0]
        slope = min(abs(delta) / (mean_abs + _EPSILON), 1.0)

        score = (
            0.35 * persistence
            + 0.25 * flip_stability
            + 0.15 * magnitude
            + 0.10 * recent_confirmation
            + 0.15 * slope
        )

        return max(0.0, min(score, 1.0))
