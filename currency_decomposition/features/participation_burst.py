from collections import defaultdict, deque
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP


class ParticipationBurstEngine:
    def __init__(self, history_size: int = 60, mad_epsilon: float = 1e-6):
        self.history_size = history_size
        self.mad_epsilon = mad_epsilon
        self._volumes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._prev_sign: dict[str, int] = {}
        self._streak: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def update(self, symbol: str, volume: float) -> None:
        self._volumes[symbol].append(volume)

    def get_burst(self, symbol: str) -> float:
        hist = list(self._volumes.get(symbol, []))
        if len(hist) < 10:
            return 0.0
        current = hist[-1]
        sorted_hist = sorted(hist)
        median = sorted_hist[len(sorted_hist) // 2]
        devs = sorted(abs(v - median) for v in hist)
        mad = devs[len(devs) // 2] + self.mad_epsilon
        return (current - median) / mad

    def get_paired_weights(self, symbols: list[str]) -> dict[str, float]:
        raw = {}
        for sym in symbols:
            burst = self.get_burst(sym)
            raw[sym] = burst
        if not raw:
            return {}
        ranked = self._cross_sectional_ranks(raw)
        weights = {}
        for sym in symbols:
            rank = ranked.get(sym, 0.0)
            weights[sym] = 0.5 + rank * 2.0
        return weights

    def _cross_sectional_ranks(self, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        sorted_items = sorted(values.items(), key=lambda x: x[1])
        n = len(sorted_items)
        return {sym: i / max(n - 1, 1) for i, (sym, _) in enumerate(sorted_items)}

    def _compute_currency_bursts_raw(self) -> dict[str, float]:
        pair_bursts = {}
        for sym in BASE_CURRENCY_MAP:
            pair_bursts[sym] = self.get_burst(sym)
        pair_ranks = self._cross_sectional_ranks(pair_bursts)
        currency_scores: dict[str, list[float]] = {c: [] for c in CURRENCY_LIST}
        for sym, (base, quote) in BASE_CURRENCY_MAP.items():
            rank = pair_ranks.get(sym, 0.5)
            score = (rank - 0.5) * 2.0
            currency_scores[base].append(score)
            currency_scores[quote].append(score)
        return {c: (sum(vs) / len(vs)) if vs else 0.0 for c, vs in currency_scores.items()}

    def get_currency_bursts(self) -> dict[str, float]:
        bursts = self._compute_currency_bursts_raw()
        self._update_persistence(bursts)
        return bursts

    def _update_persistence(self, bursts: dict[str, float]) -> None:
        for ccy, val in bursts.items():
            sign = 1 if val > 0 else (-1 if val < 0 else 0)
            prev = self._prev_sign.get(ccy, 0)
            if sign == 0:
                continue
            if sign == prev:
                self._streak[ccy] = self._streak.get(ccy, 0) + 1
                if val > self._peak.get(ccy, float("-inf")):
                    self._peak[ccy] = val
                if val < self._trough.get(ccy, float("inf")):
                    self._trough[ccy] = val
            else:
                self._prev_sign[ccy] = sign
                self._streak[ccy] = 1
                self._peak[ccy] = val
                self._trough[ccy] = val

    def get_persistence(self) -> dict[str, dict]:
        return {
            c: {
                "direction": self._prev_sign.get(c, 0),
                "streak": self._streak.get(c, 0),
                "peak": self._peak.get(c, 0.0),
                "trough": self._trough.get(c, 0.0),
            }
            for c in CURRENCY_LIST
        }