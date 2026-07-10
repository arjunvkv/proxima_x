from collections import defaultdict, deque
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP


class ParticipationBurstEngine:
    MIN_HISTORY = 10

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
        self._neutral_gap: dict[str, int] = {}

    def update(self, symbol: str, volume: float) -> None:
        self._volumes[symbol].append(volume)

    def get_burst(self, symbol: str) -> float:
        hist = list(self._volumes.get(symbol, []))
        if len(hist) < self.MIN_HISTORY:
            return 0.0
        current = hist[-1]
        sorted_hist = sorted(hist)
        median = sorted_hist[len(sorted_hist) // 2]
        devs = sorted(abs(v - median) for v in hist)
        mad = devs[len(devs) // 2] + self.mad_epsilon
        return (current - median) / mad

    def get_quality(self, symbol: str) -> dict:
        hist = list(self._volumes.get(symbol, []))
        n = len(hist)
        if n < self.MIN_HISTORY:
            return {"samples": n, "status": "cold", "needed": self.MIN_HISTORY - n}
        return {"samples": n, "status": "normal" if n >= self.history_size else "warming"}

    def _cross_sectional_ranks(self, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        sorted_items = sorted(values.items(), key=lambda x: x[1])
        n = len(sorted_items)
        return {sym: i / max(n - 1, 1) for i, (sym, _) in enumerate(sorted_items)}

    def _compute_currency_bursts_raw(self, pair_returns: dict[str, float] = None) -> dict[str, float]:
        pair_bursts = {}
        for sym in BASE_CURRENCY_MAP:
            pair_bursts[sym] = self.get_burst(sym)
        pair_ranks = self._cross_sectional_ranks(pair_bursts)
        currency_scores: dict[str, list[float]] = {c: [] for c in CURRENCY_LIST}
        for sym, (base, quote) in BASE_CURRENCY_MAP.items():
            rank = pair_ranks.get(sym, 0.5)
            score = (rank - 0.5) * 2.0
            if pair_returns:
                ret = pair_returns.get(sym, 0.0)
                direction = 1 if ret > 0 else (-1 if ret < 0 else 0)
                base_score = score * direction
                quote_score = -score * direction
                currency_scores[base].append(base_score)
                currency_scores[quote].append(quote_score)
            else:
                currency_scores[base].append(score)
                currency_scores[quote].append(score)
        return {c: (sum(vs) / len(vs)) if vs else 0.0 for c, vs in currency_scores.items()}

    def get_currency_bursts(self, pair_returns: dict[str, float] = None) -> dict[str, float]:
        bursts = self._compute_currency_bursts_raw(pair_returns)
        self._update_persistence(bursts)
        return bursts

    def _update_persistence(self, bursts: dict[str, float]) -> None:
        for ccy, val in bursts.items():
            sign = 1 if val > 0 else (-1 if val < 0 else 0)
            prev = self._prev_sign.get(ccy, 0)
            if sign == 0:
                if prev != 0:
                    self._neutral_gap[ccy] = self._neutral_gap.get(ccy, 0) + 1
                self._streak[ccy] = 0
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
                self._neutral_gap[ccy] = 0

    def get_top_burst_pairs(self, n: int = 3) -> list[str]:
        bursts = {}
        for sym in BASE_CURRENCY_MAP:
            hist = self._volumes.get(sym, [])
            bursts[sym] = abs(self.get_burst(sym)) if len(hist) >= 10 else 0.0
        sorted_pairs = sorted(bursts.items(), key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in sorted_pairs[:n]]

    def get_persistence(self) -> dict[str, dict]:
        return {
            c: {
                "direction": self._prev_sign.get(c, 0),
                "streak": self._streak.get(c, 0),
                "peak": self._peak.get(c, 0.0),
                "trough": self._trough.get(c, 0.0),
                "neutral_gap": self._neutral_gap.get(c, 0),
            }
            for c in CURRENCY_LIST
        }

    def reset(self) -> None:
        self._volumes.clear()
        self._prev_sign.clear()
        self._streak.clear()
        self._peak.clear()
        self._trough.clear()
        self._neutral_gap.clear()

    def get_state(self) -> str:
        """'cold' = most pairs below MIN_HISTORY, 'warming' = accumulating, 'active' = ready.
        Only symbols that have ever produced data are counted (excludes unavailable pairs)."""
        cold = 0
        warming = 0
        for sym in list(self._volumes.keys()):
            q = self.get_quality(sym)
            if q["status"] == "cold":
                cold += 1
            elif q["status"] == "warming":
                warming += 1
        total = cold + warming
        if total == 0:
            return "cold"
        if cold > total // 2:
            return "cold"
        if warming > total // 2:
            return "warming"
        return "active"
    def get_burst_alignment(self, symbol: str, pair_return: float) -> float:
        burst = self.get_burst(symbol)
        if burst == 0.0 or pair_return == 0.0:
            return 0.0
        direction = 1 if pair_return > 0 else -1
        return burst * direction