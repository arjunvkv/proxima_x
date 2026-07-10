from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP


class DirectionalEfficiency:
    def __init__(self):
        self._first: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._path: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._prev_der: dict[str, float] = {}
        self._prev_pair_der: dict[str, float] = {}
        self._streak: dict[str, int] = {}
        self._peak: dict[str, float] = {}
        self._trough: dict[str, float] = {}

    def update(self, symbol: str, price: float) -> None:
        if symbol not in self._first:
            self._first[symbol] = price
            self._last[symbol] = price
            self._path[symbol] = 0.0
            self._count[symbol] = 0
        else:
            self._path[symbol] += abs(price - self._last[symbol])
            self._last[symbol] = price
            self._count[symbol] += 1

    def get_der(self, symbol: str) -> float:
        if self._count.get(symbol, 0) < 2:
            return 0.0
        net = abs(self._last.get(symbol, 0.0) - self._first.get(symbol, 0.0))
        total = self._path.get(symbol, 0.0)
        return net / total if total > 0 else 0.0

    def get_all_der(self) -> dict[str, float]:
        return {s: self.get_der(s) for s in self._first}

    def get_top_pairs(self, n: int = 3) -> list[str]:
        ders = {s: self.get_der(s) for s in self._first}
        meaningful = {s: v for s, v in ders.items() if v > 0.10}
        if not meaningful:
            return []
        sorted_pairs = sorted(meaningful.items(), key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in sorted_pairs[:n]]

    def get_currency_efficiency(self, pair_returns: dict[str, float] = None) -> dict[str, float]:
        pair_der = {}
        for sym in BASE_CURRENCY_MAP:
            pair_der[sym] = self.get_der(sym)
        pair_ranks = self._cross_sectional_ranks(pair_der)
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

    def update_persistence(self, currency_der: dict[str, float]) -> None:
        for ccy, val in currency_der.items():
            prev = self._prev_der.get(ccy, 0.0)
            self._prev_der[ccy] = val
            if val > self._peak.get(ccy, float("-inf")):
                self._peak[ccy] = val
            if val < self._trough.get(ccy, float("inf")):
                self._trough[ccy] = val
            same_dir = (val > 0 and prev > 0) or (val < 0 and prev < 0) or (val == 0 and prev == 0)
            if same_dir:
                self._streak[ccy] = self._streak.get(ccy, 0) + 1
            else:
                self._streak[ccy] = 0

    def get_persistence(self) -> dict[str, dict]:
        return {
            c: {
                "value": self._prev_der.get(c, 0.0),
                "streak": self._streak.get(c, 0),
                "peak": self._peak.get(c, 0.0),
                "trough": self._trough.get(c, 0.0),
            }
            for c in CURRENCY_LIST
        }

    def get_previous_der(self, symbol: str) -> float:
        return self._prev_pair_der.get(symbol, 0.0)

    def finalize_cycle(self) -> None:
        for sym in list(self._first.keys()):
            self._prev_pair_der[sym] = self.get_der(sym)

    def reset(self) -> None:
        self._first.clear()
        self._last.clear()
        self._path.clear()
        self._count.clear()
        self._prev_der.clear()
        self._prev_pair_der.clear()
        self._streak.clear()
        self._peak.clear()
        self._trough.clear()

    @staticmethod
    def _cross_sectional_ranks(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        sorted_items = sorted(values.items(), key=lambda x: x[1])
        n = len(sorted_items)
        return {sym: i / max(n - 1, 1) for i, (sym, _) in enumerate(sorted_items)}
