"""Price-action confirmation for NME direction suggestions using M1 bar flow."""
from typing import Optional


class PriceActionConfirmer:
    def __init__(self, store):
        self.store = store
        self._lookback = 3

    def confirm(self, symbol: str, direction: float) -> float:
        bars = list(self.store._bars.get(symbol, []))
        n = len(bars)
        if n < 2:
            return 0.5
        lookback = min(self._lookback + 1, n)
        recent = bars[-lookback:]
        confirmations = 0
        total = 0
        for i in range(1, len(recent)):
            prev = recent[i - 1].mid
            curr = recent[i].mid
            if prev > 0 and curr > 0:
                bar_dir = 1 if curr > prev else -1
                hyp_dir = 1 if direction > 0 else -1
                if bar_dir == hyp_dir:
                    confirmations += 1
                total += 1
        if total == 0:
            return 0.5
        return confirmations / total
