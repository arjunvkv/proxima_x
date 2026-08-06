"""Causal market-state builder — identical history construction for backtest and live.

The core anti-lookahead invariant lives here:
  - at bar ts, `history[p]` contains ONLY closes of bars STRICTLY BEFORE ts
  - the current bar's close is appended AFTER the strategy has decided on it
  - a pair missing at ts is simply ABSENT from `bars`/`history` (never NaN/0.0)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class MarketStateBuilder:
    """Maintains per-pair closes and serves causal history to the strategy."""

    def __init__(self) -> None:
        self._closes: Dict[str, List[float]] = defaultdict(list)

    def reset(self) -> None:
        self._closes.clear()

    def history(self) -> Dict[str, List[float]]:
        """Closes strictly before the current bar (pairs present so far)."""
        return {p: closes[:] for p, closes in self._closes.items()}

    def snapshot(self) -> Dict[str, List[float]]:
        return self.history()

    def append_bar(self, bars: Dict[str, Dict]) -> None:
        """Append closes AFTER the strategy has decided (must be called last)."""
        for pair, bar in bars.items():
            close = bar.get("close")
            if close is not None and not (isinstance(close, float) and close != close):
                self._closes[pair].append(float(close))

    @property
    def closes(self) -> Dict[str, List[float]]:
        return dict(self._closes)

    def pair_close_count(self, pair: str) -> int:
        return len(self._closes.get(pair, []))
