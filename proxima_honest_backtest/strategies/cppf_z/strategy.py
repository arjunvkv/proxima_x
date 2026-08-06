"""CPPF Z≥6.0 — Cross-Pair Volatility Dislocation Strategy.

LONG-only fade of extreme 15-min drops (z-score ≤ -threshold).
Uses rolling 200-bar window for z-score estimation.
"""
from typing import Optional
import numpy as np
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

CROSS_PAIRS = ["EURAUD", "GBPAUD"]


class CPPFZStrategy(MultiPairStrategy):
    def __init__(
        self,
        pairs: Optional[list[str]] = None,
        z_threshold: float = 6.0,
        hold_bars: int = 18,
        window: int = 200,
    ):
        super().__init__({"z_threshold": z_threshold, "hold_bars": hold_bars, "window": window})
        self.pairs = list(CROSS_PAIRS)
        self.z_threshold = z_threshold
        self.hold_bars = hold_bars
        self.window = window
        self._ret15: dict[str, list[float]] = {}
        self._hist_seen: dict[str, int] = {}
        self._entry_idx: dict[str, int] = {}
        self._bar_count: dict[str, int] = {}
        self._reset_state()

    def _reset_state(self):
        for p in self.pairs:
            self._ret15[p] = []
            self._hist_seen[p] = 0
            self._bar_count[p] = 0
        self._entry_idx = {}

    def reset(self):
        super().reset()
        self._reset_state()

    def on_bars(self, context, history=None) -> list[SignalResult]:
        """Honest contract: signal from `history` (closes BEFORE current bar),
        entry at the current bar's OPEN via `entry_price` metadata.
        """
        history = history or {}
        signals = []

        for pair in self.pairs:
            if pair not in context:
                continue
            ts_bar = context[pair].get("time", context[pair].get("timestamp"))
            closes = list(history.get(pair, []))

            self._bar_count[pair] += 1

            # Extend the 15-min return series as completed bars arrive
            # (history grows by exactly one close per aligned bar).
            if len(closes) >= 4 and len(closes) > self._hist_seen.get(pair, 0):
                ret = (closes[-1] - closes[-3]) / closes[-3]
                self._ret15[pair].append(ret)
                self._hist_seen[pair] = len(closes)

            if len(self._ret15[pair]) > self.window + 10:
                self._ret15[pair] = self._ret15[pair][-(self.window + 10):]

            # Check exit (hold expiry)
            if pair in self._entry_idx:
                bars_held = self._bar_count[pair] - self._entry_idx[pair]
                if bars_held >= self.hold_bars:
                    del self._entry_idx[pair]
                    signals.append(SignalResult(
                        timestamp=ts_bar,
                        signal=0.0,
                        confidence=1.0,
                        metadata={
                            "strategy": self.name,
                            "pair": pair,
                            "action": "EXIT",
                            "reason": "hold_expired",
                        }
                    ))

            # Check entry
            if len(self._ret15[pair]) < self.window:
                continue

            data = np.array(self._ret15[pair][-self.window:])
            ret_current = data[-1]
            hist_ret = data[:-1]

            if len(hist_ret) < 60:
                continue

            mu = float(hist_ret.mean())
            sd = float(hist_ret.std())
            if sd == 0:
                continue

            z = (ret_current - mu) / sd

            if z <= -self.z_threshold and pair not in self._entry_idx:
                entry_price = float(context[pair]["open"])
                self._entry_idx[pair] = self._bar_count[pair]
                signals.append(SignalResult(
                    timestamp=ts_bar,
                    signal=1.0,
                    confidence=abs(z) / 10.0,
                    metadata={
                        "strategy": self.name,
                        "pair": pair,
                        "action": "ENTER_LONG",
                        "entry_price": entry_price,
                        "z_score": z,
                        "hold_bars": self.hold_bars,
                    }
                ))

        return signals
