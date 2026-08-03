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
        self._history: dict[str, list[float]] = {}
        self._ret15: dict[str, list[float]] = {}
        self._entry_idx: dict[str, int] = {}
        self._reset_state()

    def _reset_state(self):
        for p in self.pairs:
            self._history[p] = []
            self._ret15[p] = []
        self._entry_idx = {}

    def reset(self):
        super().reset()
        self._reset_state()

    def on_bars(self, context, history=None) -> list[SignalResult]:
        signals = []

        for pair in self.pairs:
            if pair not in context:
                continue
            close = context[pair]["close"]

            self._history[pair].append(float(close))

            hist = self._history[pair]
            if len(hist) >= 4:
                ret = (hist[-1] - hist[-4]) / hist[-4]
                self._ret15[pair].append(ret)
            else:
                self._ret15[pair].append(0.0)

            if len(self._ret15[pair]) > self.window + 10:
                self._ret15[pair] = self._ret15[pair][-(self.window + 10):]
            if len(self._history[pair]) > self.window + 10:
                self._history[pair] = self._history[pair][-(self.window + 10):]

            # Check exit
            if pair in self._entry_idx:
                bars_held = len(hist) - self._entry_idx[pair]
                if bars_held >= self.hold_bars:
                    del self._entry_idx[pair]

            # Check entry
            if len(self._ret15[pair]) < self.window:
                continue
            if pair not in context:
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

            if z <= -self.z_threshold:
                entry_price = float(close)
                self._entry_idx[pair] = len(hist) - 1
                ts_bar = context[pair].get("time", context[pair].get("timestamp"))
                signals.append(SignalResult(
                    timestamp=ts_bar,
                    signal=1.0,
                    confidence=abs(z) / 10.0,
                    metadata={
                        "pair": pair,
                        "entry_price": entry_price,
                        "z_score": z,
                        "hold_bars": self.hold_bars,
                    }
                ))

        return signals
