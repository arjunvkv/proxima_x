"""
Ultra Monster — STRICT ANTI-LOOKAHEAD adapter.

This is the honest, provable port of the rolling-ORB logic that the real
`mt5-connector/ultra_monster` implementation performs, expressed under the
honest framework's strict contract (see `validation/masked_replay.py`):

    at bar `ts` the strategy may act ONLY on:
      * `history` — closes of bars STRICTLY before `ts` (served by the engine),
      * the current bar's `open` (the bar's start).

The adapter uses ONLY `history[pair]` for its signal:
  * range = max close - min close over the `range_bars` most-recent COMPLETED
    bars (all strictly before `ts`),
  * signal = breakout of the LAST completed close above/below that range,
  * entry at the current bar's `open` (carried in signal metadata as
    `entry_price`), exit after `hold_bars` completed bars.

It NEVER reads the forming bar's close/high/low, so a masked-replay probe
(full vs NaN close/high/low) yields IDENTICAL trades -> PASS. This is the
strict-conformant target: the plain-bar `evaluate` that read the forming
bar's close and filled at the same bar's open is deliberately NOT reproduced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy


ALL_PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD",
]


class UltraMonsterHonestStrategy(MultiPairStrategy):
    """Strict-honest rolling close-range breakout, 9-pair, :00/:30 minute gate."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "range_bars": 12,
        "min_range_pips": 6.0,
        "hold_bars": 3,
        "minute_gate": (0, 30),
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._positions.clear()

    def describe(self) -> str:
        p = self.parameters
        return (
            f"UltraMonsterHonest(range={p['range_bars']}, hold={p['hold_bars']}, "
            f"min={p['min_range_pips']})"
        )

    @staticmethod
    def _pip_size(pair: str) -> float:
        return 0.01 if "JPY" in pair else 0.0001

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        ts = next((b["time"] for _, b in bars.items() if b), None)
        if ts is None:
            return signals

        minute = getattr(ts, "minute", 0)
        gate = tuple(self.parameters.get("minute_gate", (0, 30)))
        if minute not in gate:
            return signals

        p = self.parameters
        range_bars = int(p["range_bars"])
        min_pips = float(p["min_range_pips"])
        hold = int(p["hold_bars"])

        for sym, pos in list(self._positions.items()):
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= hold:
                del self._positions[sym]
                signals.append(SignalResult(
                    timestamp=ts, signal=0.0, confidence=1.0,
                    metadata={
                        "strategy": self.name, "pair": sym, "action": "EXIT",
                        "reason": "hold_expired",
                    },
                ))

        for sym in self.parameters["pairs"]:
            if sym in self._positions:
                continue
            closes = history.get(sym, [])
            if len(closes) < range_bars + 1:
                continue
            window = closes[-(range_bars + 1):-1]
            if not window:
                continue
            range_hi = max(window)
            range_lo = min(window)
            pip = self._pip_size(sym)
            if (range_hi - range_lo) / pip < min_pips:
                continue

            prev_close = closes[-1]
            if prev_close is None or (isinstance(prev_close, float) and prev_close != prev_close):
                continue

            if prev_close > range_hi:
                direction = "LONG"
            elif prev_close < range_lo:
                direction = "SHORT"
            else:
                continue

            entry_price = float(bars[sym]["open"])
            self._positions[sym] = {
                "direction": 1.0 if direction == "LONG" else -1.0,
                "bars_held": 0,
            }
            signals.append(SignalResult(
                timestamp=ts,
                signal=1.0 if direction == "LONG" else -1.0,
                confidence=0.95,
                metadata={
                    "strategy": self.name,
                    "pair": sym,
                    "action": f"ENTER_{direction}",
                    "entry_price": entry_price,
                },
            ))

        return signals
