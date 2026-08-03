from typing import Any, Dict, Optional

from proxima_honest_backtest.engine.rolling_buffer import RollingBuffer
from proxima_honest_backtest.engine.types import PointInTime, SignalResult
from proxima_honest_backtest.strategies.base import BaseStrategy


class V2zStrategy(BaseStrategy):
    """Z-score mean reversion with trailing stop (V2+z inspired).

    Default params mirror the validated CPPF config:
      lookback=50, z_entry=3.5, z_exit=1.0,
      stop_a=3.0, trig_a=1.0, gap_a=0.05,
      direction='BOTH' (or 'LONG'/'SHORT')
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "lookback": 50,
        "z_entry": 3.5,
        "z_exit": 1.0,
        "stop_a": 3.0,
        "trig_a": 1.0,
        "gap_a": 0.05,
        "direction": "BOTH",
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._position = 0       # -1 short, 0 flat, +1 long
        self._entry_price = 0.0
        self._trailing_hi = 0.0
        self._trailing_lo = 0.0
        self._bars_since_entry = 0

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0
        self._entry_price = 0.0
        self._trailing_hi = 0.0
        self._trailing_lo = 0.0
        self._bars_since_entry = 0

    def on_tick(self, tick: PointInTime, history: RollingBuffer) -> Optional[SignalResult]:
        return None

    def on_bar(self, bar: Dict[str, Any], history: RollingBuffer) -> Optional[SignalResult]:
        lookback = self.parameters["lookback"]
        if len(history) < lookback + 1:
            return None

        closes = history.get_column("close")
        if len(closes) < lookback + 1:
            return None

        past = closes[-(lookback + 1):-1]
        current = closes[-1]

        mean = sum(past) / len(past)
        variance = sum((p - mean) ** 2 for p in past) / len(past)
        std = variance ** 0.5
        if std < 1e-12:
            return None

        z = (current - mean) / std
        z_entry = self.parameters["z_entry"]
        z_exit = self.parameters["z_exit"]
        direction = self.parameters["direction"]

        self._bars_since_entry += 1

        if self._position == 0:
            return self._check_entry(z, z_entry, direction, bar["time"], current)
        elif self._position == 1:
            return self._trail_long(z, z_exit, bar["time"], current)
        else:
            return self._trail_short(z, z_exit, bar["time"], current)

    def _check_entry(self, z: float, z_entry: float, direction: str,
                     ts, price: float) -> Optional[SignalResult]:
        if z > z_entry and direction in ("BOTH", "SHORT"):
            self._position = -1
            self._entry_price = price
            self._trailing_hi = price
            self._bars_since_entry = 0
            return SignalResult(ts, -1.0, min(abs(z) / 5.0, 1.0),
                                {"action": "ENTER_SHORT", "z": z})
        if z < -z_entry and direction in ("BOTH", "LONG"):
            self._position = 1
            self._entry_price = price
            self._trailing_lo = price
            self._bars_since_entry = 0
            return SignalResult(ts, 1.0, min(abs(z) / 5.0, 1.0),
                                {"action": "ENTER_LONG", "z": z})
        return None

    def _trail_short(self, z: float, z_exit: float, ts, price: float) -> Optional[SignalResult]:
        self._trailing_hi = max(self._trailing_hi, price)
        stop_price = self._trailing_hi - self.parameters["stop_a"] * (price * self.parameters.get("gap_a", 0.05))
        trig_price = self._trailing_hi - self.parameters["trig_a"] * (price * self.parameters.get("gap_a", 0.05))

        if price <= stop_price or abs(z) < z_exit:
            self._position = 0
            return SignalResult(ts, 1.0, 0.95,
                                {"action": "EXIT_SHORT", "reason": "stop_or_revert", "z": z})
        return None

    def _trail_long(self, z: float, z_exit: float, ts, price: float) -> Optional[SignalResult]:
        self._trailing_lo = min(self._trailing_lo, price)
        stop_price = self._trailing_lo + self.parameters["stop_a"] * (price * self.parameters.get("gap_a", 0.05))
        trig_price = self._trailing_lo + self.parameters["trig_a"] * (price * self.parameters.get("gap_a", 0.05))

        if price >= stop_price or abs(z) < z_exit:
            self._position = 0
            return SignalResult(ts, -1.0, 0.95,
                                {"action": "EXIT_LONG", "reason": "stop_or_revert", "z": z})
        return None
