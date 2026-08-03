from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from proxima_honest_backtest.engine.rolling_buffer import RollingBuffer
from proxima_honest_backtest.engine.types import PointInTime, SignalResult
from proxima_honest_backtest.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    DEFAULT_PARAMS: Dict[str, Any] = {
        "lookback": 20,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "min_spread": 0.0001,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._position_side: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._last_bar_time: Optional[datetime] = None

    def on_tick(self, tick: PointInTime, history: RollingBuffer) -> Optional[SignalResult]:
        if len(history) < self.parameters["lookback"]:
            return None

        if self._last_bar_time is not None and tick.timestamp <= self._last_bar_time:
            return None

        self._last_bar_time = tick.timestamp

        return self._compute_signal(tick.price, tick.timestamp, history)

    def on_bar(self, bar: Dict[str, Any], history: RollingBuffer) -> Optional[SignalResult]:
        if len(history) < self.parameters["lookback"]:
            return None

        close = bar.get("close")
        if close is None:
            return None

        timestamp = bar.get("timestamp")
        if isinstance(timestamp, datetime):
            ts = timestamp
        elif isinstance(timestamp, (int, float)):
            ts = datetime.fromtimestamp(timestamp)
        else:
            ts = datetime.now()

        return self._compute_signal(float(close), ts, history)

    def _compute_signal(
        self, price: float, timestamp: datetime, history: RollingBuffer
    ) -> Optional[SignalResult]:
        lookback = int(self.parameters["lookback"])
        entry_z = float(self.parameters["entry_z"])
        exit_z = float(self.parameters["exit_z"])

        try:
            prices = list(history.get_column("close"))[-lookback:]
        except (KeyError, IndexError, TypeError):
            try:
                col_data = history.get_column(history.columns[0])
                prices = list(col_data)[-lookback:]
            except Exception:
                return None

        if len(prices) < lookback:
            return None

        mean = np.mean(prices)
        std = np.std(prices)
        if std == 0:
            return None

        z_score = (price - mean) / std

        spread = float(self.parameters.get("min_spread", 0.0001))

        if self._position_side is None:
            if z_score > entry_z:
                confidence = min(abs(z_score) / 5.0, 1.0)
                self._position_side = "SHORT"
                self._entry_price = price
                return SignalResult(
                    timestamp=timestamp,
                    signal=-1.0,
                    confidence=confidence,
                    metadata={
                        "strategy": self.name,
                        "z_score": z_score,
                        "side": "SHORT",
                        "spread": spread,
                    },
                )
            elif z_score < -entry_z:
                confidence = min(abs(z_score) / 5.0, 1.0)
                self._position_side = "LONG"
                self._entry_price = price
                return SignalResult(
                    timestamp=timestamp,
                    signal=1.0,
                    confidence=confidence,
                    metadata={
                        "strategy": self.name,
                        "z_score": z_score,
                        "side": "LONG",
                        "spread": spread,
                    },
                )
        else:
            if abs(z_score) < exit_z:
                side = self._position_side
                self._position_side = None
                self._entry_price = None
                return SignalResult(
                    timestamp=timestamp,
                    signal=0.0,
                    confidence=1.0,
                    metadata={
                        "strategy": self.name,
                        "z_score": z_score,
                        "side": "EXIT",
                        "prev_side": side,
                        "spread": spread,
                    },
                )

        return None

    def reset(self) -> None:
        self._position_side = None
        self._entry_price = None
        self._last_bar_time = None

    def validate_parameters(self) -> bool:
        lookback = self.parameters.get("lookback", 20)
        entry_z = self.parameters.get("entry_z", 2.0)
        exit_z = self.parameters.get("exit_z", 0.5)
        min_spread = self.parameters.get("min_spread", 0.0001)

        if not isinstance(lookback, (int, float)) or lookback < 1:
            return False
        if not isinstance(entry_z, (int, float)) or entry_z <= 0:
            return False
        if not isinstance(exit_z, (int, float)) or exit_z <= 0:
            return False
        if exit_z >= entry_z:
            return False
        if not isinstance(min_spread, (int, float)) or min_spread <= 0:
            return False
        return True

    def describe(self) -> str:
        return (
            f"MeanReversion(lookback={self.parameters['lookback']}, "
            f"entry_z={self.parameters['entry_z']}, "
            f"exit_z={self.parameters['exit_z']})"
        )
