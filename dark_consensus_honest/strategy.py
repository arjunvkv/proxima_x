from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.base import BaseStrategy


def _to_datetime(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()


class DarkConsensusStrategy(BaseStrategy):
    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ["EURJPY", "EURUSD", "GBPJPY"],
        "mag_threshold": 0.00018741,
        "hold_bars": 3,
        "session_start": 7,
        "session_end": 21,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._position: Optional[Dict[str, Any]] = None

    @property
    def in_position(self) -> bool:
        return self._position is not None

    def on_tick(self, tick, history):
        return None

    def on_bar(self, bar, history):
        return None

    def on_bars(
        self,
        bars_dict: Dict[str, float],
        prev_returns: Dict[str, float],
        timestamp: datetime,
    ) -> Optional[SignalResult]:
        pairs = self.parameters["pairs"]

        ts_dt = _to_datetime(timestamp)
        hour_utc = ts_dt.hour
        session_start = int(self.parameters["session_start"])
        session_end = int(self.parameters["session_end"])
        if not (session_start <= hour_utc < session_end):
            return None

        signs = [np.sign(prev_returns[p]) for p in pairs]
        if 0.0 in signs or len(set(signs)) != 1:
            return None

        avg_abs = float(np.mean([abs(prev_returns[p]) for p in pairs]))
        threshold = float(self.parameters["mag_threshold"])
        if avg_abs <= threshold:
            return None

        best_pair = max(pairs, key=lambda p: abs(prev_returns[p]))
        direction = 1.0 if prev_returns[best_pair] > 0 else -1.0

        confidence = min(0.99, avg_abs / threshold * 0.5)

        self._position = {
            "pair": best_pair,
            "direction": direction,
            "entry_time": ts_dt,
            "bars_held": 0,
            "entry_price": bars_dict[best_pair],
        }

        return SignalResult(
            timestamp=ts_dt,
            signal=direction,
            confidence=confidence,
            metadata={
                "strategy": self.name,
                "pair": best_pair,
                "direction": "LONG" if direction > 0 else "SHORT",
                "avg_abs_return": avg_abs,
                "returns": {p: float(prev_returns[p]) for p in pairs},
            },
        )

    def check_exit(self, timestamp: datetime) -> Optional[SignalResult]:
        if self._position is None:
            return None
        self._position["bars_held"] += 1
        if self._position["bars_held"] >= int(self.parameters["hold_bars"]):
            pos = self._position
            self._position = None
            return SignalResult(
                timestamp=timestamp,
                signal=0.0,
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "pair": pos["pair"],
                    "direction": "EXIT",
                    "prev_side": "LONG" if pos["direction"] > 0 else "SHORT",
                    "bars_held": pos["bars_held"],
                },
            )
        return None

    def reset(self) -> None:
        self._position = None

    def validate_parameters(self) -> bool:
        pairs = self.parameters.get("pairs", [])
        if not isinstance(pairs, list) or len(pairs) < 2:
            return False
        threshold = self.parameters.get("mag_threshold", 0)
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return False
        hold = self.parameters.get("hold_bars", 0)
        if not isinstance(hold, int) or hold < 1:
            return False
        return True

    def describe(self) -> str:
        params = self.parameters
        return (
            f"DarkConsensus(pairs={params['pairs']}, "
            f"threshold={params['mag_threshold']:.6f}, "
            f"hold={params['hold_bars']} bars, "
            f"session={params['session_start']}-{params['session_end']} UTC)"
        )
