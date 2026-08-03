from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from proxima_honest_backtest.engine.rolling_buffer import RollingBuffer
from proxima_honest_backtest.engine.types import PointInTime, SignalResult


class BaseStrategy(ABC):
    name: str
    parameters: Dict[str, Any]

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        if parameters is None:
            parameters = {}
        self.parameters = dict(parameters)
        self.name = self.__class__.__name__

    @abstractmethod
    def on_tick(self, tick: PointInTime, history: RollingBuffer) -> Optional[SignalResult]:
        ...

    @abstractmethod
    def on_bar(self, bar: Dict[str, Any], history: RollingBuffer) -> Optional[SignalResult]:
        ...

    def reset(self) -> None:
        pass

    def validate_parameters(self) -> bool:
        return True

    def describe(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"
