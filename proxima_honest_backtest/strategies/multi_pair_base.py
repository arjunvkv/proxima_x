from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import SignalResult


class MultiPairStrategy(ABC):
    name: str
    parameters: Dict[str, Any]

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        if parameters is None:
            parameters = {}
        self.parameters = dict(parameters)
        self.name = self.__class__.__name__

    @abstractmethod
    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        ...

    def reset(self) -> None:
        pass

    def validate_parameters(self) -> bool:
        return True

    def describe(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"
