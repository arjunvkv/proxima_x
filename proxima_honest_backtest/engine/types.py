from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, KeysView, ValuesView, ItemsView


@dataclass(frozen=True, order=True)
class PointInTime:
    timestamp: datetime
    price: float
    volume: float
    symbol: str


@dataclass(frozen=True, order=True)
class SignalResult:
    timestamp: datetime
    signal: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class Trade:
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float = 0.0
    pnl: float = 0.0


@dataclass(frozen=True, order=True)
class ExecutionReport:
    trade: Trade
    broker_profile: str
    fill_price: float
    slippage: float
    latency_ms: float
    filled: bool = True
    reject_reason: str = ""


class ReadOnlyView:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("ReadOnlyView does not support item assignment")

    def __delitem__(self, key: str) -> None:
        raise TypeError("ReadOnlyView does not support item deletion")

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ReadOnlyView({self._data})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[Any]:
        return self._data.values()

    def items(self) -> ItemsView[str, Any]:
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)
