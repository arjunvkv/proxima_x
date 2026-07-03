from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class MechanismScore:
    name: str
    category: str = ""
    information_gain: float = 0.0
    sid: float = 0.0
    sir: float = 0.0
    persistence: float = 0.0
    robustness: float = 0.0
    cross_asset_score: float = 0.0
    cross_regime_score: float = 0.0
    oos_score: float = 0.0
    simplicity: float = 0.0
    novelty: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        return (
            self.information_gain * 0.20
            + self.sid * 0.15
            + self.sir * 0.10
            + self.persistence * 0.10
            + self.robustness * 0.10
            + self.cross_asset_score * 0.10
            + self.cross_regime_score * 0.10
            + self.oos_score * 0.05
            + self.simplicity * 0.05
            + self.novelty * 0.05
        )

    @property
    def survives(self) -> bool:
        return self.information_gain > 0 and self.sid > 0 and self.sir > 0


class BaseMechanism(ABC):
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self._state: dict[str, Any] = {}

    @abstractmethod
    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_state_contribution(self) -> NDArray:
        ...

    def score(self) -> MechanismScore:
        return MechanismScore(name=self.name, category=self.category)

    def validate(self, data: dict[str, NDArray], states: NDArray) -> dict[str, bool]:
        return {"detected": True}

    def reset(self) -> None:
        self._state.clear()
