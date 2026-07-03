from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import settings
from core.state_vector import StateVector, normalize_state


@dataclass
class StateTransition:
    from_state: NDArray[np.float32]
    to_state: NDArray[np.float32]
    timestamp: int
    delta: float
    duration: int


@dataclass
class StateEngine:
    max_history: int = 100_000
    _states: List[NDArray[np.float32]] = field(default_factory=list)
    _timestamps: List[int] = field(default_factory=list)
    _transitions: deque[StateTransition] = field(default_factory=lambda: deque(maxlen=10_000))
    _current_state: Optional[NDArray[np.float32]] = None

    def update(self, state_vector: StateVector | NDArray[np.float32], timestamp: int) -> StateTransition | None:
        if isinstance(state_vector, StateVector):
            state = state_vector.as_array
        else:
            state = state_vector

        state = normalize_state(state)

        transition = None
        if self._current_state is not None:
            delta = float(np.sqrt(np.sum((state - self._current_state) ** 2)))
            duration = timestamp - (self._timestamps[-1] if self._timestamps else 0)
            transition = StateTransition(
                from_state=self._current_state,
                to_state=state,
                timestamp=timestamp,
                delta=delta,
                duration=duration,
            )
            self._transitions.append(transition)

        self._current_state = state
        self._states.append(state)
        self._timestamps.append(timestamp)

        if len(self._states) > self.max_history:
            self._states.pop(0)
            self._timestamps.pop(0)

        return transition

    @property
    def current_state(self) -> Optional[NDArray[np.float32]]:
        return self._current_state

    @property
    def recent_transitions(self) -> List[StateTransition]:
        return list(self._transitions)

    @property
    def transition_rate(self) -> float:
        if len(self._transitions) < 2:
            return 0.0
        deltas = [t.delta for t in self._transitions]
        return float(np.mean(deltas[-100:]))

    @property
    def state_stability(self) -> float:
        if len(self._states) < 10:
            return 1.0
        recent = np.array(self._states[-10:])
        mean_state = np.mean(recent, axis=0)
        std = float(np.mean(np.std(recent, axis=0)))
        return 1.0 / (1.0 + std)

    def clear(self) -> None:
        self._states.clear()
        self._timestamps.clear()
        self._transitions.clear()
        self._current_state = None
