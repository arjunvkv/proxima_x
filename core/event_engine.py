from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    STATE_CHANGE = auto()
    PRESSURE_BUILD = auto()
    PRESSURE_RELEASE = auto()
    MEMORY_FORMATION = auto()
    MEMBER_DECAY = auto()
    ECHO_TRIGGERED = auto()
    TENSION_SPIKE = auto()
    REGIME_SHIFT = auto()
    LIQUIDITY_MIGRATION = auto()
    COHORT_ALIGNMENT = auto()
    COHORT_CONFLICT = auto()


@dataclass
class Event:
    event_type: EventType
    timestamp: int
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0

    def __lt__(self, other: Event) -> bool:
        return self.timestamp < other.timestamp


EventHandler = Callable[[Event], None]


@dataclass
class EventEngine:
    _handlers: Dict[EventType, List[EventHandler]] = field(default_factory=dict)
    _history: List[Event] = field(default_factory=list)
    _max_history: int = 100_000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h is not handler]

    def emit(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for handler in self._handlers.get(event.event_type, []):
            handler(event)

    def emit_batch(self, events: List[Event]) -> None:
        for event in sorted(events):
            self.emit(event)

    @property
    def recent_events(self, n: int = 100) -> List[Event]:
        return self._history[-n:]

    def events_of_type(self, event_type: EventType) -> List[Event]:
        return [e for e in self._history if e.event_type == event_type]

    def clear(self) -> None:
        self._history.clear()
