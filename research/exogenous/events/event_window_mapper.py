"""Maps tick timestamps to event proximity states for Program VI.5.

Proximity buckets (defined relative to event timestamp):
    PRE_30M:     event.ts - 1800s  to  event.ts - 900s   (30 to 15 min before)
    PRE_15M:     event.ts - 900s   to  event.ts - 300s   (15 to 5 min before)
    PRE_5M:      event.ts - 300s   to  event.ts          (5 to 0 min before)
    EVENT_0_2M:  event.ts          to  event.ts + 120s   (0 to 2 min after)
    POST_5M:     event.ts + 120s   to  event.ts + 300s   (2 to 5 min after)
    POST_15M:    event.ts + 300s   to  event.ts + 900s   (5 to 15 min after)
    POST_30M:    event.ts + 900s   to  event.ts + 1800s  (15 to 30 min after)
    NONE:        everything else
"""

import sys
from typing import Optional

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from research.exogenous.events.schemas import EventProximityState, MacroEvent


class EventWindowMapper:
    """Maps timestamps to event proximity buckets."""

    # Proximity bucket definitions as (label, min_offset, max_offset)
    # offset is seconds relative to event timestamp; interval is [min, max)
    _BUCKETS = [
        ("PRE_30M", -1800, -900),
        ("PRE_15M", -900, -300),
        ("PRE_5M", -300, 0),
        ("EVENT_0_2M", 0, 120),
        ("POST_5M", 120, 300),
        ("POST_15M", 300, 900),
        ("POST_30M", 900, 1800),
    ]

    def __init__(self) -> None:
        pass

    # ── public API ──────────────────────────────────────────────────────────

    def classify(
        self,
        ts: float,
        events: list[MacroEvent],
        currency: str = "USD",
    ) -> EventProximityState:
        """Classify a single timestamp against a list of macro events.

        Parameters
        ----------
        ts : float
            Timestamp in unix epoch seconds.
        events : list[MacroEvent]
            Candidate macroeconomic events.
        currency : str
            Currency of interest for match detection.

        Returns
        -------
        EventProximityState
        """
        nearest_event, distance = self.find_nearest_event(ts, events)

        if nearest_event is None:
            return EventProximityState(
                bucket="NONE",
                impact=None,
                currency_match=False,
                nearest_event_name=None,
                seconds_to_event=0.0,
            )

        bucket = self._assign_bucket(distance)
        currency_match = nearest_event.currency == currency

        return EventProximityState(
            bucket=bucket,
            impact=nearest_event.impact,
            currency_match=currency_match,
            nearest_event_name=nearest_event.name,
            seconds_to_event=distance,
        )

    def classify_batch(
        self,
        timestamps: list[float],
        events: list[MacroEvent],
        currency: str = "USD",
    ) -> list[EventProximityState]:
        """Classify multiple timestamps in batch.

        Parameters
        ----------
        timestamps : list[float]
        events : list[MacroEvent]
        currency : str

        Returns
        -------
        list[EventProximityState]
        """
        return [
            self.classify(ts=ts, events=events, currency=currency)
            for ts in timestamps
        ]

    def find_nearest_event(
        self,
        ts: float,
        events: list[MacroEvent],
        max_distance: float = 3600.0,
    ) -> tuple[Optional[MacroEvent], float]:
        """Find the nearest event within *max_distance* seconds of *ts*.

        Parameters
        ----------
        ts : float
        events : list[MacroEvent]
        max_distance : float
            Maximum absolute distance in seconds (default 3600 = 1 hour).

        Returns
        -------
        tuple[MacroEvent | None, float]
            (nearest_event, distance_in_seconds).
            Distance is signed: negative = before event, positive = after event.
            If no event is within range, returns (None, 0.0).
        """
        best_event: Optional[MacroEvent] = None
        best_distance: float = float("inf")

        for ev in events:
            dist = ts - ev.ts  # negative when ts is before event
            if abs(dist) <= max_distance and abs(dist) < abs(best_distance):
                best_event = ev
                best_distance = dist

        if best_event is None:
            return (None, 0.0)

        return (best_event, best_distance)

    # ── internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _assign_bucket(distance: float) -> str:
        """Map a signed distance (seconds) to a proximity bucket label."""
        for label, lo, hi in EventWindowMapper._BUCKETS:
            if lo <= distance < hi:
                return label
        return "NONE"
