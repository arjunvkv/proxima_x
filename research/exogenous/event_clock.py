"""
EventClock — maps timestamps into exogenous time windows for Program VI.

Session definitions (UTC):
    TOKYO:     00:00–06:00
    LONDON:    07:00–15:00   (overrides NEWYORK during 12:00–15:00 overlap)
    NEWYORK:   12:00–21:00   (but 15:00–16:00 is deadzone by definition)
    DEADZONE:  06:00–07:00, 15:00–16:00, 21:00–00:00

Fixing windows (UTC):
    TOKYO_FIX  00:55–01:05
    WM_FIX     15:55–16:05   (London WM/Reuters 4 pm fix)
    NY_CUT     14:55–15:05   (NY option cut)

Rollover regime:
    21:55–22:10 UTC daily
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import datetime
from typing import Optional

try:
    import pytz
    _HAVE_PYTZ = True
except ImportError:
    _HAVE_PYTZ = False
    try:
        import zoneinfo  # Python 3.9+
        _HAVE_ZONEINFO = True
    except ImportError:
        _HAVE_ZONEINFO = False


class EventClock:
    """Maps epoch timestamps into exogenous time windows for Program VI."""

    # ── session boundaries in minutes since midnight ──────────────────────
    # TOKYO
    TOKYO_START = 0
    TOKYO_END = 6 * 60  # 06:00
    # DEADZONE 06:00–07:00
    DZ1_START = 6 * 60
    DZ1_END = 7 * 60
    # LONDON
    LONDON_START = 7 * 60
    LONDON_END = 15 * 60  # 15:00
    # DEADZONE 15:00–16:00
    DZ2_START = 15 * 60
    DZ2_END = 16 * 60
    # NEWYORK
    NY_START = 16 * 60  # effectively 12:00, but 12–15 handled by LONDON
    NY_END = 21 * 60  # 21:00
    # DEADZONE 21:00–24:00
    DZ3_START = 21 * 60
    DZ3_END = 24 * 60

    # ── fixing windows ────────────────────────────────────────────────────
    FIX_WINDOWS: dict[str, tuple[int, int]] = {
        "TOKYO_FIX": (0 * 60 + 55, 1 * 60 + 5),  # 00:55 – 01:05
        "WM_FIX": (15 * 60 + 55, 16 * 60 + 5),  # 15:55 – 16:05
        "NY_CUT": (14 * 60 + 55, 15 * 60 + 5),  # 14:55 – 15:05
    }

    # ── rollover ──────────────────────────────────────────────────────────
    ROLLOVER_START = 21 * 60 + 55  # 21:55
    ROLLOVER_END = 22 * 60 + 10  # 22:10

    # ── human-readable session names ──────────────────────────────────────
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEWYORK = "NEWYORK"
    DEADZONE = "DEADZONE"

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ts_to_minutes(ts: float) -> int:
        """Convert an epoch timestamp to total minutes since midnight UTC."""
        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.hour * 60 + dt.minute

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_session(self, ts: float) -> str:
        """Map an epoch timestamp *ts* to the active session name.

        Overlap resolution:
            - 12:00–15:00  → LONDON (LONDON has priority over NEWYORK)
        """
        m = self._ts_to_minutes(ts)

        if self.TOKYO_START <= m < self.TOKYO_END:
            return self.TOKYO
        if self.DZ1_START <= m < self.DZ1_END:
            return self.DEADZONE
        if self.LONDON_START <= m < self.LONDON_END:
            return self.LONDON  # covers 12:00–15:00 overlap w/ NY
        if self.DZ2_START <= m < self.DZ2_END:
            return self.DEADZONE
        if self.NY_START <= m < self.NY_END:
            return self.NEWYORK
        # 21:00–24:00 (and 24:00 is 0)
        return self.DEADZONE

    def in_fix(self, ts: float) -> Optional[str]:
        """Return the fixing-window name if *ts* falls inside one, else None."""
        m = self._ts_to_minutes(ts)

        for name, (start, end) in self.FIX_WINDOWS.items():
            if start <= m < end:
                return name
        return None

    def in_rollover(self, ts: float) -> bool:
        """Return True if *ts* is inside the daily rollover window."""
        m = self._ts_to_minutes(ts)
        return self.ROLLOVER_START <= m < self.ROLLOVER_END

    def describe(self, ts: float) -> dict:
        """Return a dict with *session*, *fix*, and *rollover* for *ts*."""
        return {
            "timestamp": ts,
            "session": self.get_session(ts),
            "fix": self.in_fix(ts),
            "rollover": self.in_rollover(ts),
        }
