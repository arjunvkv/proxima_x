"""Program VI — Exogenous Amplitude Discovery.

SessionOpenDetector — detects session open windows (first 30 minutes
of each major trading session in UTC).
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from datetime import datetime
from typing import Optional


class SessionOpenDetector:
    """Detects session open windows (first 30 minutes of each major session).

    Session open windows (UTC):
        TOKYO_OPEN:  00:00–00:30
        LONDON_OPEN: 08:00–08:30
        NY_OPEN:     13:00–13:30
    """

    # ── open-window boundaries in minutes since midnight ──────────────────
    TOKYO_OPEN_START = 0
    TOKYO_OPEN_END = 30  # 00:30

    LONDON_OPEN_START = 8 * 60       # 08:00
    LONDON_OPEN_END = 8 * 60 + 30    # 08:30

    NY_OPEN_START = 13 * 60          # 13:00
    NY_OPEN_END = 13 * 60 + 30       # 13:30

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ts_to_minutes(ts: float) -> int:
        """Convert an epoch timestamp to total minutes since midnight UTC."""
        dt = datetime.utcfromtimestamp(ts)
        return dt.hour * 60 + dt.minute

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def active(self, ts: float) -> bool:
        """Return True if *ts* falls within any session open window."""
        return self.active_session(ts) is not None

    def active_session(self, ts: float) -> Optional[str]:
        """Return the name of the active session open window, or None.

        Possible return values:
            - "TOKYO_OPEN"
            - "LONDON_OPEN"
            - "NY_OPEN"
            - None
        """
        m = self._ts_to_minutes(ts)

        if self.TOKYO_OPEN_START <= m < self.TOKYO_OPEN_END:
            return "TOKYO_OPEN"
        if self.LONDON_OPEN_START <= m < self.LONDON_OPEN_END:
            return "LONDON_OPEN"
        if self.NY_OPEN_START <= m < self.NY_OPEN_END:
            return "NY_OPEN"

        return None
