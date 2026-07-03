"""Program VI — Exogenous Amplitude Discovery.

RolloverDetector — detects the daily rollover window (21:55–22:10 UTC).
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from datetime import datetime


class RolloverDetector:
    """Detects the daily rollover window (21:55–22:10 UTC).

    During rollover, FX spot settlement rolls to the next value date,
    which can produce transient microstructure effects.
    """

    # ── rollover boundaries in minutes since midnight ─────────────────────
    ROLLOVER_START = 21 * 60 + 55   # 21:55
    ROLLOVER_END = 22 * 60 + 10     # 22:10

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
        """Return True if *ts* falls within the daily rollover window."""
        m = self._ts_to_minutes(ts)
        return self.ROLLOVER_START <= m < self.ROLLOVER_END

    def time_to_rollover(self, ts: float) -> float:
        """Return the number of seconds until the next rollover window.

        Returns 0.0 if *ts* is already inside the rollover window.
        """
        m = self._ts_to_minutes(ts)

        # Already in rollover
        if self.ROLLOVER_START <= m < self.ROLLOVER_END:
            return 0.0

        # Rollover occurs later today
        if m < self.ROLLOVER_START:
            return float((self.ROLLOVER_START - m) * 60)

        # Rollover occurs tomorrow (past rollover_end for today)
        seconds_remaining_today = float((24 * 60 - m) * 60)
        seconds_tomorrow = float(self.ROLLOVER_START * 60)
        return seconds_remaining_today + seconds_tomorrow
