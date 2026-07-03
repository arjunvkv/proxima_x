"""Program VI — Exogenous Amplitude Discovery.

FixingWindowDetector — detects fixing windows and option cutoff periods.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from datetime import datetime
from typing import Optional


class FixingWindowDetector:
    """Detects fixing windows and option cutoff periods.

    Fix windows (UTC):
        TOKYO_FIX:       00:55–01:05
        NY_CUT:          14:55–15:05
        WM_FIX:          15:55–16:05
        OPTIONS_CUTOFF:  09:55–10:05  (Options expiry/roll)
    """

    # ── fix-window boundaries in minutes since midnight ───────────────────
    FIX_WINDOWS: dict[str, tuple[int, int]] = {
        "TOKYO_FIX": (0 * 60 + 55, 1 * 60 + 5),    # 00:55 – 01:05
        "NY_CUT":    (14 * 60 + 55, 15 * 60 + 5),   # 14:55 – 15:05
        "WM_FIX":    (15 * 60 + 55, 16 * 60 + 5),   # 15:55 – 16:05
    }

    OPTIONS_CUTOFF_START = 9 * 60 + 55   # 09:55
    OPTIONS_CUTOFF_END = 10 * 60 + 5     # 10:05

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
        """Return True if *ts* falls within any fix or option-cut window."""
        return self.active_fix(ts) is not None

    def active_fix(self, ts: float) -> Optional[str]:
        """Return the name of the active fix window, or None.

        Possible return values:
            - "TOKYO_FIX"
            - "NY_CUT"
            - "WM_FIX"
            - "OPTIONS_CUTOFF"
            - None
        """
        m = self._ts_to_minutes(ts)

        # Check named fixing windows
        for name, (start, end) in self.FIX_WINDOWS.items():
            if start <= m < end:
                return name

        # Check options cutoff window
        if self.OPTIONS_CUTOFF_START <= m < self.OPTIONS_CUTOFF_END:
            return "OPTIONS_CUTOFF"

        return None
