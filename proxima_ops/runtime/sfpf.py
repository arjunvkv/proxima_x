"""
SFPF — State Freezing Problem Fix

Persist execution state across ticks. Maintain SES + ECL continuity across event stream.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


class StateFreezingFix:
    """Persist sovereignty execution state across restarts and event ticks."""

    def __init__(self, state_path: str = "state/sovereignty_state.json"):
        self.state_path = state_path

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------
    def persist(
        self,
        ses_state: dict,
        ecl_state: dict,
        efk_state: dict,
    ) -> bool:
        """Serialize ses_state + ecl_state + efk_state to state_path as JSON.

        Returns True on success, False on any failure.
        """
        try:
            payload = {
                "ses_state": ses_state,
                "ecl_state": ecl_state,
                "efk_state": efk_state,
            }

            dirname = os.path.dirname(self.state_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            return True

        except (OSError, IOError, TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self) -> dict:
        """Read state_path and return the three sub-dicts.

        Returns
        -------
        dict
            {
                "ses_state":  dict | None,
                "ecl_state":  dict | None,
                "efk_state":  dict | None,
                "restored":   bool,
                "restored_at": str | None    # ISO-8601 timestamp
            }
        """
        result: dict = {
            "ses_state": None,
            "ecl_state": None,
            "efk_state": None,
            "restored": False,
            "restored_at": None,
        }

        try:
            if not os.path.isfile(self.state_path):
                return result

            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            result["ses_state"] = payload.get("ses_state")
            result["ecl_state"] = payload.get("ecl_state")
            result["efk_state"] = payload.get("efk_state")
            result["restored"] = True
            result["restored_at"] = datetime.now(timezone.utc).isoformat()

            return result

        except (OSError, IOError, json.JSONDecodeError, TypeError, ValueError):
            return result

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Delete the state file.

        Returns
        -------
        dict
            {"reset": bool, "cleared_path": str}
        """
        try:
            if os.path.isfile(self.state_path):
                os.remove(self.state_path)
                return {"reset": True, "cleared_path": self.state_path}

            return {"reset": True, "cleared_path": self.state_path}

        except (OSError, IOError, PermissionError):
            return {"reset": False, "cleared_path": self.state_path}
