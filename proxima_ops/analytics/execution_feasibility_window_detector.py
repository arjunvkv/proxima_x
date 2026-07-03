"""
execution_feasibility_window_detector.py

Find WHEN the system is actually allowed to trade under current governance.
Reads wave12_cycle_log.jsonl, groups consecutive cycles by segl_state,
and reports active (ARMED) vs blocked (OBSERVE) windows.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionFeasibilityWindowDetector:
    """Detects time windows when trading is permitted under current governance."""

    # Governance states that permit execution
    ACTIVE_STATES = {"ARM", "ARMED"}

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """
        Scan the cycle log and produce a feasibility report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many most-recent cycles from the log.

        Returns
        -------
        dict
            {
                "active_windows": [{"start_cycle", "end_cycle", "duration", "trades_attempted"}, ...],
                "blocked_windows": [{"start_cycle", "end_cycle", "duration", "signals_blocked"}, ...],
                "next_feasible_trade_time_estimate": str,
                "total_active_cycles": int,
                "total_blocked_cycles": int,
                "active_ratio": float
            }
        """
        try:
            cycles = self._load_cycles(n_recent_cycles)
            if not cycles:
                return self._empty_result("No cycle data available")

            windows = self._group_consecutive_windows(cycles)

            active_windows = []
            blocked_windows = []
            total_active = 0
            total_blocked = 0

            for w in windows:
                entry = {
                    "start_cycle": w["start_cycle"],
                    "end_cycle": w["end_cycle"],
                    "duration": w["end_cycle"] - w["start_cycle"] + 1,
                }
                if w["state"] in self.ACTIVE_STATES:
                    entry["trades_attempted"] = self._count_trades_attempted(w["cycles"])
                    active_windows.append(entry)
                    total_active += entry["duration"]
                else:
                    entry["signals_blocked"] = self._count_signals_blocked(w["cycles"])
                    blocked_windows.append(entry)
                    total_blocked += entry["duration"]

            total_cycles = total_active + total_blocked
            active_ratio = round(total_active / total_cycles, 4) if total_cycles > 0 else 0.0

            next_estimate = self._estimate_next_feasible_time(cycles, windows)

            return {
                "active_windows": active_windows,
                "blocked_windows": blocked_windows,
                "next_feasible_trade_time_estimate": next_estimate,
                "total_active_cycles": total_active,
                "total_blocked_cycles": total_blocked,
                "active_ratio": active_ratio,
            }

        except Exception:
            logger.exception("ExecutionFeasibilityWindowDetector.analyze failed")
            return self._empty_result("Analysis error")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cycles(self, n_recent: int) -> list[dict[str, Any]]:
        """Read the JSONL log file and return the last *n_recent* cycles."""
        cycles: list[dict[str, Any]] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        cycles.append(json.loads(line))
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except json.JSONDecodeError as exc:
            logger.warning("JSON decode error in %s: %s", self.log_path, exc)
            return []

        # Return the most recent *n_recent* entries
        return cycles[-n_recent:] if len(cycles) > n_recent else cycles

    def _group_consecutive_windows(
        self, cycles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Group consecutive cycles by their *segl_state*.

        Returns a list of dicts:
            {"state": str, "start_cycle": int, "end_cycle": int, "cycles": [ ... ]}
        """
        windows: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_state: str | None = None

        for c in cycles:
            state = c.get("segl_state", "UNKNOWN")
            if state != current_state:
                if current:
                    windows.append(self._build_window(current, current_state))
                current = [c]
                current_state = state
            else:
                current.append(c)

        if current and current_state is not None:
            windows.append(self._build_window(current, current_state))

        return windows

    @staticmethod
    def _build_window(
        cycles: list[dict[str, Any]], state: str
    ) -> dict[str, Any]:
        return {
            "state": state,
            "start_cycle": cycles[0]["cycle"],
            "end_cycle": cycles[-1]["cycle"],
            "cycles": cycles,
        }

    @staticmethod
    def _count_trades_attempted(cycles: list[dict[str, Any]]) -> int:
        """Count cycles in this window where decision != 'HOLD'."""
        return sum(1 for c in cycles if c.get("decision", "HOLD") != "HOLD")

    @staticmethod
    def _count_signals_blocked(cycles: list[dict[str, Any]]) -> int:
        """Count OBSERVE cycles where total_signals > 0 (signals exist but are blocked)."""
        return sum(1 for c in cycles if c.get("total_signals", 0) > 0)

    def _estimate_next_feasible_time(
        self,
        all_cycles: list[dict[str, Any]],
        windows: list[dict[str, Any]],
    ) -> str:
        """
        Estimate when the next feasible trading window will open.

        Heuristic:
          - If the current/last window is ARMED, trading is already feasible → "NOW".
          - If there is a known OBSERVE → ARMED pattern, compute the average
            duration of OBSERVE windows and estimate based on that average.
          - Otherwise fall back to "UNKNOWN".
        """
        if not windows:
            return "UNKNOWN"

        last_window = windows[-1]["state"]
        if last_window in self.ACTIVE_STATES:
            return "NOW"

        # Collect durations of OBSERVE windows that were followed by ARMED
        observe_durations: list[int] = []
        for i, w in enumerate(windows):
            if w["state"] not in self.ACTIVE_STATES:
                if i + 1 < len(windows) and windows[i + 1]["state"] in self.ACTIVE_STATES:
                    observe_durations.append(w["end_cycle"] - w["start_cycle"] + 1)

        if not observe_durations:
            # No historical pattern → cannot estimate
            return "UNKNOWN"

        avg_observe_duration = int(sum(observe_durations) / len(observe_durations))

        # How long has the current OBSERVE window been running?
        current_window = windows[-1]
        elapsed = (current_window["end_cycle"] - current_window["start_cycle"]) + 1

        remaining = avg_observe_duration - elapsed
        if remaining < 0:
            remaining = 0

        # Try to produce a human-readable estimate.
        # We approximate based on typical cycle duration if available.
        cycle_times = [
            c.get("cycle_duration", 0) for c in all_cycles if c.get("cycle_duration")
        ]
        avg_cycle_sec = (
            sum(cycle_times) / len(cycle_times) if cycle_times else 30.0
        )

        remaining_seconds = remaining * avg_cycle_sec
        eta = datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)

        if remaining <= 0:
            return "IMMINENT"
        return eta.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "active_windows": [],
            "blocked_windows": [],
            "next_feasible_trade_time_estimate": reason,
            "total_active_cycles": 0,
            "total_blocked_cycles": 0,
            "active_ratio": 0.0,
        }
