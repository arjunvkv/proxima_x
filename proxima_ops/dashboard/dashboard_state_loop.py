"""
DashboardStateLoop
==================
Periodic snapshot loop controller that feeds the dashboard.

Architecture:
    Timer ──→ UnifiedStateBuilder.build() ──→ dashboard_snapshot.json
                                        └──→ dashboard_changes_log.jsonl
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from proxima_x.proxima_ops.dashboard.unified_state_builder import (
    UnifiedStateBuilder,
)

logger = logging.getLogger("proxima_ops.dashboard.state_loop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_data_dir(data_dir: str) -> str:
    """
    Resolve *data_dir* relative to the project root.

    Three levels up from ``proxima_x/proxima_ops/dashboard/`` → project root.
    If *data_dir* is absolute, return it as-is.
    """
    if os.path.isabs(data_dir):
        return data_dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(this_dir, "..", "..", "..", data_dir)
    )


# ---------------------------------------------------------------------------
# DashboardStateLoop
# ---------------------------------------------------------------------------

class DashboardStateLoop:
    """Periodically build a unified state snapshot, detect changes, and
    persist to disk for the live dashboard to consume."""

    SNAPSHOT_FILENAME = "dashboard_snapshot.json"
    CHANGELOG_FILENAME = "dashboard_changes_log.jsonl"

    def __init__(self, interval: float = 3, data_dir: str = "state"):
        self.interval = interval
        self._resolved_dir = _resolve_data_dir(data_dir)
        self.builder = UnifiedStateBuilder(data_dir=self._resolved_dir)
        self.prev_state: dict[str, Any] = {}
        self.cycle_count = 0
        self._snapshot_count = 0
        self._start_time: Optional[float] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def run(self):
        """Main loop — call :meth:`snapshot` every *interval* seconds."""
        self._running = True
        self._start_time = time.time()
        logger.info(
            "[SNAPSHOT] DashboardStateLoop starting  interval=%ss  data_dir=%s",
            self.interval,
            self._resolved_dir,
        )
        print(
            f"[SNAPSHOT] Dashboard State Loop starting — "
            f"interval={self.interval}s  data_dir={self._resolved_dir}"
        )
        while self._running:
            try:
                self.snapshot()
            except Exception as exc:
                logger.exception("[SNAPSHOT] Unhandled error in snapshot cycle: %s", exc)
                print(f"[SNAPSHOT] ERROR: {exc}")
            # Sleep in small increments so stop() responds promptly
            for _ in range(int(self.interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

        self._shutdown()

    def snapshot(self) -> dict[str, Any]:
        """Build current state, detect changes, write to file.

        Returns the envelope dict that was written to disk.
        """
        self.cycle_count += 1
        self._snapshot_count += 1

        # ---- build ----
        state = self.builder.build()

        # ---- detect changes ----
        changes = self._detect_changes(state, self.prev_state)

        # ---- data-sources health ----
        data_sources_ok = dict(
            state.get("system_health", {}).get("data_sources", {})
        )

        # ---- build envelope ----
        envelope: dict[str, Any] = {
            "snapshot_id": self._snapshot_count,
            "cycle_count": self.cycle_count,
            "uptime_seconds": round(self.uptime_seconds, 3),
            "timestamp": _now_iso(),
            "state": state,
            "changes_since_last": changes,
            "data_sources_ok": data_sources_ok,
        }

        # ---- persist ----
        self._write_snapshot(envelope, changes)

        # ---- log changes ----
        if changes:
            for c in changes:
                section = c.get("section", "?")
                field = c.get("field", "?")
                old_val = c.get("old")
                new_val = c.get("new")
                print(
                    f"[SNAPSHOT] change  {section}.{field}: "
                    f"{_short_val(old_val)} → {_short_val(new_val)}"
                )

        # keep for next cycle
        self.prev_state = state
        return envelope

    def stop(self):
        """Request a clean shutdown.  The loop will exit after the current
        iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_changes(
        self,
        current: dict[str, Any],
        previous: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Compare *current* and *previous* unified state dicts.

        Returns a list of change dicts::

            {"field": "open_positions", "old": 1, "new": 0,
             "section": "execution_state"}
        """
        changes: list[dict[str, Any]] = []

        if not previous:
            return changes  # first cycle, nothing to diff

        # Sections whose scalar values we compare
        tracked_sections = [
            "market_state",
            "execution_state",
            "risk_state",
            "performance_state",
            "pipeline_state",
        ]

        for section in tracked_sections:
            cur_sec: dict = current.get(section, {}) or {}
            prev_sec: dict = previous.get(section, {}) or {}

            if not isinstance(cur_sec, dict) or not isinstance(prev_sec, dict):
                continue

            # Collect all field names from either side
            all_keys = set(cur_sec.keys()) | set(prev_sec.keys())

            for key in all_keys:
                cur_val = cur_sec.get(key)
                prev_val = prev_sec.get(key)

                # Skip if both are identical (including both None)
                if _values_equal(cur_val, prev_val):
                    continue

                # Skip expensive comparisons for large nested structures
                if _is_large_structure(cur_val) or _is_large_structure(prev_val):
                    continue

                changes.append(
                    {
                        "field": key,
                        "old": prev_val,
                        "new": cur_val,
                        "section": section,
                    }
                )

        return changes

    def _write_snapshot(self, envelope: dict, changes: list):
        """Write the latest snapshot and append changes to the changelog."""
        snapshot_path = os.path.join(self._resolved_dir, self.SNAPSHOT_FILENAME)
        changelog_path = os.path.join(self._resolved_dir, self.CHANGELOG_FILENAME)

        os.makedirs(self._resolved_dir, exist_ok=True)

        # ---- snapshot (overwrite) ----
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write snapshot: %s", exc)

        # ---- changelog (append) ----
        if changes:
            try:
                with open(changelog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(envelope, default=str) + "\n")
            except Exception as exc:
                logger.warning("Failed to append changelog: %s", exc)

    def _shutdown(self):
        """Log final summary."""
        duration = round(self.uptime_seconds, 3)
        msg = (
            f"[SNAPSHOT] Dashboard State Loop stopped — "
            f"cycles={self.cycle_count}  snapshots={self._snapshot_count}  "
            f"uptime={duration}s"
        )
        logger.info(msg)
        print(msg)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _values_equal(a: Any, b: Any) -> bool:
    """Return True if a and b are 'equal enough' for change detection."""
    if type(a) != type(b):
        return False
    if isinstance(a, float):
        return abs(a - b) < 1e-9 if isinstance(b, float) else False
    return a == b


def _is_large_structure(val: Any) -> bool:
    """Return True if *val* is a list/dict with more than a few elements,
    indicating we should skip detailed comparison."""
    if isinstance(val, (list, tuple)):
        return len(val) > 5
    if isinstance(val, dict):
        return len(val) > 10
    return False


def _short_val(val: Any) -> str:
    """Render a value for log output, truncating long strings/lists."""
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, str):
        return val if len(val) < 80 else val[:77] + "..."
    if isinstance(val, (list, tuple)):
        return f"list[{len(val)}]"
    if isinstance(val, dict):
        return f"dict[{len(val)}]"
    return str(val)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    loop = DashboardStateLoop(interval=3)

    def _signal_handler(signum, frame):
        print(f"\n[SNAPSHOT] Signal {signum} received — shutting down...")
        loop.stop()

    import signal

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("[SNAPSHOT] Dashboard State Loop starting...")
    loop.run()
