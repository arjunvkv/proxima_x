"""
EXCEPTION DASHBOARD — Phase 7 Deliverable

Tracks runtime exceptions by type, count, last seen, and status.
"""

import traceback
import time
from collections import defaultdict


class ExceptionDashboard:
    def __init__(self, max_history: int = 100):
        self._exceptions = defaultdict(lambda: {"count": 0, "last_seen": 0, "status": "ACTIVE", "messages": []})
        self._max_history = max_history

    def record(self, exception: Exception, context: str = ""):
        ex_type = type(exception).__name__
        now = time.time()
        entry = self._exceptions[ex_type]
        entry["count"] += 1
        entry["last_seen"] = now
        entry["status"] = "ACTIVE"
        entry["messages"].append({
            "time": now,
            "message": str(exception)[:200],
            "context": context,
        })
        if len(entry["messages"]) > self._max_history:
            entry["messages"] = entry["messages"][-self._max_history:]

    def resolve(self, ex_type: str):
        if ex_type in self._exceptions:
            self._exceptions[ex_type]["status"] = "RESOLVED"

    def summary(self) -> str:
        lines = []
        lines.append("EXCEPTION DASHBOARD")
        lines.append("=" * 60)
        if not self._exceptions:
            lines.append("  No exceptions recorded.")
            return "\n".join(lines)
        lines.append(f"{'Type':<30} {'Count':<8} {'Last Seen':<20} {'Status':<12}")
        lines.append("-" * 70)
        now = time.time()
        for ex_type, data in sorted(self._exceptions.items(), key=lambda x: -x[1]["count"]):
            last_seen = f"{now - data['last_seen']:.0f}s ago" if data["last_seen"] > 0 else "NEVER"
            lines.append(f"{ex_type:<30} {data['count']:<8} {last_seen:<20} {data['status']:<12}")
        return "\n".join(lines)

    def has_active(self) -> bool:
        return any(e["status"] == "ACTIVE" and e["count"] > 0 for e in self._exceptions.values())


def demo():
    ed = ExceptionDashboard()
    try:
        1 / 0
    except Exception as e:
        ed.record(e, "test_division")
    try:
        {}["missing"]
    except Exception as e:
        ed.record(e, "test_key")
    print(ed.summary())


if __name__ == "__main__":
    demo()
