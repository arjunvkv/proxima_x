"""
P3.3: Unified Layer Synchronization Clock.

Every trading cycle has exactly ONE CycleClock instance.
All analytical layers consume this clock — no independent time.read() calls.
"""
import time as time_module
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CycleClock:
    cycle_id: int
    bar_time: float
    wall_time: float
    snapshot_id: str = ""
    _extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = f"snap_{self.cycle_id}_{int(self.wall_time)}"

    def elapsed(self, since: Optional[float] = None) -> float:
        """Seconds since given timestamp (or wall_time if None)."""
        ref = since if since is not None else self.wall_time
        return time_module.time() - ref

    def age_seconds(self) -> float:
        """How old this clock is."""
        return time_module.time() - self.wall_time

    def freeze(self) -> dict:
        """Serializable representation."""
        return {
            "cycle_id": self.cycle_id,
            "bar_time": self.bar_time,
            "wall_time": self.wall_time,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CycleClock":
        return cls(
            cycle_id=data["cycle_id"],
            bar_time=data["bar_time"],
            wall_time=data["wall_time"],
            snapshot_id=data.get("snapshot_id", ""),
        )

    def log_line(self) -> str:
        return (f"[CLOCK] cycle={self.cycle_id} "
                f"bar_time={int(self.bar_time)} "
                f"snapshot={self.snapshot_id}")
