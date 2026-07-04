from __future__ import annotations

import json
import os
import time
import numpy as np
from typing import Any


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Phase6AuditLogger:
    def __init__(self, log_path: str | None = None) -> None:
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "phase6_audit_log.jsonl",
            )
        self._log_path = log_path
        self._entries: list[dict] = []

    def log(self, event: str, state: str, metrics: dict, details: dict | None = None) -> dict:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int(time.time() * 1e6) % 1000000:06d}Z"
        entry = {
            "timestamp": ts,
            "event": event,
            "state": state,
            "metrics_snapshot": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        if details:
            entry["details"] = details
        self._entries.append(entry)
        self._flush(entry)
        return entry

    def log_transition(self, from_state: str, to_state: str, metrics: dict, reason: str) -> dict:
        return self.log("ROLLOUT_TRANSITION", to_state, metrics, {
            "from_state": from_state,
            "reason": reason,
        })

    def log_kill_switch(self, metrics: dict, reason: str) -> dict:
        return self.log("KILL_SWITCH", "SHADOW", metrics, {"reason": reason})

    def log_scale_change(self, state: str, old_mult: float, new_mult: float, metrics: dict) -> dict:
        return self.log("SCALE_CHANGE", state, metrics, {
            "old_multiplier": old_mult,
            "new_multiplier": new_mult,
        })

    def log_recovery(self, phase: str, metrics: dict) -> dict:
        return self.log("RECOVERY", phase, metrics)

    def _flush(self, entry: dict) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, cls=_SafeEncoder) + "\n")
        except Exception:
            pass

    def get_entries(self, event_type: str | None = None) -> list[dict]:
        if event_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.get("event") == event_type]

    def get_summary(self) -> dict:
        transitions = self.get_entries("ROLLOUT_TRANSITION")
        kill_switches = self.get_entries("KILL_SWITCH")
        scale_changes = self.get_entries("SCALE_CHANGE")
        recoveries = self.get_entries("RECOVERY")
        return {
            "total_entries": len(self._entries),
            "transitions": len(transitions),
            "kill_switches": len(kill_switches),
            "scale_changes": len(scale_changes),
            "recoveries": len(recoveries),
            "latest_transition": transitions[-1] if transitions else None,
            "latest_kill_switch": kill_switches[-1] if kill_switches else None,
        }
