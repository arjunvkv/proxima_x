"""ParityLogger — live-vs-replay realism scoring."""
import json
import os
from typing import Dict, List, Optional


class ParityLogger:
    def __init__(self):
        self._live: Dict[str, dict] = {}
        self._replay: Dict[str, dict] = {}

    def record_live(self, event_id: str, data: dict):
        self._live[event_id] = data

    def record_replay(self, event_id: str, data: dict):
        self._replay[event_id] = data

    def signal_parity(self) -> float:
        matched = total = 0
        for eid, ld in self._live.items():
            rd = self._replay.get(eid)
            if rd is None:
                continue
            total += 1
            if ld.get("oss_signal") == rd.get("oss_signal"):
                matched += 1
        return matched / total if total else 0.0

    def state_parity(self) -> float:
        total = drift = 0
        for eid, ld in self._live.items():
            rd = self._replay.get(eid)
            if rd is None:
                continue
            total += 1
            lb = ld.get("ecdf_bucket", 0)
            rb = rd.get("ecdf_bucket", 0)
            drift += abs(lb - rb)
        avg_drift = drift / total if total else 7.0
        return max(0.0, 1.0 - avg_drift / 7.0)

    def execution_parity(self) -> float:
        live_lat = [d.get("latency_ms", 0) for d in self._live.values()]
        replay_lat = [d.get("latency_ms", 0) for d in self._replay.values()]
        if not live_lat or not replay_lat:
            return 0.0
        mean_l = sum(live_lat) / len(live_lat)
        mean_r = sum(replay_lat) / len(replay_lat)
        if mean_l == 0:
            return 1.0 if mean_r == 0 else 0.0
        return max(0.0, 1.0 - abs(mean_l - mean_r) / mean_l)

    def timing_parity(self) -> float:
        total = drift_ms = 0
        for eid, ld in self._live.items():
            rd = self._replay.get(eid)
            if rd is None:
                continue
            total += 1
            drift_ms += abs(ld.get("decision_ts", 0) - rd.get("decision_ts", 0))
        avg_drift = drift_ms / total if total else 1000.0
        return max(0.0, 1.0 - avg_drift / 500.0)

    def lifecycle_parity(self) -> float:
        live_trades = [d for d in self._live.values() if d.get("realized_pnl") is not None]
        replay_trades = [d for d in self._replay.values() if d.get("realized_pnl") is not None]
        if not live_trades or not replay_trades:
            return 0.0
        total = min(len(live_trades), len(replay_trades))
        aligned = sum(1 for i in range(total)
                      if abs(live_trades[i].get("realized_pnl", 0) -
                             replay_trades[i].get("realized_pnl", 0)) < 0.0001)
        return aligned / total if total else 0.0

    def composite_score(self) -> dict:
        sp = self.signal_parity()
        stp = self.state_parity()
        ep = self.execution_parity()
        tp = self.timing_parity()
        lp = self.lifecycle_parity()
        composite = 0.35 * sp + 0.25 * stp + 0.20 * ep + 0.10 * tp + 0.10 * lp
        return {
            "signal": round(sp, 4),
            "state": round(stp, 4),
            "execution": round(ep, 4),
            "timing": round(tp, 4),
            "lifecycle": round(lp, 4),
            "composite": round(composite, 4),
            "grade": "PRODUCTION" if composite > 0.95 else (
                "RESEARCH" if composite > 0.90 else (
                    "USEFUL" if composite > 0.85 else "REALISM_GAP")),
            "n_match": sum(1 for eid in self._live if eid in self._replay),
            "n_live": len(self._live),
            "n_replay": len(self._replay),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.composite_score(), f, indent=2)
