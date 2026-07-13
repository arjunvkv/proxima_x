import time
import json
import os
from pathlib import Path
from typing import Optional
from narrative.overlay import narrative_health_score

LOG_DIR = Path(__file__).parent.parent / "logs" / "trade_lifecycle"

class TradeLifecycleLogger:
    """Records every cycle snapshot during a trade batch's life for post-hoc analysis."""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._file: Optional[object] = None
        self._batch_id: Optional[str] = None
        self._path: Optional[str] = None

    def open_batch(self, positions: list, narrative_state) -> None:
        if not positions:
            return
        self._batch_id = str(int(time.time() * 1000))
        leader = narrative_state.identity.leader if narrative_state else "none"
        ts_str = time.strftime("%H%M%S")
        self._path = f"{LOG_DIR}/batch_{ts_str}_{leader}.jsonl"
        self._file = open(self._path, "w", encoding="utf-8")
        entry = {
            "event": "OPEN",
            "ts": time.time(),
            "batch_id": self._batch_id,
            "narrative": self._narrative_snapshot(narrative_state),
            "positions": [self._pos_snapshot(p) for p in positions],
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def log_cycle(self, cycle: int, positions: list, narrative_state, strengths: dict,
                  bursts: dict, der: dict, graph_quality: float) -> None:
        if self._file is None:
            return
        health = narrative_health_score(narrative_state) if narrative_state else 0.5
        entry = {
            "event": "CYCLE",
            "ts": time.time(),
            "cycle": cycle,
            "batch_id": self._batch_id,
            "narrative": self._narrative_snapshot(narrative_state),
            "health": round(health, 3),
            "positions": [self._pos_snapshot(p) for p in positions],
            "strengths": {c: round(v, 6) for c, v in (strengths or {}).items()},
            "bursts": {c: round(v, 3) for c, v in (bursts or {}).items()},
            "der": {c: round(v, 3) for c, v in (der or {}).items()},
            "graph_quality": round(graph_quality, 3),
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close_batch(self, reason: str, positions: list) -> None:
        if self._file is None:
            return
        entry = {
            "event": "CLOSE",
            "ts": time.time(),
            "batch_id": self._batch_id,
            "reason": reason,
            "positions": [self._pos_snapshot(p) for p in positions],
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()
        self._file.close()
        self._file = None
        print(f"[TRADE LIFECYCLE] logged to {self._path}  reason={reason}  "
              f"pnl={sum(p.pnl or 0 for p in positions):.2f}", file=__import__("sys").stderr)
        self._path = None
        self._batch_id = None

    def discard(self) -> None:
        if self._file is not None:
            self._file.close()
            if self._path and os.path.exists(self._path):
                os.remove(self._path)
            self._file = None
            self._path = None
            self._batch_id = None

    def _narrative_snapshot(self, narrative_state):
        if narrative_state is None:
            return None
        return {
            "leader": narrative_state.identity.leader,
            "direction": narrative_state.identity.direction,
            "opponents": list(narrative_state.identity.opponents),
            "nmi": round(narrative_state.nmi, 3),
            "phase": narrative_state.phase.value,
            "age": narrative_state.age,
            "strength_delta": round(narrative_state.strength_delta, 6) if narrative_state.strength_delta is not None else None,
        }

    def _pos_snapshot(self, pos):
        return {
            "id": pos.id,
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "pnl": round(pos.pnl or 0, 2),
            "age_s": round(time.time() - pos.entry_time, 1),
        }

    @property
    def is_active(self) -> bool:
        return self._file is not None
