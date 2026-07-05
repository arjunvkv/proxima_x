"""
frame_history.py — Hybrid RAM + disk frame history for replay.

Architecture:
  RAM ring buffer  (last 60 min, fast rewind)
  Disk segments    (append-only, durable replay)

WebSocket streams:
  stream: "live"      → 10 FPS hot-path frames
  stream: "state"     → 1 FPS full snapshots
  stream: "replay"    → historical playback chunks
  stream: "control"   → mode change signals
"""
from __future__ import annotations
import json, os, time, struct, threading, zlib
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

HISTORY_DIR = Path(__file__).parent.parent.parent / "data" / "replay"
MAX_RAM_FRAMES = 36000       # 60 min at 10 FPS
MAX_RAM_SNAPSHOTS = 3600     # 60 min at 1 FPS
DISK_CHUNK_SECONDS = 300     # 5 min per chunk file
MAX_REPLAY_RESULTS = 5000    # max frames per replay query

@dataclass
class ReplayFrame:
    stream: str          # "live" | "state" | "dashboard"
    frame_id: int
    timestamp: float
    data: dict
    checksum: str = ""

    def compute_checksum(self) -> str:
        raw = json.dumps(asdict(self), default=str, sort_keys=True)
        return str(zlib.crc32(raw.encode()))

class FrameHistory:
    def __init__(self, replay_dir: Optional[Path] = None):
        self._dir = replay_dir or HISTORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_id = 0

        # RAM ring buffers
        self._live: deque[ReplayFrame] = deque(maxlen=MAX_RAM_FRAMES)
        self._snapshots: deque[ReplayFrame] = deque(maxlen=MAX_RAM_SNAPSHOTS)
        self._dashboard: deque[ReplayFrame] = deque(maxlen=MAX_RAM_SNAPSHOTS)

        # Disk writer state
        self._current_chunk: int = 0
        self._chunk_frames: int = 0
        self._chunk_start_ts: float = 0

    # ── Append ──────────────────────────────────────────────────────────

    def append_live(self, data: dict) -> ReplayFrame:
        return self._append("live", data)

    def append_snapshot(self, data: dict) -> ReplayFrame:
        return self._append("state", data)

    def append_dashboard(self, text: str) -> ReplayFrame:
        return self._append("dashboard", {"text": text, "timestamp": time.time()})

    def _append(self, stream: str, data: dict) -> ReplayFrame:
        frame = ReplayFrame(
            stream=stream,
            frame_id=self._next_id,
            timestamp=data.get("timestamp", time.time()),
            data=data,
        )
        frame.checksum = frame.compute_checksum()
        with self._lock:
            self._next_id += 1
            if stream == "live":
                self._live.append(frame)
            elif stream == "state":
                self._snapshots.append(frame)
            elif stream == "dashboard":
                self._dashboard.append(frame)
            self._write_disk(frame)
        return frame

    # ── Disk persistence ────────────────────────────────────────────────

    def _write_disk(self, frame: ReplayFrame):
        chunk_idx = int(frame.timestamp // DISK_CHUNK_SECONDS)
        chunk_file = self._dir / f"chunk_{chunk_idx}.jsonl"
        try:
            with open(chunk_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(frame), default=str) + "\n")
        except Exception:
            pass

    # ── Query ───────────────────────────────────────────────────────────

    def query(self, stream: Optional[str] = None,
              from_ts: float = 0, to_ts: float = 0,
              cursor: int = 0, max_results: int = MAX_REPLAY_RESULTS) -> tuple[list[ReplayFrame], int]:
        if to_ts <= 0:
            to_ts = time.time()
        with self._lock:
            # Search RAM first (fast path)
            sources = []
            if stream is None or stream == "live":
                sources.extend(self._live)
            if stream is None or stream == "state":
                sources.extend(self._snapshots)
            if stream is None or stream == "dashboard":
                sources.extend(self._dashboard)

            results = [
                f for f in sources
                if f.frame_id > cursor and from_ts <= f.timestamp <= to_ts
            ]
            # Sort chronologically so live/state frames interleave correctly
            results.sort(key=lambda f: f.timestamp)
            results = results[-max_results:]

            if len(results) < max_results and (stream is None or stream != "live"):
                # Fall back to disk for older frames
                results = self._query_disk(stream, from_ts, to_ts, cursor, max_results)

            new_cursor = results[-1].frame_id if results else cursor
            return results, new_cursor

    def _query_disk(self, stream: Optional[str], from_ts: float, to_ts: float,
                    cursor: int, max_results: int) -> list[ReplayFrame]:
        results = []
        chunk_start = int(from_ts // DISK_CHUNK_SECONDS)
        chunk_end = int(to_ts // DISK_CHUNK_SECONDS)
        for ci in range(chunk_start, chunk_end + 1):
            chunk_file = self._dir / f"chunk_{ci}.jsonl"
            if not chunk_file.exists():
                continue
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            rf = ReplayFrame(**d)
                            if rf.frame_id > cursor and from_ts <= rf.timestamp <= to_ts:
                                if stream is None or rf.stream == stream:
                                    results.append(rf)
                                    if len(results) >= max_results:
                                        return results
                        except Exception:
                            continue
            except Exception:
                continue
        return results

    # ── Replay payload ──────────────────────────────────────────────────

    def to_replay_payload(self, frames: list[ReplayFrame], cursor: int) -> dict:
        return {
            "stream": "replay",
            "type": "replay_chunk",
            "cursor": cursor,
            "count": len(frames),
            "timestamp": time.time(),
            "frames": [
                {
                    "stream": f.stream,
                    "frame_id": f.frame_id,
                    "timestamp": f.timestamp,
                    "data": f.data,
                }
                for f in frames
            ],
        }

    def to_range_payload(self) -> dict:
        with self._lock:
            ts_list = []
            if self._live:
                ts_list.append(self._live[0].timestamp)
                ts_list.append(self._live[-1].timestamp)
            if self._snapshots:
                if not ts_list or self._snapshots[0].timestamp < ts_list[0]:
                    ts_list.insert(0, self._snapshots[0].timestamp)
                if self._snapshots[-1].timestamp > ts_list[-1]:
                    ts_list[-1] = self._snapshots[-1].timestamp
            return {
                "stream": "control",
                "type": "history_range",
                "from_ts": ts_list[0] if len(ts_list) >= 2 else 0,
                "to_ts": ts_list[-1] if len(ts_list) >= 2 else 0,
                "live_count": len(self._live),
                "snapshot_count": len(self._snapshots),
            }

    def get_latest_live(self) -> Optional[ReplayFrame]:
        with self._lock:
            return self._live[-1] if self._live else None

# ── Global singleton ──────────────────────────────────────────────────
_history: Optional[FrameHistory] = None

def get_history() -> FrameHistory:
    global _history
    if _history is None:
        _history = FrameHistory()
    return _history
