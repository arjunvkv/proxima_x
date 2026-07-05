"""
websocket_server.py — Async WebSocket server with stream-layered design.

Streams:
  live   → 10 FPS hot-path frames
  state  → 1 FPS full snapshots
  replay → historical playback chunks (cursor-based)
  control → mode, history_range signals
"""
from __future__ import annotations
import asyncio, json, logging, math, time
from collections import deque
from threading import Lock
from typing import Optional

from ..core.shared_memory_telemetry import TelemetryCore
from ..history.frame_history import FrameHistory, get_history, MAX_REPLAY_RESULTS

logger = logging.getLogger(__name__)

class TelemetryWebSocketServer:
    def __init__(self, host="0.0.0.0", port=8765, fps=30, shm_name="proxima_telemetry"):
        self._host = host
        self._port = port
        self._interval = 1.0 / max(fps, 1)
        self._shm_name = shm_name
        self._server: Optional[asyncio.AbstractServer] = None
        self._core: Optional[TelemetryCore] = None
        self._clients: set = set()
        self._running = False
        self._history = get_history()

        # Thread-safe queues (deque + Lock for cross-thread feeding)
        self._snapshot_queue: deque = deque()
        self._dashboard_queue: deque = deque()
        self._q_lock: Lock = Lock()

    # ── External feed methods (called from trading loop thread) ────────

    def feed_snapshot(self, data: dict) -> None:
        with self._q_lock:
            if len(self._snapshot_queue) < 50:
                self._snapshot_queue.append(data)
                logger.debug("[WS_FEED] snapshot queued (qsize=%d)", len(self._snapshot_queue))
            else:
                logger.warning("[WS_FEED] snapshot queue full, dropping")

    def feed_dashboard_text(self, text: str) -> None:
        with self._q_lock:
            if len(self._dashboard_queue) < 20:
                self._dashboard_queue.append(text)
                logger.debug("[WS_FEED] dashboard queued (qsize=%d)", len(self._dashboard_queue))
            else:
                logger.warning("[WS_FEED] dashboard queue full, dropping")

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        import websockets
        try:
            self._core = TelemetryCore(self._shm_name, create=False)
        except FileNotFoundError:
            # SHM may not exist yet — create it
            self._core = TelemetryCore(self._shm_name, create=True)
            logger.info("WS server created SHM: %s", self._shm_name)
        self._running = True
        self._server = await websockets.serve(self._handle_client, self._host, self._port)
        logger.info("WS server on ws://%s:%d", self._host, self._port)
        # Launch broadcast loop as background task
        asyncio.create_task(self._broadcast_loop())

    async def stop(self):
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._core:
            try:
                self._core.unlink()
            except Exception:
                pass

    # ── Client handler ─────────────────────────────────────────────────

    async def _handle_client(self, websocket):
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)
        self._clients.add(websocket)

        # Send history range on connect
        try:
            range_payload = self._history.to_range_payload()
            await websocket.send(json.dumps(range_payload, default=str))
        except Exception:
            pass

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("type", "")
                    if msg_type == "replay_request":
                        await self._handle_replay(websocket, msg)
                    elif msg_type == "live_resume":
                        # Client is back to live mode — nothing to do server-side
                        pass
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    async def _handle_replay(self, websocket, msg: dict):
        from_ts = msg.get("from", 0)
        to_ts = msg.get("to", time.time())
        cursor = msg.get("cursor", 0)
        stream = msg.get("stream")  # None = all streams
        max_res = msg.get("max_results", MAX_REPLAY_RESULTS)

        frames, new_cursor = self._history.query(
            stream=stream, from_ts=from_ts, to_ts=to_ts,
            cursor=cursor, max_results=max_res,
        )
        payload = self._history.to_replay_payload(frames, new_cursor)

        # Mark complete if fewer than max results returned
        payload["complete"] = len(frames) < max_res
        if payload["complete"]:
            payload["type"] = "replay_complete"

        try:
            await websocket.send(json.dumps(payload, default=str))
        except Exception:
            pass

    # ── Broadcast loop ─────────────────────────────────────────────────

    async def _broadcast_loop(self):
        frame_counter = 0
        while self._running:
            start = time.monotonic()
            frame_counter += 1

            # Hot-path frame from SHM (every tick)
            await self._broadcast_live_frame()

            # Full snapshot from queue (every tick if available)
            await self._drain_queue(self._snapshot_queue, "state")
            # Dashboard text (every ~3 ticks)
            if frame_counter % 3 == 0:
                await self._drain_queue(self._dashboard_queue, "dashboard")

            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0, self._interval - elapsed))

    async def _broadcast_live_frame(self):
        if not self._core:
            return
        try:
            snap = self._core.read_snapshot()
            if snap is None:
                return

            def _f(v):
                try:
                    x = float(v)
                    return 0.0 if not math.isfinite(x) else x
                except Exception:
                    return 0.0

            vec = snap.get("engine_vector", [0.0]*32)
            payload = {
                "stream": "live",
                "type": "frame",
                "frame_id": snap.get("frame_id", 0),
                "timestamp": snap.get("timestamp", 0.0),
                "cycle_count": snap.get("cycle_counter", 0),
                "engine_vector": [_f(v) for v in vec[:32]],
                "alignment": _f(snap.get("alignment", 0)),
                "stability": _f(snap.get("stability", 0)),
                "entropy": _f(snap.get("entropy", 0)),
                "regime_state": _f(snap.get("regime_state", 0)),
                "tpi_confidence": _f(snap.get("tpi_confidence", 0)),
                "shadow_alignment": _f(snap.get("shadow_alignment", 0)),
                "sof_score": _f(snap.get("sof_score", 0)),
                "kill_switch_pressure": _f(snap.get("kill_switch_pressure", 0)),
                "rollout_progress": _f(snap.get("rollout_progress", 0)),
                "execution_intensity": _f(snap.get("execution_intensity", 0)),
                "risk_exposure": _f(snap.get("risk_exposure", 0)),
                "system_integrity": _f(snap.get("system_integrity", 0)),
            }
            self._history.append_live(payload)
            await self._broadcast(payload)
        except Exception:
            pass

    async def _drain_queue(self, q: deque, stream_name: str):
        """Drain one item from the deque and broadcast."""
        data = None
        with self._q_lock:
            if q:
                data = q.popleft()
        if data is None:
            return
        try:
            if isinstance(data, str):
                data = {"stream": stream_name, "type": stream_name, "text": data}
            else:
                data["stream"] = stream_name
            if stream_name == "state":
                self._history.append_snapshot(data)
            await self._broadcast(data)
        except Exception:
            pass

    async def _broadcast(self, data: dict):
        if not self._clients:
            return
        # Sanitize: JSON.parse rejects NaN/Inf — replace with 0
        def _clean(v):
            if isinstance(v, float):
                return 0.0 if not math.isfinite(v) else v
            if isinstance(v, dict):
                return {k: _clean(v2) for k, v2 in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return v
        data = _clean(data)
        msg = json.dumps(data, default=str)
        await asyncio.gather(
            *[self._send_safe(ws, msg) for ws in self._clients.copy()],
            return_exceptions=True,
        )

    async def _send_safe(self, websocket, msg: str):
        try:
            await websocket.send(msg)
        except Exception:
            self._clients.discard(websocket)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
