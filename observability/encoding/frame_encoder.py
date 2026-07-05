from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FramePayload:
    """Structured transport payload for WebSocket streaming."""
    frame_id: int
    timestamp: float
    engine_vector: List[float]  # 32 floats
    alignment: float
    stability: float
    entropy: float
    regime_state: float
    tpi_confidence: float
    shadow_alignment: float
    sof_score: float
    kill_switch_pressure: float
    rollout_progress: float
    execution_intensity: float
    risk_exposure: float
    system_integrity: float

    def to_json(self) -> str:
        """Serialize to JSON string for WebSocket transmission."""
        return json.dumps({
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "engine_vector": self.engine_vector,
            "alignment": self.alignment,
            "stability": self.stability,
            "entropy": self.entropy,
            "regime_state": self.regime_state,
            "tpi_confidence": self.tpi_confidence,
            "shadow_alignment": self.shadow_alignment,
            "sof_score": self.sof_score,
            "kill_switch_pressure": self.kill_switch_pressure,
            "rollout_progress": self.rollout_progress,
            "execution_intensity": self.execution_intensity,
            "risk_exposure": self.risk_exposure,
            "system_integrity": self.system_integrity,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Return as a plain dict."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "engine_vector": self.engine_vector,
            "alignment": self.alignment,
            "stability": self.stability,
            "entropy": self.entropy,
            "regime_state": self.regime_state,
            "tpi_confidence": self.tpi_confidence,
            "shadow_alignment": self.shadow_alignment,
            "sof_score": self.sof_score,
            "kill_switch_pressure": self.kill_switch_pressure,
            "rollout_progress": self.rollout_progress,
            "execution_intensity": self.execution_intensity,
            "risk_exposure": self.risk_exposure,
            "system_integrity": self.system_integrity,
        }


class FrameEncoder:
    """
    Converts between raw SHM binary frames and structured transport payloads.

    This is the ONLY module that knows about:
    - Binary SHM frame layout
    - JSON network format
    - float32 -> float conversion

    Usage:
        encoder = FrameEncoder()

        # From SHM snapshot dict:
        payload = encoder.decode_shm_snapshot(snapshot_dict)

        # From raw bytes (if reading SHM directly):
        payload = encoder.decode_raw_frame(raw_bytes)

        # To JSON:
        json_str = encoder.encode_json(payload)
    """

    # Frame layout constants (must match shared_memory_telemetry.py)
    ENGINE_VECTOR_SIZE = 32
    SCALAR_COUNT = 12
    PAD_COUNT = 1
    FRAME_FLOAT_COUNT = ENGINE_VECTOR_SIZE + SCALAR_COUNT + PAD_COUNT  # 45

    # Struct format for the full frame
    _FRAME_STRUCT = struct.Struct("<32f13f4x")  # 184 bytes

    def decode_shm_snapshot(self, snapshot: Dict[str, Any]) -> FramePayload:
        """
        Convert a TelemetryCore.read_snapshot() dict to FramePayload.

        Handles the float32 -> float conversion for JSON safety.
        """
        return FramePayload(
            frame_id=int(snapshot.get("frame_id", 0)),
            timestamp=float(snapshot.get("timestamp", 0.0)),
            engine_vector=[float(v) for v in snapshot.get("engine_vector", [0.0] * 32)],
            alignment=float(snapshot.get("alignment", 0.0)),
            stability=float(snapshot.get("stability", 0.0)),
            entropy=float(snapshot.get("entropy", 0.0)),
            regime_state=float(snapshot.get("regime_state", 0.0)),
            tpi_confidence=float(snapshot.get("tpi_confidence", 0.0)),
            shadow_alignment=float(snapshot.get("shadow_alignment", 0.0)),
            sof_score=float(snapshot.get("sof_score", 0.0)),
            kill_switch_pressure=float(snapshot.get("kill_switch_pressure", 0.0)),
            rollout_progress=float(snapshot.get("rollout_progress", 0.0)),
            execution_intensity=float(snapshot.get("execution_intensity", 0.0)),
            risk_exposure=float(snapshot.get("risk_exposure", 0.0)),
            system_integrity=float(snapshot.get("system_integrity", 0.0)),
        )

    def decode_raw_frame(self, raw_bytes: bytes) -> FramePayload:
        """
        Decode a raw 184-byte frame buffer into a FramePayload.

        This is useful for readers that access SHM directly without
        going through TelemetryCore.
        """
        if len(raw_bytes) < self._FRAME_STRUCT.size:
            raise ValueError(
                f"Raw frame too short: {len(raw_bytes)} < {self._FRAME_STRUCT.size}"
            )

        data = self._FRAME_STRUCT.unpack_from(raw_bytes, 0)

        vector = [float(v) for v in data[:32]]
        scalars = [float(v) for v in data[32:45]]  # 12 real + 1 pad

        return FramePayload(
            frame_id=0,  # Not in frame data -- caller must supply
            timestamp=0.0,
            engine_vector=vector,
            alignment=scalars[0],
            stability=scalars[1],
            entropy=scalars[2],
            regime_state=scalars[3],
            tpi_confidence=scalars[4],
            shadow_alignment=scalars[5],
            sof_score=scalars[6],
            kill_switch_pressure=scalars[7],
            rollout_progress=scalars[8],
            execution_intensity=scalars[9],
            risk_exposure=scalars[10],
            system_integrity=scalars[11],
        )

    def encode_json(self, payload: FramePayload) -> str:
        """Serialize FramePayload to JSON string."""
        return payload.to_json()

    def encode_dict(self, payload: FramePayload) -> Dict[str, Any]:
        """Serialize FramePayload to plain dict."""
        return payload.to_dict()

    def merge_full_snapshot(
        self,
        frame_payload: FramePayload,
        full_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge the hot-path frame payload with a cold-path full TelemetrySnapshot.

        The frame payload contains the 32D vector + 12 scalars (high-frequency).
        The full snapshot contains all research/dashboard details (low-frequency).

        This produces a single comprehensive JSON payload for the browser.
        """
        result = frame_payload.to_dict()
        result["full_snapshot"] = full_snapshot
        return result
