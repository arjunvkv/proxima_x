"""
decision_stream.py — Supplementary SHM channel for decision/intelligence data.

Exposes the decision layer outputs into the existing telemetry pipeline so the
browser dashboard can also display decision/intelligence data alongside raw
telemetry.

This module is standalone and does NOT require modifying existing telemetry
modules. It creates a SECOND shared memory segment ``"proxima_decision"``
that the WebSocket server can also read and broadcast.

Decision SHM format (60 bytes)
-------------------------------
    Offset  Size  Type  Field
    0        4     i     action_tendency       (enum index 0-5)
    4        4     i     regime_action_signal  (enum index 0-4)
    8        4     f     risk_bias             (-1 to +1)
    12       4     f     confidence            (0-1)
    16       4     f     health_score          (-1 to +1)
    20       4     i     anomaly_severity      (0=none … 4=CRITICAL)
    24       4     i     anomaly_count
    28       4     i     regime_from           (regime enum index)
    32       4     f     regime_probability    (0-1)
    36       4     f     regime_to             (regime enum index, packed as float)
    40       4     f     magnitude             (0-1)
    44       4     f     risk_limit            (0-1)
    48       4     f     max_slippage          (bps)
    52       4     f     timestamp
    56       4     i     frame_id

    Total: 15 fields × 4 bytes = 60 bytes

Integration note
-----------------
The ``DecisionStreamWriter`` writes to ``"proxima_decision"``, a second SHM
segment alongside the existing 432-byte telemetry frame. A
``DecisionStreamReader`` (or the ``DecisionFrameEncoder``) can be used in the
WebSocket server to also broadcast decision data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from struct import Struct
from typing import Any, Optional

__all__ = [
    "DecisionSnapshot",
    "DecisionStreamWriter",
    "DecisionStreamReader",
    "DecisionFrameEncoder",
    "create_decision_snapshot",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DECISION_SHM_NAME = "proxima_decision"

# Struct format: 15 fields, 60 bytes total
#   action_tendency (i), regime_action_signal (i), risk_bias (f), confidence (f),
#   health_score (f), anomaly_severity (i), anomaly_count (i), regime_from (i),
#   regime_probability (f), regime_to (f), magnitude (f), risk_limit (f),
#   max_slippage (f), timestamp (f), frame_id (i)
_DECISION_FORMAT = Struct("<2i2ffi2i2f2f2fi")


# ---------------------------------------------------------------------------
# DecisionSnapshot
# ---------------------------------------------------------------------------


@dataclass
class DecisionSnapshot:
    """Fixed-size binary representation of decision state.

    All numeric fields are designed to fit within the 60-byte SHM layout
    described at module level.

    Attributes
    ----------
    action_tendency : int
        Enum index 0-5 (STRONG_BUY, BUY, HOLD, REDUCE, EXIT, STRONG_SELL).
    regime_action_signal : int
        Enum index 0-4 (ESCALATE, MAINTAIN, DE_ESCALATE, PREPARE_TRANSITION,
        EMERGENCY_STOP).
    risk_bias : float
        Bias from -1.0 (max risk-averse) to +1.0 (max risk-seeking).
    confidence : float
        Overall confidence in the decision, 0.0 – 1.0.
    health_score : float
        System health score from -1.0 (critical) to +1.0 (healthy).
    anomaly_severity : int
        0 = none, 1 = LOW, 2 = MEDIUM, 3 = HIGH, 4 = CRITICAL.
    anomaly_count : int
        Number of active anomaly events.
    regime_probability : float
        Combined regime transition probability, 0.0 – 1.0.
    regime_from : int
        Source regime enum index.
    regime_to : int
        Target regime enum index.
    magnitude : float
        Position size fraction in [0.0, 1.0].
    risk_limit : float
        Maximum allowed risk in [0.0, 1.0].
    max_slippage : float
        Maximum acceptable slippage in basis points.
    timestamp : float
        Unix timestamp of this snapshot.
    frame_id : int
        Monotonically increasing frame identifier from the intelligence bus.
    """

    action_tendency: int
    regime_action_signal: int
    risk_bias: float
    confidence: float
    health_score: float
    anomaly_severity: int
    anomaly_count: int
    regime_probability: float
    regime_from: int
    regime_to: int
    magnitude: float
    risk_limit: float
    max_slippage: float
    timestamp: float
    frame_id: int


# ---------------------------------------------------------------------------
# DecisionStreamWriter
# ---------------------------------------------------------------------------


class DecisionStreamWriter:
    """Writes decision data to shared memory (hot-path safe).

    Creates (or attaches to) the ``"proxima_decision"`` shared memory segment
    and provides a single :meth:`write` call that packs a :class:`DecisionSnapshot`
    into the 60-byte binary layout.

    Usage::

        writer = DecisionStreamWriter()
        writer.write(snapshot)
        writer.close()
    """

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None
        self._init_shm()

    def _init_shm(self) -> None:
        """Create or attach to the shared memory segment."""
        try:
            self._shm = shared_memory.SharedMemory(
                name=_DECISION_SHM_NAME,
                create=True,
                size=_DECISION_FORMAT.size,
            )
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(
                name=_DECISION_SHM_NAME,
                create=False,
            )

    def write(self, snapshot: DecisionSnapshot) -> None:
        """Pack *snapshot* into the shared memory buffer.

        This is a single ``struct.pack`` call — safe for hot-path use.

        Parameters
        ----------
        snapshot : DecisionSnapshot
            The decision state to write.
        """
        buf = _DECISION_FORMAT.pack(
            snapshot.action_tendency,
            snapshot.regime_action_signal,
            snapshot.risk_bias,
            snapshot.confidence,
            snapshot.health_score,
            snapshot.anomaly_severity,
            snapshot.anomaly_count,
            snapshot.regime_from,
            snapshot.regime_probability,
            snapshot.regime_to,
            snapshot.magnitude,
            snapshot.risk_limit,
            snapshot.max_slippage,
            snapshot.timestamp,
            snapshot.frame_id,
        )
        self._shm.buf[:] = buf

    def close(self) -> None:
        """Close and unlink the shared memory segment.

        Safe to call multiple times.  Only the creating process can unlink.
        """
        if self._shm:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# DecisionStreamReader
# ---------------------------------------------------------------------------


class DecisionStreamReader:
    """Reads decision data from shared memory.

    Attaches to an existing ``"proxima_decision"`` shared memory segment
    created by :class:`DecisionStreamWriter`.

    Usage::

        reader = DecisionStreamReader()
        snapshot = reader.read()
        if snapshot is not None:
            ...
        reader.close()
    """

    def __init__(self) -> None:
        self._shm = shared_memory.SharedMemory(
            name=_DECISION_SHM_NAME,
            create=False,
        )

    def read(self) -> DecisionSnapshot | None:
        """Read and unpack the decision SHM buffer.

        Returns
        -------
        DecisionSnapshot or None
            The unpacked decision state, or ``None`` if the buffer could not
            be read (e.g. corrupted or not yet written).
        """
        try:
            data = _DECISION_FORMAT.unpack(bytes(self._shm.buf[:_DECISION_FORMAT.size]))
            return DecisionSnapshot(
                action_tendency=data[0],
                regime_action_signal=data[1],
                risk_bias=data[2],
                confidence=data[3],
                health_score=data[4],
                anomaly_severity=data[5],
                anomaly_count=data[6],
                regime_from=data[7],
                regime_probability=data[8],
                regime_to=int(data[9]),
                magnitude=data[10],
                risk_limit=data[11],
                max_slippage=data[12],
                timestamp=data[13],
                frame_id=data[14],
            )
        except Exception:
            return None

    def close(self) -> None:
        """Close the shared memory handle (does not unlink)."""
        self._shm.close()

    def cleanup(self) -> None:
        """Close and forcibly unlink the shared memory segment.

        This is a destructive operation — only call when the writer is
        known to be shut down.
        """
        self.close()
        try:
            shm = shared_memory.SharedMemory(name=_DECISION_SHM_NAME, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# DecisionFrameEncoder
# ---------------------------------------------------------------------------


class DecisionFrameEncoder:
    """Encode a :class:`DecisionSnapshot` → JSON-serialisable dict for WebSocket broadcast.

    Usage::

        snapshot = reader.read()
        if snapshot is not None:
            payload = DecisionFrameEncoder.encode(snapshot)
            # payload is a dict suitable for json.dumps()
    """

    @staticmethod
    def encode(snapshot: DecisionSnapshot) -> dict:
        """Convert *snapshot* to a JSON-friendly dictionary.

        Parameters
        ----------
        snapshot : DecisionSnapshot
            The decision state to encode.

        Returns
        -------
        dict
            A dictionary with keys ``type``, ``action``, ``regime_signal``,
            ``risk_bias``, ``confidence``, ``health``, ``anomaly_severity``,
            ``anomaly_count``, ``regime``, ``execution``, ``timestamp``,
            and ``frame_id``.
        """
        return {
            "type": "decision",
            "action": _ACTION_NAMES.get(snapshot.action_tendency, "UNKNOWN"),
            "regime_signal": _REGIME_SIGNALS.get(snapshot.regime_action_signal, "UNKNOWN"),
            "risk_bias": round(snapshot.risk_bias, 4),
            "confidence": round(snapshot.confidence, 4),
            "health": round(snapshot.health_score, 4),
            "anomaly_severity": _ANOMALY_NAMES.get(snapshot.anomaly_severity, "NONE"),
            "anomaly_count": snapshot.anomaly_count,
            "regime": {
                "from": _REGIME_NAMES.get(snapshot.regime_from, "UNKNOWN"),
                "to": _REGIME_NAMES.get(snapshot.regime_to, "UNKNOWN"),
                "probability": round(snapshot.regime_probability, 4),
            },
            "execution": {
                "magnitude": round(snapshot.magnitude, 4),
                "risk_limit": round(snapshot.risk_limit, 4),
                "max_slippage": snapshot.max_slippage,
            },
            "timestamp": snapshot.timestamp,
            "frame_id": snapshot.frame_id,
        }


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_ACTION_NAMES: dict[int, str] = {
    0: "STRONG_BUY",
    1: "BUY",
    2: "HOLD",
    3: "REDUCE",
    4: "EXIT",
    5: "STRONG_SELL",
}

_REGIME_SIGNALS: dict[int, str] = {
    0: "ESCALATE",
    1: "MAINTAIN",
    2: "DE_ESCALATE",
    3: "PREPARE_TRANSITION",
    4: "EMERGENCY_STOP",
}

_ANOMALY_NAMES: dict[int, str] = {
    0: "NONE",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "CRITICAL",
}

_REGIME_NAMES: dict[int, str] = {
    0: "SHADOW",
    1: "MICRO",
    2: "FULL",
    3: "UNKNOWN",
}


# ---------------------------------------------------------------------------
# Severity mapping (int from severity string)
# ---------------------------------------------------------------------------

_SEVERITY_TO_INT: dict[str, int] = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# ---------------------------------------------------------------------------
# Action / regime enum string-to-int helpers
# ---------------------------------------------------------------------------

_ACTION_TO_INT: dict[str, int] = {
    "STRONG_BUY": 0,
    "BUY": 1,
    "HOLD": 2,
    "REDUCE": 3,
    "EXIT": 4,
    "STRONG_SELL": 5,
}

_REGIME_SIGNAL_TO_INT: dict[str, int] = {
    "ESCALATE": 0,
    "MAINTAIN": 1,
    "DE_ESCALATE": 2,
    "PREPARE_TRANSITION": 3,
    "EMERGENCY_STOP": 4,
}

_REGIME_NAME_TO_INT: dict[str, int] = {
    "SHADOW": 0,
    "MICRO": 1,
    "FULL": 2,
    "UNKNOWN": 3,
}


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def create_decision_snapshot(
    system_decision: Any,
    execution_intent: Any,
    intelligence_frame: Any,
) -> DecisionSnapshot:
    """Build a :class:`DecisionSnapshot` from the three decision-layer objects.

    This is the primary integration point between the decision pipeline and
    the observability layer.  It extracts relevant fields from:

    * ``system_decision`` — a ``SystemDecision`` (or duck-typed equivalent)
      providing ``action_tendency``, ``regime_action_signal``, ``risk_bias``,
      ``confidence``, ``components`` (a dict that may contain
      ``health_score``), and ``timestamp``.

    * ``execution_intent`` — an ``ExecutionIntent`` (or duck-typed equivalent)
      providing ``magnitude``, ``risk_limit``, ``max_slippage``.

    * ``intelligence_frame`` — an ``IntelligenceFrame`` (or duck-typed equivalent)
      providing ``frame_id``, ``regime`` (a ``TransitionSignal`` with
      ``from_regime``, ``to_regime``, ``probability``), ``anomalies`` (a list
      of ``AnomalyEvent`` with ``severity``), and ``health`` (a
      ``SystemHealthScore`` with ``score``).

    Parameters
    ----------
    system_decision : Any
        The decision produced by ``DecisionSynthesizer.synthesize()``.
    execution_intent : Any
        The intent produced by ``ExecutionIntentTranslator.translate()``.
    intelligence_frame : Any
        The frame produced by ``IntelligenceBus.step()``.

    Returns
    -------
    DecisionSnapshot
        A fully populated snapshot ready for ``DecisionStreamWriter.write()``.
    """
    now = time.time()

    # -- Extract from system_decision ---------------------------------------
    action_tendency = _resolve_action_tendency(
        getattr(system_decision, "action_tendency", None),
    )
    regime_action_signal = _resolve_regime_signal(
        getattr(system_decision, "regime_action_signal", None),
    )
    risk_bias = float(getattr(system_decision, "risk_bias", 0.0))
    confidence = float(getattr(system_decision, "confidence", 0.0))
    decision_ts = float(getattr(system_decision, "timestamp", now))

    # health_score may live in system_decision.components
    components = getattr(system_decision, "components", None) or {}
    if isinstance(components, dict):
        health_score = float(components.get("health_score", 0.0))
    else:
        health_score = 0.0

    # -- Extract from execution_intent ---------------------------------------
    magnitude = float(getattr(execution_intent, "magnitude", 0.0))
    risk_limit = float(getattr(execution_intent, "risk_limit", 1.0))
    max_slippage = float(getattr(execution_intent, "max_slippage", 0.0))

    # -- Extract from intelligence_frame -------------------------------------
    frame_id = int(getattr(intelligence_frame, "frame_id", 0))

    regime = getattr(intelligence_frame, "regime", None)
    if regime is not None:
        regime_from = _resolve_regime_name(
            getattr(regime, "from_regime", "UNKNOWN"),
        )
        regime_to = _resolve_regime_name(
            getattr(regime, "to_regime", "UNKNOWN"),
        )
        regime_probability = float(getattr(regime, "probability", 0.0))
    else:
        regime_from = _REGIME_NAME_TO_INT.get("UNKNOWN", 3)
        regime_to = _REGIME_NAME_TO_INT.get("UNKNOWN", 3)
        regime_probability = 0.0

    anomalies = getattr(intelligence_frame, "anomalies", None) or []
    anomaly_count = len(anomalies)
    anomaly_severity = _compute_max_anomaly_severity(anomalies)

    # health_score from intelligence_frame.health overrides components value
    # if the frame has a direct health assessment
    health = getattr(intelligence_frame, "health", None)
    if health is not None:
        frame_health = float(getattr(health, "score", 0.0))
        # Use frame health if it is non-zero, otherwise keep components value
        if abs(frame_health) > 1e-9:
            health_score = frame_health

    return DecisionSnapshot(
        action_tendency=action_tendency,
        regime_action_signal=regime_action_signal,
        risk_bias=risk_bias,
        confidence=confidence,
        health_score=health_score,
        anomaly_severity=anomaly_severity,
        anomaly_count=anomaly_count,
        regime_probability=regime_probability,
        regime_from=regime_from,
        regime_to=regime_to,
        magnitude=magnitude,
        risk_limit=risk_limit,
        max_slippage=max_slippage,
        timestamp=decision_ts,
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_action_tendency(value: Any) -> int:
    """Convert an ``ActionTendency`` enum (or string or int) to an int index."""
    if value is None:
        return 2  # default HOLD
    if isinstance(value, int):
        return max(0, min(5, value))
    if isinstance(value, str):
        return _ACTION_TO_INT.get(value, 2)
    # Enum-like object
    if hasattr(value, "value"):
        return _ACTION_TO_INT.get(str(value.value), 2)
    return 2


def _resolve_regime_signal(value: Any) -> int:
    """Convert a ``RegimeActionSignal`` enum (or string or int) to an int index."""
    if value is None:
        return 1  # default MAINTAIN
    if isinstance(value, int):
        return max(0, min(4, value))
    if isinstance(value, str):
        return _REGIME_SIGNAL_TO_INT.get(value, 1)
    if hasattr(value, "value"):
        return _REGIME_SIGNAL_TO_INT.get(str(value.value), 1)
    return 1


def _resolve_regime_name(value: Any) -> int:
    """Convert a regime name (string or enum) to an int index."""
    if value is None:
        return 3  # UNKNOWN
    if isinstance(value, int):
        return max(0, min(3, value))
    if isinstance(value, str):
        return _REGIME_NAME_TO_INT.get(value, 3)
    if hasattr(value, "value"):
        return _REGIME_NAME_TO_INT.get(str(value.value), 3)
    return 3


def _compute_max_anomaly_severity(anomalies: list[Any]) -> int:
    """Compute the maximum anomaly severity from a list of anomaly events.

    Each anomaly is expected to have a ``severity`` attribute that is one of
    ``"NONE"``, ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
    Returns the highest severity encountered, or 0 (NONE) if the list is empty.
    """
    max_sev = 0
    for a in anomalies:
        sev = getattr(a, "severity", "NONE")
        if isinstance(sev, str):
            sev_int = _SEVERITY_TO_INT.get(sev, 0)
        elif isinstance(sev, int):
            sev_int = max(0, min(4, sev))
        else:
            sev_int = 0
        if sev_int > max_sev:
            max_sev = sev_int
    return max_sev
