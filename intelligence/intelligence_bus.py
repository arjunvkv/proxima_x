"""
intelligence_bus.py — Unified Intelligence Stream.

Combines all intelligence outputs (regime detector, anomaly detector, causal graph,
vector compressor, system health) into a unified IntelligenceFrame.

This is the integration point that collects all analysis results and produces
a unified view of system intelligence — running as a second parallel stream
alongside telemetry.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = [
    "IntelligenceFrame",
    "IntelligenceBus",
]


# ---------------------------------------------------------------------------
# IntelligenceFrame
# ---------------------------------------------------------------------------


@dataclass
class IntelligenceFrame:
    """Unified intelligence snapshot from all analysis engines.

    Attributes
    ----------
    frame_id : int
        Monotonically increasing frame identifier.
    timestamp : float
        Unix timestamp of the analysis.
    regime : TransitionSignal | None
        Latest regime transition signal (``None`` if not yet available).
    anomalies : list[AnomalyEvent]
        Anomaly events detected (empty list if none).
    causal_graph : CausalGraph | None
        Causal dependency graph (``None`` if not yet built).
    compressed_state : CompressedStateVector | None
        Compressed latent representation (``None`` if not yet available).
    health : SystemHealthScore | None
        System health assessment (``None`` if not yet available).
    summary : str
        Human-readable one-line summary.
    priority : str
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
    """
    frame_id: int
    timestamp: float
    regime: Optional[Any] = None
    anomalies: List[Any] = field(default_factory=list)
    causal_graph: Optional[Any] = None
    compressed_state: Optional[Any] = None
    health: Optional[Any] = None
    summary: str = ""
    priority: str = "LOW"


# ---------------------------------------------------------------------------
# IntelligenceBus
# ---------------------------------------------------------------------------


class IntelligenceBus:
    """Integration point for all intelligence analysis engines.

    Collects results from regime detector, anomaly detector, causal graph
    builder, vector compressor, and system health monitor into a unified
    :class:`IntelligenceFrame` every time :meth:`step` is called.

    Parameters
    ----------
    buffer_size : int
        Maximum number of :class:`IntelligenceFrame` entries retained in
        history (default 1000).
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        self.buffer_size = buffer_size

        # Registered engines (all optional — duck-typed)
        self._regime_detector: Optional[Any] = None
        self._anomaly_detector: Optional[Any] = None
        self._causal_builder: Optional[Any] = None
        self._vector_compressor: Optional[Any] = None
        self._health_monitor: Optional[Any] = None

        # Frame tracking
        self._frame_counter: int = 0
        self._causal_step_counter: int = 0
        self._frame_since_last_step: bool = False

        # History
        self._latest_frame: Optional[IntelligenceFrame] = None
        self._history: deque = deque(maxlen=buffer_size)

    # ── Registration ─────────────────────────────────────────────────────────

    def register_regime_detector(self, detector: Any) -> None:
        """Register a ``RegimeTransitionDetector`` instance.

        The detector must provide ``feed(frame)`` and ``detect()`` methods
        for duck typing.
        """
        self._regime_detector = detector

    def register_anomaly_detector(self, detector: Any) -> None:
        """Register an ``AnomalyDetector`` instance.

        The detector must provide ``feed(frame)`` and ``detect()`` methods
        for duck typing.
        """
        self._anomaly_detector = detector

    def register_causal_graph_builder(self, builder: Any) -> None:
        """Register a ``CausalGraphBuilder`` instance.

        The builder must provide ``feed(frame)`` and ``build_graph()``
        methods for duck typing.
        """
        self._causal_builder = builder

    def register_vector_compressor(self, compressor: Any) -> None:
        """Register a ``VectorCompressor`` instance.

        The compressor must provide:
          - ``feed(frame)``
          - ``train()``
          - ``get_regime_embedding()``
          - ``is_trained`` property
        """
        self._vector_compressor = compressor

    def register_health_monitor(self, monitor: Any) -> None:
        """Register a ``SystemHealthMonitor`` instance.

        The monitor must provide:
          - ``feed(frame)``
          - ``compute()``
          - ``feed_anomaly(anomaly)``
          - ``feed_transition(transition)``
        """
        self._health_monitor = monitor

    # ── Data ingestion ───────────────────────────────────────────────────────

    def feed(self, frame: bytes) -> None:
        """Feed a raw 432-byte SHM frame to ALL registered engines.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore``.
        """
        if self._regime_detector is not None:
            self._regime_detector.feed(frame)
        if self._anomaly_detector is not None:
            self._anomaly_detector.feed(frame)
        if self._causal_builder is not None:
            self._causal_builder.feed(frame)
        if self._vector_compressor is not None:
            self._vector_compressor.feed(frame)
        if self._health_monitor is not None:
            self._health_monitor.feed(frame)
        self._frame_since_last_step = True

    def feed_anomaly(self, anomaly: Any) -> None:
        """Feed an anomaly event to the health monitor.

        Parameters
        ----------
        anomaly : AnomalyEvent
            The anomaly event produced by :class:`AnomalyDetector`.
        """
        if self._health_monitor is not None:
            self._health_monitor.feed_anomaly(anomaly)

    def feed_transition(self, transition: Any) -> None:
        """Feed a transition signal to the health monitor.

        Parameters
        ----------
        transition : TransitionSignal
            The transition signal produced by :class:`RegimeTransitionDetector`.
        """
        if self._health_monitor is not None:
            self._health_monitor.feed_transition(transition)

    # ── Analysis step ────────────────────────────────────────────────────────

    def step(self) -> Optional[IntelligenceFrame]:
        """Run all analyses and produce a unified :class:`IntelligenceFrame`.

        Causal graph is rebuilt only every 50 steps (expensive).  All other
        engines run every step.

        Returns ``None`` if no new frame has been fed since the last step
        (i.e. :meth:`feed` has not been called since the previous :meth:`step`).

        Returns
        -------
        IntelligenceFrame or None
        """
        if not self._frame_since_last_step:
            return None
        self._frame_since_last_step = False

        self._frame_counter += 1
        timestamp = time.time()

        # 1. Regime detection
        regime = None
        if self._regime_detector is not None:
            try:
                regime = self._regime_detector.detect()
            except Exception:
                regime = None

        # 2. Anomaly detection
        anomalies: List[Any] = []
        if self._anomaly_detector is not None:
            try:
                anomalies = self._anomaly_detector.detect() or []
            except Exception:
                anomalies = []

        # Feed anomalies to health monitor
        for anomaly in anomalies:
            self.feed_anomaly(anomaly)

        # Feed transition signal to health monitor
        if regime is not None:
            self.feed_transition(regime)

        # 3. Causal graph (expensive — only every 50 steps)
        causal_graph = None
        if self._causal_builder is not None:
            self._causal_step_counter += 1
            if self._causal_step_counter >= 50:
                self._causal_step_counter = 0
                try:
                    causal_graph = self._causal_builder.build_graph()
                except Exception:
                    causal_graph = None

        # 4. Vector compression (train once, then embed)
        compressed_state = None
        if self._vector_compressor is not None:
            try:
                if not getattr(self._vector_compressor, 'is_trained', False):
                    self._vector_compressor.train()
                compressed_state = self._vector_compressor.get_regime_embedding()
            except Exception:
                compressed_state = None

        # 5. System health
        health = None
        if self._health_monitor is not None:
            try:
                health = self._health_monitor.compute()
            except Exception:
                health = None

        # Derive priority and summary
        priority = self._determine_priority(regime, anomalies, health)
        summary = self._generate_summary(
            regime, anomalies, causal_graph, compressed_state, health, priority,
        )

        frame = IntelligenceFrame(
            frame_id=self._frame_counter,
            timestamp=timestamp,
            regime=regime,
            anomalies=anomalies,
            causal_graph=causal_graph,
            compressed_state=compressed_state,
            health=health,
            summary=summary,
            priority=priority,
        )

        self._latest_frame = frame
        self._history.append(frame)
        return frame

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[IntelligenceFrame]:
        """Return the latest :class:`IntelligenceFrame` without re-running analysis.

        Returns
        -------
        IntelligenceFrame or None
        """
        return self._latest_frame

    def get_history(self, n: int = 10) -> List[IntelligenceFrame]:
        """Return the last *N* intelligence frames from the history buffer.

        Parameters
        ----------
        n : int
            Maximum number of frames to return.  If the history contains fewer
            than *n* frames, all are returned.

        Returns
        -------
        list[IntelligenceFrame]
        """
        frames = list(self._history)
        return frames[-n:] if n < len(frames) else frames

    # ── Internal: priority determination ─────────────────────────────────────

    @staticmethod
    def _determine_priority(
        regime: Any,
        anomalies: List[Any],
        health: Any,
    ) -> str:
        """Determine the overall priority from all analysis results.

        Priority logic (first match wins):
          1. **CRITICAL** — if any anomaly has severity ``"CRITICAL"`` OR
             health state is ``"CRITICAL"``.
          2. **HIGH** — if any anomaly has severity ``"HIGH"`` OR health
             state is ``"DEGRADED"``.
          3. **MEDIUM** — if any anomaly has severity ``"MEDIUM"`` OR a
             regime transition signal is present.
          4. **LOW** — otherwise.

        Parameters
        ----------
        regime : TransitionSignal or None
            Detected regime transition signal (``None`` if none).
        anomalies : list[AnomalyEvent]
            Anomaly events from the anomaly detector.
        health : SystemHealthScore or None
            Health assessment from the health monitor (``None`` if unavailable).

        Returns
        -------
        str
            One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
        """
        # ── CRITICAL ─────────────────────────────────────────────────────
        for a in anomalies:
            severity = getattr(a, 'severity', None)
            if severity == 'CRITICAL':
                return 'CRITICAL'

        if health is not None:
            state = getattr(health, 'state', None)
            if state is not None:
                state_val = _resolve_state(state)
                if state_val == 'CRITICAL':
                    return 'CRITICAL'

        # ── HIGH ──────────────────────────────────────────────────────────
        for a in anomalies:
            severity = getattr(a, 'severity', None)
            if severity == 'HIGH':
                return 'HIGH'

        if health is not None:
            state = getattr(health, 'state', None)
            if state is not None:
                state_val = _resolve_state(state)
                if state_val == 'DEGRADED':
                    return 'HIGH'

        # ── MEDIUM ────────────────────────────────────────────────────────
        for a in anomalies:
            severity = getattr(a, 'severity', None)
            if severity == 'MEDIUM':
                return 'MEDIUM'

        if regime is not None:
            return 'MEDIUM'

        # ── LOW ───────────────────────────────────────────────────────────
        return 'LOW'

    # ── Internal: summary generation ─────────────────────────────────────────

    @staticmethod
    def _generate_summary(
        regime: Any,
        anomalies: List[Any],
        causal_graph: Any,
        compressed_state: Any,
        health: Any,
        priority: str,
    ) -> str:
        """Generate a human-readable one-line summary of the intelligence frame.

        Examples
        --------
        - ``"LOW | No transition signal | Anomalies: 0 | Health: +0.72"``
        - ``"CRITICAL | Regime transition SHADOW→MICRO (p=0.83) | 2 anomalies"``
        - ``"HIGH | Entropy collapse detected | Health: -0.45"``

        Parameters
        ----------
        regime : TransitionSignal or None
        anomalies : list[AnomalyEvent]
        causal_graph : CausalGraph or None
        compressed_state : CompressedStateVector or None
        health : SystemHealthScore or None
        priority : str

        Returns
        -------
        str
        """
        parts: List[str] = [priority]

        # ── Regime ───────────────────────────────────────────────────────
        if regime is not None:
            from_r = getattr(regime, 'from_regime', '?')
            to_r = getattr(regime, 'to_regime', '?')
            prob = getattr(regime, 'probability', 0.0)
            parts.append(f"Regime transition {from_r}→{to_r} (p={prob:.2f})")
        else:
            parts.append("No transition signal")

        # ── Anomalies ────────────────────────────────────────────────────
        anomaly_count = len(anomalies)
        if anomaly_count > 0:
            worst = max(
                anomalies,
                key=lambda a: (
                    getattr(a, 'score', 0) if getattr(a, 'score', None) is not None else 0
                ),
            )
            desc = getattr(worst, 'description', '')
            if desc:
                if len(desc) > 60:
                    desc = desc[:57] + '...'
                parts.append(f"Anomalies: {anomaly_count} ({desc})")
            else:
                parts.append(f"Anomalies: {anomaly_count}")
        else:
            parts.append("Anomalies: 0")

        # ── Health ───────────────────────────────────────────────────────
        if health is not None:
            score = getattr(health, 'score', 0.0)
            if score >= 0:
                parts.append(f"Health: +{score:.2f}")
            else:
                parts.append(f"Health: {score:.2f}")
            state = getattr(health, 'state', None)
            if state is not None:
                state_val = _resolve_state(state)
                parts.append(f"State: {state_val}")
        else:
            parts.append("Health: N/A")

        # ── Causal graph (if present) ────────────────────────────────────
        if causal_graph is not None:
            edges = getattr(causal_graph, 'edges', [])
            parts.append(f"Causal edges: {len(edges)}")

        # ── Compression (if present) ─────────────────────────────────────
        if compressed_state is not None:
            ev = getattr(compressed_state, 'explained_variance', 0.0)
            parts.append(f"Compression: {ev:.1%}")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Module-level helper: resolve a health state to a string
# ---------------------------------------------------------------------------


def _resolve_state(state: Any) -> str:
    """Extract a plain string from a health state value.

    Handles both ``HealthState`` enum instances and plain strings.
    """
    if isinstance(state, str):
        return state
    if hasattr(state, 'value'):
        return str(state.value)
    return str(state)
