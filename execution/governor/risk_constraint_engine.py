"""
risk_constraint_engine.py — Mathematical safety constraints.

Applies max drawdown protection, exposure caps per regime, volatility
scaling, and entropy-based throttling to produce a RiskProfile used by
the ExecutionGovernor.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from struct import Struct
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Frame layout constants  (mirrors shared_memory_telemetry.py)
# ---------------------------------------------------------------------------

_HEADER_FORMAT = Struct("<QdQQQQ2Q")   # 64 bytes
_FRAME_FORMAT = Struct("<32f13f4x")    # 184 bytes

HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248

# Regime state-to-name mapping
_REGIME_NAMES: dict[int, str] = {
    0: "SHADOW",
    1: "MICRO",
    2: "FULL",
}

# ---------------------------------------------------------------------------
# RiskProfile
# ---------------------------------------------------------------------------


@dataclass
class RiskProfile:
    """Computed risk constraints for the current system state.

    Attributes
    ----------
    max_size : float
        Maximum position size fraction (0.0 to 1.0).
    exposure_limit : float
        Maximum risk limit (0.0 to 1.0).
    allowed_actions : list[str]
        List of allowed intent type names.
    volatility_multiplier : float
        0.0 to 1.0 (lower = more risk reduction).
    max_drawdown_risk : float
        Current drawdown-based risk cap (0.0 to 1.0).
    reasoning : list[str]
        Human-readable reasoning for each constraint.
    timestamp : float
        Unix timestamp of the evaluation.
    """
    max_size: float
    exposure_limit: float
    allowed_actions: list[str]
    volatility_multiplier: float
    max_drawdown_risk: float
    reasoning: list[str]
    timestamp: float


# ---------------------------------------------------------------------------
# RiskConstraintEngine
# ---------------------------------------------------------------------------


class RiskConstraintEngine:
    """Applies mathematical safety constraints based on SHM telemetry
    and optional intelligence context.

    The engine maintains rolling histories of raw frames and an equity-like
    proxy metric (stability).  On each call to :meth:`evaluate` it computes:

    * Max drawdown protection  (from stability proxy)
    * Exposure caps per regime (from intelligence_frame or frame scalars)
    * Volatility scaling       (from entropy variance)
    * Entropy-based throttling (from latest entropy)
    * Allowed actions          (regime + anomaly severity)

    Parameters
    ----------
    max_position_fraction : float
        Global maximum position size fraction.  Applied as an absolute upper
        bound after all other caps.
    max_exposure_fraction : float
        Global maximum exposure fraction.  Applied as an absolute upper
        bound after all other caps.
    drawdown_window : int
        Number of historical equity-proxy values used for drawdown
        calculation.
    volatility_window : int
        Number of historical entropy values used for volatility scaling.
    """

    def __init__(
        self,
        max_position_fraction: float = 1.0,
        max_exposure_fraction: float = 1.0,
        drawdown_window: int = 100,
        volatility_window: int = 50,
    ) -> None:
        self._max_position_fraction = max_position_fraction
        self._max_exposure_fraction = max_exposure_fraction
        self._drawdown_window = drawdown_window
        self._volatility_window = volatility_window

        # Raw 432-byte SHM frame history (stores last N frames)
        self._frame_history: list[bytes] = []

        # Equity-like proxy history (derived from stability scalar)
        self._equity_history: list[float] = []

    # ── Feed ─────────────────────────────────────────────────────────────────

    def feed(self, frame: bytes) -> None:
        """Feed a raw 432-byte SHM frame for historical tracking.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore``.
        """
        if len(frame) < HEADER_SIZE + FRAME_SIZE:
            return  # malformed frame — silently ignore

        self._frame_history.append(frame)

        # Extract stability as equity proxy
        stability = self._extract_stability(frame)
        self._equity_history.append(stability)

        # Trim histories to configured windows
        if len(self._frame_history) > self._drawdown_window:
            self._frame_history = self._frame_history[-self._drawdown_window:]
        if len(self._equity_history) > self._drawdown_window:
            self._equity_history = self._equity_history[-self._drawdown_window:]

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        intelligence_frame: Any = None,
        execution_intent: Any = None,
    ) -> RiskProfile:
        """Compute the current risk profile from recent history and signals.

        Parameters
        ----------
        intelligence_frame : Any, optional
            Duck-typed intelligence context.  May carry attributes:

            * ``.health`` — a ``SystemHealthScore`` (or duck) with ``.regime``
            * ``.regime`` — direct regime name string
            * ``.regime_state`` — numeric regime state (0, 1, 2)
            * ``.anomalies`` — list of anomaly objects with ``.severity``
            * ``.anomaly_severity`` — direct max-severity integer

        execution_intent : Any, optional
            Not used directly by this engine; present for interface
            compatibility with the governance pipeline.

        Returns
        -------
        RiskProfile
            The computed risk constraints.
        """
        reasoning: list[str] = []

        # ------------------------------------------------------------------
        # 1 — Current equity proxy
        # ------------------------------------------------------------------
        current_equity: float = 0.5
        if self._equity_history:
            current_equity = self._equity_history[-1]

        # ------------------------------------------------------------------
        # 2 — Max Drawdown Protection
        # ------------------------------------------------------------------
        drawdown_vals: list[float] = self._equity_history
        if len(drawdown_vals) > self._drawdown_window:
            drawdown_vals = drawdown_vals[-self._drawdown_window:]

        if drawdown_vals:
            rolling_max_val = max(drawdown_vals)
            denom = max(rolling_max_val, 0.001)
            current_drawdown = (rolling_max_val - current_equity) / denom
            max_drawdown_risk = max(0.0, min(1.0, 1.0 - (current_drawdown * 2.0)))
        else:
            current_drawdown = 0.0
            max_drawdown_risk = 1.0

        reasoning.append(
            f"drawdown_protection: drawdown={current_drawdown:.4f}, "
            f"max_drawdown_risk={max_drawdown_risk:.4f}"
        )

        # ------------------------------------------------------------------
        # 3 — Exposure Caps Per Regime
        # ------------------------------------------------------------------
        regime, regime_state = self._resolve_regime(intelligence_frame)

        if regime == "SHADOW":
            base_max_size = 0.1
            base_exposure_limit = 0.1
        elif regime == "MICRO":
            base_max_size = 0.4
            base_exposure_limit = 0.3
        elif regime == "FULL":
            base_max_size = 1.0
            base_exposure_limit = 0.8
        else:  # UNKNOWN
            base_max_size = 0.05
            base_exposure_limit = 0.05

        reasoning.append(
            f"regime_cap: regime={regime} (state={regime_state}), "
            f"base_max_size={base_max_size:.4f}, "
            f"base_exposure_limit={base_exposure_limit:.4f}"
        )

        # ------------------------------------------------------------------
        # 4 — Volatility Scaling
        # ------------------------------------------------------------------
        entropy_values: list[float] = [
            self._extract_entropy(f)
            for f in self._frame_history[-self._volatility_window:]
        ]

        if len(entropy_values) >= 2:
            mean_e = sum(entropy_values) / len(entropy_values)
            variance = sum((v - mean_e) ** 2 for v in entropy_values) / len(entropy_values)
            std_e = math.sqrt(variance)
            volatility = std_e / max(mean_e, 0.001)
            volatility_multiplier = max(0.1, min(1.0, 1.0 - volatility))
        else:
            volatility = 0.0
            volatility_multiplier = 1.0

        reasoning.append(
            f"volatility_scaling: volatility={volatility:.4f}, "
            f"volatility_multiplier={volatility_multiplier:.4f}"
        )

        # ------------------------------------------------------------------
        # 5 — Entropy-Based Throttling
        # ------------------------------------------------------------------
        latest_entropy: float = entropy_values[-1] if entropy_values else 1.0

        if latest_entropy < 0.2:
            throttle = 0.1
        elif latest_entropy < 0.3:
            throttle = 0.3
        elif latest_entropy < 0.5:
            throttle = 0.6
        elif latest_entropy < 0.7:
            throttle = 0.8
        else:
            throttle = 1.0

        reasoning.append(
            f"entropy_throttle: entropy={latest_entropy:.4f}, "
            f"throttle={throttle:.4f}"
        )

        # ------------------------------------------------------------------
        # 6 — Allowed Actions
        # ------------------------------------------------------------------
        allowed_actions = self._compute_allowed_actions(
            regime, intelligence_frame, latest_entropy
        )
        reasoning.append(f"allowed_actions: {allowed_actions}")

        # ------------------------------------------------------------------
        # 7 — Final Composite Calculation
        # ------------------------------------------------------------------
        max_size = (
            base_max_size
            * volatility_multiplier
            * max_drawdown_risk
            * throttle
        )
        exposure_limit = (
            base_exposure_limit
            * volatility_multiplier
            * max_drawdown_risk
            * throttle
        )

        # Apply global caps
        max_size = min(max_size, self._max_position_fraction)
        exposure_limit = min(exposure_limit, self._max_exposure_fraction)

        # Clamp to [0.0, 1.0]
        max_size = max(0.0, min(1.0, max_size))
        exposure_limit = max(0.0, min(1.0, exposure_limit))

        reasoning.append(
            f"composite: max_size={max_size:.6f}, "
            f"exposure_limit={exposure_limit:.6f}"
        )

        return RiskProfile(
            max_size=max_size,
            exposure_limit=exposure_limit,
            allowed_actions=allowed_actions,
            volatility_multiplier=volatility_multiplier,
            max_drawdown_risk=max_drawdown_risk,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    # ── Internal helpers — scalar extraction ─────────────────────────────────-

    @staticmethod
    def _extract_stability(frame: bytes) -> float:
        """Extract the stability scalar (index 1 of 13-float scalars array)
        from a raw 432-byte SHM frame.

        Stability serves as an equity-like proxy for drawdown calculations.
        """
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = int(hdr[2])
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)
        return float(raw[33])  # stability at scalars[1]

    @staticmethod
    def _extract_entropy(frame: bytes) -> float:
        """Extract the entropy scalar (index 2 of the 13-float scalars array)
        from a raw 432-byte SHM frame.
        """
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = int(hdr[2])
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)
        return float(raw[34])  # entropy at scalars[2]

    @staticmethod
    def _extract_regime_state(frame: bytes) -> float:
        """Extract the regime_state scalar (index 3 of the 13-float scalars
        array) from a raw 432-byte SHM frame.
        """
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = int(hdr[2])
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)
        return float(raw[35])  # regime_state at scalars[3]

    # ── Internal helpers — regime resolution ────────────────────────────────-

    def _resolve_regime(self, intelligence_frame: Any) -> tuple[str, int]:
        """Resolve the current regime name and numeric state.

        Priority:
        1. ``intelligence_frame.health.regime`` (string)
        2. ``intelligence_frame.regime`` (string)
        3. ``intelligence_frame.regime_state`` (int)
        4. Latest frame's regime_state scalar
        5. Default ``("UNKNOWN", 3)``

        Returns
        -------
        tuple[str, int]
            Regime name and numeric state.
        """
        regime: str = "UNKNOWN"
        regime_state: int = 3  # 3 = UNKNOWN sentinel

        if intelligence_frame is not None:
            # Duck-typed health attribute
            health = getattr(intelligence_frame, "health", None)
            if health is not None:
                health_regime = getattr(health, "regime", None)
                if health_regime is not None:
                    regime = str(health_regime)

            # Direct regime attribute (overrides health regime)
            direct_regime = getattr(intelligence_frame, "regime", None)
            if direct_regime is not None:
                regime = str(direct_regime)

            # Direct regime_state attribute (overrides everything)
            state_attr = getattr(intelligence_frame, "regime_state", None)
            if state_attr is not None:
                regime_state = int(state_attr)
                regime = _REGIME_NAMES.get(regime_state, regime)

        # Fallback: use latest frame's regime_state
        if regime == "UNKNOWN" and self._frame_history:
            latest_state_raw = self._extract_regime_state(self._frame_history[-1])
            regime_state = int(latest_state_raw)
            regime = _REGIME_NAMES.get(regime_state, regime)

        return regime, regime_state

    # ── Internal helpers — anomaly severity ──────────────────────────────────

    @staticmethod
    def _get_anomaly_severity(intelligence_frame: Any) -> int:
        """Extract the maximum anomaly severity as an integer.

        Severity mapping (numeric):
            * 0 = no anomaly / unknown
            * 1 = LOW
            * 2 = MEDIUM
            * 3 = HIGH / CRITICAL

        Checks in order:
        1. ``intelligence_frame.anomaly_severity`` (direct int)
        2. ``intelligence_frame.anomalies`` (list of duck-typed objects)

        Returns
        -------
        int
            The maximum severity found, or 0 if none.
        """
        if intelligence_frame is None:
            return 0

        # Direct severity attribute
        direct = getattr(intelligence_frame, "anomaly_severity", None)
        if direct is not None:
            try:
                return int(direct)
            except (ValueError, TypeError):
                pass

        # List of anomalies
        anomalies = getattr(intelligence_frame, "anomalies", None)
        if anomalies is not None and isinstance(anomalies, list) and anomalies:
            max_sev = 0
            for anomaly in anomalies:
                sev = getattr(anomaly, "severity", None)
                if sev is None:
                    continue
                try:
                    sev_int = int(sev)
                    max_sev = max(max_sev, sev_int)
                except (ValueError, TypeError):
                    sev_str = str(sev).upper()
                    if sev_str in ("HIGH", "CRITICAL"):
                        max_sev = max(max_sev, 3)
                    elif sev_str == "MEDIUM":
                        max_sev = max(max_sev, 2)
                    elif sev_str == "LOW":
                        max_sev = max(max_sev, 1)
            return max_sev

        return 0

    # ── Internal helpers — allowed actions ───────────────────────────────────

    def _compute_allowed_actions(
        self,
        regime: str,
        intelligence_frame: Any,
        latest_entropy: float,
    ) -> list[str]:
        """Determine the list of allowed actions based on regime, anomaly
        severity, and entropy level.

        Rules (applied in order, most restrictive wins):
        1. If entropy < 0.2 (collapse zone) → only reduce/exit actions
        2. If anomaly severity >= 3 (HIGH)   → only reduce/exit actions
        3. If anomaly severity == 2 (MEDIUM) → no STRONG_BUY, no SELL_SHORT
        4. If regime == SHADOW              → only HOLD / REDUCE_LIGHT /
                                               TRANSITION_PREP / EMERGENCY_STOP
        5. If regime == MICRO               → no STRONG_BUY, no SELL_SHORT,
                                               no EXIT_ALL
        6. If regime == FULL                → all actions allowed
        """
        # --- Collapse / HIGH anomaly override ---
        anomaly_severity = self._get_anomaly_severity(intelligence_frame)

        if latest_entropy < 0.2 or anomaly_severity >= 3:
            return [
                "HOLD",
                "REDUCE_LIGHT",
                "REDUCE_MODERATE",
                "REDUCE_STRONG",
                "EXIT_ALL",
                "EMERGENCY_STOP",
            ]

        # --- Start with all actions ---
        all_actions: list[str] = [
            "HOLD",
            "BUY",
            "SELL",
            "STRONG_BUY",
            "SELL_SHORT",
            "REDUCE_LIGHT",
            "REDUCE_MODERATE",
            "REDUCE_STRONG",
            "EXIT_ALL",
            "EMERGENCY_STOP",
            "TRANSITION_PREP",
        ]

        # --- MEDIUM anomaly restrictions ---
        if anomaly_severity == 2:
            all_actions = [
                a for a in all_actions
                if a not in ("STRONG_BUY", "SELL_SHORT")
            ]

        # --- Regime-based restrictions ---
        if regime == "SHADOW":
            return [
                "HOLD",
                "REDUCE_LIGHT",
                "TRANSITION_PREP",
                "EMERGENCY_STOP",
            ]

        if regime == "MICRO":
            all_actions = [
                a for a in all_actions
                if a not in ("STRONG_BUY", "SELL_SHORT", "EXIT_ALL")
            ]

        # FULL — all actions remain (no additional restrictions)

        return all_actions
