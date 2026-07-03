"""
RFG — Recovery Field Gradient

Recovery potential scalar field. High = system easily returns to trading flow.
Low = stuck in degraded attractor.

RFG(t) is a weighted combination of four inverted degradation factors:
  - CB_persistence_decay   (30 %)
  - drift_stabilization    (25 %)
  - latency_compression    (25 %)
  - signal_coherence       (20 %)

Each factor is in [0.0, 1.0] where high = good (healthy / recovered).
RFG is the weighted sum, also in [0.0, 1.0].
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import deque
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.rfg")

# ---------------------------------------------------------------------------
# RFG weights
# ---------------------------------------------------------------------------
_W_CB_PERSISTENCE = 0.30
_W_DRIFT_STABILIZATION = 0.25
_W_LATENCY_COMPRESSION = 0.25
_W_SIGNAL_COHERENCE = 0.20

# ---------------------------------------------------------------------------
# Default window for rolling degradation factors
# ---------------------------------------------------------------------------
_DEFAULT_WINDOW = 20
_MAX_CB_WINDOW = 50  # max cycles we look back for CB persistence decay


class RecoveryFieldGradient:
    """Continuous scalar field of recovery potential.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log (default
        ``"state/wave12_cycle_log.jsonl"``).
    window : int
        Rolling window size for computing stability / coherence
        metrics (default 20).
    """

    def __init__(
        self, log_path: str = "state/wave12_cycle_log.jsonl", window: int = 20
    ) -> None:
        self.log_path = log_path
        self.window = window if window > 0 else _DEFAULT_WINDOW

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the full RFG analysis.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            Schema defined in the class docstring / task specification.
        """
        try:
            records = self._load_records(n_recent_cycles)
        except Exception as exc:
            logger.error("Failed to load cycle log: %s", exc)
            return self._empty_result()

        if not records:
            logger.warning("No cycle records loaded — returning empty RFG.")
            return self._empty_result()

        try:
            return self._compute(records)
        except Exception as exc:
            logger.error("RFG analysis failed: %s", exc, exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Core analysis logic on pre-loaded records."""
        # ── 1. Per-cycle RFG trajectory ────────────────────────────────
        rfg_trajectory: dict[str, float] = {}  # "cycle_N" -> float
        rfg_by_cycle: dict[int, float] = {}    # cycle_num -> float (internal)

        # Rolling buffers (aligned with each cycle)
        cb_active_window: deque[bool] = deque(maxlen=self.window)
        mof_states_window: deque[str] = deque(maxlen=self.window)
        confirm_window: deque[int] = deque(maxlen=self.window)
        direction_window: deque[str] = deque(maxlen=self.window)

        # Per-regime accumulation (using segl_state: ARMED / OBSERVE)
        regime_samples: dict[str, list[float]] = {}

        # Track CB-active periods for time-to-recovery calculation
        # cb_active_map: cycle_num -> bool (whether CB was active that cycle)
        cb_active_map: dict[int, bool] = {}

        for i, rec in enumerate(records):
            try:
                cycle_num = rec.get("cycle", i)
                if not isinstance(cycle_num, int):
                    cycle_num = i

                # ── Determine CB active status ─────────────────────
                denial_reason = str(rec.get("denial_reason", "") or "")
                cb_active_now = "CircuitBreaker" in denial_reason or "circuit breaker" in denial_reason.lower()
                cb_active_map[cycle_num] = cb_active_now

                # ── Sub-factor 1: CB_persistence_decay (30 %) ──────
                # = 1.0 if no CB recently; otherwise decays with distance from last CB
                cb_active_window.append(cb_active_now)
                cb_persistence = self._cb_persistence_decay(
                    cb_active_window, cycle_num, records, i
                )

                # ── Sub-factor 2: drift_stabilization (25 %) ───────
                # Use mof_state stability: count changes in the rolling window
                mof_state = str(rec.get("mof_state", "NOISE") or "NOISE").upper()
                mof_states_window.append(mof_state)
                drift_stabilization = self._drift_stabilization(
                    mof_states_window, rec
                )

                # ── Sub-factor 3: latency_compression (25 %) ───────
                # Inverted proxy of confirm_cycles (high confirm = bad)
                confirm_cycles = rec.get("confirm_cycles", 0) or 0
                if not isinstance(confirm_cycles, (int, float)):
                    confirm_cycles = 0
                confirm_window.append(int(confirm_cycles))
                latency_compression = self._latency_compression(confirm_window)

                # ── Sub-factor 4: signal_coherence (20 %) ──────────
                # Fraction of cycles in window with consistent direction
                active_direction = str(rec.get("active_direction", "") or "")
                direction_window.append(active_direction)
                signal_coherence = self._signal_coherence(direction_window)

                # ── Composite RFG ──────────────────────────────────
                rfg = (
                    _W_CB_PERSISTENCE * cb_persistence
                    + _W_DRIFT_STABILIZATION * drift_stabilization
                    + _W_LATENCY_COMPRESSION * latency_compression
                    + _W_SIGNAL_COHERENCE * signal_coherence
                )
                rfg = max(0.0, min(1.0, rfg))

                rfg_trajectory[f"cycle_{cycle_num}"] = round(rfg, 4)
                rfg_by_cycle[cycle_num] = rfg

                # ── Per-regime accumulation (by segl_state) ────────
                segl = str(rec.get("segl_state", "OBSERVE") or "OBSERVE").upper()
                regime_samples.setdefault(segl, []).append(rfg)

            except Exception as inner:
                logger.debug(
                    "Skipping cycle %s due to error: %s",
                    rec.get("cycle", "?"),
                    inner,
                )
                continue

        # ── 2. Aggregate statistics ────────────────────────────────────
        rfg_values = list(rfg_by_cycle.values())
        if rfg_values:
            mean_rfg = statistics.mean(rfg_values)
        else:
            mean_rfg = 0.0

        # ── 3. RFG by regime (segl_state) ─────────────────────────────
        rfg_by_regime: dict[str, dict[str, float]] = {}
        for regime_label in ("ARMED", "OBSERVE"):
            vals = regime_samples.get(regime_label, [])
            if vals:
                r_mean = statistics.mean(vals)
                r_std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            else:
                r_mean = r_std = 0.0
            rfg_by_regime[regime_label] = {
                "mean": round(r_mean, 4),
                "std": round(r_std, 4),
            }

        # ── 4. Low / high recovery cycle counts ──────────────────────
        low_recovery_cycles = sum(1 for v in rfg_values if v < 0.3)
        high_recovery_cycles = sum(1 for v in rfg_values if v > 0.7)

        # ── 5. Time to recovery after CB trigger ─────────────────────
        time_to_recovery = self._compute_time_to_recovery(
            records, cb_active_map, rfg_by_cycle
        )

        return {
            "rfg_trajectory": rfg_trajectory,
            "mean_rfg": round(mean_rfg, 4),
            "rfg_by_regime": rfg_by_regime,
            "low_recovery_cycles": low_recovery_cycles,
            "high_recovery_cycles": high_recovery_cycles,
            "time_to_recovery_after_cb": time_to_recovery,
        }

    # ------------------------------------------------------------------
    # Sub-factor computations
    # ------------------------------------------------------------------

    @staticmethod
    def _cb_persistence_decay(
        cb_window: deque[bool],
        current_cycle: int,
        records: list[dict[str, Any]],
        current_idx: int,
    ) -> float:
        """Compute CB persistence decay factor.

        Returns 1.0 if no CB was active in the window. Otherwise
        returns 1.0 - (cycles_since_last_cb / max_cb_window), clamped
        to [0.0, 1.0].
        """
        # If no CB ever active in window, full recovery
        if not any(cb_window):
            return 1.0

        # Walk backwards from current_idx to find the most recent cycle
        # where CB was active.
        for offset in range(current_idx, -1, -1):
            if offset < len(records):
                denial = str(records[offset].get("denial_reason", "") or "")
                if "CircuitBreaker" in denial or "circuit breaker" in denial.lower():
                    cycles_since = current_cycle - records[offset].get("cycle", offset)
                    if cycles_since <= 0:
                        cycles_since = current_idx - offset
                    decay = cycles_since / _MAX_CB_WINDOW
                    return max(0.0, 1.0 - decay)

        # Should not reach here if any(cb_window) is True, but defensive
        return 1.0

    @staticmethod
    def _drift_stabilization(
        mof_window: deque[str], current_record: dict[str, Any]
    ) -> float:
        """Compute drift stabilization factor.

        Measures MoF state stability by counting state changes in the
        rolling window. Also incorporates the reconciliation drift_score
        if available.

        Returns a value in [0.0, 1.0] where high = stable (good).
        """
        if len(mof_window) < 2:
            return 1.0

        # Count state transitions in the window
        changes = 0
        window_list = list(mof_window)
        for j in range(1, len(window_list)):
            if window_list[j] != window_list[j - 1]:
                changes += 1

        # Normalize changes to [0, 1]: at most (len-1) changes possible
        max_changes = len(window_list) - 1
        change_rate = changes / max_changes if max_changes > 0 else 0.0

        # Also use reconciliation drift_score as secondary signal
        drift_score = 0.0
        try:
            reconciliation = current_record.get("reconciliation", {}) or {}
            drift_score = float(reconciliation.get("drift_score", 0.0) or 0.0)
        except (ValueError, TypeError, AttributeError):
            drift_score = 0.0

        # Drift score is already in [0, ~1] — combine with change rate
        # Weighted: 60 % mof changes, 40 % drift_score
        drift_proxy = 0.60 * change_rate + 0.40 * drift_score
        drift_proxy = max(0.0, min(1.0, drift_proxy))

        return 1.0 - drift_proxy

    @staticmethod
    def _latency_compression(confirm_window: deque[int]) -> float:
        """Compute latency compression factor.

        Uses confirm_cycles as proxy. Normalises within the rolling
        window so that the maximum observed confirm_cycles maps to 0
        (worst) and 0 maps to 1 (best / fully compressed).

        Returns a value in [0.0, 1.0] where high = low latency (good).
        """
        if not confirm_window:
            return 1.0

        current = confirm_window[-1]
        max_confirm = max(confirm_window)

        if max_confirm <= 0:
            return 1.0

        # Clamp current to [0, max_confirm]
        current = max(0, min(current, max_confirm))
        return 1.0 - (current / max_confirm)

    @staticmethod
    def _signal_coherence(direction_window: deque[str]) -> float:
        """Compute signal coherence factor.

        Fraction of cycles in the rolling window where the active
        direction is consistent (non-empty and same sign). Non-empty
        directions that all agree = high coherence (good).

        Returns a value in [0.0, 1.0].
        """
        if len(direction_window) < 2:
            return 1.0

        # Filter out empty directions
        non_empty = [d for d in direction_window if d and d.strip()]
        if len(non_empty) < 2:
            # Too few data points — default to neutral
            return 0.5

        # Check if all directions are the same
        unique = set(d.upper().strip() for d in non_empty)
        if len(unique) == 1:
            return 1.0

        # Compute majority fraction
        from collections import Counter
        counts = Counter(d.upper().strip() for d in non_empty)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(non_empty)

    # ------------------------------------------------------------------
    # Time-to-recovery after CB
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_time_to_recovery(
        records: list[dict[str, Any]],
        cb_active_map: dict[int, bool],
        rfg_by_cycle: dict[int, float],
    ) -> dict[str, float]:
        """Compute average cycles to recover after a CB trigger ends.

        A CB trigger period starts when ``denial_reason`` first contains
        "CircuitBreaker" and ends when it no longer contains it.
        Recovery time = number of cycles after the CB period ends until
        decision != "HOLD".

        Returns
        -------
        dict with "mean", "median", "max" (0.0 if no CB events found).
        """
        try:
            # Build a list of (cycle_num, is_cb_active) sorted by cycle
            sorted_pairs = sorted(
                cb_active_map.items(), key=lambda x: x[0]
            )
            if not sorted_pairs:
                return {"mean": 0.0, "median": 0.0, "max": 0.0}

            # Detect CB periods: contiguous ranges where cb_active is True
            cb_periods: list[tuple[int, int]] = []  # (start_cycle, end_cycle)
            in_period = False
            period_start = 0

            for cycle_num, active in sorted_pairs:
                if active and not in_period:
                    in_period = True
                    period_start = cycle_num
                elif not active and in_period:
                    in_period = False
                    cb_periods.append((period_start, cycle_num))

            # Close any ongoing period
            if in_period:
                cb_periods.append((period_start, sorted_pairs[-1][0]))

            if not cb_periods:
                return {"mean": 0.0, "median": 0.0, "max": 0.0}

            # For each CB period, compute recovery time
            recovery_times: list[float] = []

            # Build a lookup: cycle_num -> decision
            decision_map: dict[int, str] = {}
            for rec in records:
                c = rec.get("cycle")
                if c is not None:
                    decision_map[c] = str(rec.get("decision", "HOLD") or "HOLD")

            for start_c, end_c in cb_periods:
                # Walk forward from end_c until decision != "HOLD"
                for recovery_cycle in range(end_c + 1, end_c + 500):
                    decision = decision_map.get(recovery_cycle, "HOLD")
                    if decision != "HOLD":
                        recovery_times.append(float(recovery_cycle - end_c))
                        break
                # If never found (max 500 cycles), skip

            if not recovery_times:
                return {"mean": 0.0, "median": 0.0, "max": 0.0}

            return {
                "mean": round(statistics.mean(recovery_times), 2),
                "median": round(statistics.median(recovery_times), 2),
                "max": round(max(recovery_times), 2),
            }

        except Exception as exc:
            logger.debug("Error computing time-to-recovery: %s", exc)
            return {"mean": 0.0, "median": 0.0, "max": 0.0}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_records(self, n_recent: int) -> list[dict[str, Any]]:
        """Load up to *n_recent* cycle log entries from the JSONL file."""
        try:
            records: list[dict[str, Any]] = []
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSONL line: %.80s", stripped
                        )
                        continue
            # Return the most recent N
            return records[-n_recent:] if n_recent < len(records) else records
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Error loading cycle log: %s", exc)
            return []

    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty result dict for error / no-data cases."""
        return {
            "rfg_trajectory": {},
            "mean_rfg": 0.0,
            "rfg_by_regime": {
                "ARMED": {"mean": 0.0, "std": 0.0},
                "OBSERVE": {"mean": 0.0, "std": 0.0},
            },
            "low_recovery_cycles": 0,
            "high_recovery_cycles": 0,
            "time_to_recovery_after_cb": {
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
            },
        }


# ======================================================================
# Quick CLI demo
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rfg = RecoveryFieldGradient()
    result = rfg.analyze(n_recent_cycles=500)
    print(json.dumps(result, indent=2, default=str))
