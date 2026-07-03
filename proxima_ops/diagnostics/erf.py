import json
import logging
import math
from collections import deque
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.erf")

# ---------------------------------------------------------------------------
# MoF readiness mapping
# ---------------------------------------------------------------------------
_MOF_READINESS_MAP: dict[str, float] = {
    "INFORMATION_RICH": 0.9,
    "STRUCTURE_LIMITED": 0.5,
    "NOISE": 0.1,
}

# ---------------------------------------------------------------------------
# ERF weights
# ---------------------------------------------------------------------------
_W_SIGNAL_PERSISTENCE = 0.30
_W_VOLATILITY_COMPRESSION = 0.25
_W_CROSS_SYMBOL_ALIGNMENT = 0.25
_W_MOF_READINESS = 0.20

_DEFAULT_PHASE_THRESHOLD = 0.7


class ExecutionReadinessField:
    """Continuous scalar field of execution readiness.

    ERF(t) is a weighted combination of four sub-fields:
      - signal_persistence (30 %)
      - volatility_compression (25 %)
      - cross_symbol_alignment (25 %)
      - mof_readiness (20 %)

    Produces a value in [0.0, 1.0].
    """

    def __init__(
        self, log_path: str = "state/wave12_cycle_log.jsonl", window: int = 20
    ) -> None:
        self.log_path = log_path
        self.window = window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        """Run ERF analysis over recent cycles from the cycle log.

        Parameters
        ----------
        n_recent_cycles : int
            How many of the most recent log entries to consider.

        Returns
        -------
        dict
            Schema defined in the task specification.
        """
        try:
            records = self._load_records(n_recent_cycles)
            if not records:
                logger.warning("No cycle records loaded — returning empty ERF.")
                return self._empty_result()

            # Compute per-cycle ERF values
            erf_trajectory: dict[int, float] = {}
            # Rolling window buffers
            active_signals_window: deque[int] = deque(maxlen=self.window)
            erp_pressure_window: deque[float] = deque(maxlen=self.window)
            spread_window: deque[float] = deque(maxlen=self.window)

            # Track threshold crossings and trades on high ERF
            threshold_crossings = 0
            trades_on_high_erf = 0

            # Per-regime accumulation
            regime_samples: dict[str, list[float]] = {}

            # Determine phase threshold from execute cycles
            threshold = self._compute_phase_threshold(records)

            prev_above = False

            for i, rec in enumerate(records):
                try:
                    # -- Sub-field 1: signal_persistence --
                    active_sigs = rec.get("active_signals", 0) or 0
                    active_signals_window.append(1 if active_sigs > 0 else 0)
                    signal_persistence = (
                        sum(active_signals_window) / len(active_signals_window)
                        if active_signals_window
                        else 0.0
                    )

                    # -- Sub-field 2: volatility_compression --
                    # Use erp_pressure as a spread proxy (high pressure -> wide spread).
                    erp_pressure = rec.get("erp_pressure", 0.5) or 0.5
                    spread_proxy = 1.0 - erp_pressure  # invert: high pressure => low compression
                    erp_pressure_window.append(erp_pressure)
                    spread_window.append(spread_proxy)
                    max_spread_in_window = max(spread_window) if spread_window else 1.0
                    if max_spread_in_window > 0:
                        volatility_compression = 1.0 - (
                            spread_proxy / max_spread_in_window
                        )
                    else:
                        volatility_compression = 1.0

                    # -- Sub-field 3: cross_symbol_alignment --
                    regime_dashboard = rec.get("regime_dashboard", {}) or {}
                    directions: list[int] = []
                    for sym_data in regime_dashboard.values():
                        if isinstance(sym_data, dict):
                            d = sym_data.get("direction", 0)
                            if isinstance(d, (int, float)):
                                directions.append(int(d))
                    if directions:
                        # Majority direction count
                        pos = sum(1 for d in directions if d > 0)
                        neg = sum(1 for d in directions if d < 0)
                        neutral = sum(1 for d in directions if d == 0)
                        majority = max(pos, neg, neutral)
                        cross_symbol_alignment = majority / len(directions)
                    else:
                        cross_symbol_alignment = 0.5  # neutral when no data

                    # -- Sub-field 4: mof_readiness --
                    mof_state = rec.get("mof_state", "NOISE") or "NOISE"
                    mof_readiness = _MOF_READINESS_MAP.get(
                        mof_state.upper(), 0.1
                    )

                    # -- Composite ERF --
                    erf = (
                        _W_SIGNAL_PERSISTENCE * signal_persistence
                        + _W_VOLATILITY_COMPRESSION * volatility_compression
                        + _W_CROSS_SYMBOL_ALIGNMENT * cross_symbol_alignment
                        + _W_MOF_READINESS * mof_readiness
                    )
                    erf = max(0.0, min(1.0, erf))

                    cycle_num = rec.get("cycle", i)
                    erf_trajectory[cycle_num] = round(erf, 4)

                    # -- Threshold crossing detection --
                    is_above = erf >= threshold
                    if is_above and not prev_above:
                        threshold_crossings += 1
                    prev_above = is_above

                    # -- Trades on high ERF --
                    decision = rec.get("decision", "HOLD") or "HOLD"
                    if is_above and decision == "EXECUTE":
                        trades_on_high_erf += 1

                    # -- Per-regime accumulation --
                    regime_label = rec.get("regime", "UNKNOWN") or "UNKNOWN"
                    regime_samples.setdefault(regime_label, []).append(erf)

                except Exception as inner:
                    logger.debug(
                        "Skipping cycle %s due to error: %s",
                        rec.get("cycle", "?"),
                        inner,
                    )
                    continue

            # -- Aggregate statistics --
            erf_values = list(erf_trajectory.values())
            if erf_values:
                erf_mean = mean(erf_values)
                erf_std = stdev(erf_values) if len(erf_values) > 1 else 0.0
                erf_min = min(erf_values)
                erf_max = max(erf_values)
            else:
                erf_mean = erf_std = erf_min = erf_max = 0.0

            # -- Regime breakdown --
            erf_by_regime: dict[str, dict[str, float]] = {}
            for regime_label, vals in regime_samples.items():
                if vals:
                    r_mean = mean(vals)
                    r_std = stdev(vals) if len(vals) > 1 else 0.0
                else:
                    r_mean = r_std = 0.0
                erf_by_regime[regime_label] = {
                    "mean": round(r_mean, 4),
                    "std": round(r_std, 4),
                }

            return {
                "erf_trajectory": erf_trajectory,
                "erf_statistics": {
                    "mean": round(erf_mean, 4),
                    "std": round(erf_std, 4),
                    "min": round(erf_min, 4),
                    "max": round(erf_max, 4),
                },
                "threshold_crossings": threshold_crossings,
                "trades_on_high_erf": trades_on_high_erf,
                "erf_by_regime": erf_by_regime,
                "phase_transition_threshold": round(threshold, 4),
            }

        except Exception as exc:
            logger.error("ERF analysis failed: %s", exc, exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internals
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
                        logger.debug("Skipping malformed JSONL line: %.80s", stripped)
                        continue
            # Return the most recent N
            return records[-n_recent:] if n_recent < len(records) else records
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Error loading cycle log: %s", exc)
            return []

    def _compute_phase_threshold(
        self, records: list[dict[str, Any]]
    ) -> float:
        """Determine phase transition threshold.

        If any cycle has decision == "EXECUTE", use the median ERF
        of those execute cycles.  Otherwise default to 0.7.
        """
        try:
            execute_cycles = [
                rec for rec in records if rec.get("decision") == "EXECUTE"
            ]
            if not execute_cycles:
                return _DEFAULT_PHASE_THRESHOLD

            # Compute ERF for each execute cycle and take median
            erf_values: list[float] = []
            for rec in execute_cycles:
                try:
                    active_sigs = rec.get("active_signals", 0) or 0
                    erp_pressure = rec.get("erp_pressure", 0.5) or 0.5
                    mof_state = rec.get("mof_state", "NOISE") or "NOISE"
                    mof_readiness = _MOF_READINESS_MAP.get(
                        mof_state.upper(), 0.1
                    )

                    # Simplified signal_persistence over the single record
                    sig_persist = 1.0 if active_sigs > 0 else 0.0

                    # Simplified volatility_compression
                    spread_proxy = 1.0 - erp_pressure
                    volatility_compression = spread_proxy  # max == 1.0 here

                    # Simplified cross_symbol_alignment
                    dashboard = rec.get("regime_dashboard", {}) or {}
                    directions = []
                    for sd in dashboard.values():
                        if isinstance(sd, dict):
                            d = sd.get("direction", 0)
                            if isinstance(d, (int, float)):
                                directions.append(int(d))
                    if directions:
                        pos = sum(1 for d in directions if d > 0)
                        neg = sum(1 for d in directions if d < 0)
                        neutral = sum(1 for d in directions if d == 0)
                        majority = max(pos, neg, neutral)
                        alignment = majority / len(directions)
                    else:
                        alignment = 0.5

                    erf = (
                        _W_SIGNAL_PERSISTENCE * sig_persist
                        + _W_VOLATILITY_COMPRESSION * volatility_compression
                        + _W_CROSS_SYMBOL_ALIGNMENT * alignment
                        + _W_MOF_READINESS * mof_readiness
                    )
                    erf = max(0.0, min(1.0, erf))
                    erf_values.append(erf)
                except Exception:
                    continue

            if not erf_values:
                return _DEFAULT_PHASE_THRESHOLD

            erf_values.sort()
            n = len(erf_values)
            if n % 2 == 1:
                median = erf_values[n // 2]
            else:
                median = (erf_values[n // 2 - 1] + erf_values[n // 2]) / 2.0
            return median

        except Exception as exc:
            logger.debug("Error computing phase threshold: %s", exc)
            return _DEFAULT_PHASE_THRESHOLD

    @staticmethod
    def _empty_result() -> dict:
        """Return an empty result dict for error / no-data cases."""
        return {
            "erf_trajectory": {},
            "erf_statistics": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "threshold_crossings": 0,
            "trades_on_high_erf": 0,
            "erf_by_regime": {},
            "phase_transition_threshold": _DEFAULT_PHASE_THRESHOLD,
        }
