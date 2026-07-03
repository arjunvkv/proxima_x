"""PreTradeSnapshot: Single canonical decision object per cycle.

Pure data aggregation — reads the state of all subsystems at a given
cycle and packages it into a single explainable dict.

This is the explainability layer of the entire system: it answers
WHY a trade did or did not happen in a given cycle.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("proxima_ops.decision.pre_trade_snapshot")

# ---------------------------------------------------------------------------
# RSI computation — reuse from edge_signal_mapper if available
# ---------------------------------------------------------------------------
try:
    from proxima_x.proxima_ops.risk.edge_signal_mapper import _compute_rsi
except ImportError:
    # Fallback Wilder-smoothed RSI (same algorithm as edge_signal_mapper)
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:  # type: ignore[misc]
        """Simple Wilder-smoothed RSI fallback."""
        diffs = np.diff(closes)
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)
        avg_gain = float(np.mean(gains[:period])) if len(gains) >= period else (
            float(np.mean(gains)) if len(gains) > 0 else 0.0
        )
        avg_loss = float(np.mean(losses[:period])) if len(losses) >= period else (
            float(np.mean(losses)) if len(losses) > 0 else 0.0
        )
        rsi_arr = np.full(len(closes), 50.0, dtype=np.float64)
        for i in range(period, len(closes)):
            avg_gain = (avg_gain * (period - 1) + float(gains[i - 1])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i - 1])) / period
            if avg_loss == 0.0:
                rsi_arr[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_arr[i] = 100.0 - 100.0 / (1.0 + rs)
        return rsi_arr


class PreTradeSnapshot:
    """Build a deterministic explainability snapshot for one cycle.

    This class aggregates state from all subsystems (signal pipeline,
    governance, VEL, readiness, market data) and produces a single
    canonical snapshot dict that explains WHY a trade did or did not
    happen.

    The snapshot is a pure data structure — it does NOT make any
    decisions or modify any state.
    """

    def __init__(self) -> None:
        logger.debug("PreTradeSnapshot initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        cycle: int,
        best_signal: dict | None,
        all_signals: list,
        confirm_cycles: dict,
        governor_state: str,
        vel_allowed: bool,
        readiness: dict,
        pipeline_trace: dict,
        active_symbols: list,
        md: dict,
    ) -> dict[str, Any]:
        """Build and return the canonical pre-trade snapshot.

        Parameters
        ----------
        cycle : int
            Current cycle number.
        best_signal : dict or None
            The top signal from ``_sweep_signals()`` output, or None
            if no signal passed all gates.
        all_signals : list
            All signals produced by the fusion engine for this cycle.
        confirm_cycles : dict
            ``{symbol_direction: count}`` from executor's confirm tracking.
        governor_state : str
            Current SEGL state string (e.g. ``"ARMED"``, ``"OBSERVE"``).
        vel_allowed : bool
            Whether VEL (Volume Expansion Layer) admitted the execution.
        readiness : dict
            Result dict from ``LiveTradeReadiness.evaluate()``.
        pipeline_trace : dict
            Pipeline trace with keys ``generated``, ``threshold_gate``,
            ``confirm_gate``, ``governor_gate``, ``execution``.
        active_symbols : list
            Current SIL universe / active symbol list.
        md : dict
            Market-data dict with ``closes`` (numpy arrays keyed by
            symbol), ``prices``, etc.

        Returns
        -------
        dict
            Canonical pre-trade snapshot (see module docstring for schema).
        """
        try:
            # --- symbol_candidates ---
            symbol_candidates = self._extract_symbol_candidates(all_signals)

            # --- confirm_status ---
            confirm_status = self._build_confirm_status(confirm_cycles)

            # --- governor_status ---
            governor_status = self._build_governor_status(
                governor_state, readiness, pipeline_trace
            )

            # --- vel_status ---
            vel_status = self._build_vel_status(vel_allowed, pipeline_trace)

            # --- readiness_score ---
            readiness_score = float(readiness.get("readiness_score", 0.0))

            # --- sil_universe_size ---
            sil_universe_size = len(active_symbols) if isinstance(active_symbols, (list, tuple)) else 0

            # --- rsi_range ---
            rsi_range = self._compute_rsi_range(md)

            # --- final_decision & blocking_reason ---
            final_decision, blocking_reason = self._resolve_decision(
                readiness,
                pipeline_trace,
                governor_status,
                vel_status,
            )

            snapshot: dict[str, Any] = {
                "cycle_id": cycle,
                "symbol_candidates": symbol_candidates,
                "best_signal": best_signal,
                "confirm_status": confirm_status,
                "governor_status": governor_status,
                "vel_status": vel_status,
                "readiness_score": readiness_score,
                "sil_universe_size": sil_universe_size,
                "rsi_range": rsi_range,
                "final_decision": final_decision,
                "blocking_reason": blocking_reason,
            }

            logger.debug(
                "Pre-trade snapshot built: cycle=%d decision=%s blocking=%s",
                cycle,
                final_decision,
                blocking_reason,
            )
            return snapshot

        except Exception:
            logger.exception("PreTradeSnapshot.build failed — returning safe fallback")
            return self._safe_fallback(cycle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_symbol_candidates(all_signals: list) -> list:
        """Return unique symbols that have non-zero direction signals."""
        try:
            symbols: set[str] = set()
            for s in all_signals:
                if isinstance(s, dict) and s.get("direction", 0) != 0:
                    sym = s.get("symbol")
                    if sym:
                        symbols.add(str(sym))
            return sorted(symbols)
        except Exception:
            logger.warning("Failed to extract symbol candidates", exc_info=True)
            return []

    @staticmethod
    def _build_confirm_status(confirm_cycles: dict) -> dict:
        """Build the ``confirm_status`` sub-dict.

        Returns
        -------
        dict with keys ``active_confirm_keys``, ``max_confirm_count``,
        ``threshold_met``.
        """
        try:
            if not isinstance(confirm_cycles, dict):
                return {
                    "active_confirm_keys": [],
                    "max_confirm_count": 0,
                    "threshold_met": False,
                }

            active_keys = [k for k, v in confirm_cycles.items() if v and v > 0]
            max_count = max(confirm_cycles.values()) if confirm_cycles else 0

            return {
                "active_confirm_keys": sorted(active_keys),
                "max_confirm_count": int(max_count),
                "threshold_met": bool(max_count >= 2),
            }
        except Exception:
            logger.warning("Failed to build confirm status", exc_info=True)
            return {
                "active_confirm_keys": [],
                "max_confirm_count": 0,
                "threshold_met": False,
            }

    @staticmethod
    def _build_governor_status(
        governor_state: str,
        readiness: dict,
        pipeline_trace: dict,
    ) -> dict:
        """Build the ``governor_status`` sub-dict.

        Gathers the current SEGL state, whether execution is authorized,
        and which governor-related rules are blocking.
        """
        try:
            state_str = str(governor_state).upper() if governor_state else "UNKNOWN"
            authorized = state_str == "ARMED"

            blocking_rules: list[str] = []

            # 1. State mismatch
            if not authorized:
                blocking_rules.append(
                    f"Governor state is {governor_state} (requires ARMED)"
                )

            # 2. Readiness factors mentioning governor / SEGL / ARMED
            blocking_factors = readiness.get("blocking_factors", [])
            if isinstance(blocking_factors, list):
                for factor in blocking_factors:
                    if isinstance(factor, str) and (
                        "governor" in factor.lower()
                        or "segl" in factor.lower()
                        or "armed" in factor.lower()
                    ):
                        blocking_rules.append(factor)

            # 3. Pipeline trace governor_gate detail
            gov_gate = pipeline_trace.get("governor_gate", [])
            if isinstance(gov_gate, list):
                for entry in gov_gate:
                    if isinstance(entry, str) and "NO" in entry.upper():
                        blocking_rules.append(entry)

            return {
                "state": str(governor_state) if governor_state else "UNKNOWN",
                "authorized": authorized,
                "blocking_rules": blocking_rules,
            }
        except Exception:
            logger.warning("Failed to build governor status", exc_info=True)
            return {
                "state": "UNKNOWN",
                "authorized": False,
                "blocking_rules": ["Error building governor_status"],
            }

    @staticmethod
    def _build_vel_status(vel_allowed: bool, pipeline_trace: dict) -> dict:
        """Build the ``vel_status`` sub-dict."""
        try:
            reason: str = "allowed"
            if not vel_allowed:
                exec_entry = pipeline_trace.get("execution")
                if isinstance(exec_entry, str) and "VEL" in exec_entry:
                    reason = exec_entry
                else:
                    reason = "VEL blocked execution (rule not satisfied)"
            return {
                "allowed": vel_allowed,
                "reason": reason,
            }
        except Exception:
            logger.warning("Failed to build VEL status", exc_info=True)
            return {
                "allowed": False,
                "reason": "Error building vel_status",
            }

    @staticmethod
    def _compute_rsi_range(md: dict) -> dict:
        """Compute ``min`` / ``max`` RSI across all symbols with close data.

        Expects ``md["closes"]`` to be a dict mapping symbol to numpy
        array of close prices.
        """
        try:
            closes = md.get("closes", {})
            if not isinstance(closes, dict) or not closes:
                return {"min": 0.0, "max": 0.0}

            rsi_values: list[float] = []
            for arr in closes.values():
                if isinstance(arr, np.ndarray) and len(arr) >= 20:
                    try:
                        rsi_arr = _compute_rsi(arr)
                        if len(rsi_arr) > 0:
                            rsi_values.append(float(rsi_arr[-1]))
                    except Exception:
                        continue

            if not rsi_values:
                return {"min": 0.0, "max": 0.0}

            return {
                "min": round(float(min(rsi_values)), 2),
                "max": round(float(max(rsi_values)), 2),
            }
        except Exception:
            logger.warning("Failed to compute RSI range", exc_info=True)
            return {"min": 0.0, "max": 0.0}

    @staticmethod
    def _resolve_decision(
        readiness: dict,
        pipeline_trace: dict,
        governor_status: dict,
        vel_status: dict,
    ) -> tuple[str, str | None]:
        """Determine ``final_decision`` and ``blocking_reason``.

        Decision is ``"EXECUTE"`` only if the pipeline trace execution
        field starts with ``"EXECUTED"``. Otherwise it is ``"HOLD"``
        with an aggregated blocking reason.
        """
        try:
            exec_entry = pipeline_trace.get("execution")

            # --- EXECUTE path ---
            if isinstance(exec_entry, str) and exec_entry.startswith("EXECUTED"):
                return "EXECUTE", None

            # --- HOLD path: aggregate blocking reasons ---
            blocking_reasons: list[str] = []

            # Readiness blocking factors
            blocking_factors = readiness.get("blocking_factors", [])
            if isinstance(blocking_factors, list):
                for factor in blocking_factors:
                    if isinstance(factor, str) and factor:
                        blocking_reasons.append(factor)

            # Governor blocking rules
            gov_rules = governor_status.get("blocking_rules", [])
            if isinstance(gov_rules, list):
                for rule in gov_rules:
                    if isinstance(rule, str) and rule:
                        blocking_reasons.append(rule)

            # VEL status
            if not vel_status.get("allowed", False):
                vel_reason = vel_status.get("reason", "VEL blocked")
                if isinstance(vel_reason, str) and vel_reason:
                    blocking_reasons.append(vel_reason)

            # Pipeline trace execution denial
            if isinstance(exec_entry, str):
                if "DENIED" in exec_entry or "FAILED" in exec_entry or "NO_SIGNAL" in exec_entry:
                    blocking_reasons.append(exec_entry)

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_reasons: list[str] = []
            for r in blocking_reasons:
                if r not in seen:
                    seen.add(r)
                    unique_reasons.append(r)

            blocking_reason: str | None = (
                "; ".join(unique_reasons) if unique_reasons else None
            )
            return "HOLD", blocking_reason

        except Exception:
            logger.warning("Failed to resolve decision", exc_info=True)
            return "HOLD", "Error resolving decision"

    @staticmethod
    def _safe_fallback(cycle: int) -> dict[str, Any]:
        """Return a minimal safe snapshot when :meth:`build` fails entirely."""
        return {
            "cycle_id": cycle,
            "symbol_candidates": [],
            "best_signal": None,
            "confirm_status": {
                "active_confirm_keys": [],
                "max_confirm_count": 0,
                "threshold_met": False,
            },
            "governor_status": {
                "state": "UNKNOWN",
                "authorized": False,
                "blocking_rules": [
                    "PreTradeSnapshot.build raised an exception"
                ],
            },
            "vel_status": {
                "allowed": False,
                "reason": "PreTradeSnapshot.build raised an exception",
            },
            "readiness_score": 0.0,
            "sil_universe_size": 0,
            "rsi_range": {"min": 0.0, "max": 0.0},
            "final_decision": "HOLD",
            "blocking_reason": "PreTradeSnapshot.build raised an exception",
        }
