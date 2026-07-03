"""LiveTradeReadiness: Evaluate whether the system is capable of producing a trade.

Pure diagnostics — NEVER modifies state or interferes with execution.
Checks all preconditions and reports blocking factors.
"""

import logging
from typing import Any

logger = logging.getLogger("proxima_ops.monitoring.live_trade_readiness")

NEUTRAL_RSI_LOWER = 40.0
NEUTRAL_RSI_UPPER = 60.0
GOVERNOR_ARMED_STATE = "ARMED"
CHECK_KEYS = [
    "sil_universe_non_empty",
    "confirm_gate_active",
    "governor_armed",
    "mt5_tick_available",
    "rsi_not_all_neutral",
    "has_best_signal",
]


class LiveTradeReadiness:
    """Evaluate all preconditions required to produce a live trade.

    Each call to ``evaluate()`` returns a snapshot of readiness without
    modifying any external state or execution path.
    """

    def __init__(self) -> None:
        logger.debug("LiveTradeReadiness initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        md: dict,
        sil_active: bool,
        governor_state: str,
        confirm_cycles: dict,
        mt5_connected: bool,
        rsi_dict: dict,
    ) -> dict[str, Any]:
        """Return a readiness report.

        Parameters
        ----------
        md : dict
            Market-data dict (expected to contain at least ``"prices"``).
        sil_active : bool
            Whether the SIL universe currently has symbols loaded.
        governor_state : str
            SEGL state string (e.g. ``"ARMED"``, ``"OBSERVE"``).
        confirm_cycles : dict
            ``{symbol_direction: count}`` — non-empty means the confirm
            gate is tracking at least one candidate.
        mt5_connected : bool
            Whether the MT5 connection is active.
        rsi_dict : dict
            ``{symbol: float}`` of latest RSI values.

        Returns
        -------
        dict
            ``ready`` (bool), ``blocking_factors`` (list[str]),
            ``checks`` (dict[str, bool]), ``readiness_score`` (float).
        """
        try:
            checks: dict[str, bool] = {}

            # --- individual checks ---
            checks["sil_universe_non_empty"] = bool(sil_active)

            checks["confirm_gate_active"] = bool(confirm_cycles) and isinstance(
                confirm_cycles, dict
            )

            checks["governor_armed"] = bool(
                governor_state and governor_state.upper() == GOVERNOR_ARMED_STATE
            )

            checks["mt5_tick_available"] = bool(
                mt5_connected
                and isinstance(md, dict)
                and bool(md.get("prices"))
            )

            checks["rsi_not_all_neutral"] = self._has_non_neutral_rsi(rsi_dict)

            checks["has_best_signal"] = bool(
                md and md.get("best_signal") is not None
            )

            # --- blocking factors ---
            blocking_factors: list[str] = []
            if not checks["sil_universe_non_empty"]:
                blocking_factors.append("SIL universe is empty")
            if not checks["confirm_gate_active"]:
                blocking_factors.append("Confirm gate has no active cycles")
            if not checks["governor_armed"]:
                blocking_factors.append(
                    f"Governor (SEGL) is not ARMED (current: {governor_state})"
                )
            if not checks["mt5_tick_available"]:
                blocking_factors.append(
                    "MT5 not connected or no tick data available"
                )
            if not checks["rsi_not_all_neutral"]:
                blocking_factors.append(
                    "All symbols have neutral RSI (40-60 range)"
                )
            if not checks["has_best_signal"]:
                blocking_factors.append("No best_signal found in market data")

            ready = len(blocking_factors) == 0
            passed = sum(1 for v in checks.values() if v)
            readiness_score = round(passed / len(checks), 4) if checks else 0.0

            result: dict[str, Any] = {
                "ready": ready,
                "blocking_factors": blocking_factors,
                "checks": checks,
                "readiness_score": readiness_score,
            }

            logger.debug(
                "Readiness evaluation complete: ready=%s score=%.2f factors=%d",
                ready,
                readiness_score,
                len(blocking_factors),
            )
            return result

        except Exception:
            logger.exception("LiveTradeReadiness.evaluate failed")
            return {
                "ready": False,
                "blocking_factors": ["Readiness evaluation raised an exception"],
                "checks": {k: False for k in CHECK_KEYS},
                "readiness_score": 0.0,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_neutral_rsi(rsi: float) -> bool:
        """Return ``True`` if *rsi* falls in the neutral band."""
        return NEUTRAL_RSI_LOWER <= rsi <= NEUTRAL_RSI_UPPER

    def _has_non_neutral_rsi(self, rsi_dict: dict) -> bool:
        """Return ``True`` if at least one symbol has non-neutral RSI."""
        try:
            if not isinstance(rsi_dict, dict) or not rsi_dict:
                return False
            return any(
                not self._is_neutral_rsi(rsi)
                for rsi in rsi_dict.values()
                if isinstance(rsi, (int, float))
            )
        except Exception:
            logger.warning("Failed to check RSI neutrality", exc_info=True)
            return False
