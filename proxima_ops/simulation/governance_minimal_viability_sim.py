"""GovernanceMinimalViabilitySim: Compare current vs minimal governor (CB+VEL only).

Purpose
-------
Contrast the current live governor's performance against a *minimal_governor*
that keeps only circuit-breaker and VEL (velocity / frequency) rules while
removing SEGL OBSERVE gating.  The simulation reads historical cycle logs
and re-plays each cycle under the hypothetical rule set, then evaluates
the risk/reward trade-off using a 60 % win rate / 1.5:1 R:R assumption.

Output
------
    {
      "current_trades": int,
      "minimal_trades": int,
      "risk_delta": float,
      "drawdown_estimate": {"current": float, "minimal": float, "delta": float},
      "recommended_mode": str,
      "rationale": str
    }

    - ``risk_delta`` = (minimal_trades - current_trades) / max(total_signals, 1),
      capped at 1.0.
    - ``recovered`` (internal) = expected profit from additional trades at
      0.5 R per trade (EV = 0.6 * 1.5 - 0.4 * 1.0) * $100 risk per trade.
    - ``recommended_mode``: "AGGRESSIVE" if recovered > 5000 and risk_delta < 0.5,
      "STRICT" if recovered < 1000, else "BALANCED".
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("proxima_ops.simulation.governance_minimal_viability_sim")

# ---------------------------------------------------------------------------
# Constants for the drawdown / recovery model
# ---------------------------------------------------------------------------
_WIN_RATE = 0.6                 # 60 % historical win rate assumption
_RISK_REWARD = 1.5              # 1.5 : 1 reward-to-risk ratio
_RISK_PER_TRADE = 100.0        # notional $ risk per trade (1 % of $10 k)
_PER_TRADE_EV = _WIN_RATE * _RISK_REWARD - (1.0 - _WIN_RATE) * 1.0  # 0.5 R


class GovernanceMinimalViabilitySim:
    """Compare current system vs minimal_governor (CB+VEL only) with risk constraints.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log file (default
        ``"state/wave12_cycle_log.jsonl"``).
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self._log_path = Path(log_path)
        self._log_lines: list[dict[str, Any]] = []
        logger.debug(
            "GovernanceMinimalViabilitySim initialised with log_path=%s", log_path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Compare current system vs minimal_governor (CB+VEL only).

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            See module docstring for output schema.
        """
        # --- load data -------------------------------------------------------
        try:
            self._load_logs(n_recent_cycles)
        except Exception:
            logger.exception("Failed to load cycle log")
            return self._empty_result("Failed to load cycle log")

        # --- classify every cycle --------------------------------------------
        try:
            total_signals = len(self._log_lines)
            current_trades = 0
            segl_blocks = 0

            for record in self._log_lines:
                execution = self._get_execution(record)
                if self._is_trade(execution):
                    current_trades += 1
                elif execution.startswith("DENIED segl_state="):
                    segl_blocks += 1
                # Other outcomes (CB, VEL, confirm, NO_SIGNAL, FAILED) are
                # ignored — they either stay blocked or were never tradeable.

            # Under minimal_governor, SEGL OBSERVE blocks are removed.
            # CB and VEL blocks remain in place.
            additional_trades = segl_blocks
            minimal_trades = current_trades + additional_trades

            # --- risk delta --------------------------------------------------
            risk_delta = (minimal_trades - current_trades) / max(total_signals, 1)
            risk_delta = min(risk_delta, 1.0)  # cap at 1.0

            # --- drawdown estimate (60 % WR, 1.5:1 R:R) ----------------------
            dd_current = self._estimate_drawdown(current_trades)
            dd_minimal = self._estimate_drawdown(minimal_trades)
            dd_delta = round(dd_minimal - dd_current, 2)

            # --- recovery estimate (expected PnL from additional trades) -----
            recovered = additional_trades * _PER_TRADE_EV * _RISK_PER_TRADE

            # --- recommended mode --------------------------------------------
            recommended_mode, rationale = self._decide_mode(
                recovered=recovered,
                risk_delta=risk_delta,
                additional_trades=additional_trades,
            )

            return {
                "current_trades": current_trades,
                "minimal_trades": minimal_trades,
                "risk_delta": round(risk_delta, 4),
                "drawdown_estimate": {
                    "current": dd_current,
                    "minimal": dd_minimal,
                    "delta": dd_delta,
                },
                "recommended_mode": recommended_mode,
                "rationale": rationale,
            }

        except Exception:
            logger.exception("GovernanceMinimalViabilitySim analysis failed")
            return self._empty_result("Analysis failed — see logs")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_logs(self, n_recent_cycles: int) -> None:
        """Load up to *n_recent_cycles* log entries from the JSONL file."""
        self._log_lines = []
        path = self._log_path

        if not path.is_file():
            raise FileNotFoundError(f"Cycle log not found: {path.resolve()}")

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable log line: %s", stripped[:80])
                    continue
                self._log_lines.append(record)

        if not self._log_lines:
            raise ValueError(f"No valid log entries loaded from {path}")

        logger.debug(
            "Loaded %d total log entries; will analyse most recent %d",
            len(self._log_lines),
            n_recent_cycles,
        )

        if len(self._log_lines) > n_recent_cycles:
            self._log_lines = self._log_lines[-n_recent_cycles:]

    # ------------------------------------------------------------------
    # Drawdown model  (60 % win rate, 1.5:1 R:R)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_drawdown(n_trades: int) -> float:
        """Approximate max drawdown for *n_trades* at 60 % WR / 1.5:1 R:R.

        Uses the expected longest losing streak as a proxy for maximum
        drawdown in dollar terms (each loss = 1 R = $ *risk_per_trade*).
        """
        if n_trades <= 0:
            return 0.0

        q = 1.0 - _WIN_RATE  # probability of a loss = 0.4
        # Expected maximum consecutive losses in N trials:
        #   E[max_loss] ≈ log(N * (1 - q)) / log(1 / q)
        expected_max_losses = (
            math.log(n_trades * _WIN_RATE) / math.log(1.0 / q)
            if q > 0
            else 0.0
        )
        expected_max_losses = max(expected_max_losses, 1.0)  # at least 1

        # Each loss costs 1 R → dollar drawdown
        dd_dollars = -expected_max_losses * _RISK_PER_TRADE
        return round(dd_dollars, 2)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    @staticmethod
    def _decide_mode(
        recovered: float,
        risk_delta: float,
        additional_trades: int,
    ) -> tuple[str, str]:
        """Determine recommended governor mode based on recovery and risk."""
        if recovered > 5000 and risk_delta < 0.5:
            return "AGGRESSIVE", (
                f"Recovered ~${recovered:,.0f} from {additional_trades} "
                f"SEGL-unblocked trades with low risk delta ({risk_delta:.3f})"
            )
        if recovered < 1000:
            return "STRICT", (
                f"Insufficient recovered value (~${recovered:,.0f}) from "
                f"{additional_trades} additional trades"
            )
        return "BALANCED", (
            f"Recovered ~${recovered:,.0f} with risk delta {risk_delta:.3f} "
            f"— moderate improvement"
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_execution(record: dict[str, Any]) -> str:
        """Extract the execution outcome string from a log record."""
        pt = record.get("pipeline_trace")
        if isinstance(pt, dict):
            return str(pt.get("execution", ""))
        return ""

    @staticmethod
    def _is_blocked(execution: str) -> bool:
        """Return True if the execution outcome indicates a blocked trade."""
        if not isinstance(execution, str):
            return False
        return execution.startswith("DENIED") or execution.startswith("FAILED")

    @staticmethod
    def _is_trade(execution: str) -> bool:
        """Return True if the cycle resulted in an actual BUY/SELL trade.

        A trade is any execution that:
        - is a non-empty string,
        - is NOT blocked (DENIED/FAILED),
        - is NOT NO_SIGNAL.
        """
        if not isinstance(execution, str) or not execution:
            return False
        if GovernanceMinimalViabilitySim._is_blocked(execution):
            return False
        if execution.startswith("NO_SIGNAL"):
            return False
        return True

    # ------------------------------------------------------------------
    # Safe fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(error_hint: str = "") -> dict[str, Any]:
        """Return a safe fallback result when simulation cannot run."""
        return {
            "current_trades": 0,
            "minimal_trades": 0,
            "risk_delta": 0.0,
            "drawdown_estimate": {"current": 0.0, "minimal": 0.0, "delta": 0.0},
            "recommended_mode": "BALANCED",
            "rationale": f"Simulation unavailable — {error_hint}",
        }
