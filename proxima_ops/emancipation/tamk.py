"""
TAMK — Trade Authorization Minimal Kernel

Single binary trade permission function. Everything else is advisory.
"""


class TradeAuthorizationMinimalKernel:
    """Minimal kernel that authorizes or blocks a trade based on several
    attractor-dynamics and system-health checks."""

    def __init__(self, erf_threshold: float = 0.5, max_cb_latch_cycles: int = 100):
        self.erf_threshold = erf_threshold
        self.max_cb_latch_cycles = max_cb_latch_cycles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def authorize(
        self,
        erf: float,
        escape_energy: float,
        rfg: float,
        cb_triggered: bool,
        cb_latch_cycles: int,
        gmci_score: float,
        governor_state: str,
        mt5_connected: bool,
    ) -> dict:
        """Evaluate all gates and return an authorisation decision.

        Returns
        -------
        dict with keys:
            authorized       – bool, ALLOW (True) or BLOCK (False)
            reason           – str | None, why blocked if blocked
            checks           – dict of individual check booleans
            override_active  – bool, True when CB override is active
        """
        try:
            checks = self._compute_checks(
                erf=erf,
                escape_energy=escape_energy,
                rfg=rfg,
                cb_triggered=cb_triggered,
                cb_latch_cycles=cb_latch_cycles,
                gmci_score=gmci_score,
                governor_state=governor_state,
                mt5_connected=mt5_connected,
            )

            override_active = cb_triggered and cb_latch_cycles > self.max_cb_latch_cycles

            allowed = all(checks.values())
            reason = None if allowed else self._first_failure(checks)

            return {
                "authorized": allowed,
                "reason": reason,
                "checks": checks,
                "override_active": override_active,
            }
        except Exception:
            # On any unexpected error, default to BLOCK with a safe reason.
            return {
                "authorized": False,
                "reason": "exception_during_authorization",
                "checks": {k: False for k in self._check_keys()},
                "override_active": False,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_keys() -> list:
        return [
            "erf_above_threshold",
            "escape_energy_feasible",
            "rfg_positive",
            "cb_not_structurally_latched",
            "gmci_acceptable",
            "governor_armed",
            "mt5_connected",
        ]

    def _compute_checks(
        self,
        erf: float,
        escape_energy: float,
        rfg: float,
        cb_triggered: bool,
        cb_latch_cycles: int,
        gmci_score: float,
        governor_state: str,
        mt5_connected: bool,
    ) -> dict:
        """Compute every individual check and return them as a dict."""
        return {
            "erf_above_threshold": erf >= self.erf_threshold,
            "escape_energy_feasible": escape_energy <= 0.7,
            "rfg_positive": rfg > 0.0,
            "cb_not_structurally_latched": (
                not cb_triggered or cb_latch_cycles > self.max_cb_latch_cycles
            ),
            "gmci_acceptable": gmci_score < 0.7,
            "governor_armed": governor_state == "ARMED",
            "mt5_connected": mt5_connected is True,
        }

    def _first_failure(self, checks: dict) -> str:
        """Return a human-readable reason for the first check that failed."""
        failures = {
            "erf_above_threshold": "erf_below_threshold",
            "escape_energy_feasible": "escape_energy_too_high",
            "rfg_positive": "rfg_not_positive",
            "cb_not_structurally_latched": "circuit_breaker_active",
            "gmci_acceptable": "gmci_score_too_high",
            "governor_armed": "governor_not_armed",
            "mt5_connected": "mt5_disconnected",
        }
        for key, reason in failures.items():
            if not checks.get(key, False):
                return reason
        return None
