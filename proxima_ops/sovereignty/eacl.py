"""EACL — Execution Arbitration Collapse Layer.

Collapse all competing module outputs into one resolution.
Single-pass resolution per cycle.
"""

from __future__ import annotations

from typing import Any


class ExecutionArbitrationCollapse:
    """Single-pass arbitration layer that resolves conflicts between
    emancipated module outputs and produces a unified action vector.

    Resolution is one-shot (no iterative loops). SES result, when
    present, takes precedence for the final action.
    """

    # Thresholds used for conflict detection.
    _LOW_OPPORTUNITY_DENSITY: float = 0.5
    _HIGH_CONFIDENCE_THRESHOLD: float = 0.5

    def __init__(self) -> None:
        self._previous_action: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        dce_decision: dict[str, Any],
        tamk_result: dict[str, Any],
        era_result: dict[str, Any],
        loef_result: dict[str, Any],
        ecp_result: dict[str, Any],
        ses_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single-pass arbitration of all emancipated module outputs.

        Parameters
        ----------
        dce_decision : dict
            Output from ``DecisionCollapseEngine.collapse()``.
            Expected keys: ``action``, ``confidence``, ``symbol``, …
        tamk_result : dict
            Output from ``TradeAuthorizationMinimalKernel.authorize()``.
            Expected keys: ``authorized``, ``reason``, ``checks``, …
        era_result : dict
            Output from ``ExecutionRealityAnchor.validate()``.
            Expected keys: ``valid``, ``reality_alignment_score``, …
        loef_result : dict
            Output from ``LatentOpportunityExtractionField.compute()``.
            Expected keys: ``opportunity_density``, …
        ecp_result : dict
            Output from ``ExecutionCompressionPipeline.compress()``.
            Expected keys: ``execution_readiness``, ``action``, …
        ses_result : dict | None
            Optional output from a Sovereign Execution Supervisor.
            When present its ``action`` is used as the final action.

        Returns
        -------
        dict
            Arbitrated resolution with keys:
            ``resolved``, ``final_action``, ``conflicts_resolved``,
            ``arbitration_loops``, ``decision_oscillation``,
            ``resolution_weights``.
        """
        try:
            # ---- Conflict detection ----
            conflicts_resolved: list[str] = []

            dce_action = self._safe_get(dce_decision, "action", "HOLD")
            tamk_authorized = self._safe_get(tamk_result, "authorized", False)
            era_valid = self._safe_get(era_result, "valid", False)
            loef_density = self._safe_get(loef_result, "opportunity_density", 0.0)
            dce_confidence = self._safe_get(dce_decision, "confidence", 0.0)

            # DCE says BUY/SELL but TAMK says BLOCK
            if dce_action in ("BUY", "SELL") and not tamk_authorized:
                conflicts_resolved.append("DCE vs TAMK")

            # DCE says action but ERA says not valid
            if dce_action in ("BUY", "SELL") and not era_valid:
                conflicts_resolved.append("DCE vs ERA")

            # LOEF says low density but DCE says high confidence
            if (
                loef_density < self._LOW_OPPORTUNITY_DENSITY
                and dce_confidence > self._HIGH_CONFIDENCE_THRESHOLD
            ):
                conflicts_resolved.append("LOEF vs DCE")

            # ---- Resolution weights ----
            raw_weights: dict[str, float] = {
                "dce": 0.3 if dce_action != "HOLD" else 0.0,
                "tamk": 0.25 if tamk_authorized else 0.0,
                "era": 0.2 if era_valid else 0.0,
                "loef": 0.15 * loef_density,
                "ecp": 0.1 * self._safe_get(ecp_result, "execution_readiness", 0.0),
            }

            # Normalise so sum <= 1.0
            total = sum(raw_weights.values())
            if total > 1.0:
                weights = {k: v / total for k, v in raw_weights.items()}
            else:
                weights = dict(raw_weights)

            # ---- Resolved flag ----
            resolved = max(weights.values()) > 0.0

            # ---- Final action ----
            if ses_result is not None and "action" in ses_result:
                final_action = str(ses_result["action"])
            else:
                final_action = dce_action

            # ---- Decision oscillation ----
            oscillation = (
                self._previous_action is not None
                and dce_action in ("BUY", "SELL")
                and self._previous_action in ("BUY", "SELL")
                and dce_action != self._previous_action
            )

            # Update state
            self._previous_action = dce_action

            return {
                "resolved": resolved,
                "final_action": final_action,
                "conflicts_resolved": conflicts_resolved,
                "arbitration_loops": 1,
                "decision_oscillation": oscillation,
                "resolution_weights": weights,
            }

        except Exception:
            # Safe fallback: return a HOLD resolution
            return {
                "resolved": False,
                "final_action": "HOLD",
                "conflicts_resolved": [],
                "arbitration_loops": 1,
                "decision_oscillation": False,
                "resolution_weights": {
                    "dce": 0.0,
                    "tamk": 0.0,
                    "era": 0.0,
                    "loef": 0.0,
                    "ecp": 0.0,
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_get(d: Any, key: str, default: Any = None) -> Any:
        """Safely extract a key from a dict-like object.

        Returns *default* when *d* is not a dict, the key is missing,
        or the value is ``None`` (with a ``None`` default treated as
        absent for numeric contexts).
        """
        if not isinstance(d, dict):
            return default
        val = d.get(key, default)
        if val is None:
            return default
        return val
