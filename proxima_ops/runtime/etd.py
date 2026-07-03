"""
ETD — Execution Trigger Demultiplexer.

Single canonical execution trigger condition.
Aggregates LOEF, ERF, DCE into a unified trigger decision.
"""


class ExecutionTriggerDemux:
    """Demultiplexes execution trigger signals into a single canonical decision."""

    def __init__(
        self,
        erf_threshold: float = 0.5,
        loef_threshold: float = 0.3,
        dce_threshold: float = 0.4,
    ):
        """
        Args:
            erf_threshold: Minimum ERF value to trigger.
            loef_threshold: Minimum LOEF density to trigger (with non-HOLD DCE action).
            dce_threshold: Minimum DCE confidence to trigger (with non-HOLD DCE action).
        """
        self.erf_threshold = erf_threshold
        self.loef_threshold = loef_threshold
        self.dce_threshold = dce_threshold

    # ------------------------------------------------------------------
    def evaluate(
        self,
        erf: float,
        loef_density: float,
        dce_confidence: float,
        dce_action: str,
        tamk_authorized: bool,
    ) -> dict:
        """Evaluate all trigger signals and return a unified result dict.

        Args:
            erf: Energy Reality Field value.
            loef_density: LOEF density value.
            dce_confidence: DCE confidence score.
            dce_action: DCE action string (e.g. "BUY", "SELL", "HOLD").
            tamk_authorized: Whether TAMK authorization is present.

        Returns:
            dict with keys: trigger, trigger_source, trigger_value,
            trigger_threshold, all_signals, is_qualified.
        """
        try:
            # ----------------------------------------------------------
            # Assemble all signals
            # ----------------------------------------------------------
            all_signals = {
                "erf": erf,
                "loef": loef_density,
                "dce": dce_confidence,
                "tamk": 1.0 if tamk_authorized else 0.0,
            }

            # ----------------------------------------------------------
            # Evaluate conditions in priority order
            # ----------------------------------------------------------
            dce_ok = dce_action != "HOLD"

            trigger = False
            trigger_source = None
            trigger_value = None
            trigger_threshold = None

            # 1. ERF trigger
            if erf >= self.erf_threshold:
                trigger = True
                trigger_source = "ERF"
                trigger_value = erf
                trigger_threshold = self.erf_threshold

            # 2. LOEF trigger (requires non-HOLD DCE action)
            elif loef_density >= self.loef_threshold and dce_ok:
                trigger = True
                trigger_source = "LOEF"
                trigger_value = loef_density
                trigger_threshold = self.loef_threshold

            # 3. DCE trigger (requires non-HOLD DCE action)
            elif dce_confidence >= self.dce_threshold and dce_ok:
                trigger = True
                trigger_source = "DCE"
                trigger_value = dce_confidence
                trigger_threshold = self.dce_threshold

            # 4. TAMK-authorized trigger
            if (
                not trigger
                and tamk_authorized
                and (erf >= self.erf_threshold or loef_density >= self.loef_threshold)
            ):
                trigger = True
                trigger_source = "TAMK"
                # Use the signal that actually met its threshold (prefer ERF)
                if erf >= self.erf_threshold:
                    trigger_value = erf
                    trigger_threshold = self.erf_threshold
                else:
                    trigger_value = loef_density
                    trigger_threshold = self.loef_threshold

            # ----------------------------------------------------------
            # Qualification: trigger + permission
            # ----------------------------------------------------------
            is_qualified = trigger and tamk_authorized

            return {
                "trigger": trigger,
                "trigger_source": trigger_source,
                "trigger_value": trigger_value,
                "trigger_threshold": trigger_threshold,
                "all_signals": all_signals,
                "is_qualified": is_qualified,
            }

        except Exception:
            # Safe no-trigger fallback
            return {
                "trigger": False,
                "trigger_source": None,
                "trigger_value": 0.0,
                "trigger_threshold": 0.0,
                "all_signals": {"erf": 0.0, "loef": 0.0, "dce": 0.0, "tamk": 0.0},
                "is_qualified": False,
            }
