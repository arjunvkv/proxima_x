"""
SES — Single Execution Sovereign

Single runtime authority for MT5 order emission. No other module may veto
post-commit.  Evaluates all subsystem inputs and produces THE FINAL decision.
"""


class SingleExecutionSovereign:
    """
    Sole authority for emitting MT5 orders.

    Parameters
    ----------
    risk_caps : dict, optional
        Risk-capacity overrides (e.g., max position size per symbol).
    """

    def __init__(self, risk_caps: dict = None, live_mode: bool = True):
        self.risk_caps = risk_caps or {}
        self.live_mode = live_mode

    # ------------------------------------------------------------------
    def evaluate(
        self,
        decision: dict,
        era_result: dict,
        tamk_result: dict,
        loef_result: dict,
        signal: dict,
        mt5_tick: dict,
    ) -> dict:
        """
        Evaluate all inputs and return the sovereign ruling.

        Parameters
        ----------
        decision : dict
            From DCE.collapse() — keys: action, symbol, confidence, action_value.
        era_result : dict
            From ERA.validate() — keys: valid, reality_alignment_score,
            adjusted_price, adjusted_volume.
        tamk_result : dict
            From TAMK.authorize() — keys: authorized, override_active.
        loef_result : dict
            From LOEF.compute() — keys: opportunity_density, top_k_symbols.
        signal : dict or None
            The best_signal dict from the signal mapper — keys: symbol,
            direction, confidence, edge_id, strategy.
        mt5_tick : dict or None
            Latest tick — keys: bid, ask, time.

        Returns
        -------
        dict with keys:
            emit_order         : bool
            order_params       : dict or None
            rejection_reason   : str or None
            authority_chain    : list[str]
            sovereign_override : bool
        """
        try:
            # ---- extract signals ----
            dce_action = decision.get("action")
            dce_confidence = decision.get("confidence", 0.0)

            tamk_authorized = tamk_result.get("authorized", False)
            tamk_override = tamk_result.get("override_active", False)

            era_valid = era_result.get("valid", False)

            loef_density = loef_result.get("opportunity_density", 0.0)

            # ---- compute subsystem votes for the authority chain ----

            # DCE vote
            dce_allowed = dce_action in ("BUY", "SELL") and dce_confidence > 0
            dce_vote = "ALLOW" if dce_allowed else "BLOCK"

            # TAMK vote
            tamk_vote = "ALLOW" if (tamk_authorized or tamk_override) else "BLOCK"

            # ERA vote
            era_vote = "ALLOW" if era_valid else "BLOCK"

            # LOEF vote
            loef_vote = "ALLOW" if loef_density >= 0.12 else "BLOCK"

            # ---- sovereign rules (evaluated in order) ----
            emit = False
            rejection_reason = None
            sovereign_override = False

            # Rule 1 — TAMK safety block (unless override is active)
            if not tamk_authorized and not tamk_override:
                rejection_reason = "TAMK safety block: not authorized and no override active"

            # Rule 2 — ERA reality mismatch
            elif not era_valid:
                rejection_reason = "ERA reality mismatch: validation failed"

            # Rule 3 — No actionable decision
            elif dce_action in (None, "HOLD"):
                rejection_reason = "DCE decision is HOLD or None"

            # Rule 4 — Missing critical data
            elif signal is None or mt5_tick is None:
                rejection_reason = "Missing data: signal or mt5_tick is None"

            # Rule 5 — Low opportunity density
            elif loef_density < 0.12:
                rejection_reason = f"LOEF low opportunity density ({loef_density:.3f} < 0.12)"

            # Rule 6 — All checks passed → tentative emit
            else:
                emit = True

            # Rule 7 — LIVE mode gate (block if not live_mode)
            if emit and not self.live_mode:
                rejection_reason = "SES live_mode=False — simulation only, order blocked"
                emit = False

            # Rule 8 — Sovereign override detection
            if emit and not tamk_authorized and tamk_override:
                sovereign_override = True

            # ---- build order parameters ----
            order_params = None
            if emit:
                fallback_price = 0.0
                if mt5_tick:
                    fallback_price = mt5_tick.get("ask" if dce_action == "BUY" else "bid", 0.0)
                
                order_params = {
                    "symbol": decision.get("symbol") or (signal or {}).get("symbol"),
                    "action": dce_action,
                    "volume": era_result.get("adjusted_volume") or 0.01,
                    "price": era_result.get("adjusted_price") or fallback_price,
                    "order_type": "MARKET",
                    "slippage_points": 10,
                }

            # ---- authority chain ----
            ses_vote = "ALLOW" if emit else "BLOCK"
            authority_chain = [
                f"DCE:{dce_vote}",
                f"TAMK:{tamk_vote}",
                f"ERA:{era_vote}",
                f"LOEF:{loef_vote}",
                f"SES:{ses_vote}",
            ]

            return {
                "emit_order": emit,
                "order_params": order_params,
                "rejection_reason": rejection_reason,
                "authority_chain": authority_chain,
                "sovereign_override": sovereign_override,
            }

        except Exception as exc:
            return {
                "emit_order": False,
                "order_params": None,
                "rejection_reason": f"SES internal error: {exc}",
                "authority_chain": [
                    "DCE:BLOCK",
                    "TAMK:BLOCK",
                    "ERA:BLOCK",
                    "LOEF:BLOCK",
                    "SES:BLOCK",
                ],
                "sovereign_override": False,
            }
