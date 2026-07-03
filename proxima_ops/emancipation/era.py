"""ERA — Execution Reality Anchor.

Validate all trade intents against MT5 live constraints.
Ensure simulation-to-live alignment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ExecutionRealityAnchor:
    """Validates trade intents against live MT5 constraints.

    Provides a reality-alignment score, rejection reasons, and
    adjusted price/volume when the intent deviates from market
    reality.
    """

    def __init__(self, max_positions: int = 5, margin_fraction: float = 0.01) -> None:
        """Initialise the anchor with configurable limits.

        Parameters
        ----------
        max_positions : int
            Maximum number of concurrently open positions allowed
            (default 5).
        margin_fraction : float
            Fraction of notional value required as margin
            (default 0.01 = 1 %).
        """
        self.max_positions = max_positions
        self.margin_fraction = margin_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        action: str,
        symbol: str,
        volume: float,
        price: float,
        mt5_tick: dict,
        mt5_account: dict,
        open_positions: List[dict],
        sil_scores: Dict[str, Any],
    ) -> dict:
        """Run all reality checks and return a structured verdict.

        Parameters
        ----------
        action : str
            Trade direction — ``"BUY"`` or ``"SELL"``.
        symbol : str
            Instrument symbol (e.g. ``"EURUSD"``).
        volume : float
            Requested trade volume (lots).
        price : float
            Intended entry price.
        mt5_tick : dict
            Latest MT5 tick data (should contain *bid* and *ask*).
        mt5_account : dict
            MT5 account info (should contain *margin_free*).
        open_positions : list[dict]
            Currently open positions.
        sil_scores : dict
            Symbol-in-universe mapping from SIL (Symbol Integrity Layer).

        Returns
        -------
        dict
            Verdict dictionary with keys:
            - valid
            - rejection_reason
            - checks (per-check booleans)
            - reality_alignment_score
            - simulation_vs_live_divergence
            - adjusted_price
            - adjusted_volume
        """
        # Default safe fallback
        result: dict = {
            "valid": False,
            "rejection_reason": None,
            "checks": {
                "tick_available": False,
                "symbol_tradeable_by_mt5": False,
                "price_within_bid_ask": False,
                "account_has_margin": False,
                "position_limit_not_exceeded": False,
                "sil_symbol_still_active": False,
            },
            "reality_alignment_score": 0.0,
            "simulation_vs_live_divergence": 1.0,
            "adjusted_price": None,
            "adjusted_volume": None,
        }

        try:
            self._run_checks(
                result=result,
                action=action,
                symbol=symbol,
                volume=volume,
                price=price,
                mt5_tick=mt5_tick,
                mt5_account=mt5_account,
                open_positions=open_positions,
                sil_scores=sil_scores,
            )

            # Compute aggregate scores
            checks = result["checks"]
            passed = sum(1 for v in checks.values() if v)
            total = len(checks)
            result["reality_alignment_score"] = passed / total if total > 0 else 0.0
            result["simulation_vs_live_divergence"] = 1.0 - result["reality_alignment_score"]

            # Overall validity
            result["valid"] = all(checks.values())
            if not result["valid"]:
                result["rejection_reason"] = self._build_rejection_reason(checks)

        except Exception as exc:
            result["valid"] = False
            result["rejection_reason"] = f"ERA validation exception: {exc}"

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        result: dict,
        action: str,
        symbol: str,
        volume: float,
        price: float,
        mt5_tick: dict,
        mt5_account: dict,
        open_positions: List[dict],
        sil_scores: Dict[str, Any],
    ) -> None:
        """Populate *result['checks']* in place."""
        checks = result["checks"]

        # 1. Tick availability
        tick_available = (
            isinstance(mt5_tick, dict)
            and mt5_tick.get("bid") is not None
            and mt5_tick.get("ask") is not None
        )
        checks["tick_available"] = tick_available

        # 2. Symbol tradeable by MT5 — symbol exists and tick is not None
        checks["symbol_tradeable_by_mt5"] = isinstance(mt5_tick, dict)

        # 3. Price within bid-ask spread
        if tick_available:
            bid = mt5_tick["bid"]
            ask = mt5_tick["ask"]
            action_upper = action.upper()
            if action_upper == "BUY":
                checks["price_within_bid_ask"] = price <= ask
                if price > ask:
                    result["adjusted_price"] = ask
            elif action_upper == "SELL":
                checks["price_within_bid_ask"] = price >= bid
                if price < bid:
                    result["adjusted_price"] = bid
            else:
                checks["price_within_bid_ask"] = False

        # 4. Account has margin
        margin_free = mt5_account.get("margin_free", 0.0) if isinstance(mt5_account, dict) else 0.0
        required_margin = volume * price * self.margin_fraction
        margin_ok = margin_free > required_margin
        checks["account_has_margin"] = margin_ok

        # Adjusted volume based on available margin
        if not margin_ok and margin_free > 0 and price > 0:
            max_vol = margin_free / (price * self.margin_fraction)
            result["adjusted_volume"] = min(volume, max_vol)

        # 5. Position limit
        pos_count = len(open_positions) if isinstance(open_positions, list) else 0
        checks["position_limit_not_exceeded"] = pos_count < self.max_positions

        # 6. SIL symbol still active
        checks["sil_symbol_still_active"] = symbol in sil_scores if isinstance(sil_scores, dict) else False

    @staticmethod
    def _build_rejection_reason(checks: dict) -> str:
        """Build a human-readable rejection reason from failed checks."""
        failed = [name.replace("_", " ").title() for name, ok in checks.items() if not ok]
        if not failed:
            return None
        if len(failed) == 1:
            return f"Check failed: {failed[0]}"
        return "Checks failed: " + ", ".join(failed)
