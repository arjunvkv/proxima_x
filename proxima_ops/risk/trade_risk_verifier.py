"""RHL-2: Trade Risk Verifier — validate every order before submission."""

import logging
from typing import Optional
from proxima_ops.risk.catastrophic_stop import get_risk_stop_distance

logger = logging.getLogger("proxima_ops.risk.verifier")


def compute_stop_risk(symbol: str, volume: float, entry_price: float, sl_price: float) -> Optional[float]:
    """Compute true dollar risk = volume * pip_value_per_lot * stop_distance_pips.
    
    Returns dollar risk amount, or None if not computable.
    """
    if sl_price <= 0 or volume <= 0 or entry_price <= 0:
        return None
    sd = get_risk_stop_distance(symbol)
    pip_size = sd["pip_size"]
    pip_dist = abs(entry_price - sl_price)
    point_value_per_lot = 1000.0 / max(entry_price, 1.0) if "JPY" in symbol else 10.0
    stop_points = pip_dist / pip_size
    return volume * point_value_per_lot * stop_points


class TradeRiskVerifier:
    def __init__(self):
        self._rejected: list[dict] = []

    def verify(self, symbol: str, volume: float, entry_price: float,
               sl_price: float, account_balance: float, risk_budget: float,
               order_type: str = "BUY") -> dict:
        if account_balance <= 0:
            return self._reject(symbol, volume, "zero_balance")
        if volume <= 0:
            return self._reject(symbol, volume, "zero_volume")
        if sl_price <= 0:
            return self._reject(symbol, volume, "NO_STOP")

        actual_dollar_risk = compute_stop_risk(symbol, volume, entry_price, sl_price)
        if actual_dollar_risk is None:
            return self._reject(symbol, volume, "uncomputable_risk")

        if actual_dollar_risk > risk_budget * 1.05:
            # Budget-fit scaling: compute max volume that fits within budget
            scale_factor = risk_budget / actual_dollar_risk if actual_dollar_risk > 0 else 0
            scaled_volume = round(volume * scale_factor, 2)
            # Normalize to MT5 volume step
            try:
                import MetaTrader5 as mt5
                info = mt5.symbol_info(symbol)
                if info:
                    step = info.volume_step
                    vmin = info.volume_min
                    scaled_volume = round(scaled_volume / step) * step
                    scaled_volume = max(vmin, scaled_volume)
                    # Post-normalization recheck: volume_step rounding can push risk back over budget
                    for _ in range(10):
                        if scaled_volume <= vmin:
                            break
                        _rv = compute_stop_risk(symbol, scaled_volume, entry_price, sl_price)
                        if _rv is None or _rv <= risk_budget * 1.05:
                            break
                        scaled_volume = max(vmin, scaled_volume - step)
            except Exception:
                pass
            if scaled_volume > 0 and scaled_volume < volume:
                # Recompute risk with scaled volume
                import math
                risk_ratio = scaled_volume / volume
                scaled_risk = actual_dollar_risk * risk_ratio
                logger.info(f"[BUDGET_FIT] {symbol}: risk=${actual_dollar_risk:.2f} > budget=${risk_budget:.2f}, scaling vol {volume}->{scaled_volume} (risk={scaled_risk:.2f})")
                volume = scaled_volume
                actual_dollar_risk = scaled_risk
            else:
                return self._reject(symbol, 0, f"budget_fit_zero_volume")

        # Recalculate stop risk with scaled volume (if scaling happened)
        actual_dollar_risk = compute_stop_risk(symbol, volume, entry_price, sl_price) or actual_dollar_risk

        entry = {
            "accepted": True,
            "symbol": symbol,
            "volume": volume,
            "risk_budget": risk_budget,
            "actual_dollar_risk": round(actual_dollar_risk, 2),
            "sl_price": sl_price}
        logger.info(f"Risk verifier PASS: {symbol} vol={volume} risk=${actual_dollar_risk:.2f}")
        return entry

    def _reject(self, symbol: str, volume: float, reason: str) -> dict:
        entry = {"accepted": False, "symbol": symbol, "volume": volume, "reason": reason}
        self._rejected.append(entry)
        logger.warning(f"Risk verifier REJECT: {symbol} vol={volume} reason={reason}")
        return entry

    def rejected_count(self) -> int:
        return len(self._rejected)
