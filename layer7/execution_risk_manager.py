"""
Execution Risk Manager — Wave 3: Execution + Portfolio Safety Layer.

Provides:
- KillSwitch (P0.22): hard global safety override
- DrawdownManager (P0.23): peak equity tracking + breach detection
- LockManager: execution asymmetry repair
- convexity_corrected_mfe(): MFE/ER quality normalization
- risk_gate(): unified safety check combining all signals
"""
from typing import Dict, Optional


class KillSwitch:
    """P0.22: Hard system kill switch — single-use global halt."""

    def __init__(self):
        self._active = False
        self._reason: Optional[str] = None

    def trigger(self, reason: str) -> None:
        self._active = True
        self._reason = reason

    def reset(self) -> None:
        self._active = False
        self._reason = None

    def is_active(self) -> bool:
        return self._active

    def reason(self) -> Optional[str]:
        return self._reason


class DrawdownManager:
    """P0.23: Peak equity tracking with breach detection."""

    def __init__(self, threshold: float = 0.25):
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.threshold = threshold
        self._breached = False

    def update(self, equity: float) -> None:
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def drawdown(self) -> float:
        return (self.peak_equity - self.current_equity) / (self.peak_equity + 1e-8)

    def is_breached(self) -> bool:
        if self.drawdown() > self.threshold:
            self._breached = True
        return self._breached

    def was_breached(self) -> bool:
        return self._breached

    def reset(self) -> None:
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self._breached = False


def convexity_corrected_mfe(mfe: float, er: float, alpha: float = 0.3, max_mult: float = 2.0) -> float:
    """MFE/ER convexity correction — penalises low-efficiency high-MFE trades.
    Capped at max_mult to prevent degenerate blow-up near-zero ER.
    Applied when MFE is high but efficiency (MFE/ER) is poor.
    """
    if er < 1e-8:
        return mfe
    efficiency = mfe / er
    multiplier = min(1.0 + alpha * efficiency, max_mult)
    return mfe * multiplier


class LockManager:
    """Directional symmetry enforcement — prevents long/short bias leakage."""

    def __init__(self):
        self._locks: Dict[str, int] = {}

    def set_lock(self, symbol: str, direction: int) -> None:
        self._locks[symbol] = direction

    def can_trade(self, symbol: str, direction: int) -> bool:
        if symbol not in self._locks:
            return True
        return self._locks[symbol] != direction

    def clear(self) -> None:
        self._locks.clear()

    def state(self) -> Dict[str, int]:
        return dict(self._locks)


def risk_gate(
    drawdown: float,
    cf_block_rate: float,
    tpi_collapse: bool,
    kill_switch: KillSwitch,
    drawdown_threshold: float = 0.25,
    cf_threshold: float = 0.6,
) -> bool:
    """Unified risk gate — all safety signals must pass for execution."""
    if kill_switch.is_active():
        return False
    if drawdown > drawdown_threshold:
        return False
    if cf_block_rate > cf_threshold:
        return False
    if tpi_collapse:
        return False
    return True


class PortfolioGraph:
    """Wave 4: Cross-symbol exposure aggregation with kill threshold."""

    def __init__(self, kill_threshold: float = 1.5):
        self._exposure: Dict[str, float] = {}
        self.kill_threshold = kill_threshold

    def update(self, symbol: str, exposure_value: float) -> None:
        self._exposure[symbol] = exposure_value

    def portfolio_risk(self) -> float:
        return sum(abs(x) for x in self._exposure.values())

    def is_overexposed(self) -> bool:
        return self.portfolio_risk() > self.kill_threshold

    def exposure(self, symbol: str) -> float:
        return self._exposure.get(symbol, 0.0)

    def state(self) -> dict:
        return {
            "exposures": dict(self._exposure),
            "total_risk": round(self.portfolio_risk(), 4),
            "overexposed": self.is_overexposed(),
            "kill_threshold": self.kill_threshold,
        }
