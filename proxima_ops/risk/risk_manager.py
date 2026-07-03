"""Unified Risk Manager — aggregates all RHL modules into one interface."""

import logging
from typing import Optional
from datetime import datetime

from .catastrophic_stop import catastrophic_sl, catastrophic_tp, get_risk_stop_distance
from .trade_risk_verifier import TradeRiskVerifier, compute_stop_risk
from .risk_governor import RiskGovernor
from .exposure_controller import ExposureController
from .risk_health_monitor import RiskHealthMonitor
from .position_watchdog import PositionWatchdog
from .risk_dashboard import RiskDashboard

logger = logging.getLogger("proxima_ops.risk.manager")

STOP_RISK_CRITICAL_HEAT = 0.05  # 5% of equity at stop risk = CRITICAL


class RiskManager:
    def __init__(self):
        self.verifier = TradeRiskVerifier()
        self.governor = RiskGovernor()
        self.exposure = ExposureController()
        self.health = RiskHealthMonitor()
        self.watchdog = PositionWatchdog()
        self.dashboard = RiskDashboard(
            governor=self.governor,
            exposure=self.exposure,
            health=self.health,
            watchdog=self.watchdog,
            verifier=self.verifier)

    def set_position_manager(self, pm) -> None:
        self.watchdog.set_position_manager(pm)

    def pre_order_check(self, symbol: str, volume: float, entry_price: float,
                         account_balance: float, open_positions: list[dict],
                         risk_pct: float = 0.0025, direction: str = "BUY") -> dict:
        if not self.governor.can_trade:
            return {"allowed": False, "reason": f"entries_paused:{self.governor.summary().get('pause_reason', 'unknown')}"}

        exp = self.exposure.check(open_positions, new_symbol=symbol)
        if not exp["allowed"]:
            return {"allowed": False, "reason": f"exposure:{exp['reason']}"}

        risk_budget = account_balance * risk_pct
        sl_price = catastrophic_sl(symbol, entry_price, direction)
        verify = self.verifier.verify(symbol, volume, entry_price, sl_price,
                                       account_balance, risk_budget, direction)
        if not verify["accepted"]:
            return {"allowed": False, "reason": f"risk_verify:{verify.get('reason', 'failed')}"}

        return {"allowed": True, "reason": "", "sl": sl_price, "tp": 0.0}

    def compute_active_risk(self, open_positions: list[dict]) -> dict:
        """Compute true stop-based risk across all open positions."""
        total_stop_risk = 0.0
        for p in open_positions:
            risk_dollars = compute_stop_risk(
                p.get("symbol", ""),
                p.get("volume", 0),
                p.get("price_open", 0),
                p.get("sl", 0),
            )
            if risk_dollars is not None:
                total_stop_risk += risk_dollars
        return {"total_stop_risk": round(total_stop_risk, 2)}

    def compute_portfolio_heat(self, open_positions: list[dict], equity: float) -> float:
        heat = 0.0
        if equity > 0:
            active = self.compute_active_risk(open_positions)
            heat = active["total_stop_risk"] / equity
        return round(heat, 6)

    def post_trade_result(self, pnl: float, equity: float):
        self.governor.record_result(pnl)
        self.governor.update_unrealized(0.0, equity)
        dd = self.governor.check_equity_drawdown(equity)
        if dd.get("triggered"):
            logger.warning(f"Equity protection triggered at {dd['drawdown_pct']:.1%} drawdown")

    def health_check(self, mt5_connected: bool, account_connected: bool,
                      symbols_available: dict) -> dict:
        return self.health.check(mt5_connected, account_connected, symbols_available)

    def watchdog_check(self, mt5_positions: list[dict], ledger_positions: list[dict]) -> dict:
        return self.watchdog.verify(mt5_positions, ledger_positions)

    def dashboard_section(self, account_balance: float, open_positions: list[dict] = None) -> str:
        return self.dashboard.generate(account_balance, open_positions, risk_manager=self)

    def summary(self) -> dict:
        g = self.governor.summary()
        return {
            "risk_state": g.get("state", "HEALTHY"),
            "daily_pnl": g.get("daily_pnl", 0),
            "loss_streak": g.get("loss_streak", 0),
            "entries_paused": g.get("entries_paused", False),
            "pause_reason": g.get("pause_reason", ""),
            "rejected_orders": self.verifier.rejected_count()}
