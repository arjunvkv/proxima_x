"""RHL-9: Risk Dashboard — aggregate all RHL states into one display section."""

from typing import Optional
from .catastrophic_stop import CATASTROPHIC_STOP_PIPS
from .exposure_controller import MAX_POSITIONS_TOTAL, MAX_FX_POSITIONS, MAX_GOLD_POSITIONS, MAX_INDEX_POSITIONS
from .risk_governor import MAX_DAILY_LOSS_PCT, CONSECUTIVE_LOSS_LIMIT, EQUITY_DRAWDOWN_LIMIT

STOP_RISK_HEAT_WARN = 0.03
STOP_RISK_HEAT_CRITICAL = 0.05


class RiskDashboard:
    def __init__(self, governor=None, exposure=None, health=None, watchdog=None, verifier=None):
        self._gov = governor
        self._exp = exposure
        self._health = health
        self._watch = watchdog
        self._ver = verifier

    def generate(self, account_balance: float = 0.0, open_positions: list = None,
                 ledger_positions: list = None, risk_manager=None) -> str:
        if open_positions is None:
            open_positions = []
        if ledger_positions is None:
            ledger_positions = open_positions
        risk_budget = account_balance * 0.0025

        active_risk = {"total_stop_risk": 0.0}
        portfolio_heat = 0.0
        if risk_manager:
            active_risk = risk_manager.compute_active_risk(open_positions)
            portfolio_heat = risk_manager.compute_portfolio_heat(open_positions, account_balance)

        heat_level = "NORMAL"
        if portfolio_heat >= STOP_RISK_HEAT_CRITICAL:
            heat_level = "CRITICAL"
        elif portfolio_heat >= STOP_RISK_HEAT_WARN:
            heat_level = "WARNING"

        lines = []
        lines.append("=" * 52)
        lines.append("  RISK HARDENING LAYER")
        lines.append("=" * 52)

        g = self._gov.summary() if self._gov else {}
        e = self._exp.check(open_positions) if self._exp else {}
        h = self._health.check(True, True, {}, True) if self._health else {}
        w = self._watch.verify(open_positions, ledger_positions) if self._watch else {}

        lines.append(f"  Risk Budget:          ${risk_budget:<8.2f}")
        lines.append(f"  True Stop Risk:       ${active_risk['total_stop_risk']:<8.2f}")
        lines.append(f"  Portfolio Heat:       {portfolio_heat:.4f}  ({heat_level})")
        lines.append(f"  Daily PnL:            ${g.get('daily_pnl', 0):<8.2f}")
        lines.append(f"  Loss Streak:          {g.get('loss_streak', 0)}")
        lines.append(f"  Drawdown:             {g.get('drawdown_pct', 0)*100 if isinstance(g.get('drawdown_pct'), (int,float)) else 0:.1f}%")
        lines.append(f"  Risk State:           {self._resolve_state(g, h, w, heat_level)}")
        lines.append(f"  Entries Paused:       {'YES' if g.get('entries_paused', False) else 'NO'}")
        if g.get('pause_reason'):
            lines.append(f"  Pause Reason:         {g['pause_reason']}")

        lines.append("-" * 52)
        lines.append("  Limits Active:")
        lines.append(f"    Max Positions:       {MAX_POSITIONS_TOTAL} total, {MAX_FX_POSITIONS} FX, {MAX_GOLD_POSITIONS} gold, {MAX_INDEX_POSITIONS} index")
        lines.append(f"    Max Daily Loss:      {MAX_DAILY_LOSS_PCT:.0%} equity")
        lines.append(f"    Consecutive Loss:    {CONSECUTIVE_LOSS_LIMIT}")
        lines.append(f"    Equity Floor:        {EQUITY_DRAWDOWN_LIMIT:.0%} drawdown")
        lines.append("")
        lines.append("  Catastrophic Stops (pips):")
        for sym, pips in sorted(CATASTROPHIC_STOP_PIPS.items()):
            lines.append(f"    {sym:<12s} {pips}")
        lines.append("")
        lines.append("=" * 52)
        return "\n".join(lines)

    def _resolve_state(self, gov: dict, health: dict, watch: dict, heat_level: str = "NORMAL") -> str:
        if heat_level == "CRITICAL":
            return "CRITICAL"
        if watch.get("state") == "CRITICAL_POSITION_MISMATCH":
            return "CRITICAL"
        if health.get("state") == "BROKER_FAILURE":
            return "BROKER_FAILURE"
        if gov.get("state") in ("EQUITY_PROTECTION", "DAILY_STOP", "LOSS_STREAK_STOP"):
            return gov["state"]
        return "HEALTHY"
