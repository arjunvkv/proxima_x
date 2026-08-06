"""proxima_ops.risk.firm_risk — FTMO-style firm (prop) survivability model.

Phase 3 (apples-to-apples survival): a strategy must not just be profitable
in backtest — it must SURVIVE the firm's risk rules, or it cannot ship. This
module encodes the FTMO-style challenge rules that gate a live account and
evaluates them against any equity sequence (backtest replay OR live).

Rules modelled (FTMO-standard, all configurable):
  * Daily loss limit     — max -5% of the day's opening equity in one day
  * Max drawdown         — max -10% from trailing peak equity
  * Profit target        — account reaches +10% (challenge size)
  * Minimum trading days — must trade on >= N distinct days
  * Maximum lot          — hard per-order size cap
  * News ban window      — optional minutes around high-impact news
                           (caller gates entries; evaluator reports)

A strategy is FIRM-SURVIVING iff no rule is violated across the run;
otherwise the first violated rule is reported so the failure is actionable.
Consumed by backtest runners to reject accounts that would blow an FTMO
challenge even though gross PnL is positive.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

logger = logging.getLogger("proxima.risk.firm")

# FTMO-standard challenge defaults (10% profit target on $100k).
DEFAULT_INITIAL_BALANCE = 100_000.0
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05     # 5% daily loss limit
DEFAULT_MAX_DRAWDOWN_PCT = 0.10       # 10% max trailing drawdown
DEFAULT_PROFIT_TARGET_PCT = 0.10      # +10% challenge target
DEFAULT_MIN_TRADING_DAYS = 4          # at least 4 trading days
DEFAULT_MAX_LOT = 5.0                 # per-position lot cap

RULE_DAILY_LOSS = "DAILY_LOSS"
RULE_DRAWDOWN = "MAX_DRAWDOWN"
RULE_MAX_LOT = "MAX_LOT"
RULE_MIN_DAYS = "MIN_TRADING_DAYS"
RULE_NEWS = "NEWS_WINDOW"


@dataclass
class FirmRiskConfig:
    """FTMO-style firm rules — configure to your actual challenge."""
    initial_balance: float = DEFAULT_INITIAL_BALANCE
    max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS
    max_lot: float = DEFAULT_MAX_LOT
    news_ban_minutes: int = 0  # >0 enables the news-window rule

    @property
    def max_daily_loss_usd(self) -> float:
        return self.initial_balance * self.max_daily_loss_pct

    @property
    def max_drawdown_usd(self) -> float:
        return self.initial_balance * self.max_drawdown_pct

    @property
    def profit_target_usd(self) -> float:
        return self.initial_balance * self.profit_target_pct


@dataclass
class FirmVerdict:
    """Result of evaluating one session against the firm rules."""
    survived: bool
    reason: str = ""
    peak_equity: float = 0.0
    final_equity: float = 0.0
    trading_days: int = 0
    max_daily_loss_pct_reached: float = 0.0
    max_drawdown_pct_reached: float = 0.0
    max_lot_seen: float = 0.0
    target_hit: bool = False

    def as_dict(self) -> dict:
        return {
            "survived": self.survived,
            "reason": self.reason,
            "peak_equity": round(self.peak_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "trading_days": self.trading_days,
            "max_daily_loss_pct_reached": round(self.max_daily_loss_pct_reached, 4),
            "max_drawdown_pct_reached": round(self.max_drawdown_pct_reached, 4),
            "max_lot_seen": self.max_lot_seen,
            "target_hit": self.target_hit,
        }


def _day_change(a: date, b: date) -> bool:
    return a != b


class FirmRiskEvaluator:
    """Evaluates a sequence of (day, equity[, lot]) snapshots vs the rules.

    Snapshots are (date, equity, lot-volume-or-None). Feed post-tick or
    post-trade equity; lots are tracked for the MAX_LOT rule when provided.
    """

    def __init__(self, config: Optional[FirmRiskConfig] = None):
        self.config = config or FirmRiskConfig()
        self.reset_state()

    def reset_state(self) -> None:
        """Reset run state so the same evaluator can score a new session."""
        self._peak = self.config.initial_balance
        self._trading_days: set = set()
        self._day_open: dict = {}
        self._max_daily_drop: float = 0.0
        self._max_dd: float = 0.0
        self._max_lot: float = 0.0

    def _violation(self, rule: str, **detail) -> FirmVerdict:
        v = FirmVerdict(
            survived=False,
            reason=rule,
            peak_equity=self._peak,
            final_equity=self._last_equity,
            trading_days=len(self._trading_days),
            max_daily_loss_pct_reached=self._max_daily_drop,
            max_drawdown_pct_reached=self._max_dd,
            max_lot_seen=self._max_lot,
        )
        logger.warning(
            f"FirmRisk {rule} violated: {detail} (peak={self._peak:.2f} "
            f"equity={self._last_equity:.2f})")
        return v

    def evaluate(self, snapshots: List[Tuple]) -> FirmVerdict:
        """Evaluate a session; returns FirmVerdict. Snapshots:
        (date, equity) or (date, equity, lot)."""
        cfg = self.config
        self.reset_state()
        self._last_equity = cfg.initial_balance
        self._peak = cfg.initial_balance

        prev_day = None
        day_open = cfg.initial_balance

        for snap in snapshots:
            day = snap[0]
            eq = float(snap[1])
            lot = float(snap[2]) if len(snap) > 2 and snap[2] is not None else 0.0
            self._last_equity = eq
            self._trading_days.add(day)

            if lot > 0:
                self._max_lot = max(self._max_lot, lot)
                # MAX_LOT rule fails fast (independent of drawdown)
                if lot > cfg.max_lot:
                    self._last_label = RULE_MAX_LOT
                    return self._violation(RULE_MAX_LOT, lot=lot, limit=cfg.max_lot)

            # daily loss bucket: reset at each new trading day
            if prev_day is not None and day != prev_day:
                day_open = eq
            prev_day = day

            daily_drop = (eq - day_open) / max(cfg.initial_balance, 1.0)
            self._max_daily_drop = min(self._max_daily_drop, daily_drop)
            if daily_drop <= -cfg.max_daily_loss_pct:
                self._last_label = RULE_DAILY_LOSS
                return self._violation(
                    RULE_DAILY_LOSS, drop=round(daily_drop, 4),
                    limit=cfg.max_daily_loss_pct)

            if eq > self._peak:
                self._peak = eq
                self._last_label = None
            dd = (self._peak - eq) / max(cfg.initial_balance, 1.0)
            self._max_dd = max(self._max_dd, dd)
            if dd >= cfg.max_drawdown_pct:
                self._last_label = RULE_DRAWDOWN
                return self._violation(
                    RULE_DRAWDOWN, dd=round(dd, 4), limit=cfg.max_drawdown_pct)

        # ---- end of session ----
        final = self._last_equity
        target = cfg.initial_balance + cfg.profit_target_usd
        target_hit = final >= target

        trading_days = len(self._trading_days)
        if not target_hit and trading_days < cfg.min_trading_days:
            self._last_label = RULE_MIN_DAYS
            v = self._violation(
                RULE_MIN_DAYS, days=trading_days, required=cfg.min_trading_days)
            return v

        # survived either by hitting target or by completing min days
        v = FirmVerdict(
            survived=True,
            reason="",
            peak_equity=self._peak,
            final_equity=final,
            trading_days=trading_days,
            max_daily_loss_pct_reached=self._max_daily_drop,
            max_drawdown_pct_reached=self._max_dd,
            max_lot_seen=self._max_lot,
            target_hit=target_hit,
        )
        return v


# Convenience app-level check used by runners and the demo.
def firm_status(evaluator: FirmRiskEvaluator, snapshots: List[Tuple]) -> FirmVerdict:
    return evaluator.evaluate(snapshots)