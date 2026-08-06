"""RHL-3+4+5: Daily Loss Governor, Consecutive Loss Kill, Equity Floor."""

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.risk.governor")

MAX_DAILY_LOSS_PCT = 0.01
CONSECUTIVE_LOSS_LIMIT = 3
EQUITY_DRAWDOWN_LIMIT = 0.10
LOSS_STREAK_HALT_HOURS = 24


class RiskGovernor:
    def __init__(self, clock=None):
        # Optional injectable clock for the day-boundary key. When replaying
        # historical ticks, pass a ReplayClock so the daily-loss counter resets
        # on the TAPE's trading day, not the machine's wall-clock day (FirmRisk
        # alignment gap C5). None = legacy wall-clock behavior.
        self._clock = clock
        self._daily_loss: float = 0.0
        self._daily_unrealized: float = 0.0
        # Lazy day key: None until the first check() call. This prevents an
        # epoch-start ReplayClock from wiping the first day's accrued loss
        # on the very first check (init would snapshot 1970-01-01).
        self._today: Optional[str] = None
        self._loss_streak: int = 0
        self._peak_equity: float = 0.0
        self._start_equity: float = 0.0
        self._entries_paused: bool = False
        self._pause_reason: str = ""
        self._pause_until: Optional[datetime] = None
        self._state = "HEALTHY"
        self._last_realized_pnl: float = 0.0

    def set_start_equity(self, equity: float):
        self._start_equity = equity
        self._peak_equity = equity

    def record_result(self, pnl: float):
        self._daily_loss += pnl
        self._last_realized_pnl = pnl

        if pnl > 0:
            self._loss_streak = 0
        else:
            self._loss_streak += 1

    def update_unrealized(self, unrealized_pnl: float, equity: float):
        self._daily_unrealized = unrealized_pnl
        if equity > self._peak_equity:
            self._peak_equity = equity

    def check(self) -> dict:
        # day boundary from the injected clock (replay day) or wall clock.
        if self._clock is not None and getattr(self._clock, 'now', None):
            now = self._clock.now()
            today_str = now.date().isoformat()
        else:
            now = datetime.now()
            today_str = date.today().isoformat()
        if self._today is None:
            # first observation: adopt the key, never reset.
            self._today = today_str
        elif today_str != self._today:
            self._daily_loss = 0.0
            self._daily_unrealized = 0.0
            self._today = today_str

        if self._entries_paused:
            if self._pause_until and now >= self._pause_until:
                self._entries_paused = False
                self._pause_reason = ""
                self._state = "HEALTHY"
                logger.info("Risk governor: auto-resumed after cooldown")

        if not self._entries_paused:
            if self._start_equity <= 0:
                self._start_equity = self._peak_equity  # fallback to peak equity
            elif self._daily_loss + self._daily_unrealized <= -self._start_equity * MAX_DAILY_LOSS_PCT:
                self._entries_paused = True
                self._pause_reason = "DAILY_STOP"
                self._state = "DAILY_STOP"
                logger.warning(f"Risk governor: DAILY_STOP at ${self._daily_loss + self._daily_unrealized:.2f}")

            elif self._loss_streak >= CONSECUTIVE_LOSS_LIMIT:
                self._entries_paused = True
                self._pause_reason = "LOSS_STREAK_STOP"
                self._state = "LOSS_STREAK_STOP"
                self._pause_until = now.__class__.fromtimestamp(now.timestamp() + LOSS_STREAK_HALT_HOURS * 3600)
                logger.warning(f"Risk governor: LOSS_STREAK_STOP ({self._loss_streak} consecutive losses)")

        return {
            "state": self._state,
            "daily_pnl": round(self._daily_loss + self._daily_unrealized, 2),
            "loss_streak": self._loss_streak,
            "entries_paused": self._entries_paused,
            "pause_reason": self._pause_reason}

    def check_equity_drawdown(self, equity: float) -> dict:
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown_pct = (self._peak_equity - equity) / max(self._peak_equity, 1.0)
        if drawdown_pct >= EQUITY_DRAWDOWN_LIMIT:
            self._entries_paused = True
            self._pause_reason = "EQUITY_PROTECTION"
            self._state = "EQUITY_PROTECTION"
            logger.warning(f"Risk governor: EQUITY_PROTECTION at {drawdown_pct:.1%} drawdown")
            return {"triggered": True, "drawdown_pct": round(drawdown_pct, 4), "state": "EQUITY_PROTECTION"}
        return {"triggered": False, "drawdown_pct": round(drawdown_pct, 4), "state": self._state}

    def pause_entries(self, reason: str = "MANUAL"):
        self._entries_paused = True
        self._pause_reason = reason
        self._state = reason
        logger.warning(f"Risk governor: entries paused ({reason})")

    def resume_entries(self):
        self._entries_paused = False
        self._pause_reason = ""
        self._state = "HEALTHY"
        logger.info("Risk governor: entries resumed")

    @property
    def can_trade(self) -> bool:
        return not self._entries_paused

    def summary(self) -> dict:
        return self.check()
