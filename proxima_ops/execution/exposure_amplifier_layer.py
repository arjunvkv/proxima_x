"""Exposure Amplifier Layer — optional performance-based scaling overlay.

Adjusts final position size without affecting validation staircase or any
other pipeline component. Pure multiplier computation from recent trades,
drawdown, and volatility regime."""
import logging
import json
import os
from typing import Optional
from collections import deque

logger = logging.getLogger("proxima_ops.execution.eal")

STATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "state", "exposure_amplifier_state.json"
)
BALANCE = 25000.0
MAX_WINDOW = 10
MIN_TRADES_FOR_SCALING = 3


class ExposureAmplifierLayer:
    """Optional scaling overlay for position sizing.

    Computes a multiplier from recent trade performance, current drawdown,
    and volatility regime. Defaults to 1.0 when disabled or data is
    insufficient. Never modifies staircase state or pipeline components."""

    def __init__(self, state_path: Optional[str] = None, enabled: bool = True):
        self._state_path = state_path or STATE_PATH
        self._recent_trades: deque[float] = deque(maxlen=MAX_WINDOW)
        self._loss_streak: int = 0
        self._enabled: bool = enabled
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            logger.info("No saved eal state at %s, starting fresh", self._state_path)
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            trades = data.get("recent_trades", [])
            self._recent_trades = deque(trades, maxlen=MAX_WINDOW)
            self._loss_streak = data.get("loss_streak", 0)
            logger.info("Loaded eal state from %s (%d trades, loss_streak=%d)",
                        self._state_path, len(self._recent_trades), self._loss_streak)
        except Exception as exc:
            logger.warning("Failed to load eal state from %s: %s", self._state_path, exc)

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        data = {
            "recent_trades": list(self._recent_trades),
            "loss_streak": self._loss_streak,
            "multiplier": self.get_multiplier(
                list(self._recent_trades), 0.0, "normal"
            ),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved eal state to %s", self._state_path)
        except Exception as exc:
            logger.warning("Failed to save eal state to %s: %s", self._state_path, exc)

    # ── Enabled flag ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info("[AMPLIFIER] %s", "Enabled" if value else "Disabled")

    @property
    def loss_streak(self) -> int:
        return self._loss_streak

    # ── Public interface ───────────────────────────────────────────────

    def record_trade(self, pnl: float) -> None:
        self._recent_trades.append(pnl)
        if pnl <= 0:
            self._loss_streak += 1
        else:
            self._loss_streak = 0
        self._save_state()
        logger.debug("Recorded trade pnl=%.2f, loss_streak=%d", pnl, self._loss_streak)

    def get_loss_streak(self) -> int:
        return self._loss_streak

    def get_multiplier(
        self,
        recent_trade_pnls: Optional[list[float]] = None,
        current_drawdown: float = 0.0,
        volatility_regime: str = "normal",
    ) -> float:
        if not self._enabled:
            return 1.0

        trades = recent_trade_pnls if recent_trade_pnls is not None else list(self._recent_trades)
        if not trades:
            return 1.0

        multiplier = 1.0

        n = len(trades)
        if n >= MIN_TRADES_FOR_SCALING:
            wins = [p for p in trades if p > 0]
            losses = [p for p in trades if p <= 0]
            win_rate = len(wins) / n
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

            if win_rate < 0.4:
                multiplier = 0.8 + 0.5 * win_rate
            elif win_rate > 0.6 and avg_win > avg_loss:
                multiplier = 1.0 + 0.5 * (win_rate - 0.6)
            else:
                multiplier = 1.0

        drawdown_pct = current_drawdown / BALANCE
        if drawdown_pct > 0.05:
            multiplier *= 0.5
        elif drawdown_pct > 0.02:
            multiplier *= 0.75

        if volatility_regime == "high":
            multiplier *= 0.85

        if self._loss_streak >= 3:
            multiplier = 0.5

        multiplier = max(0.5, min(1.5, multiplier))

        return multiplier

    def describe(self) -> dict:
        trades = list(self._recent_trades)
        n = len(trades)
        wins = [p for p in trades if p > 0] if n > 0 else []
        losses = [p for p in trades if p <= 0] if n > 0 else []
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        return {
            "multiplier": self.get_multiplier(),
            "loss_streak": self._loss_streak,
            "win_rate": win_rate,
            "total_trades": n,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "recent_trades": trades,
        }

    def reset(self) -> None:
        self._recent_trades.clear()
        self._loss_streak = 0
        self._save_state()
        logger.info("Reset eal state (trades cleared, loss_streak=0)")
