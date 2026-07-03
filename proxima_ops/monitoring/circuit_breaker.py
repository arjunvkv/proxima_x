import logging
import os
import json
import time
from collections import deque

logger = logging.getLogger("proxima_ops.monitoring.circuit_breaker")


class CircuitBreaker:
    def __init__(self, drawdown_limit: float = -50.0, state_path: str = "state/circuit_breaker_state.json"):
        self._drawdown_limit = drawdown_limit
        self._state_path = state_path
        self._triggered = False
        self._trigger_reasons: list[str] = []
        self._consecutive_mt5_failures = 0
        self._slippages: deque = deque(maxlen=50)
        self._slippage_rolling_avg = 0.0
        self._vel_block_rates: deque = deque(maxlen=100)
        self._session_pnl = 0.0
        self._trade_timestamps: deque = deque()
        self._load_state()

    def _get_state_path(self) -> str:
        return self._state_path

    def _load_state(self):
        path = self._get_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                s = json.load(f)
            self._triggered = s.get("triggered", False)
            self._trigger_reasons = s.get("trigger_reasons", [])
            self._consecutive_mt5_failures = s.get("consecutive_mt5_failures", 0)
            self._slippages = deque(s.get("slippages", []), maxlen=50)
            self._slippage_rolling_avg = s.get("slippage_rolling_avg", 0.0)
            self._vel_block_rates = deque(s.get("vel_block_rates", []), maxlen=100)
            self._session_pnl = s.get("session_pnl", 0.0)
            self._trade_timestamps = deque(s.get("trade_timestamps", []))
            logger.info(f"[CIRCUIT_BREAKER] State loaded from {path}")
        except Exception as e:
            logger.warning(f"[CIRCUIT_BREAKER] Load state failed: {e}")

    def _save_state(self):
        parts = os.path.split(self._get_state_path())
        if parts[0]:
            os.makedirs(parts[0], exist_ok=True)
        s = {
            "triggered": self._triggered,
            "trigger_reasons": self._trigger_reasons,
            "consecutive_mt5_failures": self._consecutive_mt5_failures,
            "slippages": list(self._slippages),
            "slippage_rolling_avg": self._slippage_rolling_avg,
            "vel_block_rates": list(self._vel_block_rates),
            "session_pnl": self._session_pnl,
            "trade_timestamps": list(self._trade_timestamps),
        }
        try:
            with open(self._get_state_path(), "w") as f:
                json.dump(s, f, indent=2)
        except Exception as e:
            logger.warning(f"[CIRCUIT_BREAKER] Save state failed: {e}")

    def check_execution_attempt(
        self, symbol: str, direction: str, current_positions: int,
        mt5_watchdog_status: dict, vel_block_rate: float
    ) -> tuple[bool, str]:
        if self._triggered:
            return False, "Circuit breaker already triggered"

        if current_positions >= 6:
            return False, f"Max positions: {current_positions} >= 6"

        if self._consecutive_mt5_failures >= 3:
            return self._trigger_and_return("3 consecutive MT5 execution failures")

        self._vel_block_rates.append(vel_block_rate)
        if len(self._vel_block_rates) >= 20:
            avg_rate = sum(self._vel_block_rates) / len(self._vel_block_rates)
            if avg_rate > 0.9:
                return self._trigger_and_return(
                    f"VEL block rate {avg_rate:.1%} > 90% over {len(self._vel_block_rates)} cycles"
                )

        if self._session_pnl <= self._drawdown_limit:
            return self._trigger_and_return(
                f"Session PnL {self._session_pnl:.2f} <= drawdown limit {self._drawdown_limit:.2f}"
            )

        now = time.time()
        cutoff = now - 3600
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()
        if len(self._trade_timestamps) >= 5:
            return self._trigger_and_return(
                f"Trade burst: {len(self._trade_timestamps)} trades in last 60 minutes"
            )

        self._save_state()
        return True, "Allowed"

    def record_trade_result(self, trade_data: dict):
        pnl = trade_data.get("pnl", 0.0)
        self._session_pnl += pnl
        self._trade_timestamps.append(time.time())
        self._consecutive_mt5_failures = 0
        if self._session_pnl <= self._drawdown_limit:
            self._trigger(
                f"Session PnL {self._session_pnl:.2f} <= drawdown limit {self._drawdown_limit:.2f}"
            )
        self._save_state()
        logger.info(
            f"[CIRCUIT_BREAKER] Trade result: pnl={pnl:.2f}, "
            f"session_pnl={self._session_pnl:.2f}"
        )

    def record_mt5_failure(self):
        self._consecutive_mt5_failures += 1
        logger.warning(f"[CIRCUIT_BREAKER] MT5 failure #{self._consecutive_mt5_failures}")
        if self._consecutive_mt5_failures >= 3:
            self._trigger("3 consecutive MT5 execution failures")
        self._save_state()

    def record_slippage(self, slippage_pips: float):
        if len(self._slippages) >= 2:
            avg = sum(self._slippages) / len(self._slippages)
            self._slippage_rolling_avg = avg
            if slippage_pips < 0 and abs(slippage_pips) > 3 * abs(avg):
                self._trigger(
                    f"Negative slippage spike: {slippage_pips} vs rolling avg {avg:.2f}"
                )
        self._slippages.append(slippage_pips)
        self._save_state()

    def is_triggered(self) -> bool:
        return self._triggered

    def triggered_reasons(self) -> list[str]:
        return list(self._trigger_reasons)

    def reset(self):
        self._triggered = False
        self._trigger_reasons = []
        self._consecutive_mt5_failures = 0
        self._slippages = deque(maxlen=50)
        self._slippage_rolling_avg = 0.0
        self._vel_block_rates = deque(maxlen=100)
        self._session_pnl = 0.0
        self._trade_timestamps = deque()
        self._save_state()
        logger.info("[CIRCUIT_BREAKER] Reset")

    def summarize(self) -> dict:
        vel_avg = (
            sum(self._vel_block_rates) / len(self._vel_block_rates)
            if self._vel_block_rates else 0.0
        )
        now = time.time()
        trades_60m = sum(1 for t in self._trade_timestamps if t >= now - 3600)
        return {
            "triggered": self._triggered,
            "trigger_reasons": list(self._trigger_reasons),
            "consecutive_mt5_failures": self._consecutive_mt5_failures,
            "slippage_rolling_avg": round(self._slippage_rolling_avg, 4),
            "slippage_samples": len(self._slippages),
            "vel_block_rate_avg": round(vel_avg, 4),
            "vel_samples": len(self._vel_block_rates),
            "session_pnl": round(self._session_pnl, 2),
            "drawdown_limit": self._drawdown_limit,
            "trades_last_60min": trades_60m,
            "total_trades": len(self._trade_timestamps),
        }

    def _trigger_and_return(self, reason: str) -> tuple[bool, str]:
        self._trigger(reason)
        return False, f"Circuit breaker: {reason}"

    def _trigger(self, reason: str):
        if not self._triggered:
            self._triggered = True
            self._trigger_reasons.append(reason)
            logger.error(f"[CIRCUIT_BREAKER] TRIGGERED: {reason}")
            self._save_state()
