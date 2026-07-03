import json
import logging
import os
import time

logger = logging.getLogger("proxima_ops.monitoring.mt5_watchdog")

_PIP_VALUES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "EURJPY": 0.01,
    "USDJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
}


class MT5Watchdog:
    def __init__(self, state_path: str = "state/mt5_watchdog_state.json"):
        self.state_path = state_path
        self._attempt_counter = 0
        self._order_attempts: list[dict] = []
        self._order_results: list[dict] = []
        self._slippage_by_symbol: dict[str, dict] = {}
        self._failure_counts: dict[int, int] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._consecutive_failure_alerts: dict[str, bool] = {}
        self._spike_log: list[dict] = []
        self._load_state()

    @staticmethod
    def _pip_value(symbol: str) -> float:
        return _PIP_VALUES.get(symbol, 0.0001)

    def _slippage_pips(self, direction: str, requested_price: float, fill_price: float, symbol: str) -> float:
        raw = fill_price - requested_price
        if direction.upper() == "SELL":
            raw = -raw
        return raw / self._pip_value(symbol)

    def record_order_attempt(self, order_data: dict) -> int:
        self._attempt_counter += 1
        record = {
            "attempt_id": self._attempt_counter,
            "timestamp": time.time(),
            "symbol": order_data.get("symbol", ""),
            "direction": order_data.get("direction", ""),
            "requested_volume": order_data.get("volume", 0),
            "requested_price": order_data.get("price", 0),
            "order_type": order_data.get("type", "market"),
        }
        self._order_attempts.append(record)
        self._save_state()
        return self._attempt_counter

    def record_order_result(self, result: dict):
        retcode = result.get("retcode", -1)
        sym = result.get("symbol", "")
        record = {
            "timestamp": time.time(),
            "symbol": sym,
            "ticket": result.get("ticket", 0),
            "fill_price": result.get("price", 0),
            "fill_volume": result.get("volume", 0),
            "retcode": retcode,
            "comment": result.get("comment", ""),
            "requested_price": result.get("requested_price", 0),
            "direction": result.get("direction", ""),
        }
        self._order_results.append(record)

        is_done = retcode == 10009
        if not is_done:
            self._failure_counts[retcode] = self._failure_counts.get(retcode, 0) + 1
            self._consecutive_failures[sym] = self._consecutive_failures.get(sym, 0) + 1
            if self._consecutive_failures[sym] >= 3 and not self._consecutive_failure_alerts.get(sym):
                self._consecutive_failure_alerts[sym] = True
                logger.warning(
                    f"[WATCHDOG] 3+ consecutive failures on {sym} "
                    f"(count={self._consecutive_failures[sym]})"
                )
        else:
            self._consecutive_failures[sym] = 0
            self._consecutive_failure_alerts[sym] = False

        fill_price = result.get("price", 0)
        req_price = result.get("requested_price", 0)
        direction = result.get("direction", "")
        if fill_price and req_price and sym:
            slippage = self._slippage_pips(direction, req_price, fill_price, sym)
            if sym not in self._slippage_by_symbol:
                self._slippage_by_symbol[sym] = {
                    "count": 0,
                    "total_slippage_pips": 0.0,
                    "total_abs_slippage_pips": 0.0,
                    "avg_abs_slippage_pips": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "last_slippage": 0.0,
                }
            ss = self._slippage_by_symbol[sym]
            ss["count"] += 1
            ss["total_slippage_pips"] += slippage
            ss["total_abs_slippage_pips"] += abs(slippage)
            ss["avg_abs_slippage_pips"] = ss["total_abs_slippage_pips"] / ss["count"]
            if slippage > 0:
                ss["positive_count"] += 1
            elif slippage < 0:
                ss["negative_count"] += 1
            ss["last_slippage"] = slippage

            if ss["avg_abs_slippage_pips"] > 0 and abs(slippage) > 3 * ss["avg_abs_slippage_pips"]:
                spike = {
                    "timestamp": time.time(),
                    "symbol": sym,
                    "slippage_pips": round(slippage, 4),
                    "avg_abs_slippage_pips": round(ss["avg_abs_slippage_pips"], 4),
                    "threshold": round(3 * ss["avg_abs_slippage_pips"], 4),
                    "direction": direction,
                    "ticket": result.get("ticket", 0),
                }
                self._spike_log.append(spike)
                logger.warning(
                    f"[WATCHDOG] Slippage spike on {sym}: "
                    f"{slippage:.4f} pips (avg={ss['avg_abs_slippage_pips']:.4f}, "
                    f"threshold={3 * ss['avg_abs_slippage_pips']:.4f})"
                )

        self._save_state()

    def check_integrity(self) -> dict:
        issues = []
        for sym, fails in self._consecutive_failures.items():
            if fails >= 3:
                issues.append(f"Consecutive failures on {sym}: {fails}")
        for sym, ss in self._slippage_by_symbol.items():
            if ss["avg_abs_slippage_pips"] > 10:
                issues.append(f"High avg slippage on {sym}: {ss['avg_abs_slippage_pips']:.2f} pips")
        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "total_attempts": len(self._order_attempts),
            "total_results": len(self._order_results),
            "total_spikes": len(self._spike_log),
        }

    def has_critical_issue(self) -> bool:
        for fails in self._consecutive_failures.values():
            if fails >= 3:
                return True
        return False

    def summarize(self) -> dict:
        slippage_summary = {}
        for sym, ss in self._slippage_by_symbol.items():
            slippage_summary[sym] = {
                "total_orders": ss["count"],
                "avg_abs_slippage_pips": round(ss["avg_abs_slippage_pips"], 4),
                "avg_signed_slippage_pips": round(ss["total_slippage_pips"] / ss["count"], 4) if ss["count"] else 0,
                "positive_slippage_events": ss["positive_count"],
                "negative_slippage_events": ss["negative_count"],
                "last_slippage_pips": round(ss["last_slippage"], 4),
            }

        return {
            "total_attempts": len(self._order_attempts),
            "total_results": len(self._order_results),
            "slippage_by_symbol": slippage_summary,
            "failure_counts_by_code": dict(self._failure_counts),
            "consecutive_failures_by_symbol": dict(self._consecutive_failures),
            "consecutive_failure_alerts": dict(self._consecutive_failure_alerts),
            "spike_count": len(self._spike_log),
            "last_spikes": self._spike_log[-10:] if self._spike_log else [],
            "has_critical_issue": self.has_critical_issue(),
            "recent_attempts": self._order_attempts[-20:] if self._order_attempts else [],
            "recent_results": self._order_results[-20:] if self._order_results else [],
        }

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        state = {
            "attempt_counter": self._attempt_counter,
            "order_attempts": self._order_attempts[-200:],
            "order_results": self._order_results[-200:],
            "slippage_by_symbol": self._slippage_by_symbol,
            "failure_counts": dict(self._failure_counts),
            "consecutive_failures": dict(self._consecutive_failures),
            "consecutive_failure_alerts": dict(self._consecutive_failure_alerts),
            "spike_log": self._spike_log[-50:],
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                self._attempt_counter = state.get("attempt_counter", 0)
                self._order_attempts = state.get("order_attempts", [])
                self._order_results = state.get("order_results", [])
                self._slippage_by_symbol = state.get("slippage_by_symbol", {})
                self._failure_counts = {int(k): v for k, v in state.get("failure_counts", {}).items()}
                self._consecutive_failures = state.get("consecutive_failures", {})
                self._consecutive_failure_alerts = state.get("consecutive_failure_alerts", {})
                self._spike_log = state.get("spike_log", [])
            except Exception as e:
                logger.error(f"[WATCHDOG] Failed to load state: {e}")
