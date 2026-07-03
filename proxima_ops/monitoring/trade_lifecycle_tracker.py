import json
import logging
import os
import time

logger = logging.getLogger("proxima_ops.monitoring.trade_lifecycle_tracker")

STATE_PATH = "state/trade_lifecycle_state.json"


class TradeLifecycleTracker:

    def __init__(self, state_path: str = STATE_PATH):
        self._state_path = state_path
        self._vel_checks = 0
        self._vel_allowed = 0
        self._vel_blocked = 0
        self._vel_blocked_reasons = {
            "temporal_spacing": 0,
            "exposure_smoothing": 0,
            "burst_prevention": 0,
        }
        self._vel_per_symbol = {}
        self._open_trades = {}
        self._closed_trades = []
        self._trade_history = []
        self._load_state()

    def record_vel_decision(self, symbol: str, direction: str, allowed: bool, reason: str):
        self._vel_checks += 1
        if allowed:
            self._vel_allowed += 1
        else:
            self._vel_blocked += 1
            if reason in self._vel_blocked_reasons:
                self._vel_blocked_reasons[reason] += 1
        if symbol not in self._vel_per_symbol:
            self._vel_per_symbol[symbol] = {"checks": 0, "allowed": 0, "blocked": 0}
        self._vel_per_symbol[symbol]["checks"] += 1
        if allowed:
            self._vel_per_symbol[symbol]["allowed"] += 1
        else:
            self._vel_per_symbol[symbol]["blocked"] += 1
        self._save_state()
        logger.debug(f"[VEL_AUDIT] symbol={symbol} dir={direction} allowed={allowed} reason={reason}")

    def open_trade(self, trade_data: dict):
        ticket = trade_data.get("ticket")
        if ticket is None:
            logger.error("open_trade called without ticket")
            return
        now = time.time()
        record = {
            "ticket": ticket,
            "entry_timestamp": now,
            "cycle": trade_data.get("cycle", 0),
            "symbol": trade_data.get("symbol", ""),
            "direction": trade_data.get("direction", ""),
            "price": trade_data.get("price", 0.0),
            "volume": trade_data.get("volume", 0.0),
            "signal_id": trade_data.get("signal_id", ""),
            "fusion_sources": trade_data.get("fusion_sources", []),
            "confirm_path": trade_data.get("confirm_path", []),
            "governor_path": trade_data.get("governor_path", []),
            "vel_decision": trade_data.get("vel_decision", ""),
            "sl": trade_data.get("sl", 0.0),
            "tp": trade_data.get("tp", 0.0),
            "exit_timestamp": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl": None,
            "duration": None,
            "sl_correct": None,
            "tp_correct": None,
            "status": "open",
        }
        self._open_trades[ticket] = record
        self._trade_history.append(record)
        logger.info(f"[TRADE_LIFECYCLE] OPEN ticket={ticket} "
                     f"{trade_data.get('symbol', '')} "
                     f"{trade_data.get('direction', '')} "
                     f"vol={trade_data.get('volume', 0)} "
                     f"signal={trade_data.get('signal_id', '')}")
        self._save_state()

    def close_trade(self, ticket: int, exit_data: dict):
        record = self._open_trades.pop(ticket, None)
        if record is None:
            logger.warning(f"close_trade: ticket {ticket} not found in open trades")
            return
        now = time.time()
        record["exit_timestamp"] = now
        record["exit_price"] = exit_data.get("exit_price")
        record["exit_reason"] = exit_data.get("exit_reason", "unknown")
        record["pnl"] = exit_data.get("pnl")
        if record["entry_timestamp"] is not None:
            record["duration"] = now - record["entry_timestamp"]
        record["sl_correct"] = exit_data.get("sl_correct")
        record["tp_correct"] = exit_data.get("tp_correct")
        record["status"] = "closed"
        self._closed_trades.append(record)
        logger.info(f"[TRADE_LIFECYCLE] CLOSE ticket={ticket} "
                     f"{record['symbol']} pnl={record['pnl']} "
                     f"reason={record['exit_reason']}")
        self._save_state()

    def get_open_trades(self) -> list:
        return list(self._open_trades.values())

    def get_closed_trades(self) -> list:
        return list(self._closed_trades)

    def trade_history(self) -> list:
        return list(self._trade_history)

    def vel_audit_summary(self) -> dict:
        block_rate = 0.0
        if self._vel_checks > 0:
            block_rate = round(self._vel_blocked / self._vel_checks, 4)
        return {
            "total_checks": self._vel_checks,
            "total_allowed": self._vel_allowed,
            "total_blocked": self._vel_blocked,
            "block_rate": block_rate,
            "blocked_reasons": dict(self._vel_blocked_reasons),
            "per_symbol": dict(self._vel_per_symbol),
        }

    def _save_state(self):
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        state = {
            "vel": {
                "checks": self._vel_checks,
                "allowed": self._vel_allowed,
                "blocked": self._vel_blocked,
                "blocked_reasons": self._vel_blocked_reasons,
                "per_symbol": self._vel_per_symbol,
            },
            "open_trades": {str(k): v for k, v in self._open_trades.items()},
            "closed_trades": self._closed_trades,
            "trade_history": self._trade_history,
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            vel = state.get("vel", {})
            self._vel_checks = vel.get("checks", 0)
            self._vel_allowed = vel.get("allowed", 0)
            self._vel_blocked = vel.get("blocked", 0)
            self._vel_blocked_reasons = vel.get("blocked_reasons", {
                "temporal_spacing": 0,
                "exposure_smoothing": 0,
                "burst_prevention": 0,
            })
            self._vel_per_symbol = vel.get("per_symbol", {})
            self._open_trades = {int(k): v for k, v in state.get("open_trades", {}).items()}
            self._closed_trades = state.get("closed_trades", [])
            self._trade_history = state.get("trade_history", [])
            logger.info(f"[TRADE_LIFECYCLE] State restored: "
                         f"{len(self._open_trades)} open, "
                         f"{len(self._closed_trades)} closed, "
                         f"{self._vel_checks} VEL checks")
        except Exception as e:
            logger.error(f"[TRADE_LIFECYCLE] Failed to load state: {e}")
