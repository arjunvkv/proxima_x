import json
import os


class TriggerStatistics:
    def __init__(self, save_path: str = None):
        self._save_path = save_path

        # Lifetime counters (persistent across restarts)
        self.evaluated_count = 0
        self.trigger_count = 0
        self.executed_count = 0
        self.blocked_count = 0
        self.spread_blocks = 0
        self.position_blocks = 0
        self.risk_blocks = 0
        self.threshold_misses = 0
        self.symbol_stats = {}

        # Session counters (reset on each startup)
        self.session_evaluated = 0
        self.session_triggered = 0
        self.session_executed = 0
        self.session_blocked = 0
        self.session_symbols = {}

        if self._save_path:
            self.load()

    def record_evaluation(self, symbol: str, triggered: bool, blocked: bool, block_reason: str = None):
        # Lifetime
        self.evaluated_count += 1
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                "evaluated": 0, "triggered": 0, "executed": 0, "blocked": 0,
                "spread_blocks": 0, "position_blocks": 0, "risk_blocks": 0, "threshold_misses": 0
            }
        self.symbol_stats[symbol]["evaluated"] += 1

        # Session
        self.session_evaluated += 1
        if symbol not in self.session_symbols:
            self.session_symbols[symbol] = {"evaluated": 0, "triggered": 0, "executed": 0, "blocked": 0}
        self.session_symbols[symbol]["evaluated"] += 1

        if triggered:
            self.trigger_count += 1
            self.symbol_stats[symbol]["triggered"] += 1
            self.session_triggered += 1
            self.session_symbols[symbol]["triggered"] += 1

            if blocked:
                self.blocked_count += 1
                self.symbol_stats[symbol]["blocked"] += 1
                self.session_blocked += 1
                self.session_symbols[symbol]["blocked"] += 1

                if block_reason == "SPREAD":
                    self.spread_blocks += 1
                    self.symbol_stats[symbol]["spread_blocks"] += 1
                elif block_reason == "POSITION_EXISTS":
                    self.position_blocks += 1
                    self.symbol_stats[symbol]["position_blocks"] += 1
                elif block_reason in ["MAX_POSITIONS", "RISK_LIMIT"]:
                    self.risk_blocks += 1
                    self.symbol_stats[symbol]["risk_blocks"] += 1
                elif block_reason == "INVALID_SPREAD":
                    self.spread_blocks += 1
                    self.symbol_stats[symbol]["spread_blocks"] += 1
            else:
                self.executed_count += 1
                self.symbol_stats[symbol]["executed"] += 1
                self.session_executed += 1
                self.session_symbols[symbol]["executed"] += 1
        else:
            self.threshold_misses += 1
            self.symbol_stats[symbol]["threshold_misses"] += 1

        self.save()

    def session_summary(self) -> dict:
        return {
            "evaluated": self.session_evaluated,
            "triggered": self.session_triggered,
            "executed": self.session_executed,
            "blocked": self.session_blocked,
        }

    def save(self):
        if not self._save_path:
            return
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            data = {
                "evaluated_count": self.evaluated_count,
                "trigger_count": self.trigger_count,
                "executed_count": self.executed_count,
                "blocked_count": self.blocked_count,
                "spread_blocks": self.spread_blocks,
                "position_blocks": self.position_blocks,
                "risk_blocks": self.risk_blocks,
                "threshold_misses": self.threshold_misses,
                "symbol_stats": self.symbol_stats
            }
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def load(self):
        if not self._save_path or not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.evaluated_count = data.get("evaluated_count", 0)
            self.trigger_count = data.get("trigger_count", 0)
            self.executed_count = data.get("executed_count", 0)
            self.blocked_count = data.get("blocked_count", 0)
            self.spread_blocks = data.get("spread_blocks", 0)
            self.position_blocks = data.get("position_blocks", 0)
            self.risk_blocks = data.get("risk_blocks", 0)
            self.threshold_misses = data.get("threshold_misses", 0)
            self.symbol_stats = data.get("symbol_stats", {})
        except Exception:
            pass
