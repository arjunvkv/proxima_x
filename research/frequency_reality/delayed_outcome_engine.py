import os
import json
import time
import logging
from typing import Optional, Callable

logger = logging.getLogger("proxima_ops.delayed_outcome")

class DelayedOutcomeEngine:
    STATUS_PENDING = "PENDING"
    STATUS_RESOLVED = "RESOLVED"
    STATUS_FAILED = "FAILED"
    STATUS_EXPIRED = "EXPIRED"

    def __init__(self, price_provider: Callable, horizons: list[int] = None,
                 save_path: str = None):
        if horizons is None:
            horizons = [20, 50, 100]
        self._horizons = sorted(set(horizons))
        self._price_provider = price_provider
        self._save_path = save_path or os.path.join(
            os.path.dirname(__file__), "data", "pending_outcomes.json")
        self._pending = {}
        self._load()
        self._prune_expired()
        self._integrity_errors = 0

    def _assert(self, condition: bool, msg: str):
        if not condition:
            self._integrity_errors += 1
            logger.error(f"INTEGRITY: {msg}")

    def record_snapshot(self, signal_id: str, symbol: str, entry_price: float):
        self._assert(bool(signal_id), f"record_snapshot: empty signal_id")
        self._assert(bool(symbol), f"record_snapshot: empty symbol for {signal_id}")
        self._assert(entry_price > 0, f"record_snapshot: entry_price={entry_price} for {signal_id}")

        now_ts = int(time.time())
        for h in self._horizons:
            key = f"{signal_id}_h{h}"
            if key not in self._pending:
                self._pending[key] = {
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "horizon_bars": h,
                    "horizon_seconds": h * 3600,
                    "entry_timestamp": now_ts,
                    "mature_at": now_ts + h * 3600,
                    "return_h": None,
                    "status": self.STATUS_PENDING,
                }
        self._save()

    def process_matured(self) -> int:
        now_ts = int(time.time())
        resolved = 0
        for key, entry in list(self._pending.items()):
            if entry.get("status") != self.STATUS_PENDING:
                continue

            mature_at = entry["mature_at"]
            if now_ts < mature_at:
                continue

            # EXPIRED: pending more than 2x the horizon without resolution
            horizon_sec = entry.get("horizon_seconds", 72000)
            if now_ts > mature_at + horizon_sec * 2:
                self._pending[key]["status"] = self.STATUS_EXPIRED
                self._pending[key]["resolved_at"] = now_ts
                logger.warning(f"Outcome expired: {key} (pending {now_ts - entry['entry_timestamp']}s)")
                continue

            symbol = entry["symbol"]
            entry_price = entry["entry_price"]

            self._assert(entry_price > 0, f"process_matured: entry_price={entry_price} for {key}")

            current_price = self._price_provider(symbol)
            if current_price is None or current_price <= 0:
                self._pending[key]["status"] = self.STATUS_FAILED
                self._pending[key]["resolved_at"] = now_ts
                logger.warning(f"Outcome failed: {key} (price unavailable for {symbol})")
                continue

            ret = (current_price - entry_price) / entry_price
            self._pending[key]["return_h"] = round(float(ret), 6)
            self._pending[key]["status"] = self.STATUS_RESOLVED
            self._pending[key]["resolved_at"] = now_ts
            self._pending[key]["resolved_price"] = round(float(current_price), 5)

            self._assert(ret == ret, f"process_matured: NaN return for {key}")

            resolved += 1

        if resolved > 0:
            self._save()
        return resolved

    def get_return(self, signal_id: str, horizon: int) -> Optional[float]:
        key = f"{signal_id}_h{horizon}"
        entry = self._pending.get(key)
        if entry and entry["resolved"]:
            return entry["return_h"]
        return None

    def get_returns(self, signal_id: str) -> dict:
        result = {}
        for h in self._horizons:
            key = f"{signal_id}_h{h}"
            entry = self._pending.get(key)
            if entry and entry.get("status") == self.STATUS_RESOLVED:
                ret = entry.get("return_h")
                result[f"return_h{h}"] = ret
                result[f"pp_h{h}"] = 1.0 if ret is not None and ret > 0 else (
                    0.0 if ret is not None else None)
            else:
                result[f"return_h{h}"] = None
                result[f"pp_h{h}"] = None
        return result

    def pending_count(self) -> int:
        return sum(1 for e in self._pending.values() if e.get("status") == self.STATUS_PENDING)

    def resolved_count(self) -> int:
        return sum(1 for e in self._pending.values() if e.get("status") == self.STATUS_RESOLVED)

    def failed_count(self) -> int:
        return sum(1 for e in self._pending.values() if e.get("status") == self.STATUS_FAILED)

    def expired_count(self) -> int:
        return sum(1 for e in self._pending.values() if e.get("status") == self.STATUS_EXPIRED)

    def has_pending(self, signal_id: str) -> bool:
        return any(
            e.get("status") == self.STATUS_PENDING
            for k, e in self._pending.items()
            if e["signal_id"] == signal_id
        )

    def health_summary(self) -> dict:
        return {
            "pending": self.pending_count(),
            "resolved": self.resolved_count(),
            "failed": self.failed_count(),
            "expired": self.expired_count(),
            "integrity_errors": self._integrity_errors,
            "total": len(self._pending),
        }

    def _prune_expired(self, max_age_days: int = 7):
        now_ts = int(time.time())
        cutoff = now_ts - max_age_days * 86400
        before = len(self._pending)
        self._pending = {
            k: v for k, v in self._pending.items()
            if v.get("status") != self.STATUS_EXPIRED
            or v.get("resolved_at", 0) >= cutoff
        }
        after = len(self._pending)
        if after < before:
            logger.info(f"Pruned {before - after} expired outcomes ({after} remaining)")
            self._save()

    def purge(self):
        self._pending = {}
        self._save()

    def _save(self):
        try:
            parent = os.path.dirname(self._save_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            data = {k: v for k, v in self._pending.items()}
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"DelayedOutcomeEngine save error: {e}")

    def _load(self):
        try:
            if os.path.exists(self._save_path):
                with open(self._save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._pending = data
                logger.info(f"Loaded {len(self._pending)} pending outcomes")
        except Exception as e:
            logger.warning(f"DelayedOutcomeEngine load error: {e}")
