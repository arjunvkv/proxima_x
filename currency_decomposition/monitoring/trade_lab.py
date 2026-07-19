"""
trade_lab.py — Proxima CDE Trade Lab Logger
============================================
Persistent tracker for paper and live trades with full lifecycle data.
Zero engine impact: all disk I/O happens on a background daemon thread.

Storage layout:
  logs/trade_lab_index.json          — master index (one record per trade, no traj)
  logs/trade_lab/<trade_id>.jsonl    — per-trade trajectory snapshots (~100B/line)
  logs/trade_lab_cmd.json            — transient command file for web dashboard deletes
"""
import json
import time
import threading
import queue
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_TRADE_LAB_DIR = _LOG_DIR / "trade_lab"
_INDEX_FILE = _LOG_DIR / "trade_lab_index.json"
_CMD_FILE = _LOG_DIR / "trade_lab_cmd.json"


class TradeLabLogger:
    """Persistent trade recorder for paper and live trades."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index: dict = {}                         # trade_id → record dict
        self._write_queue: queue.Queue = queue.Queue(maxsize=50000)
        self._index_dirty: bool = False
        self._last_index_write: float = 0.0

        _TRADE_LAB_DIR.mkdir(parents=True, exist_ok=True)
        self._load_index()

        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="TradeLab-Writer")
        self._writer.start()

    # ── Index persistence ──────────────────────────────────────────────────

    def _load_index(self) -> None:
        try:
            if _INDEX_FILE.exists():
                with open(_INDEX_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._index = data.get("trades", {})
        except Exception:
            self._index = {}

    def _flush_index(self) -> None:
        """Queue a non-blocking index write."""
        with self._lock:
            self._index_dirty = True
        try:
            self._write_queue.put_nowait({"kind": "index"})
        except queue.Full:
            pass  # writer will catch it in the timeout fallback

    # ── Background writer ──────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._write_queue.get(timeout=2.0)
                kind = item.get("kind")

                if kind == "append":
                    tid = item["trade_id"]
                    line = item["line"]
                    path = _TRADE_LAB_DIR / f"{tid}.jsonl"
                    try:
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(line, default=str) + "\n")
                    except Exception:
                        pass

                elif kind == "index":
                    # Take a snapshot under the lock, then write outside the lock
                    with self._lock:
                        snapshot = {k: dict(v) for k, v in self._index.items()}
                        self._index_dirty = False
                        self._last_index_write = time.time()
                    try:
                        tmp = _INDEX_FILE.with_suffix(".tmp")
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump({"trades": snapshot}, f, default=str)
                        tmp.replace(_INDEX_FILE)
                    except Exception:
                        pass

            except queue.Empty:
                pass

            # Periodic flush: write dirty index if it has been at least 3.0s since last write
            with self._lock:
                dirty = self._index_dirty
                last = self._last_index_write
            if dirty and (time.time() - last) >= 3.0:
                with self._lock:
                    snapshot = {k: dict(v) for k, v in self._index.items()}
                    self._index_dirty = False
                    self._last_index_write = time.time()
                try:
                    tmp = _INDEX_FILE.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump({"trades": snapshot}, f, default=str)
                    tmp.replace(_INDEX_FILE)
                except Exception:
                    pass

    # ── Public API ──────────────────────────────────────────────────────────

    def fire_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: float,
        is_live: bool,
        entry_triggers: dict,
    ) -> str:
        """Register a new paper or live trade open. Returns the trade_id."""
        record = {
            "id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "mode": "LIVE" if is_live else "PAPER",
            "status": "OPEN",
            "entry_time": round(entry_time, 1),
            "entry_price": round(entry_price, 6),
            "entry_triggers": entry_triggers,
            "close_time": None,
            "close_price": None,
            "close_reason": None,
            "close_triggers": None,
            "peak_pnl": 0.0,
            "valley_pnl": 0.0,
            "tick_count": 0,
        }
        with self._lock:
            self._index[trade_id] = record
        self._flush_index()
        return trade_id

    def record_tick(self, trade_id: str, snapshot: dict) -> None:
        """Append a cycle snapshot to the trade's trajectory JSONL.
        snapshot keys: t (timestamp), p (price), pnl, nmi, health, pol, regime
        """
        with self._lock:
            if trade_id not in self._index:
                return
            rec = self._index[trade_id]
            pnl = snapshot.get("pnl", 0.0) or 0.0
            if pnl > rec.get("peak_pnl", 0.0):
                rec["peak_pnl"] = round(pnl, 2)
            if pnl < rec.get("valley_pnl", 0.0):
                rec["valley_pnl"] = round(pnl, 2)
            rec["tick_count"] = rec.get("tick_count", 0) + 1
            
            # Save latest values to index for live dashboard display
            rec["current_price"] = snapshot.get("p", 0.0)
            rec["current_pnl"] = round(pnl, 2)
            rec["current_nmi"] = snapshot.get("nmi", 0.0)
            rec["current_health"] = snapshot.get("health", 0.0)
            rec["current_pol"] = snapshot.get("pol", 0.0)
            rec["current_regime"] = snapshot.get("regime", "N/A")
            
            self._index_dirty = True
        try:
            self._write_queue.put_nowait({"kind": "append", "trade_id": trade_id, "line": snapshot})
        except queue.Full:
            pass  # acceptable data loss under heavy load

    def close_trade(
        self,
        trade_id: str,
        close_price: float | None,
        close_time: float,
        close_reason: str,
        close_triggers: dict,
    ) -> None:
        """Mark a trade as closed with full exit context."""
        with self._lock:
            if trade_id not in self._index:
                return
            rec = self._index[trade_id]
            rec["status"] = "CLOSED"
            rec["close_time"] = round(close_time, 1)
            rec["close_price"] = round(close_price, 6) if close_price else None
            rec["close_reason"] = close_reason
            rec["close_triggers"] = close_triggers
            self._index_dirty = True
        self._flush_index()

    def get_index_summary(self) -> dict:
        """Return a thread-safe shallow copy of the index (no trajectory data)."""
        with self._lock:
            return {k: dict(v) for k, v in self._index.items()}

    def read_trajectory(self, trade_id: str) -> list:
        """Read all trajectory points for a trade (called from web dashboard thread)."""
        path = _TRADE_LAB_DIR / f"{trade_id}.jsonl"
        result = []
        if not path.exists():
            return result
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
        return result

    def process_pending_commands(self) -> None:
        """Check and execute delete commands written by the web dashboard.
        Called once per 30s decision cycle — zero overhead when no commands pending.
        """
        if not _CMD_FILE.exists():
            return
        try:
            with open(_CMD_FILE, "r", encoding="utf-8") as f:
                cmd = json.load(f)
            try:
                _CMD_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            action = cmd.get("action")
            if action == "delete_all":
                self.delete_all()
            elif action == "delete_oldest":
                n = int(cmd.get("n", 10))
                self.delete_oldest(n)
        except Exception:
            try:
                _CMD_FILE.unlink(missing_ok=True)
            except Exception:
                pass

    def delete_all(self) -> None:
        """Delete ALL trades from index and remove trajectory files."""
        with self._lock:
            tids = list(self._index.keys())
            self._index = {}
        for tid in tids:
            try:
                (_TRADE_LAB_DIR / f"{tid}.jsonl").unlink(missing_ok=True)
            except Exception:
                pass
        self._flush_index()

    def delete_oldest(self, n: int = 10) -> None:
        """Delete the N oldest CLOSED trades by entry_time."""
        with self._lock:
            closed = sorted(
                [(k, v) for k, v in self._index.items() if v.get("status") == "CLOSED"],
                key=lambda x: x[1].get("entry_time", 0),
            )
            to_delete = [k for k, _ in closed[:n]]
            for k in to_delete:
                del self._index[k]
        for tid in to_delete:
            try:
                (_TRADE_LAB_DIR / f"{tid}.jsonl").unlink(missing_ok=True)
            except Exception:
                pass
        if to_delete:
            self._flush_index()
