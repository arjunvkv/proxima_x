import logging
import os
from datetime import datetime
from typing import Optional
from proxima_ops.config.settings import SETTINGS

logger = logging.getLogger("proxima_ops.ledger.signal")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class SignalLedger:
    def __init__(self):
        self._conn = None
        self._db_path = SETTINGS.db_path

    def _ensure_db(self):
        if self._conn is not None:
            return
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS signal_seq START 1
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id INTEGER PRIMARY KEY DEFAULT nextval('signal_seq'),
                timestamp BIGINT,
                symbol VARCHAR,
                es_percentile DOUBLE,
                residual_energy DOUBLE,
                adaptive_time DOUBLE,
                regime VARCHAR,
                threshold DOUBLE,
                signal_state VARCHAR,
                executed BOOLEAN DEFAULT FALSE,
                rejection_reason VARCHAR DEFAULT '',
                created_at TIMESTAMP DEFAULT now()
            )
        """)

    def record(self, symbol: str, es_pct: float, residual: float,
               at_val: float, regime: str, threshold: float,
               signal_state: str, executed: bool = False,
               rejection_reason: str = ""):
        self._ensure_db()
        self._conn.execute("""
            INSERT INTO signals (
                timestamp, symbol, es_percentile, residual_energy,
                adaptive_time, regime, threshold, signal_state,
                executed, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(datetime.now().timestamp()), symbol, es_pct, residual,
              at_val, regime, threshold, signal_state, int(executed),
              rejection_reason))

    def get_today(self) -> list[dict]:
        self._ensure_db()
        today_ts = int(datetime.now().timestamp()) - 86400
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp DESC",
            (today_ts,)).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    def get_recent(self, n: int = 50) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute(
            "SELECT * FROM signals ORDER BY signal_id DESC LIMIT ?",
            (n,)).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    @property
    def total_signals(self) -> int:
        self._ensure_db()
        row = self._conn.execute("SELECT COUNT(*) FROM signals").fetchone()
        return int(row[0]) if row else 0

    def summary(self) -> dict:
        self._ensure_db()
        total = self.total_signals
        executed = self._conn.execute("SELECT COUNT(*) FROM signals WHERE executed = TRUE").fetchone()
        rejected = self._conn.execute("SELECT COUNT(*) FROM signals WHERE executed = FALSE AND rejection_reason != ''").fetchone()
        return {
            "total_signals": total,
            "executed": int(executed[0]) if executed else 0,
            "rejected": int(rejected[0]) if rejected else 0}
