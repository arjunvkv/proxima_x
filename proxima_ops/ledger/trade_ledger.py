import logging
import os
from datetime import datetime
from typing import Optional
from proxima_ops.config.settings import SETTINGS

logger = logging.getLogger("proxima_ops.ledger.trade")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not installed. Install with: pip install duckdb")


class TradeLedger:
    def __init__(self):
        self._conn = None
        self._db_path = SETTINGS.db_path

    def _ensure_db(self):
        if self._conn is not None:
            return
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS trade_seq START 1
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY DEFAULT nextval('trade_seq'),
                timestamp BIGINT,
                symbol VARCHAR,
                signal_type VARCHAR,
                signal_score DOUBLE,
                energy_storage DOUBLE,
                residual_energy DOUBLE,
                adaptive_time DOUBLE,
                regime VARCHAR,
                threshold DOUBLE,
                frequency_state VARCHAR,
                persistence_forecast VARCHAR,
                entry_price DOUBLE,
                exit_price DOUBLE DEFAULT 0,
                sl DOUBLE DEFAULT 0,
                tp DOUBLE DEFAULT 0,
                duration INTEGER DEFAULT 0,
                profit_points DOUBLE DEFAULT 0,
                profit_money DOUBLE DEFAULT 0,
                mt5_ticket BIGINT DEFAULT 0,
                entry_time TIMESTAMP DEFAULT now(),
                exit_time TIMESTAMP,
                status VARCHAR DEFAULT 'OPEN'
            )
        """)
        # Schema migration: add columns that may not exist in older DBs
        for col_def in [
            "exit_reason VARCHAR DEFAULT ''",
            "min_price DOUBLE DEFAULT 0",
            "max_price DOUBLE DEFAULT 0"
        ]:
            try:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col_def}")
            except Exception:
                pass

    def record(self, symbol: str, signal_type: str, signal_score: float,
               es: float, residual: float, at_val: float, regime: str,
               threshold: float, frequency_state: str, persistence: str,
               entry_price: float, sl: float, tp: float,
               mt5_ticket: int = 0) -> int:
        self._ensure_db()
        res = self._conn.execute("""
            INSERT INTO trades (
                timestamp, symbol, signal_type, signal_score,
                energy_storage, residual_energy, adaptive_time,
                regime, threshold, frequency_state, persistence_forecast,
                entry_price, sl, tp, mt5_ticket, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            RETURNING trade_id
        """, (int(datetime.now().timestamp()), symbol, signal_type, signal_score,
              es, residual, at_val, regime, threshold, frequency_state, persistence,
              entry_price, sl, tp, mt5_ticket))
        row = res.fetchone()
        return int(row[0]) if row else 0

    def close_trade(self, trade_id: int, exit_price: float,
                    profit_points: float, profit_money: float,
                    duration: int, mt5_ticket: int = 0,
                    exit_reason: str = "",
                    min_price: float = 0.0, max_price: float = 0.0):
        self._ensure_db()
        self._conn.execute("""
            UPDATE trades SET
                exit_price = ?, profit_points = ?, profit_money = ?,
                duration = ?, exit_time = now(), status = 'CLOSED',
                exit_reason = ?, min_price = ?, max_price = ?
            WHERE trade_id = ?
        """, (exit_price, profit_points, profit_money, duration, exit_reason, min_price, max_price, trade_id))

    def close_by_ticket(self, mt5_ticket: int, exit_reason: str = "BROKER_MISSING_RECONCILE", exit_detail: str = "") -> int:
        self._ensure_db()
        # Ensure exit_detail column exists
        try:
            self._conn.execute("ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_detail VARCHAR DEFAULT ''")
        except Exception:
            pass
        self._conn.execute("""
            UPDATE trades SET
                exit_price = NULL, profit_points = NULL, profit_money = NULL,
                duration = NULL, exit_time = now(), status = 'ORPHANED',
                exit_reason = ?, exit_detail = ?
            WHERE mt5_ticket = ? AND status = 'OPEN'
        """, (exit_reason, exit_detail, mt5_ticket))
        try:
            affected = self._conn.execute("SELECT COUNT(*) FROM trades WHERE mt5_ticket = ? AND status = 'ORPHANED'", (mt5_ticket,)).fetchone()[0]
        except Exception:
            affected = 0
        if affected > 0:
            logger.warning(f"Orphaned ghost position ticket={mt5_ticket} reason={exit_reason} detail={exit_detail}")
        return affected

    def get_open(self) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute("""
            SELECT * FROM trades WHERE status = 'OPEN'
        """).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    def get_recent(self, n: int = 10) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute("""
            SELECT * FROM trades ORDER BY trade_id DESC LIMIT ?
        """, (n,)).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    def get_completed(self) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute("""
            SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY trade_id ASC
        """).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    def get_by_symbol(self, symbol: str) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY trade_id DESC",
            (symbol,)).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    @property
    def total_trades(self) -> int:
        self._ensure_db()
        row = self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        return int(row[0]) if row else 0

    @property
    def total_profit(self) -> float:
        self._ensure_db()
        row = self._conn.execute("SELECT COALESCE(SUM(profit_money), 0) FROM trades WHERE status = 'CLOSED'").fetchone()
        return float(row[0]) if row else 0.0

    def summary(self) -> dict:
        exit_reasons = {}
        try:
            rows = self._conn.execute(
                "SELECT exit_reason, COUNT(*) FROM trades WHERE status = 'CLOSED' GROUP BY exit_reason"
            ).fetchall()
            exit_reasons = {r[0] or "UNKNOWN": int(r[1]) for r in rows}
        except Exception:
            pass
        return {
            "total_trades": self.total_trades,
            "total_profit": self.total_profit,
            "open_positions": len(self.get_open()),
            "exit_reasons": exit_reasons}

    def backup(self, path: Optional[str] = None):
        self._ensure_db()
        if path is None:
            path = self._db_path + ".backup"
        self._conn.execute(f"COPY trades TO '{path}' (HEADER, DELIMITER ',')")
        logger.info(f"Trade ledger backed up to {path}")
