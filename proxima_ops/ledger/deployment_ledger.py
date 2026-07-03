import logging
import os
from datetime import datetime, date
from typing import Optional
from proxima_ops.config.settings import SETTINGS

logger = logging.getLogger("proxima_ops.ledger.deployment")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class DeploymentLedger:
    def __init__(self):
        self._conn = None
        self._db_path = SETTINGS.db_path

    def _ensure_db(self):
        if self._conn is not None:
            return
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS deployment_snapshots (
                snapshot_date DATE PRIMARY KEY,
                deployment_score DOUBLE,
                daily_pp DOUBLE,
                daily_sharpe DOUBLE,
                daily_dd DOUBLE,
                signals INTEGER,
                executions INTEGER,
                frequency_cv DOUBLE,
                open_positions INTEGER,
                account_balance DOUBLE,
                account_equity DOUBLE,
                created_at TIMESTAMP DEFAULT now()
            )
        """)

    def record(self, score: float, pp: float, sharpe: float, dd: float,
               signals: int, executions: int, freq_cv: float,
               open_positions: int, balance: float, equity: float):
        self._ensure_db()
        today = date.today()
        self._conn.execute("""
            INSERT OR REPLACE INTO deployment_snapshots (
                snapshot_date, deployment_score, daily_pp, daily_sharpe,
                daily_dd, signals, executions, frequency_cv,
                open_positions, account_balance, account_equity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (today, score, pp, sharpe, dd, signals, executions, freq_cv,
              open_positions, balance, equity))
        logger.info(f"Deployment snapshot recorded: score={score:.3f}")

    def get_recent(self, days: int = 30) -> list[dict]:
        self._ensure_db()
        rows = self._conn.execute("""
            SELECT * FROM deployment_snapshots
            ORDER BY snapshot_date DESC LIMIT ?
        """, (days,)).fetchall()
        return [dict(zip([desc[0] for desc in self._conn.description], r)) for r in rows]

    @property
    def latest(self) -> Optional[dict]:
        rows = self.get_recent(1)
        return rows[0] if rows else None

    @property
    def avg_score(self) -> float:
        rows = self.get_recent(7)
        if not rows:
            return 0.0
        return float(sum(r["deployment_score"] for r in rows)) / len(rows)

    @property
    def score_trend(self) -> str:
        rows = self.get_recent(14)
        if len(rows) < 7:
            return "STABLE"
        first_half = [r["deployment_score"] for r in rows[-7:]]
        second_half = [r["deployment_score"] for r in rows[:7]]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        diff = avg_second - avg_first
        if diff > 0.05:
            return "IMPROVING"
        elif diff < -0.05:
            return "DEGRADING"
        return "STABLE"

    def summary(self) -> dict:
        latest = self.latest
        return {
            "latest_score": latest["deployment_score"] if latest else 0.0,
            "avg_7d_score": round(self.avg_score, 3),
            "trend": self.score_trend,
            "days_recorded": len(self.get_recent(365))}
