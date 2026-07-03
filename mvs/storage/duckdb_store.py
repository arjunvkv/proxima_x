from __future__ import annotations

from pathlib import Path
from typing import Any
import duckdb
import polars as pl
import numpy as np

from mvs.models.market_plane import MARKET_DTYPE
from mvs.models.perception_plane import PERCEPTION_DTYPE
from mvs.models.action_plane import ACTION_DTYPE
from mvs.models.outcome_plane import OUTCOME_DTYPE


class TruthStore:
    def __init__(self, db_path: str = "mvs.duckdb", flush_batch_size: int = 256) -> None:
        self.db_path = Path(db_path)
        self.flush_batch_size = flush_batch_size
        self.conn = duckdb.connect(str(self.db_path))
        self._market_buf: list[np.ndarray] = []
        self._perception_buf: list[np.ndarray] = []
        self._action_buf: list[np.ndarray] = []
        self._outcome_buf: list[np.ndarray] = []
        self._init_schema()

    def _create_table(self, name: str, dtype: np.dtype) -> None:
        mapping = {"i": "BIGINT", "f": "DOUBLE", "b": "BOOLEAN", "U": "VARCHAR", "O": "VARCHAR"}
        cols = []
        for field, dt in dtype.fields.items():
            kind = dt[0].kind
            sql_type = mapping.get(kind, "VARCHAR")
            cols.append(f"{field} {sql_type}")
        sql = f"CREATE TABLE IF NOT EXISTS {name} ({','.join(cols)})"
        self.conn.execute(sql)

    def _init_schema(self) -> None:
        self._create_table("mvs_market_truth", MARKET_DTYPE)
        self._create_table("mvs_perception_truth", PERCEPTION_DTYPE)
        self._create_table("mvs_action_truth", ACTION_DTYPE)
        self._create_table("mvs_outcome_truth", OUTCOME_DTYPE)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mvs_conflicts(
                tick_id BIGINT, conflict_type VARCHAR, severity DOUBLE,
                description VARCHAR, layer VARCHAR, timestamp BIGINT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mvs_honesty(
                layer_name VARCHAR, score DOUBLE, directional_accuracy DOUBLE,
                timing_precision DOUBLE, path_alignment DOUBLE, delay_penalty DOUBLE,
                contradiction_penalty DOUBLE, sample_count BIGINT, timestamp BIGINT
            )
        """)

    def _flush_table(self, table: str, rows: list[np.ndarray]) -> None:
        if not rows:
            return
        merged = np.concatenate(rows)
        df = pl.DataFrame(merged)
        self.conn.register("tmp_table", df.to_arrow())
        self.conn.execute(f"INSERT INTO {table} SELECT * FROM tmp_table")
        self.conn.unregister("tmp_table")
        rows.clear()

    def write_market(self, rows: np.ndarray) -> None:
        self._market_buf.append(rows)
        if len(self._market_buf) >= self.flush_batch_size:
            self._flush_table("mvs_market_truth", self._market_buf)

    def write_perception(self, rows: np.ndarray) -> None:
        self._perception_buf.append(rows)
        if len(self._perception_buf) >= self.flush_batch_size:
            self._flush_table("mvs_perception_truth", self._perception_buf)

    def write_action(self, rows: np.ndarray) -> None:
        self._action_buf.append(rows)
        if len(self._action_buf) >= self.flush_batch_size:
            self._flush_table("mvs_action_truth", self._action_buf)

    def write_outcome(self, rows: np.ndarray) -> None:
        self._outcome_buf.append(rows)
        if len(self._outcome_buf) >= self.flush_batch_size:
            self._flush_table("mvs_outcome_truth", self._outcome_buf)

    def write_conflicts(self, rows: pl.DataFrame) -> None:
        self.conn.register("tmp_conflicts", rows.to_arrow())
        self.conn.execute("INSERT INTO mvs_conflicts SELECT * FROM tmp_conflicts")
        self.conn.unregister("tmp_conflicts")

    def write_honesty(self, rows: pl.DataFrame) -> None:
        self.conn.register("tmp_honesty", rows.to_arrow())
        self.conn.execute("INSERT INTO mvs_honesty SELECT * FROM tmp_honesty")
        self.conn.unregister("tmp_honesty")

    def flush(self) -> None:
        self._flush_table("mvs_market_truth", self._market_buf)
        self._flush_table("mvs_perception_truth", self._perception_buf)
        self._flush_table("mvs_action_truth", self._action_buf)
        self._flush_table("mvs_outcome_truth", self._outcome_buf)

    def _query(self, sql: str, params: tuple[Any, ...]) -> pl.DataFrame:
        return self.conn.execute(sql, params).pl()

    def query_market(self, symbol: str, start_ts: int, end_ts: int) -> pl.DataFrame:
        return self._query(
            "SELECT * FROM mvs_market_truth WHERE symbol=? AND ts_ns BETWEEN ? AND ?",
            (symbol, start_ts, end_ts),
        )

    def query_perception(self, symbol: str, start_ts: int, end_ts: int) -> pl.DataFrame:
        return self._query(
            "SELECT * FROM mvs_perception_truth WHERE symbol=? AND ts_ns BETWEEN ? AND ?",
            (symbol, start_ts, end_ts),
        )

    def query_actions(self, start_ts: int, end_ts: int) -> pl.DataFrame:
        return self._query(
            "SELECT * FROM mvs_action_truth WHERE ts_ns BETWEEN ? AND ?",
            (start_ts, end_ts),
        )

    def query_outcomes(self, symbol: str, start_ts: int, end_ts: int) -> pl.DataFrame:
        return self._query(
            "SELECT * FROM mvs_outcome_truth WHERE symbol=? AND entry_ts_ns BETWEEN ? AND ?",
            (symbol, start_ts, end_ts),
        )

    def get_trade(self, trade_id: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM mvs_outcome_truth WHERE trade_id=?", (trade_id,)
        ).fetchone()
        if row is None:
            return {}
        cols = [x[0] for x in self.conn.description]
        return dict(zip(cols, row))

    def get_conflicts(self, start_ts: int, end_ts: int) -> pl.DataFrame:
        return self._query(
            "SELECT * FROM mvs_conflicts WHERE timestamp BETWEEN ? AND ?",
            (start_ts, end_ts),
        )

    def get_honesty_scores(self) -> pl.DataFrame:
        return self.conn.execute("SELECT * FROM mvs_honesty").pl()

    def close(self) -> None:
        self.flush()
        self.conn.close()
