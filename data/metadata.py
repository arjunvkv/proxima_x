from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from config.settings import settings


@dataclass
class DatasetRecord:
    name: str
    asset: str
    timeframe: str
    source: str
    start_date: str
    end_date: str
    row_count: int
    columns: str
    hash: str
    created_at: str


@dataclass
class ExperimentRecord:
    id: str
    name: str
    description: str
    config: str
    created_at: str
    duration_seconds: float
    status: str


class MetadataStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.paths.metadata / "metadata.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                name VARCHAR PRIMARY KEY,
                asset VARCHAR,
                timeframe VARCHAR,
                source VARCHAR,
                start_date VARCHAR,
                end_date VARCHAR,
                row_count BIGINT,
                columns VARCHAR,
                hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                description VARCHAR,
                config VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds DOUBLE,
                status VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                name VARCHAR PRIMARY KEY,
                category VARCHAR,
                version VARCHAR,
                description VARCHAR,
                dependencies VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state_snapshots (
                id BIGINT PRIMARY KEY,
                timestamp BIGINT,
                state_vector VARCHAR,
                cluster_id INTEGER,
                transition_id BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def register_dataset(self, record: DatasetRecord) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO datasets
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (record.name, record.asset, record.timeframe, record.source,
              record.start_date, record.end_date, record.row_count,
              record.columns, record.hash))

    def register_experiment(self, record: ExperimentRecord) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO experiments
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        """, (record.id, record.name, record.description, record.config,
              record.duration_seconds, record.status))

    def query(self, sql: str) -> List[Dict[str, Any]]:
        return self._conn.execute(sql).fetchdf().to_dict(orient="records")

    def close(self) -> None:
        self._conn.close()
