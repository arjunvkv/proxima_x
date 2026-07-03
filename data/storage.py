from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
import pyarrow as pa

from config.settings import settings


class MarketDataStore:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or settings.paths.market_data
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_parquet(self, df: pl.DataFrame | pa.Table, path: str | Path) -> None:
        path = self.base_path / Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(df, pl.DataFrame):
            df.write_parquet(str(path), use_pyarrow=True)
        else:
            pa.parquet.write_table(df, str(path))

    def read_parquet(self, path: str | Path, lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
        path = self.base_path / Path(path)
        if lazy:
            return pl.scan_parquet(str(path))
        return pl.read_parquet(str(path))

    def list_datasets(self) -> list[str]:
        return [str(p.relative_to(self.base_path)) for p in self.base_path.rglob("*.parquet")]


class DuckDBStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.paths.research_cache / "research.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def register_parquet(self, name: str, parquet_path: str | Path) -> None:
        self._conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet_path}')")

    def query(self, sql: str) -> pl.DataFrame:
        return self._conn.execute(sql).pl()

    def query_arrow(self, sql: str) -> pa.Table:
        return self._conn.execute(sql).arrow()

    def create_table_from_parquet(self, table_name: str, parquet_path: str | Path) -> None:
        self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")

    def persist_df(self, table_name: str, df: pl.DataFrame) -> None:
        self._conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

    def close(self) -> None:
        self._conn.close()
