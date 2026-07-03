from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings
from numpy.typing import NDArray


class FeatureStore:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or settings.paths.feature_store
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pl.LazyFrame] = {}

    def store_features(
        self,
        name: str,
        df: pl.DataFrame | pl.LazyFrame,
        metadata: Optional[dict] = None,
    ) -> None:
        path = self.base_path / f"{name}.parquet"
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        df.write_parquet(str(path), use_pyarrow=True)
        if metadata:
            meta_path = self.base_path / f"{name}_meta.json"
            import orjson
            meta_path.write_bytes(orjson.dumps(metadata))

    def store_arrow(self, name: str, table: pa.Table) -> None:
        path = self.base_path / f"{name}.arrow"
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def load_features(
        self, name: str, lazy: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        path = self.base_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature set '{name}' not found at {path}")
        if lazy:
            return pl.scan_parquet(str(path))
        return pl.read_parquet(str(path))

    def load_arrow(self, name: str) -> pa.Table:
        path = self.base_path / f"{name}.arrow"
        with pa.OSFile(str(path), "rb") as source:
            reader = pa.ipc.open_file(source)
            return reader.read_all()

    def cache_lazy(self, name: str) -> Optional[pl.LazyFrame]:
        try:
            lf = self.load_features(name, lazy=True)
            self._cache[name] = lf
            return lf
        except FileNotFoundError:
            return None

    def list_features(self) -> List[str]:
        return [p.stem for p in self.base_path.glob("*.parquet") if not p.stem.endswith("_meta")]

    def get_metadata(self, name: str) -> Optional[dict]:
        meta_path = self.base_path / f"{name}_meta.json"
        if meta_path.exists():
            import orjson
            return orjson.loads(meta_path.read_bytes())
        return None

    def clear_cache(self) -> None:
        self._cache.clear()
