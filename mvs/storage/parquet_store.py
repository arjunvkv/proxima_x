from __future__ import annotations

from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import polars as pl
import numpy as np


class ParquetTruthStore:
    def __init__(self, root: str = "mvs_parquet") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, truth_plane: str, symbol: str, ts_ns: int, rows: np.ndarray) -> None:
        dt = pl.from_epoch([ts_ns], time_unit="ns").to_series()[0]
        path = self.root / truth_plane / symbol / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(pl.DataFrame(rows).to_dict(as_series=False))
        ds.write_dataset(
            table,
            base_dir=str(path),
            format="parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
