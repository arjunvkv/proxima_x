from __future__ import annotations

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator

from proxima_honest_backtest.data.providers.utils import (
    symbol_to_file_safe,
    ensure_month_dir,
    get_date_range_for_month,
)


class MT5Provider:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"

    def __init__(self, symbols: list[str] | None = None) -> None:
        if symbols is None:
            symbols = [
                "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY",
                "GBPJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
                "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD", "EURGBP",
                "EURCHF", "USDCHF", "AUDJPY",
            ]
        self.symbols = symbols
        self.connected = False

    def connect(self) -> bool:
        result = mt5.initialize()
        self.connected = result
        return result

    def disconnect(self) -> None:
        mt5.shutdown()
        self.connected = False

    def pull_ticks(
        self,
        symbol: str,
        from_date: datetime,
        to_date: Optional[datetime] = None,
        count: int = 100000,
    ) -> pd.DataFrame:
        if to_date is None:
            to_date = from_date + timedelta(days=1)

        ticks = mt5.copy_ticks_range(
            symbol, from_date, to_date, mt5.TIMEFRAME_TICK_COUNTER
        )

        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def pull_rates(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: int = mt5.TIMEFRAME_M1,
    ) -> pd.DataFrame:
        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def save_ticks(self, symbol: str, df: pd.DataFrame) -> Path:
        if df.empty:
            raise ValueError(f"Cannot save empty tick DataFrame for {symbol}")

        time_min = df["time"].min()
        year = time_min.year
        month = time_min.month

        base_path = self.DATA_DIR / "ticks"
        dir_path = ensure_month_dir(base_path, symbol, year, month)
        file_path = dir_path / f"{year}_{month:02d}.parquet"
        df.to_parquet(file_path, index=False)
        return file_path

    def save_rates(
        self, symbol: str, df: pd.DataFrame, timeframe: str = "m1"
    ) -> Path:
        if df.empty:
            raise ValueError(f"Cannot save empty rate DataFrame for {symbol}")

        time_min = df["time"].min()
        year = time_min.year
        month = time_min.month

        base_path = self.DATA_DIR / timeframe
        dir_path = ensure_month_dir(base_path, symbol, year, month)
        file_path = dir_path / f"{year}_{month:02d}.parquet"
        df.to_parquet(file_path, index=False)
        return file_path

    def load_ticks(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        base_path = self.DATA_DIR / "ticks" / symbol_to_file_safe(symbol)
        file_path = base_path / f"{year}_{month:02d}.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def load_rates(
        self, symbol: str, year: int, month: int, timeframe: str = "m1"
    ) -> pd.DataFrame:
        base_path = self.DATA_DIR / timeframe / symbol_to_file_safe(symbol)
        file_path = base_path / f"{year}_{month:02d}.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def get_available_data(self, symbol: str | None = None) -> dict:
        result: dict = {}

        for data_type in sorted(
            p.name for p in self.DATA_DIR.iterdir() if p.is_dir()
        ):
            type_path = self.DATA_DIR / data_type
            if not type_path.exists():
                continue

            result[data_type] = {}
            for symbol_dir in sorted(type_path.iterdir()):
                if not symbol_dir.is_dir():
                    continue
                sym = symbol_dir.name
                if symbol is not None and sym != symbol:
                    continue

                months = sorted(
                    p.stem for p in symbol_dir.glob("*.parquet")
                )
                if months:
                    result[data_type][sym] = months

        return result

    def iter_ticks_range(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        chunk_size: int = 50000,
    ) -> Iterator[pd.DataFrame]:
        current = from_date
        while current < to_date:
            year = current.year
            month = current.month
            df = self.load_ticks(symbol, year, month)
            if df.empty:
                current = get_date_range_for_month(year, month)[1]
                continue

            month_end = get_date_range_for_month(year, month)[1]
            chunk_start = current
            chunk_end = min(to_date, month_end)

            mask = (df["time"] >= chunk_start) & (df["time"] < chunk_end)
            chunk = df.loc[mask].copy()

            if len(chunk) > chunk_size:
                for i in range(0, len(chunk), chunk_size):
                    yield chunk.iloc[i : i + chunk_size]
            elif len(chunk) > 0:
                yield chunk

            current = month_end