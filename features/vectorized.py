from __future__ import annotations

from typing import Optional

import polars as pl


class FeatureGenerator:
    def __init__(self, lazy: bool = True):
        self.lazy = lazy

    def compute_rolling_stats(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        price_col: str = "close",
        windows: list[int] = None,
    ) -> pl.LazyFrame:
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        price = pl.col(price_col).shift(1)
        for w in windows:
            df = df.with_columns(
                price.rolling_mean(w).alias(f"{price_col}_ma_{w}"),
                price.rolling_std(w).alias(f"{price_col}_std_{w}"),
                price.rolling_min(w).alias(f"{price_col}_min_{w}"),
                price.rolling_max(w).alias(f"{price_col}_max_{w}"),
            )
        return df

    def compute_rolling_ratios(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        price_col: str = "close",
        windows: list[int] = None,
    ) -> pl.LazyFrame:
        if windows is None:
            windows = [5, 10, 20, 50]
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        price = pl.col(price_col)
        for w in windows:
            ma = price.rolling_mean(w).shift(1)
            df = df.with_columns(
                (price / ma).alias(f"{price_col}_ratio_{w}")
            )
        return df

    def compute_returns(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        price_col: str = "close",
        periods: list[int] = None,
    ) -> pl.LazyFrame:
        if periods is None:
            periods = [1, 5, 10, 20]
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        price = pl.col(price_col)
        for p in periods:
            df = df.with_columns(
                (price / price.shift(p)).log().alias(f"return_{p}")
            )
        return df

    def compute_volume_features(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        volume_col: str = "volume",
        windows: list[int] = None,
    ) -> pl.LazyFrame:
        if windows is None:
            windows = [5, 10, 20, 50]
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        vol = pl.col(volume_col)
        for w in windows:
            ma = vol.rolling_mean(w).shift(1)
            std = vol.rolling_std(w).shift(1)
            df = df.with_columns(
                ((vol - ma) / std).alias(f"{volume_col}_z_{w}"),
                (vol / ma).alias(f"{volume_col}_ratio_{w}"),
            )
        return df

    def compute_high_low_features(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        windows: list[int] = None,
    ) -> pl.LazyFrame:
        if windows is None:
            windows = [5, 10, 20]
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        high = pl.col(high_col)
        low = pl.col(low_col)
        close = pl.col(close_col)
        range_val = high - low
        df = df.with_columns(
            range_val.alias("range"),
            ((close - low) / range_val).alias("position_in_range"),
        )
        for w in windows:
            range_ma = range_val.rolling_mean(w).shift(1)
            df = df.with_columns(
                (range_val / range_ma).alias(f"range_ratio_{w}")
            )
        return df

    def generate_all(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        df = self.compute_rolling_stats(df)
        df = self.compute_rolling_ratios(df)
        df = self.compute_returns(df)
        df = self.compute_volume_features(df)
        df = self.compute_high_low_features(df)
        return df
