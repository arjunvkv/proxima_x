"""PropagationMapper — maps source asset state to target forward returns.

Uses the StateExtractor's per-minute state signatures (``oss_bucket``)
and ForwardSurface's forward-return columns to quantify cross-asset
propagation — conditional win rates, profit factors, amplitude expansion,
transfer entropy, and optimal lead-lag structure.
"""
import sys; sys.path.insert(0, ".")

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from research.dpl_x.state_extractor import StateExtractor

try:
    from research.dpl_x.forward_surface import ForwardSurface
    _HAS_FORWARD_SURFACE = True
except ImportError:
    ForwardSurface = None  # type: ignore[assignment]
    _HAS_FORWARD_SURFACE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PropagationResult:
    """Aggregated cross-asset propagation metrics.

    Attributes
    ----------
    source : str
        Source asset symbol (whose state drives propagation).
    target : str
        Target asset symbol (whose forward returns are affected).
    horizon : int
        Forward-return horizon in seconds.
    n : int
        Number of aligned (state, forward-return) samples.
    wr : float
        Overall win rate — fraction of aligned samples with positive
        forward return.
    pf : float
        Overall profit factor, adjusted by win/loss rates.
    aer : float
        Amplitude expansion ratio — mean conditional standard deviation
        divided by unconditional standard deviation.
    transfer_entropy : float
        Mutual information between discretised source state and target
        forward-return bins (a proxy for information flow).
    lag_score : int
        Lag (in seconds) that maximises absolute correlation between
        source ``oss_bucket`` and target forward return, tested over
        0–10 minutes.
    expected_return : float
        Unconditional mean forward return.
    pct_return : float
        Same as *expected_return* (kept for compatibility).
    effects : dict
        Per-bucket breakdown keyed by ``oss_bucket`` with fields
        ``n``, ``win_rate``, ``loss_rate``, ``profit_factor``, ``aer``,
        ``expected_return``, ``pct_return``, ``std_return``.
    """
    source: str
    target: str
    horizon: int
    n: int
    wr: float
    pf: float
    aer: float
    transfer_entropy: float
    lag_score: int
    expected_return: float
    pct_return: float
    effects: dict  # bucket_id -> dict


# ---------------------------------------------------------------------------
# PropagationMapper
# ---------------------------------------------------------------------------

class PropagationMapper:
    """Maps source asset state signatures to target forward returns.

    The mapper aligns source-state timestamps with pre-computed (or
    locally computed) target forward returns, then computes:

    * Conditional win rates and profit factors per state bucket.
    * Amplitude expansion ratio (AER) — how much larger state-conditional
      return dispersion is vs the unconditional distribution.
    * Transfer entropy via discretized returns and state labels.
    * Optimal lead-lag structure by scanning forward shifts of source
      timestamps.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._forward_surface: Optional[ForwardSurface] = (
            ForwardSurface() if _HAS_FORWARD_SURFACE else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(
        self,
        source_df: pl.DataFrame,
        target_df: pl.DataFrame,
        source_symbol: str,
        target_symbol: str,
        horizon: int,
    ) -> PropagationResult:
        """Map source asset state to target forward returns.

        Parameters
        ----------
        source_df : pl.DataFrame
            Source asset DataFrame with columns ``ts``, ``oss_bucket``,
            and optionally other state features (from :class:`StateExtractor`).
        target_df : pl.DataFrame
            Target asset DataFrame with columns ``ts``, ``price`` *or*
            with pre-computed ``forward_<horizon>`` columns (from
            :class:`ForwardSurface`).
        source_symbol : str
            Symbol name for the source asset.
        target_symbol : str
            Symbol name for the target asset.
        horizon : int
            Forward-return horizon **in seconds** (e.g. ``300`` for 5 min).

        Returns
        -------
        PropagationResult
        """
        # -- 1. Obtain target forward returns ----------------------------
        target_fwd = self._resolve_forward_returns(target_df, horizon)
        if len(target_fwd) == 0:
            return self._empty_result(source_symbol, target_symbol, horizon)

        # -- 2. Align source state timestamps to target forward returns --
        aligned = self._align_source_to_target(source_df, target_fwd)
        if len(aligned) == 0:
            return self._empty_result(source_symbol, target_symbol, horizon)

        n = len(aligned)

        # -- 3. Overall win rate ----------------------------------------
        wr_val = aligned.select((pl.col("forward_return") > 0).mean()).item()
        wr = float(wr_val) if wr_val is not None else 0.0
        loss_rate = 1.0 - wr

        # -- 4. Overall profit factor -----------------------------------
        pos_sum_val = (
            aligned.filter(pl.col("forward_return") > 0)
            .select(pl.sum("forward_return"))
            .item()
        )
        neg_sum_val = (
            aligned.filter(pl.col("forward_return") < 0)
            .select(pl.sum("forward_return"))
            .item()
        )
        pos_sum = float(pos_sum_val) if pos_sum_val is not None else 0.0
        neg_sum = float(neg_sum_val) if neg_sum_val is not None else 0.0

        denom = abs(neg_sum) * loss_rate
        pf = (pos_sum * wr) / denom if denom > 1e-15 else 0.0

        # -- 5. Expected return -----------------------------------------
        exp_ret_val = aligned.select(pl.mean("forward_return")).item()
        expected_return = float(exp_ret_val) if exp_ret_val is not None else 0.0

        # -- 6. Amplitude expansion ratio (AER) -------------------------
        overall_std_val = aligned.select(pl.std("forward_return")).item()
        if overall_std_val is None or overall_std_val < 1e-15:
            aer = 1.0
            overall_std = 1.0
        else:
            overall_std = float(overall_std_val)
            bucket_std_val = (
                aligned.group_by("oss_bucket")
                .agg(pl.std("forward_return").alias("std_cond"))
                .select(pl.mean("std_cond"))
                .item()
            )
            bstd = float(bucket_std_val) if bucket_std_val is not None else 0.0
            aer = bstd / overall_std if overall_std > 1e-15 else 1.0

        # -- 7. Transfer entropy ----------------------------------------
        state_buckets: list[int] = aligned["oss_bucket"].to_list()
        fwd_vals: list[float] = aligned["forward_return"].to_list()
        te = self._compute_transfer_entropy(state_buckets, fwd_vals)

        # -- 8. Optimal lag ---------------------------------------------
        lag_score = self._find_optimal_lag(source_df, target_fwd, horizon)

        # -- 9. Per-bucket effects --------------------------------------
        effects = self._compute_effects(aligned, overall_std)

        return PropagationResult(
            source=source_symbol,
            target=target_symbol,
            horizon=horizon,
            n=n,
            wr=wr,
            pf=pf,
            aer=aer,
            transfer_entropy=float(te),
            lag_score=lag_score,
            expected_return=expected_return,
            pct_return=expected_return,
            effects=effects,
        )

    # ------------------------------------------------------------------
    # Forward-return resolution
    # ------------------------------------------------------------------

    def _resolve_forward_returns(
        self, target_df: pl.DataFrame, horizon: int
    ) -> pl.DataFrame:
        """Return a DataFrame with columns ``ts``, ``forward_return``.

        Priority
        --------
        1. If *target_df* already has a column ``forward_{horizon}``,
           use that (e.g. from :class:`ForwardSurface`).
        2. If :class:`ForwardSurface` is available, try to build forward
           returns using its internal method.
        3. Fall back to local computation from ``price`` / ``close``.
        """
        # Priority 1: pre-computed column
        col = f"forward_{horizon}"
        if col in target_df.columns:
            return target_df.select(
                pl.col("ts"),
                pl.col(col).alias("forward_return"),
            ).drop_nulls("forward_return")

        # Priority 2: ForwardSurface exists but we don't have replay
        # parameters here, so skip directly to local computation.
        # (ForwardSurface.build() requires symbol / date range that the
        # caller manages externally.)

        # Priority 3: local computation from price
        return self._compute_forward_returns_locally(target_df, horizon)

    # ------------------------------------------------------------------
    # Local forward-return computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_forward_returns_locally(
        target_df: pl.DataFrame, horizon: int
    ) -> pl.DataFrame:
        """Compute forward returns from price data using join_asof.

        For each timestamp, finds the price at ``ts + horizon`` (nearest
        neighbour within tolerance) and computes
        ``(future_price - price) / price``.
        """
        target = target_df.sort("ts").unique(subset=["ts"], maintain_order=True)
        if len(target) < 2:
            return pl.DataFrame(schema={"ts": pl.Int64, "forward_return": pl.Float64})

        # Locate a price column
        price_col: Optional[str] = None
        for candidate in ("price", "close", "mid", "last"):
            if candidate in target.columns:
                price_col = candidate
                break
        if price_col is None:
            return pl.DataFrame(schema={"ts": pl.Int64, "forward_return": pl.Float64})

        target = target.with_columns(
            pl.col(price_col).alias("_price"),
            (pl.col("ts") + horizon).alias("_future_ts"),
        )

        future = target.select(
            pl.col("ts").alias("_match_ts"),
            pl.col("_price").alias("_future_price"),
        )

        # If tolerance is set too tight many rows will be dropped; use a
        # generous tolerance (5× the horizon) to keep as many pairs as
        # possible, then let downstream drop_nulls clean up.
        tolerance = max(horizon * 5, 300)  # at least 5 minutes

        target = target.join_asof(
            future,
            left_on="_future_ts",
            right_on="_match_ts",
            strategy="nearest",
            tolerance=tolerance,
        ).with_columns(
            (
                (pl.col("_future_price") - pl.col("_price")) / pl.col("_price")
            ).alias("forward_return")
        ).drop_nulls("forward_return")

        return target.select("ts", "forward_return")

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_source_to_target(
        source_df: pl.DataFrame, target_fwd: pl.DataFrame
    ) -> pl.DataFrame:
        """Align source state timestamps to target forward returns.

        Uses a merge-asof join (nearest match within 30 seconds) so each
        source state snapshot is paired with the closest target forward
        return available.
        """
        source = source_df.sort("ts").unique(subset=["ts"], maintain_order=True)
        if len(source) == 0 or len(target_fwd) == 0:
            return pl.DataFrame()

        if "oss_bucket" not in source.columns:
            return pl.DataFrame()

        aligned = source.join_asof(
            target_fwd,
            on="ts",
            strategy="nearest",
            tolerance=30,
        ).drop_nulls("forward_return")

        return aligned

    # ------------------------------------------------------------------
    # Transfer entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_transfer_entropy(
        state_buckets: list[int],
        forward_returns: list[float],
        n_bins: int = 5,
    ) -> float:
        """Compute transfer entropy from source state to target returns.

        Discretises *forward_returns* into ``n_bins`` quantile-based bins
        (labelled *-large*, *-small*, *0*, *+small*, *+large*) and
        computes the mutual information:

        .. math::

            TE = \\sum_{s,f} P(s,f) \\log_2 \\frac{P(s,f)}{P(s) P(f)}

        which measures how much information source state provides about
        the direction / magnitude of the target forward return.

        Parameters
        ----------
        state_buckets : list[int]
            Integer state labels (typically ``oss_bucket``, 0–9).
        forward_returns : list[float]
            Raw forward returns in decimal form.
        n_bins : int
            Number of quantile bins (default 5).

        Returns
        -------
        float
            Transfer entropy in bits.
        """
        if len(state_buckets) < 2:
            return 0.0

        fwd = np.array(forward_returns, dtype=np.float64)
        if np.std(fwd) < 1e-15:
            return 0.0

        # Discretise forward returns using quantile bin edges
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        bin_edges = np.percentile(fwd, percentiles)
        bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])
        fwd_bins = np.digitize(fwd, bin_edges) - 1  # 0 .. n_bins-1

        states = np.array(state_buckets, dtype=int)
        n_unique_states = int(np.max(states)) + 1 if len(states) > 0 else 10

        # Joint distribution P(state, forward_bin)
        joint, _, _ = np.histogram2d(
            states,
            fwd_bins,
            bins=[np.arange(0, n_unique_states + 1), np.arange(0, n_bins + 1)],
        )
        joint = joint / joint.sum()

        # Marginals
        p_state = joint.sum(axis=1)
        p_fwd = joint.sum(axis=0)

        # Mutual information (proxy for transfer entropy)
        mi = 0.0
        for i in range(len(p_state)):
            if p_state[i] <= 0:
                continue
            for j in range(len(p_fwd)):
                if p_fwd[j] <= 0:
                    continue
                if joint[i, j] > 0:
                    mi += joint[i, j] * math.log2(
                        joint[i, j] / (p_state[i] * p_fwd[j])
                    )

        return float(mi)

    # ------------------------------------------------------------------
    # Optimal lead-lag scan
    # ------------------------------------------------------------------

    def _find_optimal_lag(
        self,
        source_df: pl.DataFrame,
        target_fwd: pl.DataFrame,
        horizon: int,
        max_lag_minutes: int = 10,
    ) -> int:
        """Find the lag (seconds) that maximises |correlation|.

        Shifts source timestamps forward by each candidate lag (0 to
        ``max_lag_minutes`` minutes, in 1-minute steps), re-aligns with
        target forward returns, and measures the absolute Pearson
        correlation between ``oss_bucket`` and ``forward_return``.

        Returns the lag (in seconds) that yields the strongest absolute
        correlation, or 0 if no meaningful signal is found.
        """
        if len(target_fwd) < 5:
            return 0

        source = source_df.sort("ts").unique(subset=["ts"], maintain_order=True)
        if len(source) < 2 or "oss_bucket" not in source.columns:
            return 0

        best_lag_sec = 0
        best_abs_corr: float = -1.0

        for lag_min in range(max_lag_minutes + 1):
            lag_sec = lag_min * 60
            shifted = source.with_columns(
                (pl.col("ts") + lag_sec).alias("_lag_ts")
            )

            aligned = shifted.join_asof(
                target_fwd,
                left_on="_lag_ts",
                right_on="ts",
                strategy="nearest",
                tolerance=60,
            ).drop_nulls("forward_return")

            if len(aligned) < 5:
                continue

            corr_val = aligned.select(
                pl.corr("oss_bucket", "forward_return")
            ).item()

            if corr_val is not None and not math.isnan(float(corr_val)):
                corr_abs = abs(float(corr_val))
                if corr_abs > best_abs_corr:
                    best_abs_corr = corr_abs
                    best_lag_sec = lag_sec

        return best_lag_sec

    # ------------------------------------------------------------------
    # Per-bucket effects
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_effects(
        aligned_df: pl.DataFrame, overall_std: float
    ) -> dict:
        """Build a detailed per-bucket breakdown of propagation effects.

        For each unique ``oss_bucket`` present in *aligned_df* the
        following metrics are computed:

        * ``n`` — number of samples
        * ``win_rate`` — fraction of forward returns > 0
        * ``loss_rate`` — complement of win rate
        * ``profit_factor`` — ``(sum_pos * win_rate) / abs(sum_neg * loss_rate)``
        * ``aer`` — conditional std / *overall_std*
        * ``expected_return`` — mean forward return
        * ``pct_return`` — same as expected_return
        * ``std_return`` — conditional standard deviation
        """
        if len(aligned_df) == 0:
            return {}

        if overall_std < 1e-15:
            overall_std = 1.0

        buckets = aligned_df.group_by("oss_bucket").agg(
            [
                pl.count().alias("n"),
                pl.mean("forward_return").alias("mean_return"),
                pl.std("forward_return").alias("std_return"),
                (pl.col("forward_return") > 0).mean().alias("win_rate"),
                pl.when(pl.col("forward_return") > 0)
                .then(pl.col("forward_return"))
                .otherwise(0)
                .sum()
                .alias("sum_pos"),
                pl.when(pl.col("forward_return") < 0)
                .then(pl.col("forward_return"))
                .otherwise(0)
                .sum()
                .alias("sum_neg"),
            ]
        )

        effects: dict[int, dict[str, Any]] = {}
        for row in buckets.iter_rows(named=True):
            b = int(row["oss_bucket"])
            n_bucket = int(row["n"]) if row["n"] is not None else 0
            wr = float(row["win_rate"]) if row["win_rate"] is not None else 0.0
            loss_rate = 1.0 - wr
            sum_pos = float(row["sum_pos"]) if row["sum_pos"] is not None else 0.0
            sum_neg = float(row["sum_neg"]) if row["sum_neg"] is not None else 0.0

            # Conditional profit factor
            denom = abs(sum_neg) * loss_rate
            pf = (sum_pos * wr) / denom if denom > 1e-15 else 0.0

            std_cond = (
                float(row["std_return"]) if row["std_return"] is not None else 0.0
            )
            aer = std_cond / overall_std if overall_std > 1e-15 else 1.0

            mean_ret = (
                float(row["mean_return"]) if row["mean_return"] is not None else 0.0
            )

            effects[b] = {
                "n": n_bucket,
                "win_rate": wr,
                "loss_rate": loss_rate,
                "profit_factor": pf,
                "aer": aer,
                "expected_return": mean_ret,
                "pct_return": mean_ret,
                "std_return": std_cond,
            }

        return effects

    # ------------------------------------------------------------------
    # Empty result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(source: str, target: str, horizon: int) -> PropagationResult:
        """Return a zeroed-out result when no samples are available."""
        return PropagationResult(
            source=source,
            target=target,
            horizon=horizon,
            n=0,
            wr=0.0,
            pf=0.0,
            aer=1.0,
            transfer_entropy=0.0,
            lag_score=0,
            expected_return=0.0,
            pct_return=0.0,
            effects={},
        )
