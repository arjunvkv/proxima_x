"""RegimeAlignmentEngine — tests whether source asset microstructure state
predicts target asset regime transitions (volatility regimes, not just price
direction).

This module aligns source-state snapshots (``oss_bucket``) with target price
data, computes rolling volatility and expansion regimes for the target, and
quantifies how much each source state elevates (or suppresses) the probability
of the target being in a particular regime.
"""

import sys; sys.path.insert(0, ".")

import math
from typing import Optional

import numpy as np
import polars as pl

from research.cep.session_partition import SessionPartitioner


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def align_to_regime_matrix(
    source_df: pl.DataFrame,
    target_df: pl.DataFrame,
    regimes: list[str],
) -> pl.DataFrame:
    """Create a joint DataFrame with source state and target regime columns.

    Performs an as-of join (30-second tolerance) on ``ts`` so that each
    source state snapshot is paired with the closest target row that carries
    the requested *regimes* columns.

    Parameters
    ----------
    source_df : pl.DataFrame
        Source asset DataFrame with at least columns ``ts`` and
        ``oss_bucket``.
    target_df : pl.DataFrame
        Target asset DataFrame with at least columns ``ts`` and each column
        named in *regimes*.
    regimes : list[str]
        Names of the regime columns in *target_df* to align (e.g.
        ``["vol_regime", "expansion_regime"]``).

    Returns
    -------
    pl.DataFrame
        Joined DataFrame with source columns (including ``oss_bucket``) and
        the requested *regimes* columns.  Rows where any regime column is
        null are dropped.
    """
    source = source_df.sort("ts").unique(subset=["ts"], maintain_order=True)
    target = target_df.sort("ts").unique(subset=["ts"], maintain_order=True)

    if "oss_bucket" not in source.columns:
        return pl.DataFrame()

    available = [c for c in regimes if c in target.columns]
    if not available:
        return pl.DataFrame()

    cols = ["ts"] + available
    target_sub = target.select(cols)

    aligned = source.join_asof(
        target_sub,
        on="ts",
        strategy="nearest",
        tolerance=30,
    ).drop_nulls(subset=available)

    return aligned


# ---------------------------------------------------------------------------
# RegimeAlignmentEngine
# ---------------------------------------------------------------------------

class RegimeAlignmentEngine:
    """Quantify how source asset microstructure states predict target regimes.

    The engine aligns per-minute source state signatures (``oss_bucket``)
    with target price data, computes rolling volatility and expansion
    regimes for the target, and returns a score dict that includes the
    regime transfer matrix, average lift effect, and best/worst buckets.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._partitioner = SessionPartitioner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        source_df: pl.DataFrame,
        target_df: pl.DataFrame,
        source_symbol: str,
        target_symbol: str,
    ) -> dict:
        """Compute regime alignment metrics between source and target.

        Parameters
        ----------
        source_df : pl.DataFrame
            Source asset DataFrame with columns ``ts``, ``oss_bucket``
            (and optionally other state features).
        target_df : pl.DataFrame
            Target asset DataFrame with columns ``ts``, ``price`` (or
            ``close`` / ``mid`` / ``last``).
        source_symbol : str
            Symbol name for the source asset (used for diagnostics).
        target_symbol : str
            Symbol name for the target asset.

        Returns
        -------
        dict
            Keys:

            - ``regime_transfer_matrix`` : list[list[float]]
                ``P(target_vol_regime_j | source_bucket_i)`` as a 2-D list
                with rows = source buckets 0-9, columns = vol regimes
                0 (low), 1 (normal), 2 (high).
            - ``alignment_score`` : float
                Mean conditional high-vol probability across all buckets
                (a simple aggregate measure of regime predictability).
            - ``ale`` : float
                Average lift effect —
                ``mean_i[ P(high_vol | bucket_i) / P(high_vol) ]``
                weighted by bucket frequency.
            - ``best_bucket`` : int
                Source bucket with the highest regime lift.
            - ``worst_bucket`` : int
                Source bucket with the lowest regime lift.
            - ``expansion_transfer_matrix`` : list[list[float]]
                ``P(target_expansion_regime_j | source_bucket_i)``.
            - ``n_aligned`` : int
                Number of successfully aligned (source, target) samples.
            - ``source_symbol`` : str
            - ``target_symbol`` : str
        """
        # -- 1. Validate & deduplicate -------------------------------------
        if not self._valid(source_df, target_df):
            return self._empty_score(source_symbol, target_symbol)

        source = source_df.sort("ts").unique(subset=["ts"], maintain_order=True)
        target = target_df.sort("ts").unique(subset=["ts"], maintain_order=True)

        if len(source) < 2 or len(target) < 2:
            return self._empty_score(source_symbol, target_symbol)

        # Locate price column in target
        price_col: Optional[str] = None
        for candidate in ("price", "close", "mid", "last"):
            if candidate in target.columns:
                price_col = candidate
                break
        if price_col is None:
            return self._empty_score(source_symbol, target_symbol)

        # -- 2. Align source states to target timestamps ------------------
        aligned = source.join_asof(
            target.select("ts", pl.col(price_col).alias("_price")),
            on="ts",
            strategy="nearest",
            tolerance=30,
        ).drop_nulls("_price")

        if len(aligned) < 2:
            return self._empty_score(source_symbol, target_symbol)

        aligned_ts = aligned["ts"].to_list()
        aligned_buckets: list[int] = aligned["oss_bucket"].to_list()

        # -- 3. Compute rolling regimes for each aligned point -----------
        # We need the full target price history (sorted by ts) to build
        # look-back windows correctly.
        target_sorted = target.sort("ts")
        target_all_prices: list[float] = target_sorted[price_col].to_list()
        target_all_ts: list[int] = target_sorted["ts"].to_list()

        vol_regimes: list[int] = []
        exp_regimes: list[int] = []

        for i in range(len(aligned)):
            ats = aligned_ts[i]
            # Find the insertion point in target_all_ts
            pos = self._find_le(target_all_ts, ats)
            if pos < 0:
                pos = 0
            # Include prices up to this position
            window_prices = target_all_prices[: pos + 1]
            vr = self._volatility_regime(window_prices)
            er = self._expansion_regime(window_prices)
            vol_regimes.append(vr)
            exp_regimes.append(er)

        # -- 4. Compute unconditional probabilities -----------------------
        n_total = len(vol_regimes)
        p_high_vol = sum(1 for r in vol_regimes if r == 2) / n_total
        if p_high_vol < 1e-15:
            p_high_vol = 1e-15  # avoid division by zero

        p_expand = sum(1 for r in exp_regimes if r == 2) / n_total
        if p_expand < 1e-15:
            p_expand = 1e-15

        # -- 5. Per-bucket conditional probabilities ----------------------
        # Collect (oss_bucket, vol_regime, expansion_regime) triples
        n_buckets = 10  # oss_bucket is 0..9
        bucket_counts = [0] * n_buckets
        bucket_vol_counts = [[0, 0, 0] for _ in range(n_buckets)]
        bucket_exp_counts = [[0, 0, 0] for _ in range(n_buckets)]

        for b, vr, er in zip(aligned_buckets, vol_regimes, exp_regimes):
            if 0 <= b < n_buckets:
                bucket_counts[b] += 1
                bucket_vol_counts[b][vr] += 1
                bucket_exp_counts[b][er] += 1

        # -- 6. Regime transfer matrix (volatility) -----------------------
        transfer_matrix: list[list[float]] = []
        for b in range(n_buckets):
            cnt = bucket_counts[b]
            if cnt == 0:
                transfer_matrix.append([0.0, 0.0, 0.0])
            else:
                row = [bucket_vol_counts[b][r] / cnt for r in range(3)]
                transfer_matrix.append(row)

        # -- 7. Expansion transfer matrix ---------------------------------
        expansion_matrix: list[list[float]] = []
        for b in range(n_buckets):
            cnt = bucket_counts[b]
            if cnt == 0:
                expansion_matrix.append([0.0, 0.0, 0.0])
            else:
                row = [bucket_exp_counts[b][r] / cnt for r in range(3)]
                expansion_matrix.append(row)

        # -- 8. Conditional high-vol probability per bucket ---------------
        # P(high_vol | bucket)
        cond_high_vol: list[float] = []
        for b in range(n_buckets):
            cnt = bucket_counts[b]
            if cnt == 0:
                cond_high_vol.append(p_high_vol)  # default to unconditional
            else:
                cond_high_vol.append(bucket_vol_counts[b][2] / cnt)

        # -- 9. Regime alignment score & lift (ALE) -----------------------
        # alignment_score = mean of conditional high-vol prob across buckets
        alignment_score = float(np.mean(cond_high_vol))

        # ALE = average of P(high_vol|bucket) / P(high_vol) weighted by
        #       bucket frequency
        lifts: list[float] = []
        lift_weights: list[float] = []
        for b in range(n_buckets):
            cnt = bucket_counts[b]
            if cnt > 0:
                lifts.append(cond_high_vol[b] / p_high_vol)
                lift_weights.append(cnt / n_total)
        if lifts:
            ale = float(np.average(lifts, weights=lift_weights))
        else:
            ale = 1.0

        # -- 10. Best / worst bucket (by lift) ----------------------------
        bucket_lifts: list[tuple[int, float]] = []
        for b in range(n_buckets):
            if bucket_counts[b] > 0:
                bucket_lifts.append((b, cond_high_vol[b] / p_high_vol))
            else:
                bucket_lifts.append((b, 1.0))

        best_bucket = max(bucket_lifts, key=lambda x: x[1])[0]
        worst_bucket = min(bucket_lifts, key=lambda x: x[1])[0]

        return {
            "regime_transfer_matrix": transfer_matrix,
            "expansion_transfer_matrix": expansion_matrix,
            "alignment_score": float(alignment_score),
            "ale": float(ale),
            "best_bucket": int(best_bucket),
            "worst_bucket": int(worst_bucket),
            "n_aligned": n_total,
            "source_symbol": source_symbol,
            "target_symbol": target_symbol,
        }

    # ------------------------------------------------------------------
    # Regime classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _volatility_regime(prices: list[float], window: int = 20) -> int:
        """Classify the most recent price window into a volatility regime.

        Compares the realised volatility of the most recent ``window``
        returns to that of the preceding ``window`` returns.  If the recent
        vol is significantly higher → regime 2 (high); significantly lower
        → regime 0 (low); otherwise → regime 1 (normal).

        Parameters
        ----------
        prices : list[float]
            Price series in chronological order.  Need at least
            ``2 * window + 1`` entries for a reliable baseline comparison.
        window : int
            Look-back window length in price observations (default 20).

        Returns
        -------
        int
            0 (low vol), 1 (normal vol), or 2 (high vol).
        """
        min_required = 2 * window + 1
        if len(prices) < min_required:
            # Fall back to a simpler classification when data is scarce
            return RegimeAlignmentEngine._volatility_regime_short(prices, window)

        # Compute log returns for the baseline (older) window and the
        # recent window.
        older = prices[-(2 * window) : -window]
        recent = prices[-window:]

        older_rets = np.diff(np.log(np.array(older, dtype=np.float64)))
        recent_rets = np.diff(np.log(np.array(recent, dtype=np.float64)))

        older_vol = float(np.std(older_rets, ddof=1))
        recent_vol = float(np.std(recent_rets, ddof=1))

        baseline = older_vol if older_vol > 1e-15 else 1e-15
        ratio = recent_vol / baseline

        if ratio > 1.5:
            return 2  # high vol
        elif ratio < 0.67:
            return 0  # low vol
        return 1  # normal vol

    @staticmethod
    def _volatility_regime_short(prices: list[float], window: int) -> int:
        """Fallback vol classifier for short price series.

        Uses a ratio of recent volatility to the mean absolute return.
        """
        if len(prices) < window + 1:
            return 1  # neutral — insufficient data

        recent = prices[-window:]
        returns = np.diff(np.log(np.array(recent, dtype=np.float64)))
        vol = float(np.std(returns, ddof=1))
        mean_abs = float(np.mean(np.abs(returns)))
        if mean_abs < 1e-15:
            return 1

        ratio = vol / mean_abs
        # For normally distributed zero-mean returns the expected ratio
        # vol / mean_abs is about 1.25.
        if ratio > 1.6:
            return 2
        elif ratio < 0.9:
            return 0
        return 1

    @staticmethod
    def _expansion_regime(prices: list[float], window: int = 20) -> int:
        """Classify the most recent price window into an expansion regime.

        Uses a t-statistic of the mean log-return over the recent window.
        If significantly positive → regime 2 (expanding); significantly
        negative → regime 0 (contracting); otherwise → regime 1 (neutral).

        Parameters
        ----------
        prices : list[float]
            Price series in chronological order.  Need at least
            ``window + 1`` entries.
        window : int
            Look-back window length in price observations (default 20).

        Returns
        -------
        int
            0 (contracting), 1 (neutral), or 2 (expanding).
        """
        if len(prices) < window + 1:
            return 1  # neutral — insufficient data

        recent = prices[-(window + 1) :]
        returns = np.diff(np.log(np.array(recent, dtype=np.float64)))
        n = len(returns)
        if n < 2:
            return 1

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))

        if std_ret < 1e-15:
            return 1  # flat prices → neutral

        t_stat = mean_ret / (std_ret / math.sqrt(n))
        if t_stat > 1.0:
            return 2  # expanding
        elif t_stat < -1.0:
            return 0  # contracting
        return 1  # neutral

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid(source_df: pl.DataFrame, target_df: pl.DataFrame) -> bool:
        """Check that both DataFrames have the required columns and rows."""
        if len(source_df) == 0 or len(target_df) == 0:
            return False
        for col in ("ts", "oss_bucket"):
            if col not in source_df.columns:
                return False
        if "ts" not in target_df.columns:
            return False
        return True

    @staticmethod
    def _find_le(sorted_arr: list[int], value: int) -> int:
        """Return the largest index ``i`` s.t. ``sorted_arr[i] <= value``.

        Returns -1 if *value* is smaller than every element.
        """
        if not sorted_arr or value < sorted_arr[0]:
            return -1
        if value >= sorted_arr[-1]:
            return len(sorted_arr) - 1
        lo, hi = 0, len(sorted_arr) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sorted_arr[mid] <= value:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @staticmethod
    def _empty_score(
        source_symbol: str = "",
        target_symbol: str = "",
    ) -> dict:
        """Return a zeroed-out score dict when no alignment is possible."""
        return {
            "regime_transfer_matrix": [[0.0, 0.0, 0.0] for _ in range(10)],
            "expansion_transfer_matrix": [[0.0, 0.0, 0.0] for _ in range(10)],
            "alignment_score": 0.0,
            "ale": 1.0,
            "best_bucket": 0,
            "worst_bucket": 0,
            "n_aligned": 0,
            "source_symbol": source_symbol,
            "target_symbol": target_symbol,
        }
