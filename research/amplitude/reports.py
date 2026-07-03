"""
Phase V — Structural Amplitude Discovery
=========================================
Report export — generates all surface, promotion, exceedance, and regime
findings into CSV, JSON, and human-readable text files.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import json
import os
from datetime import datetime
from typing import Any

import polars as pl

from research.amplitude.amplitude_surface import AmplitudeSurfaceEngine
from research.amplitude.spread_exceedance import SpreadExceedanceModel
from research.amplitude.promotion_engine import AmplitudePromotionEngine
from research.amplitude.regime_mapper import AmplitudeRegimeMapper
from research.amplitude.schemas import AmplitudeSurfaceEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SURFACE_COLS = [
    "state_hash",
    "horizon",
    "n",
    "mean_abs",
    "median_abs",
    "p90_abs",
    "spread_multiple",
    "aer",
]

_PROMOTION_COLS = [
    "state_hash",
    "horizon",
    "aer",
    "spread_multiple",
    "n",
    "tier",
    "promoted",
    "stability",
]

_HEATMAP_COLS = [
    "regime_id",
    "horizon",
    "mean_aer",
    "mean_spread_multiple",
    "n_observations",
]

_EXCEED_CURVE_COLS = [
    "state_hash",
    "k",
    "probability",
    "n",
]

_BEST_STATE_COLS = [
    "state_hash",
    "horizon",
    "spread_multiple",
    "aer",
    "exceed_prob_k3",
    "n",
]

_BEST_STATES_MIN_SPREAD_MULTIPLE = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export(results: pl.DataFrame, output_dir: str = "reports") -> dict[str, str]:
    """Generate all Phase V report exports from the amplitude *results*.

    Parameters
    ----------
    results : pl.DataFrame
        Amplitude surface results with per-(state_hash, horizon) statistics.
        Expected to contain at minimum the columns:

        - ``state_hash``     — composite state key
        - ``horizon``        — forward-looking horizon (seconds)
        - ``n``              — observation count
        - ``mean_abs``       — mean absolute price move (bps)
        - ``median_abs``     — median absolute price move (bps)
        - ``p90_abs``        — 90th percentile absolute move (bps)
        - ``spread_multiple``— mean_abs / spread ratio
        - ``aer``            — amplitude expansion ratio

        May also contain:

        - ``tier``, ``promoted`` — promotion columns (optional; computed
          via ``AmplitudePromotionEngine`` if absent)
        - ``regime_id``           — amplitude regime label (optional)
        - ``exceed_*``            — exceedance probability columns such as
          ``exceed_1.0``, ``exceed_3.0`` (optional)

    output_dir : str
        Directory to write report files into.  Created if it does not
        exist (default ``"reports"``).

    Returns
    -------
    dict[str, str]
        Mapping of logical report name to the absolute (or relative) file
        path written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Delegate to per-report builders so each section is independently
    # testable and handles the empty-DataFrame case uniformly.
    is_empty = len(results) == 0

    # ---- 1. Surface Summary ------------------------------------------------
    surface_path = os.path.join(output_dir, "amplitude_surface_summary.csv")
    if is_empty:
        _write_empty_csv(surface_path, _SURFACE_COLS)
    else:
        _build_surface_summary(results).write_csv(surface_path)

    # ---- 2. Promotions Report ----------------------------------------------
    promos_path = os.path.join(output_dir, "amplitude_promotions.csv")
    if is_empty:
        _write_empty_csv(promos_path, _PROMOTION_COLS)
    else:
        _build_promotions(results).write_csv(promos_path)

    # ---- 3. Regime Heatmap -------------------------------------------------
    heatmap_path = os.path.join(output_dir, "amplitude_regime_heatmap.csv")
    if is_empty or "regime_id" not in results.columns:
        _write_empty_csv(heatmap_path, _HEATMAP_COLS)
    else:
        _build_regime_heatmap(results).write_csv(heatmap_path)

    # ---- 4. Spread Exceedance Curve ----------------------------------------
    exceed_path = os.path.join(output_dir, "amplitude_spread_exceedance_curve.csv")
    if is_empty:
        _write_empty_csv(exceed_path, _EXCEED_CURVE_COLS)
    else:
        _build_exceedance_curve(results).write_csv(exceed_path)

    # ---- 5. Best States Report ---------------------------------------------
    best_path = os.path.join(output_dir, "amplitude_best_states.csv")
    if is_empty:
        _write_empty_csv(best_path, _BEST_STATE_COLS)
    else:
        _build_best_states(results).write_csv(best_path)

    # ---- 6. Summary JSON ---------------------------------------------------
    json_path = os.path.join(output_dir, "amplitude_summary.json")
    summary = _build_summary(results) if not is_empty else _empty_summary()
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=4)

    # ---- 7. Executive Summary (human-readable) -----------------------------
    exec_path = os.path.join(output_dir, "amplitude_executive_summary.txt")
    text = _build_executive_summary(summary) if not is_empty else _empty_executive_summary()
    with open(exec_path, "w") as fh:
        fh.write(text)

    return {
        "surface_summary": surface_path,
        "promotions": promos_path,
        "regime_heatmap": heatmap_path,
        "exceedance_curve": exceed_path,
        "best_states": best_path,
        "summary_json": json_path,
        "executive_summary": exec_path,
    }


# ---------------------------------------------------------------------------
# Per-report builders  (all accept a non-empty DataFrame)
# ---------------------------------------------------------------------------


def _build_surface_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Build the surface-summary CSV payload.

    Columns: state_hash, horizon, n, mean_abs, median_abs, p90_abs,
             spread_multiple, aer
    Sorted by spread_multiple descending.
    """
    avail = [c for c in _SURFACE_COLS if c in df.columns]
    return df.select(avail).sort("spread_multiple", descending=True)


def _build_promotions(df: pl.DataFrame) -> pl.DataFrame:
    """Build the promotions CSV payload.

    Uses ``AmplitudePromotionEngine`` to evaluate entries if ``tier`` and
    ``promoted`` columns are not present in *df*.
    """
    has_promo = {"tier", "promoted"}.issubset(df.columns)

    if has_promo:
        promoted = df.filter(pl.col("promoted") == True)
    else:
        engine = AmplitudePromotionEngine()
        entries = [_row_to_surface_entry(r) for r in df.iter_rows(named=True)]
        promo_df = engine.evaluate_batch(entries)
        promoted = promo_df.filter(pl.col("promoted") == True)

    # Attach stability (fraction of exceed probabilities > 0.3)
    if "stability" in promoted.columns:
        pass  # already present
    else:
        exceed_cols = sorted([c for c in df.columns if c.startswith("exceed_")])
        if exceed_cols:
            # Build stability expression and compute on original df, then join
            cond_sum = sum((pl.col(c) > 0.3).cast(pl.Float64) for c in exceed_cols)
            stab_expr = cond_sum / len(exceed_cols)
            stab_df = df.select(["state_hash", "horizon"] + exceed_cols)
            stab_df = stab_df.with_columns(stab_expr.alias("stability"))
            stab_df = stab_df.select(["state_hash", "horizon", "stability"])
            promoted = promoted.join(stab_df, on=["state_hash", "horizon"], how="left")
        else:
            promoted = promoted.with_columns(pl.lit(0.0).alias("stability"))

    # Ensure stability is never None (fill with 0.0)
    if "stability" in promoted.columns:
        promoted = promoted.with_columns(
            pl.col("stability").fill_null(0.0)
        )

    avail = [c for c in _PROMOTION_COLS if c in promoted.columns]
    return promoted.select(avail)


def _build_regime_heatmap(df: pl.DataFrame) -> pl.DataFrame:
    """Build the regime heatmap CSV payload.

    Groups by ``regime_id`` and ``horizon``, computing mean AER, mean
    spread-multiple, and observation bucket count.

    Requires ``regime_id`` column in *df*.
    """
    return (
        df.group_by(["regime_id", "horizon"])
        .agg([
            pl.col("aer").mean().alias("mean_aer"),
            pl.col("spread_multiple").mean().alias("mean_spread_multiple"),
            pl.len().alias("n_observations"),
        ])
        .sort(["regime_id", "horizon"])
    )


def _build_exceedance_curve(df: pl.DataFrame) -> pl.DataFrame:
    """Build the spread-exceedance curve CSV payload.

    Unpivots ``exceed_*`` columns into a long-format table with columns
    ``state_hash``, ``k``, ``probability``, ``n``.

    Returns an empty DataFrame (with correct schema) if no exceedance
    columns are found.
    """
    exceed_cols = sorted([c for c in df.columns if c.startswith("exceed_")])
    if not exceed_cols:
        return pl.DataFrame(
            {"state_hash": [], "k": [], "probability": [], "n": []},
            schema={"state_hash": pl.Utf8, "k": pl.Float64, "probability": pl.Float64, "n": pl.Int64},
        )

    id_vars = ["state_hash"]
    if "n" in df.columns:
        id_vars.append("n")

    melted = df.unpivot(
        index=id_vars,
        on=exceed_cols,
        variable_name="k_str",
        value_name="probability",
    )

    # Parse numeric k from column name, e.g. "exceed_3.0" -> 3.0
    k_expr = pl.col("k_str").str.strip_prefix("exceed_").cast(pl.Float64)
    melted = melted.with_columns(k_expr.alias("k")).drop("k_str")

    return melted.select(["state_hash", "k", "probability", "n"]).sort(["state_hash", "k"])


def _build_best_states(df: pl.DataFrame) -> pl.DataFrame:
    """Build the best-states CSV payload.

    Filters to rows where ``spread_multiple > 2.0``, includes the
    ``exceed_prob_k3`` probability (read from ``exceed_3.0`` if present),
    and sorts by spread_multiple descending.
    """
    best = df.filter(pl.col("spread_multiple") > _BEST_STATES_MIN_SPREAD_MULTIPLE)

    if "exceed_3.0" in best.columns:
        best = best.with_columns(pl.col("exceed_3.0").alias("exceed_prob_k3"))
    elif "exceed_prob_k3" not in best.columns:
        best = best.with_columns(pl.lit(None, dtype=pl.Float64).alias("exceed_prob_k3"))

    avail = [c for c in _BEST_STATE_COLS if c in best.columns]
    return best.select(avail).sort("spread_multiple", descending=True)


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _build_summary(df: pl.DataFrame) -> dict[str, Any]:
    """Compute the machine-readable summary JSON dict from a non-empty *df*."""
    # Counts
    total_buckets = df.select(pl.col("state_hash").n_unique()).item()
    total_horizons = df.select(pl.col("horizon").n_unique()).item()

    if "n" in df.columns:
        total_observations = df.select(pl.col("n").sum()).item()
    else:
        total_observations = len(df)

    # Promotion counts
    if "tier" in df.columns:
        promotions_tier1 = df.filter(pl.col("tier") == 1).height
        promotions_tier2 = df.filter(pl.col("tier") == 2).height
    else:
        promotions_tier1 = 0
        promotions_tier2 = 0

    # Best row (by spread_multiple)
    if "spread_multiple" in df.columns and df.height > 0:
        best_row = df.sort("spread_multiple", descending=True).row(0, named=True)
        best_spread_multiple = float(best_row.get("spread_multiple", 0.0))
        best_aer = float(best_row.get("aer", 0.0))
        best_state_hash = str(best_row.get("state_hash", ""))
        best_horizon = int(best_row.get("horizon", 0))
    else:
        best_spread_multiple = 0.0
        best_aer = 0.0
        best_state_hash = ""
        best_horizon = 0

    # Verdict
    if promotions_tier1 > 0:
        verdict = "STRONG_BUY — Tier 1 promotions identified"
    elif promotions_tier2 > 0:
        verdict = "CAUTIOUS_BUY — Tier 2 promotions only"
    elif best_spread_multiple > 2.0:
        verdict = "SPECULATIVE — High spread multiple but below promotion thresholds"
    else:
        verdict = "NO_TRADE — Insufficient amplitude signal"

    return {
        "total_buckets": total_buckets,
        "total_horizons": total_horizons,
        "total_observations": total_observations,
        "promotions_tier1": promotions_tier1,
        "promotions_tier2": promotions_tier2,
        "best_spread_multiple": best_spread_multiple,
        "best_aer": best_aer,
        "best_state_hash": best_state_hash,
        "best_horizon": best_horizon,
        "program_verdict": verdict,
    }


def _empty_summary() -> dict[str, Any]:
    """Summary dict for an empty input DataFrame."""
    return {
        "total_buckets": 0,
        "total_horizons": 0,
        "total_observations": 0,
        "promotions_tier1": 0,
        "promotions_tier2": 0,
        "best_spread_multiple": 0.0,
        "best_aer": 0.0,
        "best_state_hash": "",
        "best_horizon": 0,
        "program_verdict": "NO_DATA — Empty input DataFrame",
    }


def _build_executive_summary(summary: dict[str, Any]) -> str:
    """Build human-readable executive summary text from the JSON *summary* dict."""
    lines = [
        "=" * 70,
        "AMPLITUDE STRUCTURAL DISCOVERY \u2014 EXECUTIVE SUMMARY",
        "=" * 70,
        f"  Generated             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "  \u2014\u2014 Overview \u2014\u2014",
        f"  Total observations     : {summary['total_observations']:,}",
        f"  Surface buckets        : {summary['total_buckets']}",
        f"  Horizons               : {summary['total_horizons']}",
        "",
        "  \u2014\u2014 Promotions \u2014\u2014",
        f"  Tier 1 promotions      : {summary['promotions_tier1']}",
        f"  Tier 2 promotions      : {summary['promotions_tier2']}",
        "",
        "  \u2014\u2014 Best Results \u2014\u2014",
        f"  Best spread multiple   : {summary['best_spread_multiple']:.4f}",
        f"  Best AER               : {summary['best_aer']:.4f}",
        f"  Best state hash        : {summary['best_state_hash']}",
        f"  Best horizon           : {summary['best_horizon']}s",
        "",
        "  \u2014\u2014 Verdict \u2014\u2014",
        f"  {summary['program_verdict']}",
        "=" * 70,
    ]
    return "\n".join(lines)


def _empty_executive_summary() -> str:
    """Executive summary text for an empty input DataFrame."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "=" * 70 + "\n"
        "AMPLITUDE STRUCTURAL DISCOVERY \u2014 EXECUTIVE SUMMARY\n"
        + "=" * 70 + "\n"
        f"  Generated  : {now}\n"
        "\n"
        "  No data was present in the input DataFrame.\n"
        "  All report files contain headers only.\n"
        "  Run the amplitude pipeline first to generate observations.\n"
        + "=" * 70 + "\n"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_empty_csv(path: str, columns: list[str]) -> None:
    """Write a CSV with only a header row (no data rows)."""
    pl.DataFrame({c: [] for c in columns}).write_csv(path)


def _row_to_surface_entry(row: dict[str, Any]) -> AmplitudeSurfaceEntry:
    """Convert a DataFrame row dict into an ``AmplitudeSurfaceEntry``.

    Exceed-probability columns are detected by the ``exceed_`` prefix and
    packed into the ``exceed_prob`` dict with numeric keys.
    """
    exceed_prob: dict[float, float] = {}
    for colname, value in row.items():
        if colname.startswith("exceed_"):
            try:
                k = float(colname.split("_", 1)[1])
                exceed_prob[k] = float(value) if value is not None else 0.0
            except (ValueError, IndexError):
                pass

    return AmplitudeSurfaceEntry(
        state_hash=str(row.get("state_hash", "")),
        horizon=int(row.get("horizon", 0)),
        n=int(row.get("n", 0)),
        mean_abs=float(row.get("mean_abs", 0.0)),
        median_abs=float(row.get("median_abs", 0.0)),
        std_abs=float(row.get("std_abs", 0.0)),
        p90_abs=float(row.get("p90_abs", 0.0)),
        spread_multiple=float(row.get("spread_multiple", 0.0)),
        aer=float(row.get("aer", 0.0)),
        exceed_prob=exceed_prob,
    )
