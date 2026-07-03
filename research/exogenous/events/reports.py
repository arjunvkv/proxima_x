"""
Program VI.5 — Event Alignment Reports
==========================================
Export event promotion and surface evaluation results to human-readable
reports and machine-readable summaries.

Typical usage::

    from research.exogenous.events.reports import export
    paths = export(results_df, output_dir="reports")
    print(paths)
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

import json
import os
from typing import Any

import polars as pl


def export(
    results: pl.DataFrame, output_dir: str = "reports"
) -> dict[str, str]:
    """Generate report files from event promotion evaluation results.

    Produces five files:

    1. ``event_surface_summary.csv``   — all surface entries sorted by spread_multiple desc
    2. ``event_promotions.csv``        — only promoted entries (tier > 0)
    3. ``event_bucket_summary.csv``    — grouped by event_bucket with aggregate stats
    4. ``event_summary.json``          — machine-readable summary dictionary
    5. ``event_executive_summary.txt`` — human-readable verdict with pass/fail per bucket

    Parameters
    ----------
    results : pl.DataFrame
        Evaluation results produced by :class:`EventPromotionEngine`
        enriched with surface metadata (at minimum columns
        ``key``, ``horizon``, ``aer``, ``spread_multiple``, ``n``,
        ``tier``, ``promoted``, ``p2x``, ``p3x``, ``event_bucket``,
        ``event_impact``).
    output_dir : str
        Directory where report files are written (created if absent).

    Returns
    -------
    dict[str, str]
        Mapping from report name to absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths: dict[str, str] = {}

    # Ensure event_bucket column exists for grouping
    df = results.clone()
    if "event_bucket" not in df.columns and "key" in df.columns:
        df = df.with_columns(
            pl.col("key").str.split("|").list.get(0).alias("event_bucket")
        )

    # ------------------------------------------------------------------
    # 1. Surface summary — all entries sorted by spread_multiple desc
    # ------------------------------------------------------------------
    surface_csv = os.path.join(output_dir, "event_surface_summary.csv")
    if "spread_multiple" in df.columns and not df.is_empty():
        sorted_df = df.sort("spread_multiple", descending=True)
        sorted_df.write_csv(surface_csv)
    else:
        df.write_csv(surface_csv)
    paths["surface_summary"] = os.path.abspath(surface_csv)

    # ------------------------------------------------------------------
    # 2. Promotions — only tier > 0
    # ------------------------------------------------------------------
    prom_csv = os.path.join(output_dir, "event_promotions.csv")
    if "tier" in df.columns and not df.is_empty():
        prom_df = df.filter(pl.col("tier") > 0)
        if not prom_df.is_empty():
            prom_df.write_csv(prom_csv)
        else:
            df.head(0).write_csv(prom_csv)
    else:
        df.head(0).write_csv(prom_csv)
    paths["promotions"] = os.path.abspath(prom_csv)

    # ------------------------------------------------------------------
    # 3. Bucket summary — grouped by event_bucket
    # ------------------------------------------------------------------
    bucket_csv = os.path.join(output_dir, "event_bucket_summary.csv")
    if "event_bucket" in df.columns and not df.is_empty():
        bucket_agg = df.group_by("event_bucket").agg([
            pl.col("aer").mean().alias("mean_aer"),
            pl.col("spread_multiple").mean().alias("mean_sm"),
            pl.col("p2x").mean().alias("mean_p2x"),
            pl.col("p3x").mean().alias("mean_p3x"),
            pl.col("n").sum().alias("total_n"),
            pl.len().alias("n_entries"),
            (pl.col("promoted").sum() / pl.len()).alias("promotion_rate"),
        ]).sort("mean_sm", descending=True)
        bucket_agg.write_csv(bucket_csv)
    else:
        pl.DataFrame({
            "event_bucket": pl.Series([], dtype=pl.Utf8),
            "mean_aer": pl.Series([], dtype=pl.Float64),
            "mean_sm": pl.Series([], dtype=pl.Float64),
            "mean_p2x": pl.Series([], dtype=pl.Float64),
            "mean_p3x": pl.Series([], dtype=pl.Float64),
            "total_n": pl.Series([], dtype=pl.Int64),
            "n_entries": pl.Series([], dtype=pl.Int64),
            "promotion_rate": pl.Series([], dtype=pl.Float64),
        }).write_csv(bucket_csv)
    paths["bucket_summary"] = os.path.abspath(bucket_csv)

    # ------------------------------------------------------------------
    # 4. Machine-readable JSON summary
    # ------------------------------------------------------------------
    summary_json_path = os.path.join(output_dir, "event_summary.json")
    summary: dict[str, Any] = _build_machine_summary(df)
    with open(summary_json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    paths["summary_json"] = os.path.abspath(summary_json_path)

    # ------------------------------------------------------------------
    # 5. Human-readable executive summary
    # ------------------------------------------------------------------
    exec_txt_path = os.path.join(
        output_dir, "event_executive_summary.txt"
    )
    executive_text = _build_executive_summary(df, summary)
    with open(exec_txt_path, "w") as fh:
        fh.write(executive_text)
    paths["executive_summary"] = os.path.abspath(exec_txt_path)

    _print_report_index(paths)
    return paths


# ===================================================================
# Internal helpers
# ===================================================================


def _build_machine_summary(results: pl.DataFrame) -> dict[str, Any]:
    """Construct a machine-readable summary dictionary."""
    total = len(results)
    n_promoted = _safe_count(results, "promoted", True) if total > 0 else 0
    n_tier1 = _safe_count(results, "tier", 1) if total > 0 else 0
    n_tier2 = _safe_count(results, "tier", 2) if total > 0 else 0

    return {
        "total_entries": total,
        "n_promoted": n_promoted,
        "n_tier1": n_tier1,
        "n_tier2": n_tier2,
        "promotion_rate": round(n_promoted / total, 4) if total > 0 else 0.0,
        "mean_aer": _safe_mean(results, "aer"),
        "mean_spread_multiple": _safe_mean(results, "spread_multiple"),
        "mean_p2x": _safe_mean(results, "p2x"),
        "mean_p3x": _safe_mean(results, "p3x"),
        "max_spread_multiple": _safe_max(results, "spread_multiple"),
        "max_aer": _safe_max(results, "aer"),
        "n_buckets": (
            results["event_bucket"].n_unique()
            if "event_bucket" in results.columns and total > 0
            else 0
        ),
    }


def _build_executive_summary(
    results: pl.DataFrame, summary: dict[str, Any]
) -> str:
    """Build a human-readable executive summary with explicit pass/fail per bucket."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("PROGRAM VI.5 — EXTERNAL EVENT ALIGNMENT LAYER")
    lines.append("Executive Summary")
    lines.append("=" * 64)
    lines.append("")

    if results.is_empty():
        lines.append(
            "No entries were evaluated — the results DataFrame is empty."
        )
        lines.append("")
        lines.append("=" * 64)
        return "\n".join(lines)

    lines.append(
        f"Total surface entries evaluated:  {summary['total_entries']}"
    )
    lines.append(
        f"Promoted entries:                {summary['n_promoted']}"
    )
    lines.append(
        f"  Tier 1:                        {summary['n_tier1']}"
    )
    lines.append(
        f"  Tier 2:                        {summary['n_tier2']}"
    )
    lines.append(
        f"Promotion rate:                  {summary['promotion_rate']:.2%}"
    )
    lines.append("")
    lines.append(
        f"Mean AER:                       {summary['mean_aer']:.4f}"
    )
    lines.append(
        f"Mean spread multiple:            "
        f"{summary['mean_spread_multiple']:.4f}"
    )
    lines.append(
        f"Mean P(|move| > 2x spread):      {summary['mean_p2x']:.4f}"
    )
    lines.append(
        f"Mean P(|move| > 3x spread):      {summary['mean_p3x']:.4f}"
    )
    lines.append(
        f"Max spread multiple:             {summary['max_spread_multiple']:.4f}"
    )
    lines.append(
        f"Max AER:                         {summary['max_aer']:.4f}"
    )
    lines.append(
        f"Distinct event buckets:           {summary['n_buckets']}"
    )
    lines.append("")

    # --- Event bucket breakdown with explicit pass/fail ---
    if "event_bucket" in results.columns:
        lines.append("Event Bucket Breakdown:")
        lines.append("-" * 64)
        try:
            bucket_stats = results.group_by("event_bucket").agg([
                pl.col("aer").mean().alias("mean_aer"),
                pl.col("spread_multiple").mean().alias("mean_sm"),
                pl.col("p2x").mean().alias("mean_p2x"),
                pl.col("p3x").mean().alias("mean_p3x"),
                pl.col("tier").max().alias("max_tier"),
                pl.col("promoted").sum().alias("n_promoted"),
                pl.len().alias("n"),
            ]).sort("mean_sm", descending=True)

            # Define pass/fail thresholds
            header = (
                f"{'Bucket':<16s} {'n':>5s} {'AER':>7s} {'SM':>7s} "
                f"{'P2x':>7s} {'P3x':>7s} {'Prm':>4s} {'Status':>8s}"
            )
            lines.append(header)
            lines.append("-" * 64)

            for row in bucket_stats.iter_rows(named=True):
                bucket = row["event_bucket"]
                bucket_n = row["n"]
                bucket_aer = row["mean_aer"]
                bucket_sm = row["mean_sm"]
                bucket_p2x = row["mean_p2x"]
                bucket_p3x = row["mean_p3x"]
                bucket_max_tier = row["max_tier"]
                bucket_n_promoted = row["n_promoted"]

                # Determine pass/fail status
                if bucket_max_tier >= 1:
                    status = "PASS"
                elif bucket_max_tier >= 2:
                    status = "PASS"
                else:
                    status = "FAIL"

                # Add asterisk for borderline
                if bucket_aer >= 1.5 and status == "FAIL":
                    status = "MARGNL"

                lines.append(
                    f"{bucket:<16s} {bucket_n:>5d} {bucket_aer:>7.3f} "
                    f"{bucket_sm:>7.2f} {bucket_p2x:>7.3f} "
                    f"{bucket_p3x:>7.3f} {bucket_n_promoted:>4d} "
                    f"{status:>8s}"
                )

        except Exception:
            lines.append("  (Could not compute bucket breakdown)")
        lines.append("")

    # --- Verdict ---
    lines.append("Verdict:")
    lines.append("-" * 40)
    if summary["n_tier1"] > 0:
        lines.append(
            f"PASS — {summary['n_tier1']} Tier-1 entries identified. "
            "External event alignment shows strong conditional structure "
            "with high AER and tail exceedance probabilities."
        )
    elif summary["n_tier2"] > 0:
        lines.append(
            f"BORDERLINE — {summary['n_tier2']} Tier-2 entries identified. "
            "Some conditional structure exists but criteria are marginal "
            "and may not generalise out-of-sample."
        )
    elif summary["n_promoted"] > 0:
        lines.append(
            "WEAK — Promoted entries exist but none reach Tier 1 or Tier 2."
        )
    else:
        lines.append(
            "FAIL — No promoted entries. "
            "The event amplitude surface did not identify any "
            "conditional windows with sufficient signal-to-noise."
        )

    lines.append("")
    lines.append("Top 5 entries (by spread_multiple):")
    lines.append("-" * 40)
    try:
        if "spread_multiple" in results.columns:
            sort_col = "spread_multiple"
            top5 = results.sort(sort_col, descending=True).head(5)
            for row in top5.iter_rows(named=True):
                lines.append(
                    f"  key={str(row.get('key', '?')):>40s}  "
                    f"h={row.get('horizon', 0):>4d}s  "
                    f"SM={row.get('spread_multiple', 0):.2f}  "
                    f"AER={row.get('aer', 0):.2f}  "
                    f"tier={row.get('tier', 0)}"
                )
    except Exception:
        pass

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


def _safe_count(
    df: pl.DataFrame, column: str, value: Any
) -> int:
    """Count rows where *column* equals *value*."""
    if column not in df.columns or df.is_empty():
        return 0
    try:
        return df.filter(pl.col(column) == value).height
    except Exception:
        return 0


def _safe_mean(df: pl.DataFrame, column: str) -> float:
    """Return the mean of a column, or 0.0 if unavailable."""
    if column not in df.columns or df.is_empty():
        return 0.0
    try:
        val = df[column].mean()
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def _safe_max(df: pl.DataFrame, column: str) -> float:
    """Return the max of a column, or 0.0 if unavailable."""
    if column not in df.columns or df.is_empty():
        return 0.0
    try:
        val = df[column].max()
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def _print_report_index(paths: dict[str, str]) -> None:
    """Print the generated report index."""
    print("[EventReports] Generated reports:")
    for name, path in paths.items():
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"  {name:<25s} -> {path}  ({size:,} bytes)")
