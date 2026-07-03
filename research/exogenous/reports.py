"""
Program VI — Exogenous Amplitude Discovery Reports
====================================================
Export promotion-evaluation results to human-readable reports and
machine-readable summaries.

Typical usage::

    from research.exogenous.reports import export
    paths = export(results_df, output_dir="reports")
    print(paths)
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import json
import os
from typing import Any

import polars as pl


def export(
    results: pl.DataFrame, output_dir: str = "reports"
) -> dict[str, str]:
    """Generate report files from exogenous promotion evaluation results.

    Produces six files:

    1. ``exogenous_surface_summary.csv``  — all entries sorted by spread_multiple desc
    2. ``exogenous_promotions.csv``       — only promoted entries (tier > 0)
    3. ``exogenous_session_summary.csv``   — grouped by session with aggregate stats
    4. ``exogenous_best_windows.csv``      — entries with spread_multiple > 3.0
    5. ``exogenous_summary.json``          — machine-readable summary dictionary
    6. ``exogenous_executive_summary.txt`` — human-readable verdict

    Parameters
    ----------
    results : pl.DataFrame
        Evaluation results produced by :class:`ExogenousPromotionEngine`
        enriched with surface metadata (at minimum columns
        ``exogenous_key``, ``horizon``, ``session``, ``aer``,
        ``spread_multiple``, ``n``, ``tier``, ``promoted``, ``mean_abs``,
        ``median_abs``, ``std_abs``, ``p90_abs``).
    output_dir : str
        Directory where report files are written (created if absent).

    Returns
    -------
    dict[str, str]
        Mapping from report name to absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Surface summary — all entries sorted by spread_multiple desc
    # ------------------------------------------------------------------
    surface_csv = os.path.join(output_dir, "exogenous_surface_summary.csv")
    sort_col = "spread_multiple"
    if sort_col in results.columns and not results.is_empty():
        sorted_df = results.sort(sort_col, descending=True)
        sorted_df.write_csv(surface_csv)
    else:
        results.write_csv(surface_csv)
    paths["surface_summary"] = os.path.abspath(surface_csv)

    # ------------------------------------------------------------------
    # 2. Promotions — only tier > 0
    # ------------------------------------------------------------------
    prom_csv = os.path.join(output_dir, "exogenous_promotions.csv")
    if "tier" in results.columns and not results.is_empty():
        prom_df = results.filter(pl.col("tier") > 0)
        if not prom_df.is_empty():
            prom_df.write_csv(prom_csv)
        else:
            # Write header-only CSV
            results.head(0).write_csv(prom_csv)
    else:
        results.head(0).write_csv(prom_csv)
    paths["promotions"] = os.path.abspath(prom_csv)

    # ------------------------------------------------------------------
    # 3. Session summary — grouped by session
    # ------------------------------------------------------------------
    session_csv = os.path.join(output_dir, "exogenous_session_summary.csv")
    if "session" in results.columns and not results.is_empty():
        session_agg = results.group_by("session").agg([
            pl.col("aer").mean().alias("mean_aer"),
            pl.col("spread_multiple").mean().alias("mean_sm"),
            pl.count().alias("n"),
        ]).sort("mean_sm", descending=True)
        session_agg.write_csv(session_csv)
    else:
        # Write empty result with expected columns
        pl.DataFrame({
            "session": pl.Series([], dtype=pl.Utf8),
            "mean_aer": pl.Series([], dtype=pl.Float64),
            "mean_sm": pl.Series([], dtype=pl.Float64),
            "n": pl.Series([], dtype=pl.Int64),
        }).write_csv(session_csv)
    paths["session_summary"] = os.path.abspath(session_csv)

    # ------------------------------------------------------------------
    # 4. Best windows — spread_multiple > 3.0
    # ------------------------------------------------------------------
    best_csv = os.path.join(output_dir, "exogenous_best_windows.csv")
    if "spread_multiple" in results.columns and not results.is_empty():
        best_df = results.filter(pl.col("spread_multiple") > 3.0)
        if not best_df.is_empty():
            best_df = best_df.sort("spread_multiple", descending=True)
            best_df.write_csv(best_csv)
        else:
            results.head(0).write_csv(best_csv)
    else:
        results.head(0).write_csv(best_csv)
    paths["best_windows"] = os.path.abspath(best_csv)

    # ------------------------------------------------------------------
    # 5. Machine-readable JSON summary
    # ------------------------------------------------------------------
    summary_json_path = os.path.join(output_dir, "exogenous_summary.json")
    summary: dict[str, Any] = _build_machine_summary(results)
    with open(summary_json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    paths["summary_json"] = os.path.abspath(summary_json_path)

    # ------------------------------------------------------------------
    # 6. Human-readable executive summary
    # ------------------------------------------------------------------
    exec_txt_path = os.path.join(output_dir, "exogenous_executive_summary.txt")
    executive_text = _build_executive_summary(results, summary)
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
        "max_spread_multiple": _safe_max(results, "spread_multiple"),
        "n_sessions": (
            results["session"].n_unique()
            if "session" in results.columns and total > 0
            else 0
        ),
    }


def _build_executive_summary(
    results: pl.DataFrame, summary: dict[str, Any]
) -> str:
    """Build a human-readable executive summary string."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("PROGRAM VI — EXOGENOUS AMPLITUDE DISCOVERY")
    lines.append("Executive Summary")
    lines.append("=" * 64)
    lines.append("")

    if results.is_empty():
        lines.append("No entries were evaluated — the results DataFrame is empty.")
        lines.append("")
        lines.append("=" * 64)
        return "\n".join(lines)

    lines.append(f"Total surface entries evaluated:  {summary['total_entries']}")
    lines.append(f"Promoted entries:                {summary['n_promoted']}")
    lines.append(f"  Tier 1:                        {summary['n_tier1']}")
    lines.append(f"  Tier 2:                        {summary['n_tier2']}")
    lines.append(f"Promotion rate:                  {summary['promotion_rate']:.2%}")
    lines.append("")
    lines.append(f"Mean AER:                       {summary['mean_aer']:.4f}")
    lines.append(f"Mean spread multiple:            {summary['mean_spread_multiple']:.4f}")
    lines.append(f"Max spread multiple:             {summary['max_spread_multiple']:.4f}")
    lines.append(f"Distinct sessions:               {summary['n_sessions']}")
    lines.append("")

    # --- Session breakdown ---
    if "session" in results.columns:
        lines.append("Session Breakdown:")
        lines.append("-" * 40)
        try:
            session_stats = results.group_by("session").agg([
                pl.col("aer").mean().alias("mean_aer"),
                pl.col("spread_multiple").mean().alias("mean_sm"),
                pl.col("tier").max().alias("max_tier"),
                pl.count().alias("n"),
            ]).sort("n", descending=True)
            for row in session_stats.iter_rows(named=True):
                lines.append(
                    f"  {row['session']:<12s}  n={row['n']:>4d}  "
                    f"AER={row['mean_aer']:.3f}  "
                    f"SM={row['mean_sm']:.3f}  "
                    f"max_tier={row['max_tier']}"
                )
        except Exception:
            pass
        lines.append("")

    # --- Verdict ---
    lines.append("Verdict:")
    lines.append("-" * 40)
    if summary["n_tier1"] > 0:
        lines.append(
            f"PASS — {summary['n_tier1']} Tier-1 entries identified. "
            "Exogenous amplitude surface shows strong conditional structure."
        )
    elif summary["n_tier2"] > 0:
        lines.append(
            f"BORDERLINE — {summary['n_tier2']} Tier-2 entries identified. "
            "Some conditional structure exists but criteria are marginal."
        )
    elif summary["n_promoted"] > 0:
        lines.append(
            "WEAK — Promoted entries exist but none reach Tier 1 or Tier 2."
        )
    else:
        lines.append(
            "FAIL — No promoted entries. "
            "The exogenous amplitude surface did not identify any "
            "conditional windows with sufficient signal."
        )

    lines.append("")
    lines.append(f"Top 3 entries (by spread_multiple):")
    lines.append("-" * 40)
    try:
        if "spread_multiple" in results.columns:
            top3 = results.sort("spread_multiple", descending=True).head(3)
            for row in top3.iter_rows(named=True):
                lines.append(
                    f"  key={row.get('exogenous_key', '?'):>40s}  "
                    f"h={row.get('horizon', 0):>4d}s  "
                    f"SM={row.get('spread_multiple', 0):.2f}  "
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
    print("[ExogenousReports] Generated reports:")
    for name, path in paths.items():
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"  {name:<25s} -> {path}  ({size:,} bytes)")
