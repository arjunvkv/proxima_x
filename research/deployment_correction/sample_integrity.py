"""
PROBLEM 5: SAMPLE INTEGRITY
============================
Hard statistical sample gates. Suppress conclusive classifications
until minimum sample sizes are achieved.

Phases:
  trades < 30:   EARLY_VALIDATION
  trades < 100:  COLLECTING_EVIDENCE
  trades < 300:  INTERMEDIATE_VALIDATION
  trades >= 300: FULL_VALIDATION

Suppress: LIVE_DEPLOYABLE, ALPHA_DECAYING, RESEARCH_ARTIFACT
Replace with: INSUFFICIENT_EVIDENCE
"""

import os
import json
import sys
import math
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

CLASSIFICATIONS_TO_SUPPRESS = [
    "LIVE_DEPLOYABLE",
    "ALPHA_DECAYING",
    "ALPHA_CONFIRMED",
    "RESEARCH_ARTIFACT",
    "PRODUCTION_READY",
]

SUPPRESSION_REPLACEMENT = "INSUFFICIENT_EVIDENCE"


def _load_trades():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from proxima_ops.ledger.trade_ledger import TradeLedger
        tl = TradeLedger()
        tl._ensure_db()
        r = tl._conn.execute("SELECT * FROM trades ORDER BY trade_id ASC").fetchall()
        return [dict(zip([desc[0] for desc in tl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Warning: could not load trades: {e}")
        return []


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def confidence_interval(n_success, n_total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_success / n_total
    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denominator
    return p, centre - margin, centre + margin


def main():
    print("=" * 62)
    print("PROBLEM 5: SAMPLE INTEGRITY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    trades = _load_trades()
    funnel = _load_funnel_stats()
    
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
    all_trades = len(trades)
    n_closed = len(closed_trades)
    
    # Funnel counts
    if isinstance(funnel, dict) and "counts" in funnel:
        funnel_counts = funnel["counts"]
        funnel_total_signals = funnel_counts.get("GENERATED", 0)
        funnel_executed = funnel_counts.get("POSITION_OPENED", 0)
    else:
        funnel_counts = {}
        funnel_total_signals = 0
        funnel_executed = 0
    
    # Determine phase
    if all_trades < 30:
        phase = "EARLY_VALIDATION"
        phase_num = 1
    elif all_trades < 100:
        phase = "COLLECTING_EVIDENCE"
        phase_num = 2
    elif all_trades < 300:
        phase = "INTERMEDIATE_VALIDATION"
        phase_num = 3
    else:
        phase = "FULL_VALIDATION"
        phase_num = 4
    
    # Check if any suppressed classifications exist
    active_classifications = []
    suppressed = []
    
    # Look for classification files in the project
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))):
        for f in files:
            if f.endswith(".md") or f.endswith(".json"):
                path = os.path.join(root, f)
                try:
                    with open(path) as fh:
                        content = fh.read()
                        for cls in CLASSIFICATIONS_TO_SUPPRESS:
                            if cls in content:
                                suppressed.append((cls, path))
                                active_classifications.append(cls)
                except:
                    pass
                if len(suppressed) > 20:
                    break
        if len(suppressed) > 20:
            break
    
    # Compute win rate CI
    if n_closed > 0:
        wins = sum(1 for t in closed_trades if (t.get("profit_points", 0) or 0) > 0)
        p_hat, ci_low, ci_high = confidence_interval(wins, n_closed)
    else:
        wins, p_hat, ci_low, ci_high = 0, 0.0, 0.0, 0.0
    
    print(f"  Total trade records: {all_trades}")
    print(f"  Open trades: {len(open_trades)}")
    print(f"  Closed trades: {n_closed}")
    print(f"  Funnel signals: {funnel_total_signals}")
    print()
    print(f"  Phase: {phase} (Phase {phase_num}/4)")
    print(f"  Win rate: {wins}/{n_closed} = {100*p_hat:.1f}%")
    print(f"  95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print(f"  CI width: {100*(ci_high-ci_low):.1f}%")
    print()
    print(f"  Active suppressed classifications: {len(suppressed)}")
    for cls, path in suppressed[:10]:
        print(f"    {cls} -> {SUPPRESSION_REPLACEMENT} (in {os.path.relpath(path)})")
    print()
    
    # Minimum sample to achieve each target precision
    target_widths = {0.05: 5, 0.10: 10, 0.15: 15, 0.20: 20}
    min_samples = {}
    for width_pct, width_abs in target_widths.items():
        width = width_abs / 100
        z = 1.96
        # n = (z^2 * p * (1-p)) / (width/2)^2
        # worst case: p = 0.5
        n_needed = math.ceil((z**2 * 0.25) / (width / 2)**2)
        min_samples[width_pct] = n_needed
    
    print(f"  Minimum trades needed for various precision levels:")
    for width, n_needed in sorted(min_samples.items()):
        status = "MET" if all_trades >= n_needed else f"NEED {n_needed - all_trades} MORE"
        print(f"    +/-{width*100:.0f}% precision: {n_needed} trades ({status})")
    print()
    
    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "SAMPLE_INTEGRITY_REPORT.md")
    
    lines = []
    lines.append("# SAMPLE INTEGRITY REPORT — Deployment Correction Problem 5")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Validation Phase")
    lines.append("")
    lines.append(f"**Phase {phase_num}/4: {phase}**")
    lines.append("")
    lines.append(f"- Total trade records: {all_trades}")
    lines.append(f"- Open positions: {len(open_trades)}")
    lines.append(f"- Closed trades: {n_closed}")
    lines.append(f"- Funnel signals generated: {funnel_total_signals}")
    lines.append("")
    
    lines.append("## Phase Definitions")
    lines.append("")
    lines.append("| Phase | Trades | Allowed Conclusions |")
    lines.append("|-------|--------|--------------------|")
    lines.append(f"| 1. EARLY_VALIDATION | < 30 | No conclusions. Logging only. |")
    lines.append(f"| 2. COLLECTING_EVIDENCE | 30-99 | Directional observations. No statistical claims. |")
    lines.append(f"| 3. INTERMEDIATE_VALIDATION | 100-299 | Preliminary metrics with large confidence intervals. |")
    lines.append(f"| 4. FULL_VALIDATION | 300+ | Confident metrics. Valid comparisons possible. |")
    lines.append("")
    
    lines.append("## Suppressed Classifications")
    lines.append("")
    lines.append(f"The following classifications are **SUPPRESSED** until Phase 4 (300+ trades):")
    lines.append("")
    for cls in CLASSIFICATIONS_TO_SUPPRESS:
        lines.append(f"- `{cls}` -> `{SUPPRESSION_REPLACEMENT}`")
    lines.append("")
    lines.append(f"{len(suppressed)} instances found across project files will need to be replaced.")
    lines.append("")
    
    lines.append("## Win Rate Confidence")
    lines.append("")
    lines.append(f"**Current win rate:** {wins}/{n_closed} = {100*p_hat:.1f}%")
    lines.append(f"**95% CI:** [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    lines.append(f"**CI width:** {100*(ci_high-ci_low):.1f}%")
    lines.append("")
    lines.append("A +/-5% confidence interval requires ~385 trades (worst case p=0.5).")
    lines.append("At current trade rates, this will take additional days/weeks of trading.")
    lines.append("")
    
    lines.append("## Recommendations")
    lines.append("")
    lines.append(f"1. Do NOT make live trading decisions based on {phase} data.")
    lines.append(f"2. All prior classification files containing `LIVE_DEPLOYABLE`, `ALPHA_DECAYING`, etc. should be tagged as `INSUFFICIENT_EVIDENCE`.")
    lines.append(f"3. Continue collecting data. Target: 300+ closed trades before any conclusive analysis.")
    lines.append(f"4. Focus on infrastructure stability and deployment correction rather than alpha evaluation.")
    lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

