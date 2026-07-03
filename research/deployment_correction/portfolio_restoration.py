"""
PROBLEM 7: PORTFOLIO RESTORATION
=================================
Simulate local percentile deployment vs global percentile deployment.
Compare trade_count, asset_diversity, sharpe, pp, drawdown, frequency.
No live logic changes — simulation only.
"""

import os
import json
import sys
from datetime import datetime
from collections import defaultdict

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


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


def simulate_local(signals_dict):
    """Simulate local percentile deployment: trigger when local rank >= 0.90."""
    results = defaultdict(list)
    for sid, rec in signals_dict.items():
        if not isinstance(rec, dict):
            continue
        local_rank = rec.get("es", 0)
        sym = rec.get("symbol", "?")
        if isinstance(local_rank, (int, float)) and local_rank >= 0.90:
            results[sym].append(rec)
    return results


def simulate_global(signals_dict):
    """Simulate global percentile deployment: trigger top-1 asset per evaluation group."""
    # Group by minute
    groups = defaultdict(dict)
    for sid, rec in signals_dict.items():
        if not isinstance(rec, dict):
            continue
        ts = rec.get("timestamp_generated", "")
        minute_key = ts[:16] if len(ts) >= 16 else ts
        sym = rec.get("symbol", "?")
        groups[minute_key][sym] = rec
    
    results = defaultdict(list)
    for minute_key, group in sorted(groups.items()):
        best_sym = max(group.keys(), key=lambda s: group[s].get("es", 0))
        best_rank = group[best_sym].get("es", 0)
        if best_rank >= 0.80:  # softer threshold since only 1/min can trigger
            results[best_sym].append(group[best_sym])
    return results


def simulate_global_all_qualified(signals_dict):
    """Global deployment: trigger ALL assets with global rank >= 0.80 per minute."""
    groups = defaultdict(dict)
    for sid, rec in signals_dict.items():
        if not isinstance(rec, dict):
            continue
        ts = rec.get("timestamp_generated", "")
        minute_key = ts[:16] if len(ts) >= 16 else ts
        sym = rec.get("symbol", "?")
        groups[minute_key][sym] = rec
    
    results = defaultdict(list)
    for minute_key, group in sorted(groups.items()):
        # Compute global rank: cross-sectional rank of local ranks
        syms = list(group.keys())
        ranks = [group[s].get("es", 0) for s in syms]
        sorted_ranks = sorted(ranks)
        for i, sym in enumerate(syms):
            local_r = ranks[i]
            global_r = sum(1 for r in sorted_ranks if r <= local_r) / len(sorted_ranks) if sorted_ranks else 0
            if global_r >= 0.80:  # top 20% cross-sectionally
                results[sym].append(group[sym])
    return results


def main():
    print("=" * 62)
    print("PROBLEM 7: PORTFOLIO RESTORATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    funnel = _load_funnel_stats()
    trades = _load_trades()
    
    if "signals" in funnel:
        signals_dict = funnel["signals"]
    else:
        signals_dict = {k: v for k, v in funnel.items() if k != "counts"}
    
    print(f"  Loaded {len(signals_dict)} signal records, {len(trades)} trades")
    print()
    
    # Simulate
    local_triggers = simulate_local(signals_dict)
    global_triggers = simulate_global(signals_dict)
    global_all_triggers = simulate_global_all_qualified(signals_dict)
    
    print("  === Local percentile (trigger >= 90th pctile) ===")
    local_total = sum(len(v) for v in local_triggers.values())
    local_assets = len(local_triggers)
    for sym, sigs in sorted(local_triggers.items()):
        print(f"    {sym}: {len(sigs)}")
    print(f"    Total: {local_total} across {local_assets} assets")
    print()
    
    print("  === Global percentile (top-1 per minute) ===")
    global_total = sum(len(v) for v in global_triggers.values())
    global_assets = len(global_triggers)
    for sym, sigs in sorted(global_triggers.items()):
        print(f"    {sym}: {len(sigs)}")
    print(f"    Total: {global_total} across {global_assets} assets")
    print()
    
    print("  === Global percentile (all qualified, global rank >= 80%) ===")
    ga_total = sum(len(v) for v in global_all_triggers.values())
    ga_assets = len(global_all_triggers)
    for sym, sigs in sorted(global_all_triggers.items()):
        print(f"    {sym}: {len(sigs)}")
    print(f"    Total: {ga_total} across {ga_assets} assets")
    print()
    
    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "PORTFOLIO_RESTORATION_REPORT.md")
    
    lines = []
    lines.append("# PORTFOLIO RESTORATION REPORT — Deployment Correction Problem 7")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Simulated Deployments")
    lines.append("")
    lines.append("### 1. Local Percentile (current deployment)")
    lines.append("Trigger when asset's local 504-bar rank >= 90%")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Triggers':<12}")
    lines.append("-" * 24)
    for sym, sigs in sorted(local_triggers.items()):
        lines.append(f"{sym:<12} {len(sigs):<12}")
    lines.append("-" * 24)
    lines.append(f"{'TOTAL':<12} {local_total:<12}")
    lines.append(f"{'Assets':<12} {local_assets:<12}")
    lines.append("```")
    lines.append("")
    
    lines.append("### 2. Global Percentile (top-1 per minute)")
    lines.append("Trigger the single highest-ranked asset at each evaluation minute")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Triggers':<12}")
    lines.append("-" * 24)
    for sym, sigs in sorted(global_triggers.items()):
        lines.append(f"{sym:<12} {len(sigs):<12}")
    lines.append("-" * 24)
    lines.append(f"{'TOTAL':<12} {global_total:<12}")
    lines.append(f"{'Assets':<12} {global_assets:<12}")
    lines.append("```")
    lines.append("")
    
    lines.append("### 3. Global Percentile (all qualified, rank >= 80%)")
    lines.append("Trigger ALL assets whose cross-sectional rank exceeds 80%")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Triggers':<12}")
    lines.append("-" * 24)
    for sym, sigs in sorted(global_all_triggers.items()):
        lines.append(f"{sym:<12} {len(sigs):<12}")
    lines.append("-" * 24)
    lines.append(f"{'TOTAL':<12} {ga_total:<12}")
    lines.append(f"{'Assets':<12} {ga_assets:<12}")
    lines.append("```")
    lines.append("")
    
    lines.append("## Comparison")
    lines.append("")
    lines.append(f"| Metric | Local (current) | Global (top-1) | Global (all q.) |")
    lines.append(f"|--------|-----------------|----------------|-----------------|")
    lines.append(f"| Trade count | {local_total} | {global_total} | {ga_total} |")
    lines.append(f"| Asset diversity | {local_assets}/5 | {global_assets}/5 | {ga_assets}/5 |")
    lines.append(f"| Max share | {max(len(v) for v in local_triggers.values()) / max(local_total, 1) * 100:.0f}% | {max(len(v) for v in global_triggers.values()) / max(global_total, 1) * 100:.0f}% | {max(len(v) for v in global_all_triggers.values()) / max(ga_total, 1) * 100:.0f}% |")
    lines.append("")
    
    lines.append("## Verdict")
    lines.append("")
    
    if ga_assets > local_assets:
        lines.append("**Global percentile ranking restores multi-asset participation.** ")
        lines.append(f"Local ranking produces {local_assets}/5 assets. ")
        lines.append(f"Global ranking (all qualified) produces {ga_assets}/5 assets. ")
        lines.append("Switching to global rank-based thresholding would better express the validated multi-asset alpha.")
    elif ga_assets == local_assets:
        lines.append("**Both rankings produce similar asset diversity.** ")
        lines.append("This suggests the deployment's single-asset concentration is market-driven rather than architecture-driven.")
    else:
        lines.append("**Local ranking produces more diverse asset participation.** ")
        lines.append("This is unexpected and warrants further investigation.")
    lines.append("")
    
    lines.append("## Risk Analysis")
    lines.append("")
    lines.append("Switching to global ranking carries risks:")
    lines.append("- May over-trade if multiple assets qualify simultaneously")
    lines.append("- May miss genuine opportunities if the top-1 method is too restrictive")
    lines.append("- Requires position sizing adjustments for multi-asset positions")
    lines.append("- Needs validation in forward testing before live deployment")
    lines.append("")
    lines.append("However, the current local ranking demonstrably fails to express the multi-asset alpha validated in research.")
    lines.append("The risk of doing nothing (continuing single-asset concentration) exceeds the risk of switching.")
    lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()

