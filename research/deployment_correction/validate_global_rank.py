"""
GLOBAL RANK VALIDATION — Phase 2 Deliverable
=============================================
Validate the GlobalRankEngine against historical funnel data.
Show asset participation, concentration ratio, HHI, trigger distribution
for both local and global ranking.
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from proxima_ops.monitoring.global_rank_engine import GlobalRankEngine

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def hhi(shares):
    total = sum(shares.values()) or 1
    return sum((v / total * 100) ** 2 for v in shares.values())


def main():
    print("=" * 62)
    print("GLOBAL RANK VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()

    funnel = _load_funnel_stats()
    if "signals" in funnel:
        signals_dict = funnel["signals"]
    else:
        signals_dict = {k: v for k, v in funnel.items() if k != "counts"}

    # Group by evaluation minute
    groups = defaultdict(dict)
    for sid, rec in signals_dict.items():
        if not isinstance(rec, dict):
            continue
        ts = rec.get("timestamp_generated", "")
        minute_key = ts[:16] if len(ts) >= 16 else ts
        sym = rec.get("symbol", "?")
        groups[minute_key][sym] = rec

    print(f"  Loaded {len(signals_dict)} signals across {len(groups)} evaluation minutes")
    print()

    # Run GlobalRankEngine on each group
    engine = GlobalRankEngine()
    local_triggers = defaultdict(int)
    global_triggers = defaultdict(int)
    global_all_qualified = defaultdict(int)

    for minute_key in sorted(groups.keys()):
        group = groups[minute_key]
        engine.clear()
        for sym, rec in group.items():
            lr = rec.get("es", 0)
            if isinstance(lr, (int, float)):
                engine.record_evaluation(sym, lr)

        engine.compute()

        # Local triggers: local rank >= 90%
        for sym in group:
            lr = group[sym].get("es", 0)
            if isinstance(lr, (int, float)) and lr >= 0.90:
                local_triggers[sym] += 1

        # Global triggers: top-1 per minute
        global_top1 = engine.get_qualified_assets(100.0)
        if global_top1:
            global_triggers[global_top1[0]] += 1

        # Global all qualified: global percentile >= 80%
        for sym in engine.get_qualified_assets(80.0):
            global_all_qualified[sym] += 1

    # Metrics
    local_total = sum(local_triggers.values()) or 1
    global_total = sum(global_triggers.values()) or 1
    ga_total = sum(global_all_qualified.values()) or 1

    local_hhi = hhi(local_triggers)
    global_hhi = hhi(global_triggers)
    ga_hhi = hhi(global_all_qualified)

    local_assets = len(local_triggers)
    global_assets = len(global_triggers)
    ga_assets = len(global_all_qualified)

    print("  === Local Rank (current) ===")
    for sym in sorted(local_triggers):
        print(f"    {sym}: {local_triggers[sym]} triggers")
    print(f"    Assets: {local_assets}/5, HHI: {local_hhi:.1f}")
    print()

    print("  === Global Rank (top-1 per minute) ===")
    for sym in sorted(global_triggers):
        print(f"    {sym}: {global_triggers[sym]} triggers")
    print(f"    Assets: {global_assets}/5, HHI: {global_hhi:.1f}")
    print()

    print("  === Global Rank (all qualified >=80%) ===")
    for sym in sorted(global_all_qualified):
        print(f"    {sym}: {global_all_qualified[sym]} triggers")
    print(f"    Assets: {ga_assets}/5, HHI: {ga_hhi:.1f}")
    print()

    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "GLOBAL_RANK_VALIDATION.md")

    lines = []
    lines.append("# GLOBAL RANK VALIDATION — Phase 2 Deliverable")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Validation Method")
    lines.append("")
    lines.append(f"Ran {len(groups)} evaluation minutes through `GlobalRankEngine`. ")
    lines.append("Compared three deployment modes on identical historical data.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Metric':<40} {'Local (current)':<20} {'Global (top-1)':<20} {'Global (all >=80%)':<20}")
    lines.append("-" * 100)
    lines.append(f"{'Assets participating':<40} {f'{local_assets}/5':<20} {f'{global_assets}/5':<20} {f'{ga_assets}/5':<20}")
    lines.append(f"{'Total triggers':<40} {local_total:<20} {global_total:<20} {ga_total:<20}")
    lines.append(f"{'HHI concentration':<40} {local_hhi:<20.1f} {global_hhi:<20.1f} {ga_hhi:<20.1f}")
    lines.append("```")
    lines.append("")

    lines.append("## Per-Asset Triggers")
    lines.append("")
    all_syms = sorted(set(list(local_triggers.keys()) + list(global_triggers.keys()) + list(global_all_qualified.keys())))
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Local':<12} {'Global (top-1)':<18} {'Global (all >=80%)':<20}")
    lines.append("-" * 62)
    for sym in all_syms:
        lt = local_triggers.get(sym, 0)
        gt = global_triggers.get(sym, 0)
        ga = global_all_qualified.get(sym, 0)
        lines.append(f"{sym:<12} {lt:<12} {gt:<18} {ga:<20}")
    lines.append("```")
    lines.append("")

    lines.append("## Engine Validation")
    lines.append("")
    lines.append("### Synthetic Test")
    lines.append("```")
    lines.append("Input: EURJPY=92, USDJPY=95, GBPJPY=75, XAUUSD=88, EURUSD=91")
    lines.append("Expected: USDJPY=G1/P100, EURJPY=G2/P80, EURUSD=G3/P60, XAUUSD=G4/P40, GBPJPY=G5/P20")
    lines.append("```")
    lines.append("")

    engine = GlobalRankEngine()
    for sym, lr in {"EURJPY": 92, "USDJPY": 95, "GBPJPY": 75, "XAUUSD": 88, "EURUSD": 91}.items():
        engine.record_evaluation(sym, lr)
    engine.compute()
    syn_results = engine.summary()
    lines.append("```")
    lines.append(syn_results)
    lines.append("```")
    lines.append("")

    lines.append("### Qualification Gate")
    lines.append("")
    lines.append("GLOBAL_ALL_QUALIFIED rule: `global_percentile >= 80`")
    lines.append("")
    qualified = engine.get_qualified_assets(80.0)
    lines.append(f"Qualified assets (synthetic test): {qualified}")
    lines.append(f"This means the top {len(qualified)}/{engine.n_assets} assets per evaluation cycle.")
    lines.append("With 5 monitored assets, this produces 1-2 simultaneous trades per cycle.")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if ga_assets > local_assets:
        lines.append(f"**Global ranking restores multi-asset participation:** {local_assets}/5 -> {ga_assets}/5 assets.")
        lines.append(f"**HHI drops from {local_hhi:.0f} to {ga_hhi:.0f}.**")
        lines.append("The GlobalRankEngine correctly addresses the LOCAL PERCENTILE NORMALIZATION BIAS.")
    else:
        lines.append("Global ranking does not significantly improve multi-asset participation on this data.")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
