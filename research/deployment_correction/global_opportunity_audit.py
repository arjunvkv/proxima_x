"""
PROBLEM 1: GLOBAL OPPORTUNITY AUDIT
=====================================
Re-analyse signal records computing both local and global percentile ranks.
Compare trigger rates to determine whether local normalization creates
artificial EURUSD dominance.

Requirement: for every evaluation record compute:
  timestamp, symbol, raw_es, local_rank, global_rank, at_rank,
  trigger_local, trigger_global
"""

import os
import sys
import json
import math
from datetime import datetime
from collections import defaultdict


REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        print(f"  funnel_stats.json not found at {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        # funnel_stats.json has {"signals": {"SIG_...": {...}, ...}, "counts": {...}}
        if "signals" in raw:
            signals = []
            for sid, record in raw["signals"].items():
                record["signal_id"] = sid
                signals.append(record)
            return signals
        # Legacy format: flat dict of signal_id -> record
        signals = []
        for sid, record in raw.items():
            if sid == "counts":
                continue
            record["signal_id"] = sid
            signals.append(record)
        return signals
    return raw


def _load_observability_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "observability_stats.json")
    if not os.path.exists(path):
        print(f"  observability_stats.json not found at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _load_live_trades():
    """Try to load live trades from DuckDB trade ledger."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from proxima_ops.ledger.trade_ledger import TradeLedger
        tl = TradeLedger()
        tl._ensure_db()
        r = tl._conn.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY trade_id ASC").fetchall()
        return [dict(zip([desc[0] for desc in tl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Warning: could not load trade ledger: {e}")
        return []


def _load_signal_ledger():
    """Try to load signal records from DuckDB."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from proxima_ops.ledger.signal_ledger import SignalLedger
        sl = SignalLedger()
        sl._ensure_db()
        r = sl._conn.execute("SELECT * FROM signals ORDER BY signal_id DESC").fetchall()
        return [dict(zip([desc[0] for desc in sl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Warning: could not load signal ledger: {e}")
        return []


def compute_global_rank(current_es, all_es_values):
    """ECDF-based rank across ALL assets (cross-asset)."""
    if not all_es_values:
        return 0.0
    return float(sum(all_es_values <= current_es)) / len(all_es_values)


def analyze_funnel_signals(funnel_signals, stats):
    """Analyze funnel signals for local vs global rank comparison.
    
    The funnel_stats 'es' field stores the LOCAL percentile rank.
    For global rank, we need raw ES, which IS NOT persisted.
    We reconstruct it by comparing local ranks across assets
    at the same timestamp group.
    """
    # Group signals by timestamp (minute resolution)
    eval_groups = defaultdict(dict)
    for s in funnel_signals:
        ts = s.get("timestamp_generated", "")
        # Round to minute
        minute_key = ts[:16] if len(ts) >= 16 else ts
        sym = s.get("symbol", "?")
        eval_groups[minute_key][sym] = s

    print(f"  Loaded {len(funnel_signals)} funnel signal records in {len(eval_groups)} evaluation groups")

    # For each group, compute what global rank would be
    global_trigger_count = defaultdict(int)
    local_trigger_count = defaultdict(int)
    group_results = []

    for minute_key, group in sorted(eval_groups.items()):
        # Gather local ranks per symbol
        local_ranks = {}
        for sym, s in group.items():
            es_val = s.get("es", 0)
            if isinstance(es_val, (int, float)):
                local_ranks[sym] = es_val

        if not local_ranks:
            continue

        # Simulate global rank: the es field stores LOCAL percentile (0-1).
        # For a true cross-asset rank, we'd need raw ES, which we don't have.
        # Instead, we use the local rank as a proxy for "how extreme is this asset
        # relative to its own history" and then rank those extremes cross-sectionally.
        #
        # Method: the local rank IS comparable in one sense: it measures how
        # extreme the asset is relative to itself. We rank the local ranks to
        # get a "relative extremeness" rank.
        all_ranks = list(local_ranks.values())
        sorted_ranks = sorted(all_ranks)

        for sym, lr in local_ranks.items():
            # Global rank = rank of this asset's local rank among all assets' local ranks
            gr = float(sum(1 for r in sorted_ranks if r <= lr)) / len(sorted_ranks) if sorted_ranks else 0.0

            # If using local rank: trigger if local >= 0.90
            trigger_local = lr >= 0.90

            # If using global cross-asset rank: the threshold concept changes.
            # With 5 assets, global rank of 0.80 means this asset is in the top 20%.
            # For a single-asset trade, the top asset at each minute would trigger.
            trigger_global_max = gr >= 0.80  # top 20% cross-sectionally
            trigger_global_top1 = gr >= 0.99  # only the single highest-ranked asset

            if trigger_local:
                local_trigger_count[sym] += 1

            # For global: top-1 per minute
            top_sym = max(local_ranks, key=local_ranks.get)
            if trigger_global_top1:
                global_trigger_count[sym] += 1

            group_results.append({
                "minute": minute_key,
                "symbol": sym,
                "local_rank": lr,
                "global_rank": gr,
                "trigger_local": trigger_local,
                "trigger_global_max": trigger_global_max,
                "trigger_global_top1": trigger_global_top1,
                "is_top_asset": sym == top_sym,
            })

    return group_results, local_trigger_count, global_trigger_count, len(eval_groups)


def generate_report(group_results, local_triggers, global_triggers, stats, trades, signals, n_groups):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "GLOBAL_RANK_REPORT.md")

    # Stats from observability
    total_eval = stats.get("evaluated_count", 0)
    total_trig = stats.get("trigger_count", 0)
    total_exec = stats.get("executed_count", 0)
    symbol_stats = stats.get("symbol_stats", {})

    lines = []
    lines.append("# GLOBAL RANK REPORT — Deployment Correction Problem 1")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- **Observability stats:** {total_eval} evaluations, {total_trig} triggers, {total_exec} executions")
    lines.append(f"- **Funnel signal records:** {len(group_results)} evaluation-level records")
    lines.append(f"- **DuckDB signal records:** {len(signals)}")
    lines.append(f"- **DuckDB closed trades:** {len(trades)}")
    lines.append("")

    # Q1: Local triggers by asset
    lines.append("## Q1: How many triggers by asset using LOCAL rank?")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Evaluations':<14} {'Local Triggers':<16} {'Executed':<12} {'Trigger %':<10}")
    lines.append("-" * 64)
    for sym in sorted(symbol_stats.keys()):
        ss = symbol_stats[sym]
        ev = ss.get("evaluated", 0)
        tr = ss.get("triggered", 0)
        ex = ss.get("executed", 0)
        pct = f"{100*tr/ev:.1f}%" if ev > 0 else "N/A"
        lines.append(f"{sym:<12} {ev:<14} {tr:<16} {ex:<12} {pct:<10}")
    lines.append("```")
    lines.append("")

    # Q2: Global triggers by asset (from funnel analysis)
    lines.append("## Q2: How many triggers by asset using GLOBAL rank?")
    lines.append("")
    lines.append("**Method:** Global rank = cross-sectional rank of local percentile ranks at each evaluation minute. `trigger_global_top1` = asset has the highest local rank among all 5 assets at that minute.")
    lines.append("")
    lines.append("```")
    all_syms = sorted(set(r["symbol"] for r in group_results))
    lines.append(f"{'Asset':<12} {'Local Triggers':<18} {'Global Top-1 Triggers':<24} {'Minute Share':<14}")
    lines.append("-" * 68)
    for sym in all_syms:
        lc = local_triggers.get(sym, 0)
        gc = global_triggers.get(sym, 0)
        total_minutes = sum(global_triggers.values())
        share = f"{100*gc/total_minutes:.1f}%" if total_minutes > 0 else "N/A"
        lines.append(f"{sym:<12} {lc:<18} {gc:<24} {share:<14}")
    lines.append("```")
    lines.append("")

    # Q3: Cross-asset participation
    lines.append("## Q3: Does global ranking restore cross-asset participation?")
    lines.append("")

    local_concentration = max(local_triggers.values()) / sum(local_triggers.values()) if sum(local_triggers.values()) > 0 else 0
    global_concentration = max(global_triggers.values()) / sum(global_triggers.values()) if sum(global_triggers.values()) > 0 else 0
    local_diverse = sum(1 for v in local_triggers.values() if v > 0)
    global_diverse = sum(1 for v in global_triggers.values() if v > 0)

    lines.append(f"- Assets that trigger under LOCAL rank: {local_diverse}/5")
    lines.append(f"- Assets that trigger under GLOBAL rank (top-1): {global_diverse}/5")
    lines.append(f"- Local concentration ratio: {local_concentration:.2f} (1.0 = single asset)")
    lines.append(f"- Global concentration ratio: {global_concentration:.2f}")
    lines.append("")

    if global_diverse > local_diverse:
        lines.append("**CONCLUSION: Global ranking restores cross-asset participation.**")
        lines.append(f"Under local ranking, only {local_diverse}/5 assets ever trigger. Under global ranking, {global_diverse}/5 would trigger. This confirms LOCAL PERCENTILE NORMALIZATION BIAS as the root cause of single-asset concentration.")
    elif local_diverse == global_diverse and local_diverse == 1:
        lines.append("**CONCLUSION: Single-asset dominance persists under both rankings.**")
        lines.append("This suggests the dominance is MARKET-DRIVEN rather than normalization-driven. However, the top-1 global trigger method naturally produces single-asset concentration by design. A softer threshold (e.g., global rank >= 0.80 for all qualifying assets) should be tested.")
    else:
        lines.append(f"**CONCLUSION: Cross-asset participation changes from {local_diverse}/5 to {global_diverse}/5 assets.**")

    lines.append("")

    # Q4: AAE comparison
    lines.append("## Q4: Which ranking better matches AAE portfolio findings?")
    lines.append("")
    lines.append("AAE RQ5 portfolio analysis found expected Sharpe per asset:")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'AAE Sharpe':<14} {'PP':<10} {'n_signals':<12}")
    lines.append("-" * 48)
    aae_expected = {
        "EURJPY": {"sharpe": 0.693, "pp": 0.738, "n": 183},
        "USDJPY": {"sharpe": 0.326, "pp": 0.654, "n": 182},
        "GBPJPY": {"sharpe": 0.526, "pp": 0.678, "n": 183},
        "XAUUSD": {"sharpe": 0.177, "pp": 0.489, "n": 176},
        "EURUSD": {"sharpe": 0.0, "pp": 0.0, "n": 0},  # EURUSD not in AAE
    }
    for sym, vals in aae_expected.items():
        if vals["n"] > 0:
            lines.append(f"{sym:<12} {vals['sharpe']:<14.3f} {vals['pp']:<10.3f} {vals['n']:<12}")
    lines.append("```")
    lines.append("")

    aae_assets = {k for k, v in aae_expected.items() if v["n"] > 0}
    local_assets = set(local_triggers.keys())
    global_assets = set(global_triggers.keys())
    aae_local_overlap = len(aae_assets & local_assets)
    aae_global_overlap = len(aae_assets & global_assets)

    lines.append(f"- AAE has {len(aae_assets)} assets with positive alpha expectation")
    lines.append(f"- Local ranking triggers {local_assets} ({aae_local_overlap}/{len(aae_assets)} overlap with AAE)")
    lines.append(f"- Global ranking triggers {global_assets} ({aae_global_overlap}/{len(aae_assets)} overlap with AAE)")
    lines.append("")

    if aae_global_overlap > aae_local_overlap:
        lines.append("**CONCLUSION: Global ranking better aligns with AAE findings.** The AAE validated multi-asset alpha across EURJPY, USDJPY, GBPJPY, and XAUUSD. Global ranking would distribute triggers across these assets, while local ranking concentrates on whichever asset happens to be at an extreme percentile.")
    else:
        lines.append("**CONCLUSION: Both rankings show limited alignment with AAE.** The AAE validated portfolio-based multi-asset signal generation. Neither local nor global percentile thresholding inherently produces portfolio-like diversification. This suggests the architecture needs a fundamentally different approach to cross-asset signal integration, not just a different normalization method.")

    lines.append("")

    # Final summary
    lines.append("---")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Data span:** {len(group_results)} evaluation-level records analyzed across {n_groups} time groups")
    lines.append("")
    lines.append(f"**Local rank triggers:**")
    for sym, cnt in sorted(local_triggers.items(), key=lambda x: -x[1]):
        lines.append(f"  - {sym}: {cnt}")
    lines.append("")
    lines.append(f"**Global rank (top-1) triggers:**")
    for sym, cnt in sorted(global_triggers.items(), key=lambda x: -x[1]):
        lines.append(f"  - {sym}: {cnt}")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")
    return report_path


def main():
    print("=" * 62)
    print("PROBLEM 1: GLOBAL OPPORTUNITY AUDIT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()

    print("Loading data sources...")
    stats = _load_observability_stats()
    funnel = _load_funnel_stats()
    trades = _load_live_trades()
    signals = _load_signal_ledger()

    print(f"  Observability stats: {stats.get('evaluated_count', 0)} evaluations")
    print(f"  Funnel records: {len(funnel)}")
    print(f"  DuckDB trades: {len(trades)}")
    print(f"  DuckDB signals: {len(signals)}")
    print()

    print("Analyzing local vs global rank triggers...")
    results, local_triggers, global_triggers, n_groups = analyze_funnel_signals(funnel, stats)

    print(f"  Analyzed {len(results)} evaluation records across {n_groups} time groups")
    print(f"  Local triggers by asset: {dict(local_triggers)}")
    print(f"  Global (top-1) triggers by asset: {dict(global_triggers)}")
    print()

    print("Generating report...")
    generate_report(results, local_triggers, global_triggers, stats, trades, signals, n_groups)


if __name__ == "__main__":
    main()

