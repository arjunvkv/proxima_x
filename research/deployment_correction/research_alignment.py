"""
PROBLEM 2: RESEARCH ALIGNMENT
=============================
Compare AAE-validated asset contribution vs live deployment asset contribution.
Compute alignment_score: 0 = completely different strategy, 1 = identical strategy.
"""

import os
import json
import sys
from datetime import datetime
from collections import defaultdict

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# AAE RQ5: Portfolio Builder — validated asset contribution
AAE_EXPECTED = {
    "EURJPY": {"sharpe": 0.693, "pp": 0.738, "n_signals": 183},
    "USDJPY": {"sharpe": 0.326, "pp": 0.654, "n_signals": 182},
    "GBPJPY": {"sharpe": 0.526, "pp": 0.678, "n_signals": 183},
    "XAUUSD": {"sharpe": 0.177, "pp": 0.489, "n_signals": 176},
}
AAE_ASSETS = set(AAE_EXPECTED.keys())
# EURUSD is NOT in AAE research — deployment is trading it instead of the researched assets


def _load_observability_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "observability_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}, {}
    # Extract signal records
    signals_dict = raw.get("signals", raw)
    counts = raw.get("counts", {})
    return signals_dict, counts


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


def compute_alignment(aae_data, live_executions, live_trades):
    """
    Compute alignment score between AAE expected and live deployment.
    
    Dimensions:
    1. Asset overlap: What fraction of AAE-researched assets appear in live execution?
    2. Contribution similarity: How similar are the execution shares?
    3. EURUSD substitution penalty: Is EURUSD replacing researched assets?
    
    Returns 0.0 (no alignment) to 1.0 (perfect alignment).
    """
    scores = {}
    
    # D1: Asset overlap
    live_assets = set(live_executions.keys())
    aae_assets_set = set(aae_data.keys())
    overlap = len(aae_assets_set & live_assets)
    total_aae = len(aae_assets_set)
    # Also count EURUSD as a substitution (negative signal if EURUSD > 0 and AAE assets < overlap)
    eu_share = live_executions.get("EURUSD", 0)
    d1_overlap = overlap / total_aae if total_aae > 0 else 0
    # Penalize if EURUSD dominates while AAE assets are absent
    d1_eu_penalty = max(0, (eu_share - 0.2)) * 0.5  # up to -0.5 penalty if EURUSD > 20%
    d1_score = max(0, d1_overlap - d1_eu_penalty)
    scores["asset_overlap"] = d1_score
    
    # D2: Contribution similarity (cosine similarity of normalized execution vectors)
    all_assets = sorted(set(list(aae_data.keys()) + list(live_executions.keys())))
    aae_vec = []
    live_vec = []
    total_aae_signals = sum(v["n_signals"] for v in aae_data.values())
    total_live_exec = sum(live_executions.values()) if sum(live_executions.values()) > 0 else 1
    
    for a in all_assets:
        if a in aae_data:
            aae_vec.append(aae_data[a]["n_signals"] / total_aae_signals)
        else:
            aae_vec.append(0.0)
        if a in live_executions:
            live_vec.append(live_executions[a] / total_live_exec)
        else:
            live_vec.append(0.0)
    
    # Cosine similarity
    dot = sum(av * lv for av, lv in zip(aae_vec, live_vec))
    norm_aae = sum(av ** 2 for av in aae_vec) ** 0.5
    norm_live = sum(lv ** 2 for lv in live_vec) ** 0.5
    d2_score = dot / (norm_aae * norm_live) if norm_aae * norm_live > 0 else 0
    scores["contribution_similarity"] = d2_score
    
    # D3: EURUSD substitution
    # If EURUSD has high share but wasn't in AAE research, that's a substitution signal
    eu_share_of_live = live_executions.get("EURUSD", 0) / total_live_exec
    aae_share_of_live = sum(live_executions.get(a, 0) for a in aae_assets_set) / total_live_exec
    if eu_share_of_live > aae_share_of_live and len(live_executions) > 1:
        d3_score = 0.0  # EURUSD has replaced researched assets
    elif eu_share_of_live > 0.5:
        d3_score = 0.2  # EURUSD dominates but AAE assets also present
    else:
        d3_score = 1.0  # No substitution
    scores["substitution_penalty"] = d3_score
    
    # Weighted final score
    weights = {"asset_overlap": 0.4, "contribution_similarity": 0.4, "substitution_penalty": 0.2}
    final = sum(scores[k] * weights[k] for k in weights)
    
    return final, scores


def main():
    print("=" * 62)
    print("PROBLEM 2: RESEARCH ALIGNMENT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    stats = _load_observability_stats()
    signals_dict, counts = _load_funnel_stats()
    trades = _load_trades()
    
    # Build live execution counts per asset
    symbol_stats = stats.get("symbol_stats", {})
    live_executions = {}
    live_triggers = {}
    live_evaluations = {}
    for sym, ss in symbol_stats.items():
        live_executions[sym] = ss.get("executed", 0)
        live_triggers[sym] = ss.get("triggered", 0)
        live_evaluations[sym] = ss.get("evaluated", 0)
    
    # Compute funnel-based execution counts for cross-reference
    funnel_exec = defaultdict(int)
    for sid, rec in signals_dict.items():
        if isinstance(rec, dict) and rec.get("final_state") == "POSITION_OPENED":
            funnel_exec[rec.get("symbol", "?")] += 1
    
    print(f"  Live executions: {dict(live_executions)}")
    print(f"  Funnel executions: {dict(funnel_exec)}")
    print(f"  Live triggers: {dict(live_triggers)}")
    print(f"  Live evaluations: {dict(live_evaluations)}")
    print()
    
    print("  AAE expected contribution (RQ5 Portfolio Builder):")
    for sym, v in sorted(AAE_EXPECTED.items()):
        print(f"    {sym}: sharpe={v['sharpe']:.3f}, pp={v['pp']:.3f}, n_signals={v['n_signals']}")
    print()
    
    # Compute alignment
    alignment, scores = compute_alignment(AAE_EXPECTED, live_executions, trades)
    
    print(f"  Alignment score: {alignment:.4f}")
    print(f"  Dimension scores: {scores}")
    print()
    
    # Classification
    if alignment >= 0.8:
        classification = "RESEARCH_ALIGNED"
        detail = "Deployment faithfully expresses the validated multi-asset alpha strategy."
    elif alignment >= 0.5:
        classification = "PARTIALLY_ALIGNED"
        detail = "Deployment expresses some aspects of the research but has significant distortions."
    elif alignment >= 0.2:
        classification = "DEPLOYMENT_DISTORTED"
        detail = "Deployment deviates substantially from the researched strategy. Correction needed."
    else:
        classification = "COMPLETELY_DIVERGED"
        detail = "Deployment is trading a completely different strategy from what was researched. EURUSD-only concentration with no multi-asset participation."
    
    print(f"  Classification: {classification}")
    print(f"  Detail: {detail}")
    print()
    
    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "RESEARCH_ALIGNMENT_REPORT.md")
    
    lines = []
    lines.append("# RESEARCH ALIGNMENT REPORT — Deployment Correction Problem 2")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- **AAE RQ5 Portfolio Builder** (EURJPY run)")
    lines.append(f"- **Observability stats:** {stats.get('evaluated_count', 0)} evaluations, {stats.get('trigger_count', 0)} triggers, {stats.get('executed_count', 0)} executions")
    lines.append(f"- **Funnel signal records:** {len(signals_dict)}")
    lines.append(f"- **DuckDB trades:** {len(trades)}")
    lines.append("")
    
    lines.append("## AAE Expected Asset Contribution (RQ5)")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Sharpe':<10} {'PP':<10} {'Signals':<10}")
    lines.append("-" * 42)
    for sym, v in sorted(AAE_EXPECTED.items()):
        lines.append(f"{sym:<12} {v['sharpe']:<10.3f} {v['pp']:<10.3f} {v['n_signals']:<10}")
    lines.append("```")
    lines.append("")
    lines.append("**Key insight:** EURUSD is NOT in the AAE research. The validated alpha is across EURJPY, USDJPY, GBPJPY, and XAUUSD. " +
                 "EURUSD-only deployment represents a fundamental departure from the researched strategy.")
    lines.append("")
    
    lines.append("## Live Deployment Asset Contribution")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Evaluations':<14} {'Triggers':<12} {'Executions':<14} {'Exec %':<10}")
    lines.append("-" * 62)
    total_exec = sum(live_executions.values()) or 1
    for sym in sorted(live_executions.keys()):
        ev = live_evaluations.get(sym, 0)
        tr = live_triggers.get(sym, 0)
        ex = live_executions.get(sym, 0)
        pct = f"{100*ex/total_exec:.1f}%"
        lines.append(f"{sym:<12} {ev:<14} {tr:<12} {ex:<14} {pct:<10}")
    lines.append("```")
    lines.append("")
    
    lines.append("## Cross-Reference: AAE Assets vs Live Assets")
    lines.append("")
    r_assets = set(AAE_ASSETS)
    l_assets = set(live_executions.keys())
    
    lines.append(f"- AAE-researched assets: {sorted(r_assets)}")
    lines.append(f"- Live deployment assets: {sorted(l_assets)}")
    lines.append(f"- Overlap: {sorted(r_assets & l_assets)}")
    lines.append(f"- Missing from live: {sorted(r_assets - l_assets)}")
    lines.append(f"- Extra in live: {sorted(l_assets - r_assets)}")
    lines.append("")
    
    lines.append("## Dimension Scores")
    lines.append("")
    for k, v in scores.items():
        lines.append(f"- **{k}:** {v:.4f}")
    lines.append("")
    
    lines.append(f"## Alignment Score: {alignment:.4f}")
    lines.append("")
    lines.append(f"**Classification: {classification}**")
    lines.append(f"**Detail:** {detail}")
    lines.append("")
    
    lines.append("## Verdict")
    lines.append("")
    
    if alignment < 0.5:
        lines.append("The live deployment is NOT faithfully expressing the validated research. ")
        lines.append(f"Research built multi-asset alpha across {len(AAE_ASSETS)} assets (EURJPY, USDJPY, GBPJPY, XAUUSD). ")
        if live_executions.get("EURUSD", 0) > 0:
            lines.append("The deployment has substituted an unresearched asset (EURUSD) for the validated portfolio. ")
        lines.append("This is a deployment-layer distortion, not an alpha failure.")
    else:
        lines.append("The deployment partially reflects the validated research. ")
        lines.append("However, distortions in asset allocation and execution share remain.")
    
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("Before changing ANY research, the deployment must be corrected to express the validated multi-asset strategy. ")
    lines.append("The alignment score must reach >= 0.8 before any research conclusions about alpha decay or performance can be trusted.")
    lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

