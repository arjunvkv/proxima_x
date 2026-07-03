"""
PROBLEM 4: OPPORTUNITY CONCENTRATION
=====================================
Measure trigger share, execution share, and profit share per asset.
Compute HHI concentration score.

Classification: DIVERSIFIED, MODERATELY_CONCENTRATED, HIGHLY_CONCENTRATED, SINGLE_ASSET_DEPENDENT
"""

import os
import json
import sys
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _load_observability_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "observability_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_trades():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from proxima_ops.ledger.trade_ledger import TradeLedger
        tl = TradeLedger()
        tl._ensure_db()
        r = tl._conn.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY trade_id ASC").fetchall()
        return [dict(zip([desc[0] for desc in tl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Warning: could not load trades: {e}")
        return []


def hhi(shares):
    """Herfindahl-Hirschman Index: sum of squared shares (0-10000 scale)."""
    total = sum(shares.values()) or 1
    return sum((v / total * 100) ** 2 for v in shares.values())


def classify_hhi(hhi_score, n_assets):
    if n_assets <= 1:
        return "SINGLE_ASSET_DEPENDENT"
    if hhi_score >= 5000:
        return "HIGHLY_CONCENTRATED"
    if hhi_score >= 2500:
        return "MODERATELY_CONCENTRATED"
    return "DIVERSIFIED"


def main():
    print("=" * 62)
    print("PROBLEM 4: OPPORTUNITY CONCENTRATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    stats = _load_observability_stats()
    trades = _load_trades()
    
    symbol_stats = stats.get("symbol_stats", {})
    
    # Build shares
    trigger_shares = {sym: ss.get("triggered", 0) for sym, ss in symbol_stats.items()}
    execution_shares = {sym: ss.get("executed", 0) for sym, ss in symbol_stats.items()}
    
    # Profit shares from trades
    profit_shares = {}
    for t in trades:
        sym = t.get("symbol", "?")
        profit = t.get("profit_points", 0) or 0
        profit_shares[sym] = profit_shares.get(sym, 0) + profit
    
    total_trades = len(trades)
    n_assets_with_trigger = sum(1 for v in trigger_shares.values() if v > 0)
    n_assets_with_exec = sum(1 for v in execution_shares.values() if v > 0)
    
    # HHI scores
    trigger_hhi = hhi(trigger_shares)
    exec_hhi = hhi(execution_shares)
    profit_hhi = hhi(profit_shares) if profit_shares else 0
    
    trigger_class = classify_hhi(trigger_hhi, n_assets_with_trigger)
    exec_class = classify_hhi(exec_hhi, n_assets_with_exec)
    
    total_triggers = sum(trigger_shares.values())
    total_execs = sum(execution_shares.values())
    
    print(f"  Total trades: {total_trades}")
    print(f"  Trigger shares: {dict(trigger_shares)}")
    print(f"  Execution shares: {dict(execution_shares)}")
    print(f"  Profit shares: {dict(profit_shares)}")
    print()
    print(f"  Trigger HHI: {trigger_hhi:.1f} ({trigger_class})")
    print(f"  Execution HHI: {exec_hhi:.1f} ({exec_class})")
    print(f"  Profit HHI: {profit_hhi:.1f}")
    print()
    
    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "OPPORTUNITY_CONCENTRATION_REPORT.md")
    
    lines = []
    lines.append("# OPPORTUNITY CONCENTRATION REPORT — Deployment Correction Problem 4")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Trigger Share by Asset")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Triggers':<12} {'Share %':<10}")
    lines.append("-" * 34)
    dominant = max(trigger_shares, key=trigger_shares.get) if trigger_shares else "?"
    for sym in sorted(trigger_shares.keys()):
        v = trigger_shares[sym]
        pct = f"{100*v/total_triggers:.1f}%" if total_triggers > 0 else "N/A"
        marker = " <- dominant" if sym == dominant else ""
        lines.append(f"{sym:<12} {v:<12} {pct:<10}{marker}")
    lines.append("```")
    lines.append(f"**HHI: {trigger_hhi:.1f} | Classification: {trigger_class}**")
    lines.append("")
    
    lines.append("## Execution Share by Asset")
    lines.append("")
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Executions':<12} {'Share %':<10}")
    lines.append("-" * 34)
    dominant = max(execution_shares, key=execution_shares.get) if execution_shares else "?"
    for sym in sorted(execution_shares.keys()):
        v = execution_shares[sym]
        pct = f"{100*v/total_execs:.1f}%" if total_execs > 0 else "N/A"
        marker = " <- dominant" if sym == dominant else ""
        lines.append(f"{sym:<12} {v:<12} {pct:<10}{marker}")
    lines.append("```")
    lines.append(f"**HHI: {exec_hhi:.1f} | Classification: {exec_class}**")
    lines.append("")
    
    lines.append("## Profit Share by Asset")
    lines.append("")
    lines.append("```")
    if profit_shares:
        total_profit = sum(profit_shares.values())
        for sym in sorted(profit_shares.keys()):
            v = profit_shares[sym]
            pct = f"{100*v/total_profit:.1f}%"
            lines.append(f"  {sym}: {v:+.2f} pts ({pct})")
    else:
        lines.append("  No profit data available (0 trades)")
    lines.append("```")
    lines.append("")
    
    lines.append("## Concentration Classification")
    lines.append("")
    aggregated = max(trigger_hhi, exec_hhi, profit_hhi)
    if aggregated >= 5000:
        final = "SINGLE_ASSET_DEPENDENT"
    elif aggregated >= 3000:
        final = "HIGHLY_CONCENTRATED"
    elif aggregated >= 2000:
        final = "MODERATELY_CONCENTRATED"
    else:
        final = "DIVERSIFIED"
    
    lines.append(f"- Trigger HHI: {trigger_hhi:.1f} ({trigger_class})")
    lines.append(f"- Execution HHI: {exec_hhi:.1f} ({exec_class})")
    lines.append(f"- Profit HHI: {profit_hhi:.1f}")
    lines.append(f"- **Final: {final}**")
    lines.append("")
    
    if "SINGLE" in final:
        lines.append("The deployment is dependent on a single asset. This contradicts the multi-asset portfolio alpha validated in AAE RQ5.")
    elif "HIGHLY" in final:
        lines.append("The deployment is highly concentrated. The multi-asset diversification validated in research is not being expressed.")
    elif "MODERATELY" in final:
        lines.append("Moderate concentration exists but some diversification is present.")
    else:
        lines.append("The deployment is well-diversified across assets, consistent with AAE research.")
    lines.append("")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

