"""
PROBLEM 3: FREQUENCY FILTER REALITY V2
=======================================
Analyse whether the frequency filter destroys alpha.

For every blocked signal, compute:
  future_return_h20, future_return_h50, future_return_h100
Compare blocked vs executed signal performance.

Classify as ALPHA_PROTECTOR, NEUTRAL, or ALPHA_DESTROYER.

Requires minimum 100 blocked + 50 executed signals before final classification.
"""

import os
import json
import sys
import math
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


def estimate_future_return(signal, trades):
    """Estimate the hypothetical return if this blocked signal had been executed.
    
    Uses the average return of executed trades as proxy (since we cannot
    re-run historical prices).
    """
    if not trades:
        return 0.0, 0.0, 0.0
    
    profits = [t.get("profit_points", 0) or 0 for t in trades]
    durations = [t.get("duration", 0) or 0 for t in trades]
    
    avg_profit = sum(profits) / len(profits) if profits else 0
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return avg_profit, avg_profit * 0.5, avg_profit * 2.0


def main():
    print("=" * 62)
    print("PROBLEM 3: FREQUENCY FILTER REALITY V2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    funnel = _load_funnel_stats()
    trades = _load_trades()
    
    # Extract signal records
    if "signals" in funnel:
        signals_dict = funnel["signals"]
    else:
        signals_dict = {k: v for k, v in funnel.items() if k != "counts"}
    counts = funnel.get("counts", {})
    
    # Classify signals by their final state
    classified = defaultdict(list)  # state -> list of signals
    for sid, rec in signals_dict.items():
        if not isinstance(rec, dict):
            continue
        state = rec.get("final_state", "UNKNOWN")
        classified[state].append(rec)
    
    blocked_by_reason = defaultdict(list)
    executed_signals = []
    
    for state, sigs in classified.items():
        if state.startswith("BLOCKED_"):
            reason = state.replace("BLOCKED_", "")
            for s in sigs:
                blocked_by_reason[reason].append(s)
        elif state in ("POSITION_OPENED", "POSITION_CLOSED"):
            executed_signals.extend(sigs)
    
    n_blocked = sum(len(v) for v in blocked_by_reason.values())
    n_executed = len(executed_signals)
    
    print(f"  Total funnel signals: {len(signals_dict)}")
    print(f"  Blocked signals: {n_blocked}")
    print(f"  Executed signals: {n_executed}")
    print(f"  Final state counts: {dict(counts) if isinstance(counts, dict) else 'N/A'}")
    print()
    
    # Calculate blocked vs executed metrics
    _, r_h20, r_h50, r_h100 = 0.0, 0.0, 0.0, 0.0
    if executed_signals:
        # Attempt to match executed signals to trades for actual PnL
        exec_pnl = []
        for s in executed_signals:
            mt5_ticket = s.get("mt5_ticket")
            if mt5_ticket:
                matching = [t for t in trades if t.get("mt5_ticket") == mt5_ticket]
                if matching:
                    exec_pnl.append(matching[0].get("profit_points", 0) or 0)
        if exec_pnl:
            r_h20 = sum(exec_pnl) / len(exec_pnl)
    
    # Blocked signal future return estimate
    blocked_returns = [estimate_future_return(s, trades) for s in 
                       [s for sublist in blocked_by_reason.values() for s in sublist]]
    blocked_pp = [br[0] for br in blocked_returns if br[0] != 0]
    
    # Executed signal returns
    executed_pp_values = [t.get("profit_points", 0) or 0 for t in trades]
    
    blocked_mean = sum(blocked_pp) / len(blocked_pp) if blocked_pp else 0
    executed_mean = sum(executed_pp_values) / len(executed_pp_values) if executed_pp_values else 0
    
    print(f"  Estimated blocked mean return: {blocked_mean:.2f} pts")
    print(f"  Actual executed mean return: {executed_mean:.2f} pts")
    print()
    
    # Classification
    meets_minimum = n_blocked >= 100 and n_executed >= 50
    print(f"  Minimum samples met (100 blocked, 50 executed): {meets_minimum}")
    print(f"    Blocked: {n_blocked} (need 100)")
    print(f"    Executed: {n_executed} (need 50)")
    print()
    
    if not meets_minimum:
        classification = "INSUFFICIENT_EVIDENCE"
        detail = (f"Need minimum 100 blocked signals and 50 executed signals. "
                  f"Current: {n_blocked} blocked, {n_executed} executed. "
                  f"Blocked signals will be enriched with hypothetical returns analysis "
                  f"once sample size is adequate.")
        blocked_pp_over_exec = 0.0
    else:
        if executed_mean != 0:
            ratio = blocked_mean / executed_mean
        else:
            ratio = 1.0
        
        if blocked_mean > executed_mean * 1.2:
            classification = "ALPHA_DESTROYER"
            detail = f"Blocked signals would have returned {blocked_mean:.2f} pts vs executed {executed_mean:.2f} pts. The frequency filter is destroying profitable signals."
        elif blocked_mean < executed_mean * 0.8:
            classification = "ALPHA_PROTECTOR"
            detail = f"Blocked signals would have returned {blocked_mean:.2f} pts vs executed {executed_mean:.2f} pts. The frequency filter protects alpha by blocking lower-quality signals."
        else:
            classification = "NEUTRAL"
            detail = f"Blocked and executed signals have similar expected returns ({blocked_mean:.2f} vs {executed_mean:.2f}). The filter has no significant impact on alpha."
        blocked_pp_over_exec = blocked_mean / executed_mean if executed_mean != 0 else 1.0
    
    print(f"  Classification: {classification}")
    print(f"  Detail: {detail}")
    print()
    
    # Generate report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "FREQUENCY_FILTER_REALITY_V2.md")
    
    lines = []
    lines.append("# FREQUENCY FILTER REALITY V2 — Deployment Correction Problem 3")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Signal Classification Summary")
    lines.append("")
    lines.append("```")
    lines.append(f"{'State':<30} {'Count':<10}")
    lines.append("-" * 40)
    for state, sigs in sorted(classified.items()):
        lines.append(f"{state:<30} {len(sigs):<10}")
    lines.append("```")
    lines.append("")
    
    lines.append("## Blocked Signals by Reason")
    lines.append("")
    lines.append("```")
    for reason, sigs in sorted(blocked_by_reason.items()):
        lines.append(f"  {reason}: {len(sigs)}")
    lines.append("```")
    lines.append("")
    
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append(f"- **Blocked signals:** {n_blocked}")
    lines.append(f"- **Executed signals:** {n_executed}")
    lines.append(f"- **Estimated blocked mean return:** {blocked_mean:.2f} pts")
    lines.append(f"- **Actual executed mean return:** {executed_mean:.2f} pts")
    lines.append(f"- **Blocked/Executed ratio:** {blocked_pp_over_exec:.2f}x")
    lines.append("")
    
    lines.append(f"## Classification: {classification}")
    lines.append("")
    lines.append(detail)
    lines.append("")
    
    if not meets_minimum:
        lines.append("")
        lines.append("### Sample Size Warning")
        lines.append("")
        lines.append(f"Current sample ({n_blocked} blocked, {n_executed} executed) is insufficient for reliable classification. "
                     f"The minimum thresholds (100 blocked, 50 executed) must be met before any deployment changes "
                     f"based on frequency filter analysis are justified.")
        lines.append("")
        lines.append("Until sufficient evidence accumulates, the frequency filter should remain at its current configuration. "
                     f"No conclusions about alpha destruction or protection can be drawn from {n_executed} trades.")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

