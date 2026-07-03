"""
PROBLEM 6: FREQUENCY TARGET REALITY V2
=======================================
Estimate monthly trades per asset and total using actual observed rates.
Determine if the deployment will realistically achieve 20-60 trades/month.
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


def _load_funnel_stats():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "proxima_ops", "data", "funnel_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def main():
    print("=" * 62)
    print("PROBLEM 6: FREQUENCY TARGET REALITY V2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
    
    stats = _load_observability_stats()
    funnel = _load_funnel_stats()
    
    symbol_stats = stats.get("symbol_stats", {})
    total_seconds = 3600  # ~1 hour of data in observability stats
    
    # Extract execution timestamps from funnel for duration estimate
    if "signals" in funnel:
        signals_dict = funnel["signals"]
    else:
        signals_dict = {k: v for k, v in funnel.items() if k != "counts"}
    
    # Find timestamps of the first and last signal
    timestamps = []
    for sid, rec in signals_dict.items():
        ts = rec.get("timestamp_generated", "")
        if ts:
            timestamps.append(ts)
    
    if len(timestamps) >= 2:
        timestamps.sort()
        from datetime import datetime as dt
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        try:
            t_first = dt.strptime(timestamps[0][:26], fmt)
            t_last = dt.strptime(timestamps[-1][:26], fmt)
            data_span_hours = (t_last - t_first).total_seconds() / 3600
        except:
            data_span_hours = 1.0
    else:
        data_span_hours = 1.0
    
    print(f"  Data span: {data_span_hours:.2f} hours")
    print()
    
    # Monthly projections
    hours_per_month = 730  # avg
    scale_factor = hours_per_month / data_span_hours if data_span_hours > 0 else 730
    
    lines = []
    lines.append("# FREQUENCY TARGET REALITY V2 — Deployment Correction Problem 6")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Monthly Trade Projections")
    lines.append("")
    lines.append(f"**Data span:** {data_span_hours:.2f} hours (observed)")
    lines.append(f"**Scale factor:** {scale_factor:.1f}x to monthly")
    lines.append("")
    
    lines.append("```")
    lines.append(f"{'Asset':<12} {'Triggers':<12} {'Executed':<12} {'Est. Trig/Mo':<15} {'Est. Exec/Mo':<15}")
    lines.append("-" * 66)
    
    total_trig_monthly = 0
    total_exec_monthly = 0
    
    for sym in sorted(symbol_stats.keys()):
        ss = symbol_stats[sym]
        trig = ss.get("triggered", 0)
        exec_ = ss.get("executed", 0)
        trig_mo = trig * scale_factor
        exec_mo = exec_ * scale_factor
        total_trig_monthly += trig_mo
        total_exec_monthly += exec_mo
        lines.append(f"{sym:<12} {trig:<12} {exec_:<12} {trig_mo:<15.1f} {exec_mo:<15.1f}")
    
    lines.append("-" * 66)
    lines.append(f"{'TOTAL':<12} {'':<12} {'':<12} {total_trig_monthly:<15.1f} {total_exec_monthly:<15.1f}")
    lines.append("```")
    lines.append("")
    
    # Cap the total at realistic levels (these are clearly extrapolated from short sample)
    if data_span_hours < 10:
        lines.append("**WARNING:** Data span is too short ({:.1f}h) for reliable monthly extrapolation.".format(data_span_hours))
        lines.append("The monthly estimates above are extrapolated from a short observation window.")
        lines.append("Actual monthly rates will differ significantly due to market regime variation.")
        lines.append("")
    
    lines.append("## Target Assessment")
    lines.append("")
    lines.append("Research expectation: **20-60 trades/month**")
    lines.append(f"Projected total: **{total_exec_monthly:.0f} trades/month**")
    lines.append("")
    
    if total_exec_monthly >= 20 and total_exec_monthly <= 60:
        classification = "ON_TARGET"
        detail = f"Projected {total_exec_monthly:.0f} trades/month falls within the 20-60 target range."
    elif total_exec_monthly > 60:
        classification = "OVER_TRADING"
        detail = f"Projected {total_exec_monthly:.0f} trades/month exceeds the 60/month target. Risk of over-trading."
    elif total_exec_monthly >= 10:
        classification = "BELOW_TARGET"
        detail = f"Projected {total_exec_monthly:.0f} trades/month is below the 20/month target. May need more assets or lower thresholds."
    else:
        classification = "CRITICALLY_LOW"
        detail = f"Projected {total_exec_monthly:.0f} trades/month is critically below target. The engine is not generating enough trade flow."
    
    lines.append(f"**Classification: {classification}**")
    lines.append(detail)
    lines.append("")
    
    lines.append("## Per-Asset Monthly Execution Rate")
    lines.append("")
    if total_exec_monthly > 0:
        for sym in sorted(symbol_stats.keys()):
            ss = symbol_stats[sym]
            exec_ = ss.get("executed", 0)
            exec_mo = exec_ * scale_factor
            if exec_mo > 0:
                lines.append(f"- {sym}: {exec_mo:.1f}/month")
        lines.append("")
    
    if sum(ss.get("executed", 0) for ss in symbol_stats.values()) <= 1:
        lines.append("**CONCLUSION:** The current sample is too small for any meaningful frequency projection. ")
        lines.append("The estimated 20-60 trades/month target cannot be evaluated with existing data. ")
        lines.append("Continue running the engine until at least 30 trades have been observed before reassessing.")
    
    with open(os.path.join(REPORTS_DIR, "FREQUENCY_TARGET_REALITY_V2.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {os.path.join(REPORTS_DIR, 'FREQUENCY_TARGET_REALITY_V2.md')}")


if __name__ == "__main__":
    main()
