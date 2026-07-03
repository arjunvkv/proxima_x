"""
PROXIMA LIVE OUTCOME AUDIT

Audits every completed trade from DuckDB ledger.
For each trade: entry/exit time, hold duration, ES/AT ranks,
entry/exit price, PnL, exit reason, excursion.
"""

import sys
import os
import json
from datetime import datetime
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from proxima_ops.ledger.trade_ledger import TradeLedger


def run_audit() -> dict:
    try:
        ledger = TradeLedger()
        trades = ledger.get_completed()
    except Exception as e:
        print(f"Error accessing trade ledger (may be locked by another process): {e}")
        return {"total_trades": 0, "trades": [], "aggregates": {
            "avg_hold_bars": 0, "avg_adverse_exc_pct": 0, "avg_favorable_exc_pct": 0,
            "max_adverse_exc_pct": 0, "max_favorable_exc_pct": 0,
            "total_profitable_bars": 0, "total_losing_bars": 0,
            "total_profitable_trades": 0, "total_losing_trades": 0,
            "h20_exit_count": 0, "non_h20_exit_count": 0,
            "h20_compliance_pct": 0, "win_rate_pct": 0, "classification": "NO_DATA"
        }}

    if not trades:
        print("No completed trades found in ledger.")
        return {"total_trades": 0, "trades": [], "aggregates": {}}

    output_lines = []
    output_lines.append("=" * 62)
    output_lines.append("PROXIMA LIVE OUTCOME AUDIT")
    output_lines.append(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Total Completed Trades: {len(trades)}")
    output_lines.append("=" * 62)
    output_lines.append("")

    per_trade = []
    total_bars = 0
    total_profitable_bars = 0
    total_losing_bars = 0
    total_profitable = 0
    total_losing = 0
    h20_exits = 0
    non_h20_exits = 0
    total_ae = 0.0
    total_fe = 0.0
    max_ae = 0.0
    max_fe = 0.0

    for t in trades:
        trade_id = t.get("trade_id", 0)
        symbol = t.get("symbol", "?")
        entry_time = t.get("entry_time", "")
        exit_time = t.get("exit_time", "")
        duration_sec = t.get("duration", 0)
        duration_bars = max(1, duration_sec // 3600)
        es_rank = t.get("signal_score", 0.0)
        at_rank = t.get("adaptive_time", 0.0)
        entry_price = t.get("entry_price", 0.0)
        exit_price = t.get("exit_price", 0.0)
        pnl = t.get("profit_money", 0.0)
        exit_reason = t.get("exit_reason", "UNKNOWN") or "UNKNOWN"
        min_px = t.get("min_price", 0.0)
        max_px = t.get("max_price", 0.0)

        # Compute excursion
        if entry_price > 0 and min_px > 0:
            adverse_exc = abs((min_px - entry_price) / entry_price) * 100
        else:
            adverse_exc = 0.0
        if entry_price > 0 and max_px > 0:
            favorable_exc = abs((max_px - entry_price) / entry_price) * 100
        else:
            favorable_exc = 0.0

        total_bars += duration_bars
        if pnl > 0:
            total_profitable += 1
            total_profitable_bars += duration_bars
        else:
            total_losing += 1
            total_losing_bars += duration_bars

        if exit_reason == "H20":
            h20_exits += 1
        else:
            non_h20_exits += 1

        total_ae += adverse_exc
        total_fe += favorable_exc
        max_ae = max(max_ae, adverse_exc)
        max_fe = max(max_fe, favorable_exc)

        entry_ts = entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(entry_time, "strftime") else str(entry_time)[:16]
        exit_ts = exit_time.strftime("%Y-%m-%d %H:%M") if hasattr(exit_time, "strftime") and exit_time else "OPEN"

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        ae_str = f"{adverse_exc:.2f}%"
        fe_str = f"{favorable_exc:.2f}%"

        output_lines.append(f"Trade #{trade_id} | {symbol}")
        output_lines.append(f"  Entry: {entry_ts} | Exit: {exit_ts} | Duration: {duration_bars} bars")
        output_lines.append(f"  ES Rank: {es_rank:.2%} | AT Rank: {at_rank:.2%}")
        output_lines.append(f"  Entry Price: {entry_price:.5f} | Exit Price: {exit_price:.5f}")
        output_lines.append(f"  PnL: {pnl_str} | Exit Reason: {exit_reason}")
        output_lines.append(f"  Adverse Exc: {ae_str} | Favorable Exc: {fe_str}")
        output_lines.append("")

        per_trade.append({
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "duration_bars": duration_bars,
            "es_rank": round(es_rank, 4),
            "at_rank": round(at_rank, 4),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "adverse_exc_pct": round(adverse_exc, 2),
            "favorable_exc_pct": round(favorable_exc, 2)
        })

    n = max(len(trades), 1)
    avg_bars = total_bars / n
    avg_ae = total_ae / n
    avg_fe = total_fe / n

    profitable_pct = (total_profitable / n) * 100
    losing_pct = (total_losing / n) * 100

    h20_compliance = (h20_exits / n) * 100

    output_lines.append("=" * 62)
    output_lines.append("AGGREGATE METRICS")
    output_lines.append("=" * 62)
    output_lines.append(f"  1. Average Hold Duration:       {avg_bars:.1f} bars")
    output_lines.append(f"  2. Average Adverse Excursion:   {avg_ae:.2f}%")
    output_lines.append(f"  3. Average Favorable Excursion: {avg_fe:.2f}%")
    output_lines.append(f"  4. Maximum Adverse Excursion:   {max_ae:.2f}%")
    output_lines.append(f"  5. Maximum Favorable Excursion: {max_fe:.2f}%")
    output_lines.append(f"  6. Time Spent Profitable:       {total_profitable_bars} bars ({total_profitable} trades)")
    output_lines.append(f"  7. Time Spent Losing:           {total_losing_bars} bars ({total_losing} trades)")
    output_lines.append(f"  8. H20 Exit Compliance Rate:    {h20_compliance:.1f}% ({h20_exits}/{n})")
    output_lines.append("")
    output_lines.append(f"  Win Rate: {profitable_pct:.1f}% | Loss Rate: {losing_pct:.1f}%")
    output_lines.append(f"  H20 Exits: {h20_exits} | Non-H20 Exits: {non_h20_exits}")
    output_lines.append("")

    # Classification
    if h20_compliance >= 90:
        classification = "H20_HORIZON_ACHIEVED"
    elif h20_compliance >= 70:
        classification = "H20_PARTIALLY_ACHIEVED"
    else:
        classification = "H20_MISSED"

    output_lines.append(f"Classification: {classification}")
    output_lines.append("=" * 62)

    print("\n".join(output_lines))

    aggregates = OrderedDict([
        ("avg_hold_bars", round(avg_bars, 1)),
        ("avg_adverse_exc_pct", round(avg_ae, 2)),
        ("avg_favorable_exc_pct", round(avg_fe, 2)),
        ("max_adverse_exc_pct", round(max_ae, 2)),
        ("max_favorable_exc_pct", round(max_fe, 2)),
        ("total_profitable_bars", total_profitable_bars),
        ("total_losing_bars", total_losing_bars),
        ("total_profitable_trades", total_profitable),
        ("total_losing_trades", total_losing),
        ("h20_exit_count", h20_exits),
        ("non_h20_exit_count", non_h20_exits),
        ("h20_compliance_pct", round(h20_compliance, 1)),
        ("win_rate_pct", round(profitable_pct, 1)),
        ("classification", classification)
    ])

    report = {
        "audit_timestamp": datetime.now().isoformat(),
        "total_trades": len(trades),
        "classification": classification,
        "trades": per_trade,
        "aggregates": dict(aggregates)
    }

    # Write JSON
    out_json = os.path.join(os.path.dirname(__file__), "..", "..", "live_outcome_audit_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results written to: {out_json}")

    return report


def generate_audit_md(report: dict) -> str:
    if not report or not report.get("trades"):
        return "# PROXIMA LIVE OUTCOME AUDIT\n\nNo completed trades found.\n"

    lines = []
    lines.append("# PROXIMA LIVE OUTCOME AUDIT")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Completed Trades:** {report['total_trades']}")
    lines.append(f"**Classification:** {report['classification']}")
    lines.append("")

    agg = report["aggregates"]
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| # | Metric | Value |")
    lines.append("|---|--------|-------|")
    lines.append(f"| 1 | Average Hold Duration | {agg['avg_hold_bars']} bars |")
    lines.append(f"| 2 | Average Adverse Excursion | {agg['avg_adverse_exc_pct']}% |")
    lines.append(f"| 3 | Average Favorable Excursion | {agg['avg_favorable_exc_pct']}% |")
    lines.append(f"| 4 | Maximum Adverse Excursion | {agg['max_adverse_exc_pct']}% |")
    lines.append(f"| 5 | Maximum Favorable Excursion | {agg['max_favorable_exc_pct']}% |")
    lines.append(f"| 6 | Time Spent Profitable | {agg['total_profitable_bars']} bars ({agg['total_profitable_trades']} trades) |")
    lines.append(f"| 7 | Time Spent Losing | {agg['total_losing_bars']} bars ({agg['total_losing_trades']} trades) |")
    lines.append(f"| 8 | H20 Exit Compliance Rate | {agg['h20_compliance_pct']}% ({agg['h20_exit_count']}/{agg['h20_exit_count'] + agg['non_h20_exit_count']}) |")
    lines.append("")
    lines.append(f"**Win Rate:** {agg['win_rate_pct']}% | **Loss Rate:** {100 - agg['win_rate_pct']:.1f}%")
    lines.append("")

    # Answer the key questions
    cls = report["classification"]
    if cls == "H20_HORIZON_ACHIEVED":
        lines.append("## Verdict: ✅ Trades are reaching H20 horizon")
        lines.append(f"")
        lines.append(f"**{agg['h20_compliance_pct']}%** of trades exit at H20 (20 bars). The research exit horizon is the dominant exit path.")
    elif cls == "H20_PARTIALLY_ACHIEVED":
        lines.append("## Verdict: ⚠️ Partial H20 compliance")
        lines.append(f"")
        lines.append(f"Only {agg['h20_compliance_pct']}% of trades reach H20. Some exits are driven by other subsystems.")
    else:
        lines.append("## Verdict: ❌ Trades are NOT reaching H20 horizon")
        lines.append("")
        lines.append(f"Only {agg['h20_compliance_pct']}% of trades exit at H20. Trades are being closed early by another subsystem.")

    # Exit reason breakdown
    exit_reasons = {}
    for t in report["trades"]:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    lines.append("")
    lines.append("## Exit Reason Breakdown")
    lines.append("")
    lines.append("| Reason | Count | % |")
    lines.append("|--------|-------|---|")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / max(report["total_trades"], 1) * 100
        lines.append(f"| {reason} | {count} | {pct:.1f}% |")
    lines.append("")

    # Per-trade table
    lines.append("## Per-Trade Detail")
    lines.append("")
    lines.append("| ID | Symbol | Entry | Exit | Bars | ES | AT | Entry $ | Exit $ | PnL | Reason | AE% | FE% |")
    lines.append("|----|--------|-------|------|------|----|----|---------|---------|------|--------|-----|-----|")
    for t in report["trades"]:
        pnl_str = f"+${t['pnl']}" if t['pnl'] >= 0 else f"-${abs(t['pnl'])}"
        lines.append(f"| {t['trade_id']} | {t['symbol']} | {t['entry_time']} | {t['exit_time']} | {t['duration_bars']} | {t['es_rank']:.0%} | {t['at_rank']:.0%} | {t['entry_price']:.5f} | {t['exit_price']:.5f} | {pnl_str} | {t['exit_reason']} | {t['adverse_exc_pct']}% | {t['favorable_exc_pct']}% |")
    lines.append("")

    max_ae = agg["max_adverse_exc_pct"]
    if max_ae > 10:
        lines.append(f"## ⚠️ Warning: Max Adverse Excursion is {max_ae}%")
        lines.append("")
        lines.append("Some trades experienced significant adverse price movement. Review position sizing and stop placement.")

    return "\n".join(lines)


def main():
    report = run_audit()

    md = generate_audit_md(report)
    out_md = os.path.join(os.path.dirname(__file__), "..", "..", "LIVE_OUTCOME_AUDIT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Audit written to: {out_md}")


if __name__ == "__main__":
    main()
