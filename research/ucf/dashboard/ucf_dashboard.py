from __future__ import annotations

import json
import math
import time
from typing import Any

from ..core.unified_conviction_field import UnifiedConvictionField


def _direction_str(direction: int) -> str:
    if direction > 0:
        return "BUY"
    if direction < 0:
        return "SELL"
    return "NET"


def _conviction_color(value: float) -> str:
    if value >= 0.8:
        return "#00cc66"
    if value >= 0.6:
        return "#88cc44"
    if value >= 0.4:
        return "#cccc33"
    if value >= 0.2:
        return "#cc8800"
    return "#cc3333"


def _coherence_color(value: float) -> str:
    if value >= 0.7:
        return "#00cc66"
    if value >= 0.4:
        return "#cccc33"
    return "#cc3333"


class UCFDashboard:

    def render_field_snapshot(self, field_result: dict[str, Any]) -> str:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(field_result["timestamp"]),
        )
        lines: list[str] = []
        lines.append(f"UCF Conviction Field @ {timestamp}")
        lines.append("=" * 76)
        lines.append(
            f"{'Symbol':<10} | {'Conviction':<10} | {'Dir':<4} | {'Stability':<10} | {'Agreement':<10} | {'Tech%':<6} | {'Fund%':<6} | {'Exp%':<6}"
        )
        lines.append("-" * 76)

        field = field_result.get("field", {})
        if field:
            for symbol in sorted(field.keys()):
                entry = field[symbol]
                conv = entry.get("conviction_score", 0.0)
                direction = entry.get("direction", 0)
                stab = entry.get("stability", 0.0)
                agreement = entry.get("agreement", 0.0)
                comp = entry.get("component_breakdown", {})
                tc = comp.get("technical_contribution", 0.0)
                fc = comp.get("fundamental_contribution", 0.0)
                ec = comp.get("exposure_contribution", 0.0)
                total_comp = tc + fc + ec
                if total_comp > 0:
                    tech_pct = tc / total_comp * 100
                    fund_pct = fc / total_comp * 100
                    exp_pct = ec / total_comp * 100
                else:
                    tech_pct = fund_pct = exp_pct = 0.0

                dir_str = _direction_str(direction)
                agreement_str = f"{agreement:+.2f}"
                lines.append(
                    f"{symbol:<10} | {conv:<10.2f} | {dir_str:<4} | "
                    f"{stab:<10.2f} | {agreement_str:<10} | "
                    f"{tech_pct:>5.0f}% | {fund_pct:>5.0f}% | {exp_pct:>5.0f}%"
                )

        lines.append("=" * 76)

        coherence = field_result.get("field_coherence", 0.0)
        dominant = _direction_str(field_result.get("dominant_direction", 0))
        weights = field_result.get("weights", {})
        tw = weights.get("technical_weight", 0.0)
        fw = weights.get("fundamental_weight", 0.0)
        mw = weights.get("macro_weight", 0.0)
        ew = weights.get("exposure_weight", 0.0)

        lines.append(
            f"Field Coherence: {coherence:.2f} | Dominant: {dominant} | "
            f"Weights: T={tw:.2f} F={fw:.2f} M={mw:.2f} E={ew:.2f}"
        )

        return "\n".join(lines)

    def render_weight_dynamics(
        self, weight_history: list[dict[str, Any]]
    ) -> str:
        recent = weight_history[-10:] if len(weight_history) > 10 else weight_history
        lines: list[str] = []
        lines.append(f"Weight Evolution (last {len(recent)})")
        lines.append(
            f"{'Time':<8} | {'Tech':<6} | {'Fund':<6} | {'Macro':<6} | {'Exp':<6}"
        )
        lines.append("-" * 42)
        for i, entry in enumerate(recent):
            label = f"t-{len(recent) - i}"
            tw = entry.get("technical_weight", 0.0)
            fw = entry.get("fundamental_weight", 0.0)
            mw = entry.get("macro_weight", 0.0)
            ew = entry.get("exposure_weight", 0.0)
            lines.append(
                f"{label:<8} | {tw:<6.2f} | {fw:<6.2f} | {mw:<6.2f} | {ew:<6.2f}"
            )
        return "\n".join(lines)

    def render_regime_overlay(
        self, regime: str, stability: float, field: dict[str, Any]
    ) -> str:
        lines: list[str] = []
        lines.append(f"Regime Overlay: {regime.upper()}")
        lines.append(f"Regime Stability: {stability:.2f}")
        adapted_count = sum(
            1 for entry in field.values() if entry.get("regime_adapted", False)
        )
        lines.append(f"Regime-adapted symbols: {adapted_count}/{len(field)}")
        if field:
            avg_conviction = (
                sum(entry.get("conviction_score", 0.0) for entry in field.values())
                / len(field)
            )
            avg_stability = (
                sum(entry.get("stability", 0.0) for entry in field.values())
                / len(field)
            )
        else:
            avg_conviction = 0.0
            avg_stability = 0.0
        lines.append(
            f"Avg Conviction: {avg_conviction:.2f} | Avg Stability: {avg_stability:.2f}"
        )
        lines.append("-" * 50)
        directions: dict[str, int] = {}
        for symbol, entry in field.items():
            dir_key = _direction_str(entry.get("direction", 0))
            directions[dir_key] = directions.get(dir_key, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(directions.items()))
        lines.append(f"Direction distribution: {dist}")
        return "\n".join(lines)

    def export_field_snapshot(
        self, field_result: dict[str, Any]
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "timestamp": field_result.get("timestamp", 0.0),
            "regime": field_result.get("regime", "neutral"),
            "field_coherence": field_result.get("field_coherence", 0.0),
            "dominant_direction": _direction_str(
                field_result.get("dominant_direction", 0)
            ),
            "weights": dict(field_result.get("weights", {})),
            "symbols": [],
        }
        field = field_result.get("field", {})
        for symbol in sorted(field.keys()):
            entry = field[symbol]
            snapshot["symbols"].append(
                {
                    "symbol": symbol,
                    "conviction_score": entry.get("conviction_score", 0.0),
                    "direction": _direction_str(entry.get("direction", 0)),
                    "stability": entry.get("stability", 0.0),
                    "agreement": entry.get("agreement", 0.0),
                    "entropy": entry.get("entropy", 0.0),
                    "regime_adapted": entry.get("regime_adapted", False),
                    "component_breakdown": dict(
                        entry.get("component_breakdown", {})
                    ),
                }
            )
        return snapshot

    def export_html_report(
        self,
        field_result: dict[str, Any],
        weight_history: list[dict[str, Any]] | None = None,
        coherence_history: list[float] | None = None,
    ) -> str:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(field_result.get("timestamp", time.time())),
        )
        field = field_result.get("field", {})
        weights = field_result.get("weights", {})
        regime = field_result.get("regime", "neutral")
        coherence = field_result.get("field_coherence", 0.0)
        dominant = _direction_str(field_result.get("dominant_direction", 0))

        tw = weights.get("technical_weight", 0.0)
        fw = weights.get("fundamental_weight", 0.0)
        mw = weights.get("macro_weight", 0.0)
        ew = weights.get("exposure_weight", 0.0)
        regime_adapted_count = sum(
            1 for e in field.values() if e.get("regime_adapted", False)
        )

        ch = coherence_history or []
        if len(ch) > 1:
            trend_range = abs(ch[-1] - ch[0])
            coherence_trend = "Stable" if trend_range < 0.05 else "Volatile"
        else:
            coherence_trend = "N/A"

        heatmap_rows = ""
        for symbol in sorted(field.keys()):
            entry = field[symbol]
            conv = entry.get("conviction_score", 0.0)
            direction = _direction_str(entry.get("direction", 0))
            stab = entry.get("stability", 0.0)
            agr = entry.get("agreement", 0.0)
            color = _conviction_color(conv)
            heatmap_rows += (
                f"<tr>"
                f"<td>{symbol}</td>"
                f"<td style='background:{color};color:#000'>{conv:.2f}</td>"
                f"<td>{direction}</td>"
                f"<td>{stab:.2f}</td>"
                f"<td>{agr:+.2f}</td>"
                f"</tr>\n"
            )

        comp_section = ""
        for symbol in sorted(field.keys()):
            entry = field[symbol]
            comp = entry.get("component_breakdown", {})
            tc = comp.get("technical_contribution", 0.0)
            fc = comp.get("fundamental_contribution", 0.0)
            ec = comp.get("exposure_contribution", 0.0)
            total_c = tc + fc + ec
            if total_c > 0:
                tp = tc / total_c * 100
                fp = fc / total_c * 100
                ep = ec / total_c * 100
            else:
                tp = fp = ep = 0.0
            comp_section += (
                f"<div class='comp-row'>"
                f"<span class='comp-label'>{symbol}</span>"
                f"<div class='comp-bar'>"
                f"<div class='comp-segment' style='width:{tp:.1f}%;background:#3399ff' title='Tech {tp:.1f}%'></div>"
                f"<div class='comp-segment' style='width:{fp:.1f}%;background:#33cc66' title='Fund {fp:.1f}%'></div>"
                f"<div class='comp-segment' style='width:{ep:.1f}%;background:#cc6633' title='Exp {ep:.1f}%'></div>"
                f"</div></div>\n"
            )

        weight_rows = ""
        if weight_history:
            recent_wh = (
                weight_history[-10:]
                if len(weight_history) > 10
                else weight_history
            )
            for i, entry in enumerate(recent_wh):
                label = f"t-{len(recent_wh) - i}"
                w_tw = entry.get("technical_weight", 0.0)
                w_fw = entry.get("fundamental_weight", 0.0)
                w_mw = entry.get("macro_weight", 0.0)
                w_ew = entry.get("exposure_weight", 0.0)
                weight_rows += (
                    f"<tr>"
                    f"<td>{label}</td>"
                    f"<td>{w_tw:.2f}</td>"
                    f"<td>{w_fw:.2f}</td>"
                    f"<td>{w_mw:.2f}</td>"
                    f"<td>{w_ew:.2f}</td>"
                    f"</tr>\n"
                )

        gauge_color = _coherence_color(coherence)
        gauge_pct = coherence * 100

        html = (
            "<!DOCTYPE html>\n"
            "<html lang='en'>\n"
            "<head>\n"
            "<meta charset='UTF-8'>\n"
            "<title>UCF Dashboard</title>\n"
            "<style>\n"
            "*{margin:0;padding:0;box-sizing:border-box}\n"
            "body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI','DejaVu Sans',Arial,sans-serif;padding:20px}\n"
            "h1{color:#58a6ff;font-size:24px;margin-bottom:5px}\n"
            "h2{color:#58a6ff;font-size:18px;margin:20px 0 10px 0;border-bottom:1px solid #30363d;padding-bottom:5px}\n"
            ".header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding:15px;background:#161b22;border-radius:8px;border:1px solid #30363d}\n"
            ".header-info{color:#8b949e;font-size:13px}\n"
            "table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}\n"
            "th{text-align:left;padding:8px 10px;background:#161b22;color:#58a6ff;border:1px solid #30363d}\n"
            "td{padding:8px 10px;border:1px solid #30363d}\n"
            "tr:hover{background:#1c2128}\n"
            ".comp-row{display:flex;align-items:center;margin:6px 0;gap:10px}\n"
            ".comp-label{width:80px;font-size:12px;color:#8b949e;flex-shrink:0}\n"
            ".comp-bar{display:flex;height:22px;flex-grow:1;border-radius:4px;overflow:hidden;background:#21262d}\n"
            ".comp-segment{height:100%;transition:width 0.3s}\n"
            ".gauge-container{margin:15px 0;padding:15px;background:#161b22;border-radius:8px;border:1px solid #30363d}\n"
            ".gauge-bar{height:28px;border-radius:14px;background:#21262d;overflow:hidden;position:relative}\n"
            ".gauge-fill{height:100%;border-radius:14px;transition:width 0.5s;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;color:#000}\n"
            ".gauge-label{display:flex;justify-content:space-between;margin-top:5px;font-size:12px;color:#8b949e}\n"
            ".regime-badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold;text-transform:uppercase}\n"
            ".regime-risk_on{background:#1a3a1a;color:#3fb950}\n"
            ".regime-risk_off{background:#3a1a1a;color:#f85149}\n"
            ".regime-transition{background:#3a2a1a;color:#d29922}\n"
            ".regime-neutral{background:#1a1a3a;color:#58a6ff}\n"
            ".grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}\n"
            ".card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px;margin-bottom:15px}\n"
            ".metric{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}\n"
            ".metric-label{color:#8b949e}\n"
            ".metric-value{color:#c9d1d9;font-weight:bold}\n"
            ".footer{text-align:center;margin-top:30px;padding:15px;font-size:11px;color:#484f58;border-top:1px solid #30363d}\n"
            "</style>\n"
            "</head>\n"
            "<body>\n"
            "<div class='header'>\n"
            "<div>\n"
            "<h1>UCF Conviction Field</h1>\n"
            f"<div class='header-info'>{timestamp} | Regime: <span class='regime-badge regime-{regime}'>{regime}</span></div>\n"
            "</div>\n"
            "<div style='text-align:right'>\n"
            f"<div style='font-size:28px;font-weight:bold;color:#58a6ff'>{coherence:.2f}</div>\n"
            "<div style='font-size:11px;color:#8b949e'>Field Coherence</div>\n"
            "</div>\n"
            "</div>\n"
            "<div class='grid-2'>\n"
            "<div class='card'>\n"
            "<h2>Conviction Heatmap</h2>\n"
            "<table>\n"
            "<tr><th>Symbol</th><th>Conviction</th><th>Dir</th><th>Stability</th><th>Agreement</th></tr>\n"
            f"{heatmap_rows}"
            "</table>\n"
            "</div>\n"
            "<div class='card'>\n"
            "<h2>Weight Snapshot</h2>\n"
            f"<div class='metric'><span class='metric-label'>Technical</span><span class='metric-value'>{tw:.2f}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Fundamental</span><span class='metric-value'>{fw:.2f}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Macro</span><span class='metric-value'>{mw:.2f}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Exposure</span><span class='metric-value'>{ew:.2f}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Dominant Direction</span><span class='metric-value' style='color:#58a6ff'>{dominant}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Regime-Adapted</span><span class='metric-value'>{regime_adapted_count}/{len(field)}</span></div>\n"
            "</div>\n"
            "</div>\n"
            "<div class='card'>\n"
            "<h2>Component Decomposition</h2>\n"
            "<div style='display:flex;gap:15px;margin-bottom:10px;font-size:11px;color:#8b949e'>\n"
            "<span><span style='display:inline-block;width:12px;height:12px;background:#3399ff;border-radius:2px;vertical-align:middle;margin-right:4px'></span>Technical</span>\n"
            "<span><span style='display:inline-block;width:12px;height:12px;background:#33cc66;border-radius:2px;vertical-align:middle;margin-right:4px'></span>Fundamental</span>\n"
            "<span><span style='display:inline-block;width:12px;height:12px;background:#cc6633;border-radius:2px;vertical-align:middle;margin-right:4px'></span>Exposure</span>\n"
            "</div>\n"
            f"{comp_section}"
            "</div>\n"
            "<div class='grid-2'>\n"
            "<div class='card'>\n"
            "<h2>Weight Evolution</h2>\n"
            "<table>\n"
            "<tr><th>Time</th><th>Tech</th><th>Fund</th><th>Macro</th><th>Exp</th></tr>\n"
            f"{weight_rows}"
            "</table>\n"
            "</div>\n"
            "<div class='card'>\n"
            "<h2>Coherence Gauge</h2>\n"
            "<div class='gauge-container'>\n"
            "<div class='gauge-bar'>\n"
            f"<div class='gauge-fill' style='width:{gauge_pct:.1f}%;background:{gauge_color}'>{coherence:.2f}</div>\n"
            "</div>\n"
            "<div class='gauge-label'>\n"
            "<span>0.0</span>\n"
            "<span>0.5</span>\n"
            "<span>1.0</span>\n"
            "</div>\n"
            "</div>\n"
            f"<div class='metric'><span class='metric-label'>Regime</span><span class='metric-value'>{regime}</span></div>\n"
            f"<div class='metric'><span class='metric-label'>Coherence Trend</span><span class='metric-value'>{coherence_trend}</span></div>\n"
            "</div>\n"
            "</div>\n"
            f"<div class='footer'>UCF Dashboard &mdash; Generated {timestamp}</div>\n"
            "</body>\n"
            "</html>"
        )

        return html


class UCFConsoleRenderer:

    def render_field_table(self, field_result: dict[str, Any]) -> str:
        field = field_result.get("field", {})
        lines: list[str] = []
        lines.append(
            f"{'Symbol':<10} | {'Conviction':<10} | {'Dir':<4} | {'Stability':<10} | {'Agreement':<10}"
        )
        lines.append("-" * 54)
        if field:
            for symbol in sorted(field.keys()):
                entry = field[symbol]
                conv = entry.get("conviction_score", 0.0)
                direction = _direction_str(entry.get("direction", 0))
                stab = entry.get("stability", 0.0)
                agreement = entry.get("agreement", 0.0)
                agreement_str = f"{agreement:+.2f}"
                lines.append(
                    f"{symbol:<10} | {conv:<10.2f} | {direction:<4} | "
                    f"{stab:<10.2f} | {agreement_str:<10}"
                )
        return "\n".join(lines)

    def render_summary(self, field_result: dict[str, Any]) -> str:
        coherence = field_result.get("field_coherence", 0.0)
        dominant = _direction_str(field_result.get("dominant_direction", 0))
        field = field_result.get("field", {})
        top_symbol = ""
        top_conviction = -1.0
        for symbol, entry in field.items():
            conv = entry.get("conviction_score", 0.0)
            if conv > top_conviction:
                top_conviction = conv
                top_symbol = symbol
        regime = field_result.get("regime", "neutral")
        return (
            f"[UCF] Coherence={coherence:.2f} Dir={dominant} "
            f"Top={top_symbol}({top_conviction:.2f}) Regime={regime}"
        )
