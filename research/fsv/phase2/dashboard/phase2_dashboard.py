from typing import Optional
from ...core.fsv_schema import FundamentalStateVector
from ...dashboard.fsv_dashboard_spec import FSVDashboardSpec, FSVDashboardRenderer
from ..core.fundamental_ranker import FundamentalRanker
from ..core.symbol_comparator import FundamentalComparator
from ..core.regime_context import RegimeContextClassifier
from ..integration.fundamental_selector import FundamentalSelector
import time
import json
import math


class Phase2Dashboard:

    def __init__(self, fsv_dashboard: FSVDashboardSpec = None) -> None:
        self.fsv_dashboard: FSVDashboardSpec = fsv_dashboard or FSVDashboardSpec()
        self.selector: FundamentalSelector = FundamentalSelector()
        self.comparator: FundamentalComparator = FundamentalComparator()
        self.ranker: FundamentalRanker = FundamentalRanker()
        self.regime_classifier: RegimeContextClassifier = RegimeContextClassifier()
        self.selection_history: list[dict] = []

    def capture_selection_state(
        self,
        top3: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
        convictions: dict[str, float],
    ) -> dict:
        selection: dict = self.selector.select_best(top3, fsves, directions, convictions)
        record: dict = {
            "timestamp": time.time(),
            "top3": list(top3),
            "selected_symbol": selection.get("selected_symbol", ""),
            "confidence": selection.get("recommendation", {}).get("confidence", 0.0),
            "reason": selection.get("recommendation", {}).get("reason", ""),
            "ranking_vector": dict(selection.get("ranking_vector", {})),
            "modulation_applied": bool(selection.get("modulation_applied", False)),
            "fallback_used": bool(selection.get("fallback_used", False)),
        }
        self.selection_history.append(record)
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
        return selection

    def export_selection_snapshot(
        self,
        top3: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
        convictions: dict[str, float],
    ) -> dict:
        selection_result: dict = self.selector.select_best(top3, fsves, directions, convictions)
        comparison: dict = self.comparator.compare_symbols(top3, fsves, directions)
        environment: str = comparison.get("environment", "neutral")
        regime: str = self.regime_classifier.classify(fsves)
        ranked_list: list[dict] = []
        candidates: dict[str, tuple[FundamentalStateVector, int]] = {}
        for sym in top3:
            if sym in fsves and sym in directions:
                candidates[sym] = (fsves[sym], directions[sym])
        if candidates:
            ranked_list = self.ranker.rank_symbols(candidates)
        ranking_vector: dict[str, float] = {}
        if ranked_list:
            ranking_vector = self.ranker.get_ranking_vector(ranked_list)
        adjusted_convictions: dict[str, float] = {}
        if ranked_list:
            adjusted_convictions = self.ranker.compute_conviction_adjustment(ranked_list, convictions)
        best_symbol: str = selection_result.get("selected_symbol", top3[0]) if top3 else ""
        influence: float = self.selector.get_selection_influence(best_symbol, ranking_vector)
        history_timestamps: list[float] = [h["timestamp"] for h in self.selection_history[-20:]]
        snapshot: dict = {
            "timestamp": time.time(),
            "top3_symbols": list(top3),
            "selection_result": selection_result,
            "comparison": comparison,
            "environment": environment,
            "regime": regime,
            "ranking_vector": ranking_vector,
            "adjusted_convictions": adjusted_convictions,
            "selection_influence": influence,
            "history_length": len(self.selection_history),
            "history_timestamps": history_timestamps,
        }
        return snapshot

    def generate_selection_timeline(self, limit: int = 50) -> list[dict]:
        recent: list[dict] = self.selection_history[-limit:] if self.selection_history else []
        timeline: list[dict] = []
        for entry in recent:
            timeline.append({
                "timestamp": entry.get("timestamp", 0.0),
                "selected_symbol": entry.get("selected_symbol", ""),
                "top3": list(entry.get("top3", [])),
                "confidence": entry.get("confidence", 0.0),
                "influence": entry.get("influence", 0.0),
            })
        return timeline

    def get_selection_metrics(self) -> dict:
        if not self.selection_history:
            return {
                "frequency_per_symbol": {},
                "average_confidence": 0.0,
                "influence_trend": [],
                "selection_stability": 1.0,
                "total_selections": 0,
            }
        freq: dict[str, int] = {}
        total_conf: float = 0.0
        changes: int = 0
        last_selected: str = ""
        influence_values: list[float] = []
        for entry in self.selection_history:
            sym: str = entry.get("selected_symbol", "")
            freq[sym] = freq.get(sym, 0) + 1
            total_conf += entry.get("confidence", 0.0)
            inf: float = entry.get("influence", 0.0)
            influence_values.append(inf)
            if last_selected and sym != last_selected:
                changes += 1
            last_selected = sym
        n: int = len(self.selection_history)
        avg_conf: float = total_conf / n if n > 0 else 0.0
        stability: float = 1.0 - (changes / n) if n > 0 else 1.0
        stability = max(0.0, min(1.0, stability))
        return {
            "frequency_per_symbol": freq,
            "average_confidence": avg_conf,
            "influence_trend": influence_values[-50:],
            "selection_stability": stability,
            "total_selections": n,
        }

    def export_html_report(
        self,
        top3: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
        convictions: dict[str, float],
        fsv_snapshot: dict = None,
    ) -> str:
        snapshot: dict = self.export_selection_snapshot(top3, fsves, directions, convictions)
        selection_result: dict = snapshot.get("selection_result", {})
        comparison: dict = snapshot.get("comparison", {})
        environment: str = snapshot.get("environment", "neutral")
        regime: str = snapshot.get("regime", "neutral")
        ranking_vector: dict = snapshot.get("ranking_vector", {})
        adjusted_convictions: dict = snapshot.get("adjusted_convictions", {})
        influence: float = snapshot.get("selection_influence", 0.0)
        ranked_list: list[dict] = []
        candidates: dict[str, tuple[FundamentalStateVector, int]] = {}
        for sym in top3:
            if sym in fsves and sym in directions:
                candidates[sym] = (fsves[sym], directions[sym])
        if candidates:
            ranked_list = self.ranker.rank_symbols(candidates)
        recommendation: dict = selection_result.get("recommendation", {})
        best_symbol: str = recommendation.get("best_symbol", top3[0]) if top3 else ""
        confidence: float = recommendation.get("confidence", 0.0)
        reason: str = recommendation.get("reason", "")
        timeline: list[dict] = self.generate_selection_timeline(20)
        metrics: dict = self.get_selection_metrics()

        ranking_rows: str = ""
        for i, r in enumerate(ranked_list):
            sym: str = r.get("symbol", "")
            score: float = r.get("fundamental_score", 0.0)
            rconf: float = r.get("confidence", 0.0)
            aligned: bool = r.get("direction_alignment", False)
            dir_sym: str = "YES" if aligned else "NO"
            rank_cls: str = "rank-best" if i == 0 else ("rank-second" if i == 1 else "rank-third")
            ranking_rows += (
                f"<tr class=\"{rank_cls}\">"
                f"<td>{i + 1}</td>"
                f"<td>{sym}</td>"
                f"<td>{score:.4f}</td>"
                f"<td>{rconf:.4f}</td>"
                f"<td>{dir_sym}</td>"
                f"</tr>\n"
            )

        bar_charts: str = ""
        if ranking_vector:
            max_weight: float = max(ranking_vector.values()) if ranking_vector else 1.0
            if max_weight <= 0.0:
                max_weight = 1.0
            sorted_vec: list[tuple[str, float]] = sorted(ranking_vector.items(), key=lambda x: x[1], reverse=True)
            for sym, weight in sorted_vec:
                pct: float = (weight / max_weight) * 100.0
                bar_color: str = "#4caf50" if sym == best_symbol else "#e94560"
                bar_charts += (
                    f"<div style=\"margin:4px 0;display:flex;align-items:center;\">"
                    f"<span style=\"width:80px;font-size:12px;\">{sym}</span>"
                    f"<div style=\"flex:1;background:#2a2a4a;border-radius:4px;height:20px;margin:0 8px;\">"
                    f"<div style=\"width:{pct:.1f}%;background:{bar_color};height:20px;border-radius:4px;"
                    f"display:flex;align-items:center;justify-content:flex-end;padding-right:4px;"
                    f"font-size:10px;color:#fff;min-width:30px;\">{weight:.3f}</div>"
                    f"</div></div>\n"
                )

        adv_rows: str = ""
        for sym in top3:
            base_val: float = convictions.get(sym, 0.0)
            adj_val: float = adjusted_convictions.get(sym, base_val)
            delta: float = adj_val - base_val
            delta_sym: str = "+" if delta >= 0 else ""
            adv_rows += (
                f"<tr>"
                f"<td>{sym}</td>"
                f"<td>{base_val:.4f}</td>"
                f"<td>{adj_val:.4f}</td>"
                f"<td style=\"color:{'#4caf50' if delta >= 0 else '#f44336'}\">{delta_sym}{delta:.4f}</td>"
                f"</tr>\n"
            )

        hist_rows: str = ""
        for entry in timeline:
            ts: float = entry.get("timestamp", 0.0)
            ts_str: str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "N/A"
            sel_sym: str = entry.get("selected_symbol", "")
            sel_conf: float = entry.get("confidence", 0.0)
            sel_inf: float = entry.get("influence", 0.0)
            hist_rows += (
                f"<tr>"
                f"<td>{ts_str}</td>"
                f"<td>{sel_sym}</td>"
                f"<td>{sel_conf:.4f}</td>"
                f"<td>{sel_inf:.4f}</td>"
                f"</tr>\n"
            )

        freq_rows: str = ""
        freq: dict = metrics.get("frequency_per_symbol", {})
        sorted_freq: list[tuple[str, int]] = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for sym, cnt in sorted_freq:
            freq_rows += (
                f"<tr>"
                f"<td>{sym}</td>"
                f"<td>{cnt}</td>"
                f"</tr>\n"
            )

        divergences: dict = comparison.get("divergences", {})
        div_rows: str = ""
        for pair, val in divergences.items():
            div_rows += (
                f"<tr>"
                f"<td>{pair}</td>"
                f"<td>{val:.4f}</td>"
                f"</tr>\n"
            )

        influence_pct: float = influence * 100.0
        influence_color: str = "#4caf50" if influence > 0.6 else ("#ff9800" if influence > 0.3 else "#f44336")

        env_color: str = "#4caf50"
        if environment == "risk_off":
            env_color = "#f44336"
        elif environment == "mixed":
            env_color = "#ff9800"
        elif environment == "neutral":
            env_color = "#9e9e9e"

        regime_color: str = "#4caf50"
        if regime == "risk_off":
            regime_color = "#f44336"
        elif regime == "transition":
            regime_color = "#ff9800"
        elif regime == "neutral":
            regime_color = "#9e9e9e"

        fsv_table_rows: str = ""
        if fsv_snapshot is not None:
            symbols_data: dict = fsv_snapshot.get("symbols", {}) if isinstance(fsv_snapshot, dict) else {}
            for sym in top3:
                sdata: dict = symbols_data.get(sym, {})
                if not sdata:
                    continue
                bias: float = sdata.get("bias_alignment", 0.0)
                macro: float = sdata.get("macro_pressure", 0.0)
                sent: float = sdata.get("sentiment_gradient", 0.0)
                risk: float = sdata.get("event_risk", 0.5)
                stab: float = sdata.get("regime_stability", 0.5)
                score: float = sdata.get("composite_macro_score", 0.0)
                fsv_table_rows += (
                    f"<tr>"
                    f"<td>{sym}</td>"
                    f"<td style=\"color:{'#4caf50' if bias >= 0 else '#f44336'}\">{bias:+.3f}</td>"
                    f"<td style=\"color:{'#4caf50' if macro >= 0 else '#f44336'}\">{macro:+.3f}</td>"
                    f"<td style=\"color:{'#4caf50' if sent >= 0 else '#f44336'}\">{sent:+.3f}</td>"
                    f"<td>{risk:.3f}</td>"
                    f"<td>{stab:.3f}</td>"
                    f"<td>{score:+.3f}</td>"
                    f"</tr>\n"
                )

        html: str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 2 — FSV Selection Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1, h2, h3 {{ color: #ffffff; margin: 16px 0 8px; }}
h1 {{ font-size: 24px; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
h2 {{ font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-top: 24px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #333; }}
th {{ background: #16213e; color: #e94560; font-weight: 600; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:hover {{ background: #16213e; }}
.rank-best {{ background: rgba(76, 175, 80, 0.08); }}
.rank-second {{ background: rgba(255, 152, 0, 0.05); }}
.rank-third {{ background: rgba(244, 67, 54, 0.03); }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 12px 0 24px; }}
.stat-card {{ background: #16213e; padding: 12px; border-radius: 8px; }}
.stat-label {{ font-size: 11px; color: #888; }}
.stat-value {{ font-size: 18px; font-weight: bold; margin-top: 4px; }}
.selection-box {{ background: #16213e; border-left: 4px solid #e94560; padding: 16px; margin: 12px 0 24px; border-radius: 4px; }}
.selection-box .best {{ font-size: 20px; color: #4caf50; font-weight: bold; }}
.selection-box .reason {{ font-size: 13px; color: #aaa; margin-top: 8px; }}
.selection-box .meta {{ font-size: 12px; color: #888; margin-top: 4px; }}
.gauge-container {{ background: #2a2a4a; border-radius: 8px; height: 24px; margin: 8px 0 24px; overflow: hidden; position: relative; }}
.gauge-fill {{ height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: #fff; transition: width 0.3s; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0 24px; }}
.card {{ background: #16213e; padding: 16px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Phase 2 — FSV Selection Dashboard</h1>

<div class="stats-grid">
<div class="stat-card"><div class="stat-label">Environment</div><div class="stat-value" style="color:{env_color};">{environment}</div></div>
<div class="stat-card"><div class="stat-label">Regime</div><div class="stat-value" style="color:{regime_color};">{regime}</div></div>
<div class="stat-card"><div class="stat-label">Total Selections</div><div class="stat-value">{metrics.get('total_selections', 0)}</div></div>
<div class="stat-card"><div class="stat-label">Avg Confidence</div><div class="stat-value">{metrics.get('average_confidence', 0.0):.4f}</div></div>
<div class="stat-card"><div class="stat-label">Stability</div><div class="stat-value">{metrics.get('selection_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Influence</div><div class="stat-value" style="color:{influence_color};">{influence:.4f}</div></div>
</div>

<div class="selection-box">
<div class="best">Selected: {best_symbol}</div>
<div class="meta">Confidence: {confidence:.4f} &mdash; Influence: {influence:.4f}</div>
<div class="reason">{reason}</div>
</div>

<h2>Top-3 Ranking</h2>
<table>
<thead><tr><th>Rank</th><th>Symbol</th><th>Score</th><th>Confidence</th><th>Direction Aligned</th></tr></thead>
<tbody>{ranking_rows}</tbody>
</table>

{f"<h2>Fundamental State Vectors (Top-3)</h2><table><thead><tr><th>Symbol</th><th>Bias</th><th>Macro</th><th>Sentiment</th><th>Risk</th><th>Stability</th><th>Composite</th></tr></thead><tbody>{fsv_table_rows}</tbody></table>" if fsv_table_rows else ""}

<h2>Ranking Vector Distribution</h2>
<div style="margin:12px 0 24px;">{bar_charts}</div>

<h2>Selection Recommendation</h2>
<table>
<thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Best Symbol</td><td><strong>{best_symbol}</strong></td></tr>
<tr><td>Confidence</td><td>{confidence:.4f}</td></tr>
<tr><td>Reason</td><td style="text-align:left;">{reason}</td></tr>
<tr><td>Runner Up</td><td>{recommendation.get('runner_up', '')}</td></tr>
<tr><td>Ranking Spread</td><td>{recommendation.get('ranking_spread', 0.0):.4f}</td></tr>
<tr><td>Modulation Applied</td><td>{'YES' if selection_result.get('modulation_applied') else 'NO'}</td></tr>
<tr><td>Fallback Used</td><td>{'YES' if selection_result.get('fallback_used') else 'NO'}</td></tr>
</tbody>
</table>

<h2>Adjusted Convictions</h2>
<table>
<thead><tr><th>Symbol</th><th>Base</th><th>Adjusted</th><th>Delta</th></tr></thead>
<tbody>{adv_rows}</tbody>
</table>

<h2>Environment &amp; Regime Classification</h2>
<div class="grid-2">
<div class="card">
<div style="font-size:14px;color:#888;">Environment</div>
<div style="font-size:24px;font-weight:bold;color:{env_color};">{environment}</div>
</div>
<div class="card">
<div style="font-size:14px;color:#888;">Regime</div>
<div style="font-size:24px;font-weight:bold;color:{regime_color};">{regime}</div>
</div>
</div>

<h2>Influence Gauge</h2>
<div class="gauge-container">
<div class="gauge-fill" style="width:{influence_pct:.1f}%;background:{influence_color};">{influence:.3f}</div>
</div>

<h2>Symbol Divergences</h2>
<table>
<thead><tr><th>Pair</th><th>Divergence</th></tr></thead>
<tbody>{div_rows}</tbody>
</table>

<h2>Selection History (last 20)</h2>
<table>
<thead><tr><th>Time</th><th>Symbol</th><th>Confidence</th><th>Influence</th></tr></thead>
<tbody>{hist_rows}</tbody>
</table>

<h2>Selection Frequency</h2>
<table>
<thead><tr><th>Symbol</th><th>Selections</th></tr></thead>
<tbody>{freq_rows}</tbody>
</table>
</body>
</html>"""
        return html


class Phase2ConsoleRenderer:

    @staticmethod
    def render_selection(selection_result: dict) -> str:
        recommendation: dict = selection_result.get("recommendation", {})
        selected: str = recommendation.get("best_symbol", "")
        confidence: float = recommendation.get("confidence", 0.0)
        reason: str = recommendation.get("reason", "")
        runner_up: str = recommendation.get("runner_up", "")
        spread: float = recommendation.get("ranking_spread", 0.0)
        fallback: bool = selection_result.get("fallback_used", False)
        modulated: bool = selection_result.get("modulation_applied", False)
        lines: list[str] = [
            "=" * 48,
            "  PHASE 2 — FUNDAMENTAL SELECTION",
            "=" * 48,
            f"  Selected:     {selected}",
            f"  Confidence:   {confidence:.4f}",
            f"  Runner Up:    {runner_up}",
            f"  Spread:       {spread:.4f}",
            f"  Reason:       {reason}",
            f"  Modulated:    {'YES' if modulated else 'NO'}",
            f"  Fallback:     {'YES' if fallback else 'NO'}",
            "=" * 48,
        ]
        return "\n".join(lines)

    @staticmethod
    def render_ranking(ranked_list: list[dict]) -> str:
        if not ranked_list:
            return "No ranking data available."
        header: str = f"{'Rank':>5} {'Symbol':<10} {'Score':>8} {'Confidence':>12} {'Direction':>10}"
        sep: str = "-" * len(header)
        lines: list[str] = [header, sep]
        for i, r in enumerate(ranked_list):
            sym: str = r.get("symbol", "")
            score: float = r.get("fundamental_score", 0.0)
            conf: float = r.get("confidence", 0.0)
            aligned: bool = r.get("direction_alignment", False)
            dir_str: str = "ALIGNED" if aligned else "MISALIGN"
            lines.append(
                f"{i + 1:>5d} {sym:<10s} {score:>8.4f} {conf:>12.4f} {dir_str:>10s}"
            )
        return "\n".join(lines)

    @staticmethod
    def render_comparison(comparison_result: dict) -> str:
        environment: str = comparison_result.get("environment", "neutral")
        divergences: dict = comparison_result.get("divergences", {})
        alignment: dict = comparison_result.get("alignment", {})
        ranked_symbols: list = comparison_result.get("ranked_symbols", [])
        lines: list[str] = [
            "=" * 48,
            "  FUNDAMENTAL COMPARISON",
            "=" * 48,
            f"  Environment:    {environment}",
            f"  Ranked Symbols: {', '.join(ranked_symbols)}",
            "",
            "  DIVERGENCES",
            "-" * 48,
        ]
        for pair, val in divergences.items():
            lines.append(f"    {pair:<20s} {val:>8.4f}")
        lines.append("")
        lines.append("  ALIGNMENT SCORES")
        lines.append("-" * 48)
        for sym, data in alignment.items():
            score: float = data.get("composite_alignment", 0.0)
            risk_env: str = data.get("risk_environment", "?")
            lines.append(f"    {sym:<10s} score={score:+.4f}  env={risk_env}")
        lines.append("=" * 48)
        return "\n".join(lines)

    @staticmethod
    def render_history(history: list[dict]) -> str:
        if not history:
            return "No selection history available."
        header: str = f"{'Time':>10} {'Symbol':<10} {'Confidence':>12} {'Influence':>10}"
        sep: str = "-" * len(header)
        lines: list[str] = [
            "=" * 48,
            "  SELECTION HISTORY (last {})".format(len(history)),
            "=" * 48,
            header,
            sep,
        ]
        for entry in history:
            ts: float = entry.get("timestamp", 0.0)
            ts_str: str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "N/A"
            sym: str = entry.get("selected_symbol", "")
            conf: float = entry.get("confidence", 0.0)
            inf: float = entry.get("influence", 0.0)
            lines.append(
                f"{ts_str:>10s} {sym:<10s} {conf:>12.4f} {inf:>10.4f}"
            )
        return "\n".join(lines)
