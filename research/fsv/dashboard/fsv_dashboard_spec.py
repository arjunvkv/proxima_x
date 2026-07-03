from typing import Optional
from ..core.fsv_schema import FundamentalStateVector, NormalizedEvent
import time
import json
import statistics
import math
import copy


class FSVDashboardSpec:

    def __init__(self, engine=None) -> None:
        self.engine = engine
        self.history: list[dict] = []
        self.max_history: int = 10000

    def export_snapshot(self, state_map: dict[str, FundamentalStateVector]) -> dict:
        now = time.time()
        symbols = {}
        for sym, fsv in state_map.items():
            age = now - fsv.last_update_ts if fsv.last_update_ts else 0.0
            composite = self._compute_composite(fsv)
            risk = self._classify_risk(fsv.event_risk)
            symbols[sym] = {
                "bias_alignment": fsv.bias_alignment,
                "macro_pressure": fsv.macro_pressure,
                "sentiment_gradient": fsv.sentiment_gradient,
                "event_risk": fsv.event_risk,
                "regime_stability": fsv.regime_stability,
                "last_update_ts": fsv.last_update_ts,
                "decay_lambda": fsv.decay_lambda,
                "age_seconds": age,
                "composite_macro_score": composite,
                "risk_classification": risk,
            }

        bias_vals = [s["bias_alignment"] for s in symbols.values()]
        macro_vals = [s["macro_pressure"] for s in symbols.values()]
        sent_vals = [s["sentiment_gradient"] for s in symbols.values()]
        risk_vals = [s["event_risk"] for s in symbols.values()]
        regime_vals = [s["regime_stability"] for s in symbols.values()]

        n = len(symbols)
        mean_bias = statistics.mean(bias_vals) if bias_vals else 0.0
        mean_macro = statistics.mean(macro_vals) if macro_vals else 0.0
        mean_sent = statistics.mean(sent_vals) if sent_vals else 0.0
        mean_risk = statistics.mean(risk_vals) if risk_vals else 0.0
        mean_regime = statistics.mean(regime_vals) if regime_vals else 0.0
        vol_index = statistics.stdev(bias_vals) if len(bias_vals) >= 2 else 0.0

        risk_dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for s in symbols.values():
            risk_dist[s["risk_classification"]] += 1

        snapshot = {
            "timestamp": now,
            "symbol_count": n,
            "symbols": symbols,
            "aggregate_stats": {
                "mean_bias_alignment": mean_bias,
                "mean_macro_pressure": mean_macro,
                "mean_sentiment_gradient": mean_sent,
                "mean_event_risk": mean_risk,
                "mean_regime_stability": mean_regime,
                "volatility_index": vol_index,
                "risk_distribution": risk_dist,
            },
        }

        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return snapshot

    def generate_metrics(self, state_history: list[dict] = None) -> dict:
        if state_history is None:
            state_history = self.history

        if not state_history:
            return {
                "time_range": [0.0, 0.0],
                "snapshot_count": 0,
                "stability_metrics": {
                    "bias_alignment_stability": 1.0,
                    "macro_pressure_stability": 1.0,
                    "sentiment_gradient_stability": 1.0,
                    "event_risk_stability": 1.0,
                    "regime_stability_stability": 1.0,
                    "overall_stability": 1.0,
                },
                "event_metrics": {
                    "total_events_processed": 0,
                    "events_per_symbol": {},
                    "sources_breakdown": {},
                    "event_type_breakdown": {},
                },
                "health_score": 0.0,
            }

        timestamps = [s["timestamp"] for s in state_history]
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        snapshot_count = len(state_history)

        def _rolling_stability(field: str) -> float:
            vals: list[float] = []
            for snap in state_history:
                agg = snap.get("aggregate_stats", {})
                v = agg.get(field, 0.0)
                vals.append(v)
            if len(vals) < 2:
                return 1.0
            diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
            std_val = statistics.stdev(diffs) if len(diffs) >= 2 else (diffs[0] if diffs else 0.0)
            stab = 1.0 - std_val
            return max(0.0, min(1.0, stab))

        stab_bias = _rolling_stability("mean_bias_alignment")
        stab_macro = _rolling_stability("mean_macro_pressure")
        stab_sent = _rolling_stability("mean_sentiment_gradient")
        stab_risk = _rolling_stability("mean_event_risk")
        stab_regime = _rolling_stability("mean_regime_stability")
        overall = (stab_bias + stab_macro + stab_sent + stab_risk + stab_regime) / 5.0

        event_metrics: dict = {
            "total_events_processed": 0,
            "events_per_symbol": {},
            "sources_breakdown": {},
            "event_type_breakdown": {},
        }

        if self.engine is not None and hasattr(self.engine, "event_log"):
            log = getattr(self.engine, "event_log", [])
            event_metrics["total_events_processed"] = len(log)
            for ev in log:
                sym = ev.symbol
                event_metrics["events_per_symbol"][sym] = event_metrics["events_per_symbol"].get(sym, 0) + 1
                src = ev.source if ev.source else "unknown"
                event_metrics["sources_breakdown"][src] = event_metrics["sources_breakdown"].get(src, 0) + 1
                etype = ev.event_type if ev.event_type else "UNKNOWN"
                event_metrics["event_type_breakdown"][etype] = event_metrics["event_type_breakdown"].get(etype, 0) + 1

        latest_snapshot = state_history[-1] if state_history else {}
        health = self._compute_health_score(latest_snapshot, {
            "stability_metrics": {
                "bias_alignment_stability": stab_bias,
                "macro_pressure_stability": stab_macro,
                "sentiment_gradient_stability": stab_sent,
                "event_risk_stability": stab_risk,
                "regime_stability_stability": stab_regime,
                "overall_stability": overall,
            },
            "event_metrics": event_metrics,
        })

        return {
            "time_range": [first_ts, last_ts],
            "snapshot_count": snapshot_count,
            "stability_metrics": {
                "bias_alignment_stability": stab_bias,
                "macro_pressure_stability": stab_macro,
                "sentiment_gradient_stability": stab_sent,
                "event_risk_stability": stab_risk,
                "regime_stability_stability": stab_regime,
                "overall_stability": overall,
            },
            "event_metrics": event_metrics,
            "health_score": health,
        }

    def symbol_heatmap_data(self, state_map: dict[str, FundamentalStateVector]) -> list[dict]:
        result = []
        for sym, fsv in state_map.items():
            composite = self._compute_composite(fsv)
            result.append({
                "symbol": sym,
                "bias_alignment": fsv.bias_alignment,
                "macro_pressure": fsv.macro_pressure,
                "sentiment_gradient": fsv.sentiment_gradient,
                "event_risk": fsv.event_risk,
                "regime_stability": fsv.regime_stability,
                "composite": composite,
                "intensity": abs(composite),
            })
        return result

    def event_timeline(self, events: list[NormalizedEvent]) -> list[dict]:
        cum_impact: dict[str, float] = {}
        timeline = []
        for ev in events:
            cum_impact[ev.symbol] = cum_impact.get(ev.symbol, 0.0) + ev.impact_weight
            timeline.append({
                "timestamp": ev.timestamp,
                "symbol": ev.symbol,
                "event_type": ev.event_type,
                "surprise_score": ev.surprise_score,
                "direction_bias": ev.direction_bias,
                "impact_weight": ev.impact_weight,
                "source": ev.source,
                "cumulative_impact": cum_impact[ev.symbol],
            })
        return timeline

    def modulation_impact_report(
        self,
        symbol: str,
        conviction_history: list[float],
        fsv_history: list[FundamentalStateVector],
    ) -> dict:
        if not conviction_history or not fsv_history:
            return {
                "symbol": symbol,
                "mean_base_conviction": 0.0,
                "mean_adjusted_conviction": 0.0,
                "mean_modulation": 0.0,
                "max_modulation": 0.0,
                "min_modulation": 0.0,
                "modulation_volatility": 0.0,
                "total_adjustment_pct": 0.0,
            }

        modulations = [self._compute_composite(fsv) for fsv in fsv_history]
        min_len = min(len(conviction_history), len(modulations))
        conviction_history = conviction_history[:min_len]
        modulations = modulations[:min_len]

        base_convictions = [c - m for c, m in zip(conviction_history, modulations)]

        mean_base = statistics.mean(base_convictions) if base_convictions else 0.0
        mean_adj = statistics.mean(conviction_history) if conviction_history else 0.0
        mean_mod = statistics.mean(modulations) if modulations else 0.0
        max_mod = max(modulations) if modulations else 0.0
        min_mod = min(modulations) if modulations else 0.0
        mod_vol = statistics.stdev(modulations) if len(modulations) >= 2 else 0.0

        total_abs_base = sum(abs(b) for b in base_convictions) if base_convictions else 1.0
        total_abs_mod = sum(abs(m) for m in modulations) if modulations else 0.0
        total_adj_pct = (total_abs_mod / total_abs_base) * 100.0 if total_abs_base != 0 else 0.0

        return {
            "symbol": symbol,
            "mean_base_conviction": mean_base,
            "mean_adjusted_conviction": mean_adj,
            "mean_modulation": mean_mod,
            "max_modulation": max_mod,
            "min_modulation": min_mod,
            "modulation_volatility": mod_vol,
            "total_adjustment_pct": total_adj_pct,
        }

    def export_full_state(self, engine) -> dict:
        state_map = {}
        event_log: list[NormalizedEvent] = []
        if hasattr(engine, "state_map"):
            state_map = engine.state_map
        if hasattr(engine, "event_log"):
            event_log = engine.event_log

        snapshot = self.export_snapshot(state_map)
        metrics = self.generate_metrics()
        timeline = self.event_timeline(event_log)

        now = time.time()
        total_symbols = len(state_map)
        avg_age = 0.0
        if state_map:
            ages = [now - s.last_update_ts for s in state_map.values()]
            avg_age = statistics.mean(ages) if ages else 0.0

        has_recent_data = avg_age < 300.0
        has_stable_metrics = metrics["stability_metrics"]["overall_stability"] > 0.5
        has_event_flow = metrics["event_metrics"]["total_events_processed"] > 0

        status_flags = {
            "has_recent_data": has_recent_data,
            "has_stable_metrics": has_stable_metrics,
            "has_event_flow": has_event_flow,
        }

        return {
            "snapshot": snapshot,
            "metrics": metrics,
            "event_timeline": timeline,
            "status_flags": status_flags,
            "health_score": metrics["health_score"],
        }

    def generate_html_report(self, data: dict) -> str:
        snapshot = data.get("snapshot", {})
        metrics = data.get("metrics", {})
        timeline = data.get("event_timeline", [])
        health = data.get("health_score", 0.0)
        flags = data.get("status_flags", {})

        health_pct = f"{health:.1f}%"

        def _risk_color(risk: str) -> str:
            if risk == "low":
                return "#4caf50"
            elif risk == "medium":
                return "#ff9800"
            return "#f44336"

        def _val_color(v: float) -> str:
            if v > 0.0:
                return "#4caf50"
            elif v < 0.0:
                return "#f44336"
            return "#9e9e9e"

        symbols = snapshot.get("symbols", {})
        agg = snapshot.get("aggregate_stats", {})
        stab = metrics.get("stability_metrics", {})

        rows = ""
        for sym, sdata in symbols.items():
            bias = sdata.get("bias_alignment", 0.0)
            macro = sdata.get("macro_pressure", 0.0)
            sent = sdata.get("sentiment_gradient", 0.0)
            risk = sdata.get("event_risk", 0.5)
            regime = sdata.get("regime_stability", 0.5)
            age = sdata.get("age_seconds", 0.0)
            score = sdata.get("composite_macro_score", 0.0)
            rclass = sdata.get("risk_classification", "medium")
            rcolor = _risk_color(rclass)
            age_str = f"{age:.0f}s" if age < 60 else f"{age / 60:.1f}m"
            if age >= 3600:
                age_str = f"{age / 3600:.1f}h"
            rows += (
                f"<tr>"
                f"<td>{sym}</td>"
                f"<td style='color:{_val_color(bias)}'>{bias:+.3f}</td>"
                f"<td style='color:{_val_color(macro)}'>{macro:+.3f}</td>"
                f"<td style='color:{_val_color(sent)}'>{sent:+.3f}</td>"
                f"<td>{risk:.3f}</td>"
                f"<td>{regime:.3f}</td>"
                f"<td>{age_str}</td>"
                f"<td>{score:+.3f}</td>"
                f"<td style='color:{rcolor};font-weight:bold'>{rclass}</td>"
                f"</tr>\n"
            )

        tl_rows = ""
        for ev in timeline[-50:]:
            direction = "+" if ev.get("direction_bias", 0) >= 0 else ""
            tl_rows += (
                f"<tr>"
                f"<td>{ev.get('timestamp', 0.0):.1f}</td>"
                f"<td>{ev.get('symbol', '')}</td>"
                f"<td>{ev.get('event_type', '')}</td>"
                f"<td>{ev.get('surprise_score', 0.0):+.3f}</td>"
                f"<td>{direction}{ev.get('direction_bias', 0.0):+.3f}</td>"
                f"<td>{ev.get('impact_weight', 0.0):.3f}</td>"
                f"<td>{ev.get('source', '')}</td>"
                f"<td>{ev.get('cumulative_impact', 0.0):.3f}</td>"
                f"</tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FSV Dashboard Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1, h2, h3 {{ color: #ffffff; margin: 16px 0 8px; }}
h1 {{ font-size: 24px; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
h2 {{ font-size: 18px; }}
.health {{ display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
.health.good {{ background: #4caf50; color: #fff; }}
.health.fair {{ background: #ff9800; color: #fff; }}
.health.poor {{ background: #f44336; color: #fff; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 13px; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #333; }}
th {{ background: #16213e; color: #e94560; font-weight: 600; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:hover {{ background: #16213e; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 12px 0 24px; }}
.stat-card {{ background: #16213e; padding: 12px; border-radius: 8px; }}
.stat-label {{ font-size: 11px; color: #888; }}
.stat-value {{ font-size: 18px; font-weight: bold; margin-top: 4px; }}
.flags {{ display: flex; gap: 8px; margin: 12px 0; }}
.flag {{ padding: 4px 12px; border-radius: 12px; font-size: 12px; }}
.flag.ok {{ background: #4caf50; color: #fff; }}
.flag.warn {{ background: #f44336; color: #fff; }}
</style>
</head>
<body>
<h1>FSV Dashboard Report</h1>
<div class="flags">
<span class="flag {'ok' if flags.get('has_recent_data') else 'warn'}">Recent Data</span>
<span class="flag {'ok' if flags.get('has_stable_metrics') else 'warn'}">Stable Metrics</span>
<span class="flag {'ok' if flags.get('has_event_flow') else 'warn'}">Event Flow</span>
<span class="health {'good' if health >= 80 else 'fair' if health >= 50 else 'poor'}">Health: {health_pct}</span>
</div>

<h2>Symbol States</h2>
<table>
<thead><tr><th>Symbol</th><th>Bias</th><th>Macro</th><th>Sentiment</th><th>Risk</th><th>Stability</th><th>Age</th><th>Score</th><th>Risk Class</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>Aggregate Stats</h2>
<div class="stats-grid">
<div class="stat-card"><div class="stat-label">Mean Bias Alignment</div><div class="stat-value">{agg.get('mean_bias_alignment', 0.0):+.3f}</div></div>
<div class="stat-card"><div class="stat-label">Mean Macro Pressure</div><div class="stat-value">{agg.get('mean_macro_pressure', 0.0):+.3f}</div></div>
<div class="stat-card"><div class="stat-label">Mean Sentiment Gradient</div><div class="stat-value">{agg.get('mean_sentiment_gradient', 0.0):+.3f}</div></div>
<div class="stat-card"><div class="stat-label">Mean Event Risk</div><div class="stat-value">{agg.get('mean_event_risk', 0.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Mean Regime Stability</div><div class="stat-value">{agg.get('mean_regime_stability', 0.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Volatility Index</div><div class="stat-value">{agg.get('volatility_index', 0.0):.3f}</div></div>
</div>

<h2>Stability Metrics</h2>
<div class="stats-grid">
<div class="stat-card"><div class="stat-label">Bias Alignment Stability</div><div class="stat-value">{stab.get('bias_alignment_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Macro Pressure Stability</div><div class="stat-value">{stab.get('macro_pressure_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Sentiment Gradient Stability</div><div class="stat-value">{stab.get('sentiment_gradient_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Event Risk Stability</div><div class="stat-value">{stab.get('event_risk_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Regime Stability Stability</div><div class="stat-value">{stab.get('regime_stability_stability', 1.0):.3f}</div></div>
<div class="stat-card"><div class="stat-label">Overall Stability</div><div class="stat-value">{stab.get('overall_stability', 1.0):.3f}</div></div>
</div>

<h2>Event Timeline (last 50)</h2>
<table>
<thead><tr><th>Timestamp</th><th>Symbol</th><th>Type</th><th>Surprise</th><th>Direction</th><th>Impact</th><th>Source</th><th>Cum. Impact</th></tr></thead>
<tbody>{tl_rows}</tbody>
</table>

<h2>Risk Distribution</h2>
<div class="stats-grid">
<div class="stat-card" style="border-left: 3px solid #4caf50"><div class="stat-label">Low Risk</div><div class="stat-value">{agg.get('risk_distribution', {}).get('low', 0)}</div></div>
<div class="stat-card" style="border-left: 3px solid #ff9800"><div class="stat-label">Medium Risk</div><div class="stat-value">{agg.get('risk_distribution', {}).get('medium', 0)}</div></div>
<div class="stat-card" style="border-left: 3px solid #f44336"><div class="stat-label">High Risk</div><div class="stat-value">{agg.get('risk_distribution', {}).get('high', 0)}</div></div>
</div>
</body>
</html>"""
        return html

    def _classify_risk(self, event_risk: float) -> str:
        if event_risk < 0.3:
            return "low"
        if event_risk < 0.6:
            return "medium"
        return "high"

    def _compute_composite(self, fsv: FundamentalStateVector) -> float:
        return (
            fsv.bias_alignment * 0.3
            + fsv.macro_pressure * 0.25
            + fsv.sentiment_gradient * 0.15
            - (fsv.event_risk - 0.5) * 0.2
            + (fsv.regime_stability - 0.5) * 0.1
        )

    def _compute_health_score(self, snapshot: dict, metrics: dict) -> float:
        stab = metrics.get("stability_metrics", {})
        event_m = metrics.get("event_metrics", {})
        overall_stab = stab.get("overall_stability", 0.5)
        total_events = event_m.get("total_events_processed", 0)

        symbols_data = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
        n_symbols = len(symbols_data)
        coverage = min(1.0, n_symbols / 10.0)

        now = time.time()
        ages = []
        for sdata in symbols_data.values():
            if isinstance(sdata, dict):
                ts = sdata.get("last_update_ts", 0.0)
                if ts:
                    ages.append(now - ts)
        freshness = 0.0
        if ages:
            avg_age = statistics.mean(ages)
            freshness = max(0.0, min(1.0, 1.0 - avg_age / 3600.0))

        throughput = min(1.0, total_events / 100.0)

        health = (
            coverage * 30.0
            + freshness * 25.0
            + overall_stab * 30.0
            + throughput * 15.0
        )
        return max(0.0, min(100.0, health))


class FSVDashboardRenderer:

    @staticmethod
    def render_state_table(state_map: dict[str, FundamentalStateVector]) -> str:
        if not state_map:
            return "No symbols in state map."

        header = f"{'Symbol':<10} {'Bias':>8} {'Macro':>8} {'Sentiment':>10} {'Risk':>8} {'Stability':>10} {'Age':>8} {'Score':>8}"
        sep = "-" * len(header)
        now = time.time()
        lines = [header, sep]

        for sym, fsv in sorted(state_map.items()):
            age = now - fsv.last_update_ts if fsv.last_update_ts else 0.0
            age_str = f"{age:.0f}s" if age < 60 else f"{age / 60:.1f}m"
            if age >= 3600:
                age_str = f"{age / 3600:.1f}h"
            composite = (
                fsv.bias_alignment * 0.3
                + fsv.macro_pressure * 0.25
                + fsv.sentiment_gradient * 0.15
                - (fsv.event_risk - 0.5) * 0.2
                + (fsv.regime_stability - 0.5) * 0.1
            )
            lines.append(
                f"{sym:<10} {fsv.bias_alignment:>+8.3f} {fsv.macro_pressure:>+8.3f} "
                f"{fsv.sentiment_gradient:>+10.3f} {fsv.event_risk:>8.3f} "
                f"{fsv.regime_stability:>10.3f} {age_str:>8} {composite:>+8.3f}"
            )

        return "\n".join(lines)

    @staticmethod
    def render_summary(snapshot: dict) -> str:
        ts = snapshot.get("timestamp", 0.0)
        n = snapshot.get("symbol_count", 0)
        agg = snapshot.get("aggregate_stats", {})

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "N/A"

        lines = [
            f"FSV Dashboard Summary -- {ts_str}",
            f"Symbols tracked: {n}",
            f"Mean Bias Alignment:      {agg.get('mean_bias_alignment', 0.0):>+8.3f}",
            f"Mean Macro Pressure:      {agg.get('mean_macro_pressure', 0.0):>+8.3f}",
            f"Mean Sentiment Gradient:  {agg.get('mean_sentiment_gradient', 0.0):>+8.3f}",
            f"Mean Event Risk:          {agg.get('mean_event_risk', 0.0):>8.3f}",
            f"Mean Regime Stability:    {agg.get('mean_regime_stability', 0.0):>8.3f}",
            f"Volatility Index:         {agg.get('volatility_index', 0.0):>8.3f}",
        ]

        risk_dist = agg.get("risk_distribution", {})
        lines.append(
            f"Risk Distribution: Low={risk_dist.get('low', 0)} "
            f"Med={risk_dist.get('medium', 0)} High={risk_dist.get('high', 0)}"
        )

        return "\n".join(lines)

    @staticmethod
    def render_test_results(results: dict) -> str:
        if not results:
            return "No test results to display."

        lines = []
        lines.append("FSV Dashboard Test Results")
        lines.append("=" * 40)

        total = 0
        passed = 0

        for key, value in results.items():
            total += 1
            if value is True or (isinstance(value, dict) and value.get("passed", False)):
                passed += 1
                lines.append(f"  [PASS] {key}")
            else:
                detail = ""
                if isinstance(value, dict):
                    detail = f" -- {value.get('reason', '')}"
                lines.append(f"  [FAIL] {key}{detail}")

        lines.append("=" * 40)
        pass_rate = (passed / total) * 100.0 if total > 0 else 0.0
        lines.append(f"Passed: {passed}/{total} ({pass_rate:.1f}% pass rate)")

        return "\n".join(lines)
