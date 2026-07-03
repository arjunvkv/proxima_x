import json
from datetime import datetime

class LiveDashboard:
    def __init__(self):
        self._daily_reports: list[str] = []

    def generate_report(self, pipeline_report: dict, day: int = 0) -> str:
        lines = []
        lines.append("# LIVE VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Day: {day}")
        lines.append("")

        ds = pipeline_report.get("deployment_score", {})
        lines.append("## DEPLOYMENT SCORE")
        lines.append(f"Score: {ds.get('current_score', 0.0):.3f}")
        lines.append(f"Classification: {ds.get('classification', 'N/A')}")
        lines.append(f"Trend: {ds.get('trend', 'N/A')}")
        lines.append("")

        st = pipeline_report.get("signal_tracker", {})
        lines.append("## SIGNALS")
        lines.append(f"Total: {st.get('total_signals', 0)}")
        lines.append(f"Per Day: {st.get('avg_per_day', 0)}")
        lines.append(f"Per Week: {st.get('avg_per_week', 0)}")
        lines.append(f"Per Month: {st.get('avg_per_month', 0)}")
        lines.append(f"State Dist: {st.get('state_distribution', {})}")
        lines.append("")

        pm = pipeline_report.get("performance_monitor", {})
        lines.append("## PERFORMANCE")
        for window, label in [("last_7d", "7 Day"), ("last_30d", "30 Day"), ("last_90d", "90 Day")]:
            w = pm.get(window, {})
            lines.append(f"**{label}**")
            lines.append(f"  Sharpe: {w.get('sharpe', 0.0):.3f}")
            lines.append(f"  PP: {w.get('pp', 0.0):.3f}")
            lines.append(f"  Return: {w.get('return', 0.0):.4f}")
            lines.append(f"  DD: {w.get('dd', 0.0):.4f}")
        lines.append(f"Classification: {pm.get('classification', 'N/A')}")
        lines.append("")

        fm = pipeline_report.get("frequency_monitor", {})
        lines.append("## FREQUENCY")
        lines.append(f"Target: {fm.get('target', 30)}/mo")
        lines.append(f"Actual: {fm.get('actual_frequency', 0)}/mo")
        lines.append(f"CV: {fm.get('frequency_cv', 0):.3f}")
        lines.append(f"On Target: {fm.get('on_target', False)}")
        lines.append("")

        rm = pipeline_report.get("residual_monitor", {})
        lines.append("## RESIDUAL vs ES")
        lines.append(f"ES Predictive Power: {rm.get('es_predictive_power', 0):.3f}")
        lines.append(f"Residual Predictive Power: {rm.get('residual_predictive_power', 0):.3f}")
        lines.append(f"Ratio: {rm.get('ratio', 0):.3f}")
        lines.append(f"Residual Beats ES: {rm.get('residual_beats_es', False)}")
        lines.append(f"ES Hypothetical Sharpe: {rm.get('es_hypothetical_sharpe', 0):.3f}")
        lines.append(f"Residual Hypothetical Sharpe: {rm.get('residual_hypothetical_sharpe', 0):.3f}")
        lines.append(f"Trades Tracked: {rm.get('n_trades', 0)}")
        lines.append("")

        psm = pipeline_report.get("persistence_monitor", {})
        lines.append("## PERSISTENCE")
        lines.append(f"DA: {psm.get('directional_accuracy', 0):.3f}")
        lines.append(f"Duration Error: {psm.get('duration_error', 0):.1f}")
        lines.append(f"Forecast Decay: {psm.get('forecast_decay', 0):.3f}")
        lines.append("")

        ad = pipeline_report.get("anomaly_detector", {})
        lines.append("## ANOMALIES")
        lines.append(f"Total: {ad.get('total_anomalies', 0)}")
        lines.append(f"By Type: {ad.get('by_type', {})}")
        lines.append(f"By Severity: {ad.get('by_severity', {})}")
        lines.append("")

        dd = pipeline_report.get("drift_detector", {})
        lines.append("## DRIFT")
        lines.append(f"Stable: {dd.get('stable', 0)}")
        lines.append(f"Drifting: {dd.get('drifting', 0)}")
        lines.append(f"Broken: {dd.get('broken', 0)}")

        report = "\n".join(lines)
        self._daily_reports.append(report)
        return report

    def save_report(self, report: str, path: str = "LIVE_VALIDATION_REPORT.md"):
        with open(path, "w") as f:
            f.write(report)
        return path
