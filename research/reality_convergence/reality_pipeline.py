import logging
from datetime import datetime

logger = logging.getLogger("proxima_ops.rce.pipeline")


class RealityConvergencePipeline:
    def __init__(self, expectation_engine, reality_engine, divergence_detector,
                 convergence_tracker, alpha_transfer, operational_friction,
                 deployment_health, anomaly_detector, reality_classifier):
        self._exp = expectation_engine
        self._real = reality_engine
        self._div = divergence_detector
        self._conv = convergence_tracker
        self._ate = alpha_transfer
        self._friction = operational_friction
        self._health = deployment_health
        self._anomaly = anomaly_detector
        self._classifier = reality_classifier

    def report(self) -> str:
        ate = self._ate.summary()
        conv = self._conv.check()
        friction = self._friction.summary()
        div_scan = self._div.full_scan()
        health = self._health.compute()
        anomaly = self._anomaly.check()
        cls = self._classifier.classify()

        lines = []
        lines.append("=" * 52)
        lines.append(f"  REALITY CONVERGENCE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 52)
        lines.append("")
        lines.append("  ALPHA TRANSFER")
        lines.append(f"  ATE:                    {ate['ate']}")
        lines.append(f"  Classification:         {ate['ate_classification']}")
        lines.append(f"  PP Transfer:            {ate['pp_transfer']}")
        lines.append(f"  Expected Sharpe:        {ate['expected_sharpe']}")
        lines.append(f"  Observed Sharpe:        {ate['observed_sharpe']}")
        lines.append("")
        lines.append("  FREQUENCY CONVERGENCE")
        lines.append(f"  Match:                  {conv['frequency_stable']}")
        lines.append(f"  Expected:               {conv['expected_monthly']}/mo")
        lines.append(f"  Observed:               {conv['observed_monthly']}/mo")
        lines.append(f"  Error:                  {conv['frequency_error_pct']:.2%}")
        lines.append(f"  Drift:                  {conv['frequency_drift']}")
        lines.append("")
        lines.append("  EXECUTION FRICTION")
        lines.append(f"  Friction Index:         {friction['friction_index']}")
        lines.append(f"  Spread Cost:            {friction['spread_cost']}")
        lines.append(f"  Latency Cost:           {friction['latency_cost']}")
        lines.append(f"  Blocked Signal Cost:    {friction['blocked_signal_cost']}")
        lines.append(f"  Missed Opp Cost:        {friction['missed_opportunity_cost']}")
        lines.append("")
        lines.append("  DIVERGENCE")
        lines.append(f"  Divergence Score:       {div_scan.get('divergence_score', 0)}")
        lines.append(f"  Alerts:                 {div_scan.get('alert_count', 0)}")
        lines.append("")
        lines.append("  DEPLOYMENT HEALTH")
        lines.append(f"  Health Index:           {health['health_index']}")
        lines.append(f"  Status:                 {health['classification']}")
        lines.append("")
        lines.append("  ANOMALIES")
        lines.append(f"  Count:                  {anomaly['anomaly_count']}")
        lines.append(f"  Severity:               {anomaly['severity']}")
        for a in anomaly.get("anomalies", []):
            lines.append(f"   - {a['type']}: {a['detail']} ({a['severity']})")
        lines.append("")
        lines.append("  REALITY CLASSIFICATION")
        lines.append(f"  Classification:         {cls['classification']}")
        lines.append(f"  Confidence:             {cls['confidence']}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def summary(self) -> dict:
        ate = self._ate.summary()
        friction = self._friction.summary()
        div_scan = self._div.full_scan()
        health = self._health.compute()
        cls = self._classifier.classify()
        return {
            "ate": ate.get("ate", 0.0),
            "health_index": health.get("health_index", 100.0),
            "divergence_score": div_scan.get("divergence_score", 0.0),
            "friction_index": friction.get("friction_index", 0.0),
            "classification": cls.get("classification", "UNKNOWN")
        }
