"""
REALITY CONVERGENCE ENGINE
python run_reality_convergence.py

Modes:
  report     — Full RCE report (default)
  status     — Quick status
  ate        — Alpha Transfer Efficiency
  freq       — Frequency convergence
  friction   — Operational friction
  divergence — Divergence scan
  health     — Deployment health
  anomaly    — Anomaly detection
  classify   — Final reality classification

Observation and diagnosis only. No modifications.
"""
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.reality_convergence.expectation_engine import ExpectationEngine
from research.reality_convergence.reality_engine import RealityEngine
from research.reality_convergence.divergence_detector import DivergenceDetector
from research.reality_convergence.convergence_tracker import ConvergenceTracker
from research.reality_convergence.alpha_transfer import AlphaTransfer
from research.reality_convergence.operational_friction import OperationalFriction
from research.reality_convergence.deployment_health import DeploymentHealth
from research.reality_convergence.anomaly_detector import AnomalyDetector
from research.reality_convergence.reality_classifier import RealityClassifier
from research.reality_convergence.reality_pipeline import RealityConvergencePipeline


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"

    exp = ExpectationEngine()
    real = RealityEngine()
    div = DivergenceDetector(exp, real)
    conv = ConvergenceTracker(exp, real)
    ate = AlphaTransfer(exp, real)
    friction = OperationalFriction(None, None, real)
    health = DeploymentHealth(alpha_transfer=ate, friction=friction, convergence=conv, divergence=div)
    anomaly = AnomalyDetector()
    cls = RealityClassifier(alpha_transfer=ate, divergence=div, friction=friction, health=health, anomaly=anomaly)
    pipeline = RealityConvergencePipeline(exp, real, div, conv, ate, friction, health, anomaly, cls)

    if mode == "report":
        print(pipeline.report())
    elif mode == "status":
        c = cls.classify()
        h = health.compute()
        print(f"{c['classification']} | ATE={c['ate']} | Health={h['health_index']} | Conf={c['confidence']}")
    elif mode == "ate":
        print(ate.summary())
    elif mode == "freq":
        print(conv.check())
    elif mode == "friction":
        print(friction.summary())
    elif mode == "divergence":
        s = div.full_scan()
        print(f"Score: {s.get('divergence_score')}, Alerts: {s.get('alert_count')}")
        for a in div.alerts():
            print(f"  {a.get('metric')}: expected={a.get('expected')}, observed={a.get('observed')}, dev={a.get('deviation_pct'):.2%}")
    elif mode == "health":
        print(health.compute())
    elif mode == "anomaly":
        print(anomaly.check())
    elif mode == "classify":
        c = cls.classify()
        print(f"Classification: {c['classification']}")
        print(f"Confidence: {c['confidence']}")
        print(f"ATE: {c['ate']}")
        print(f"Divergence: {c['divergence_score']}")
        print(f"Friction: {c['friction_index']}")
        print(f"Health: {c['health_status']}")
        print(f"Anomaly: {c['anomaly_severity']}")
    else:
        print(f"Unknown mode: {mode}")
        print("Available: report, status, ate, freq, friction, divergence, health, anomaly, classify")


if __name__ == "__main__":
    main()
