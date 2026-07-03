"""
DEPLOYMENT REALITY LAB
python run_deployment_reality.py

Modes:
  report     — Full DRL report (default)
  status     — Quick status
  asr        — Alpha Survival Ratio
  exec       — Execution quality
  latency    — Latency analysis
  spread     — Spread reality
  decay      — Signal decay
  bve        — Blocked vs executed
  dd         — Drawdown forensics
  regime     — Regime reality
  classify   — Final adjudication

Observatory only. No alpha modifications.
"""
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.monitoring.performance_monitor import OpsPerformanceMonitor
from proxima_ops.monitoring.deployment_score import DeploymentScore
from research.frequency_reality.blocked_signal_tracker import BlockedSignalTracker
from research.frequency_reality.executed_signal_tracker import ExecutedSignalTracker
from research.deployment_reality.latency_analysis import LatencyAnalysis
from research.deployment_reality.spread_reality import SpreadReality
from research.deployment_reality.signal_decay import SignalDecay
from research.deployment_reality.execution_quality import ExecutionQuality
from research.deployment_reality.blocked_vs_executed import BlockedVsExecuted
from research.deployment_reality.drawdown_forensics import DrawdownForensics
from research.deployment_reality.regime_reality import RegimeReality
from research.deployment_reality.deployment_classifier import DeploymentClassifier
from research.deployment_reality.deployment_pipeline import DeploymentRealityPipeline


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"

    perf = OpsPerformanceMonitor()
    score = DeploymentScore()
    blocked = BlockedSignalTracker()
    executed = ExecutedSignalTracker()
    latency = LatencyAnalysis()
    spread = SpreadReality()
    decay = SignalDecay()
    exec_q = ExecutionQuality()
    bve = BlockedVsExecuted(blocked, executed)
    dd = DrawdownForensics()
    regime = RegimeReality()
    classifier = DeploymentClassifier(perf, score, latency, spread, decay,
                                      exec_q, bve, dd, regime)
    pipeline = DeploymentRealityPipeline(latency, spread, decay, exec_q,
                                         bve, dd, regime, classifier)

    if mode == "report":
        print(pipeline.report())
    elif mode == "status":
        c = classifier.classify()
        print(f"ASR: {c['asr']:.3f} | {c['classification']} | Trades: {c['n_trades']} | Ready: {c['adjudication_ready']}")
    elif mode == "asr":
        print(f"Alpha Survival Ratio: {classifier.alpha_survival_ratio():.3f}")
    elif mode == "exec":
        print(exec_q.summary())
    elif mode == "latency":
        print(latency.summary())
    elif mode == "spread":
        print(spread.summary())
    elif mode == "decay":
        print(decay.summary())
    elif mode == "bve":
        print(bve.compare())
    elif mode == "dd":
        print(dd.summary())
    elif mode == "regime":
        print(regime.summary())
    elif mode == "classify":
        c = classifier.classify()
        print(f"Classification: {c['classification']}")
        print(f"ASR: {c['asr']}")
        print(f"Exec Quality: {c['execution_quality']}")
        print(f"Trend: {c['score_trend']}")
        print(f"Trades: {c['n_trades']}")
        print(f"Ready: {c['adjudication_ready']}")
    else:
        print(f"Unknown mode: {mode}")
        print("Available: report, status, asr, exec, latency, spread, decay, bve, dd, regime, classify")


if __name__ == "__main__":
    main()
