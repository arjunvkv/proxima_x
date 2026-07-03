"""
FREQUENCY REALITY AUDIT
python run_frequency_reality.py

Modes:
  report     — Print full audit report (default)
  status     — Quick status
  rq3        — Opportunity cost comparison
  rq4        — Extreme signal analysis
  rq5        — Controller simulation
  rq6        — Alpha Destruction Ratio
  rq7        — Asset-level impact
  rq8        — Regime impact
  rq9        — Leakage rate
  rq10       — Final adjudication

No modifications to trading logic. Audit only.
"""
import sys
import os
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.frequency_reality.blocked_signal_tracker import BlockedSignalTracker
from research.frequency_reality.executed_signal_tracker import ExecutedSignalTracker
from research.frequency_reality.future_return_engine import FutureReturnEngine
from research.frequency_reality.frequency_cost_analysis import FrequencyCostAnalysis
from research.frequency_reality.frequency_classifier import FrequencyClassifier
from research.frequency_reality.frequency_pipeline import FrequencyRealityPipeline


def _rates_provider(symbol):
    return None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"

    blocked = BlockedSignalTracker()
    executed = ExecutedSignalTracker()
    future = FutureReturnEngine(_rates_provider)
    analysis = FrequencyCostAnalysis(blocked, executed)
    classifier = FrequencyClassifier(analysis)
    pipeline = FrequencyRealityPipeline(blocked, executed, future, analysis, classifier)

    if mode == "report":
        print(pipeline.report())
    elif mode == "status":
        print(f"Blocked: {blocked.count()}, Executed: {executed.count()}")
    elif mode == "rq3":
        for h in ["h20", "h50", "h100"]:
            oc = analysis.opportunity_cost(h)
            print(f"\n--- {h} ---")
            print(f"Blocked: mean={oc['blocked']['mean_return']:.6f}, pp={oc['blocked']['pp']:.2%}, count={oc['blocked']['count']}")
            print(f"Executed: mean={oc['executed']['mean_return']:.6f}, pp={oc['executed']['pp']:.2%}, count={oc['executed']['count']}")
    elif mode == "rq4":
        ex = analysis.extreme_analysis()
        print(json.dumps(ex, indent=2))
    elif mode == "rq5":
        sim = analysis.controller_simulation([])
        print(f"Controller ON:  count={sim['controller_on']['count']}, pp={sim['controller_on']['pp']:.2%}")
        print(f"Controller OFF: count={sim['controller_off']['count']}, pp={sim['controller_off']['pp']:.2%}")
    elif mode == "rq6":
        adr = analysis.alpha_destruction_ratio()
        print(f"Alpha Destruction Ratio: {adr:.3f}")
        if adr < 0.15:
            print("Classification: Controller harmless")
        elif adr < 0.50:
            print("Classification: Controller removing significant alpha")
        else:
            print("Classification: Controller destroys most alpha")
    elif mode == "rq7":
        imp = analysis.asset_level_impact()
        for sym, data in imp.items():
            print(f"\n{sym}:")
            print(f"  Blocked: count={data['blocked_count']}, mean_ret={data['blocked_mean_return']:.6f}, pp={data['blocked_pp']:.2%}")
            print(f"  Executed: count={data['executed_count']}, mean_ret={data['executed_mean_return']:.6f}, pp={data['executed_pp']:.2%}")
    elif mode == "rq8":
        imp = analysis.regime_impact()
        for reg, data in imp.items():
            print(f"\n{reg}:")
            print(f"  Blocked: count={data['blocked_count']}, mean_ret={data['blocked_mean_return']:.6f}")
            print(f"  Executed: count={data['executed_count']}, mean_ret={data['executed_mean_return']:.6f}")
    elif mode == "rq9":
        le = analysis.leakage_rate()
        print(f"Blocked Total: {le['blocked_total']}")
        print(f"Profitable Blocked: {le['blocked_profitable']}")
        print(f"Leakage Rate: {le['leakage_rate']}%")
    elif mode == "rq10":
        c = classifier.classify()
        print(f"Classification: {c['classification']}")
        print(f"Confidence: {c['confidence']}")
        print(f"ADR: {c['adr']}")
    else:
        print(f"Unknown mode: {mode}")
        print("Available: report, status, rq3-rq10")


if __name__ == "__main__":
    main()
