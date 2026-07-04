from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proxima_ops.orchestration.proxima_orchestrator import ProximaOrchestrator


def main() -> None:
    max_cycles = 20
    if len(sys.argv) > 1:
        try:
            max_cycles = int(sys.argv[1])
        except ValueError:
            pass
    orchestrator = ProximaOrchestrator()
    orchestrator.run(max_cycles=max_cycles)
    log = orchestrator.cycle_manager.cycle_log
    if log:
        last = log[-1]
        print(f"Cycles: {len(log)}")
        print(f"Last cycle: alignment={last['ucf_alignment']:.3f} state={last['phase6']['state']}")
        print(f"STR-E samples: {last.get('stre', {}).get('samples', 0)}")
        print(f"Kill switch: {last['phase6']['kill_switch_triggered']}")
        print(f"Multiplier: {last['phase6']['multiplier']}")
        print(f"Stability tier: {last['phase6']['stability_tier']}")


if __name__ == "__main__":
    main()
