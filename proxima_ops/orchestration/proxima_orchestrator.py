from __future__ import annotations

from typing import Any

from proxima_ops.orchestration.runtime_state import RuntimeState
from proxima_ops.orchestration.cycle_evaluator import CycleEvaluator
from proxima_ops.orchestration.cycle_manager import CycleManager


class ProximaOrchestrator:
    def __init__(self) -> None:
        self._runtime = RuntimeState()
        self._evaluator = CycleEvaluator(self._runtime)
        self._cycle_manager = CycleManager(self._runtime, self._evaluator)
        self._running = False

    @property
    def runtime(self) -> RuntimeState:
        return self._runtime

    @property
    def evaluator(self) -> CycleEvaluator:
        return self._evaluator

    @property
    def cycle_manager(self) -> CycleManager:
        return self._cycle_manager

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._evaluator.save_state()

    def run(self, max_cycles: int = 0) -> None:
        self.start()
        try:
            self._cycle_manager.run_loop(max_cycles=max_cycles)
        finally:
            self.stop()
