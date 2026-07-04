from __future__ import annotations

from typing import Any

from .replay_orchestrator import ReplayOrchestrator


class ReplayEngine:
    def __init__(self) -> None:
        self.orchestrator: ReplayOrchestrator = ReplayOrchestrator()

    def run(
        self,
        symbols: list[str] | None = None,
        num_ticks: int = 1000,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        ticks: list[dict[str, Any]] = self.orchestrator.generate_synthetic_ticks(
            num_ticks, symbols
        )
        self.orchestrator.load_ticks(ticks)
        return self.orchestrator.run_replay(batch_size)

    def run_with_ticks(
        self, ticks: list[dict[str, Any]], batch_size: int = 100
    ) -> dict[str, Any]:
        self.orchestrator.load_ticks(ticks)
        return self.orchestrator.run_replay(batch_size)
