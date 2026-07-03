"""Counterfactual Activation Simulator.

Replays existing pipeline trace data and answers "what if" questions
about parameter changes without making any real changes to the running system.
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("proxima_ops.monitoring.counterfactual_simulator")


class CounterfactualSimulator:
    """Replay trace data through modified decision logic.

    Supports built-in scenarios (confirm_1, vel_off, threshold_035,
    confirm_1_vel_off) and arbitrary custom parameter combinations.
    """

    BUILTIN_SCENARIOS = ["confirm_1", "vel_off", "threshold_035", "confirm_1_vel_off"]

    def __init__(self):
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_trace(self, path: str = "state/live_pipeline_trace.jsonl") -> list[dict]:
        """Load trace entries from a JSONL file.

        Args:
            path: Path to the JSONL trace file.

        Returns:
            List of parsed entry dicts.
        """
        resolved = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            path,
        ) if not os.path.isabs(path) else path

        resolved = os.path.normpath(resolved)

        if not os.path.exists(resolved):
            self._logger.warning("Trace file not found at %s, trying relative", resolved)
            resolved = path

        entries: list[dict] = []
        with open(resolved, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        self._logger.info("Loaded %d trace entries from %s", len(entries), resolved)
        return entries

    def simulate(self, entries: list[dict], scenarios: list[str]) -> dict:
        """Run selected counterfactual scenarios.

        Args:
            entries: List of trace entry dicts.
            scenarios: List of scenario names to simulate.

        Returns:
            Dict with baseline stats and per-scenario results.
        """
        baseline_trades, baseline_cycles = self._baseline(entries)
        total_cycles = len(entries)

        result: dict[str, Any] = {
            "total_cycles": total_cycles,
            "baseline_trades": baseline_trades,
            "scenarios": {},
        }

        scenario_results = {}
        for sc in scenarios:
            sc_key = sc
            if sc == "confirm_1":
                scenario_results[sc_key] = self._simulate_confirm_1(entries, baseline_cycles)
            elif sc == "vel_off":
                scenario_results[sc_key] = self._simulate_vel_off(entries, baseline_cycles)
            elif sc == "threshold_035":
                scenario_results[sc_key] = self._simulate_threshold_035(entries, baseline_cycles)
            elif sc == "confirm_1_vel_off":
                scenario_results[sc_key] = self._simulate_confirm_1_vel_off(entries, baseline_cycles)
            else:
                self._logger.warning("Unknown scenario: %s", sc)
                continue

        result["scenarios"] = scenario_results

        if scenario_results:
            dominant = max(scenario_results, key=lambda k: scenario_results[k]["trades"])
            if scenario_results[dominant]["trades"] > 0:
                result["dominant_scenario"] = dominant

        return result

    def scenario_summary(self, entries: list[dict]) -> dict:
        """Run all built-in scenarios and return a summary table.

        Args:
            entries: List of trace entry dicts.

        Returns:
            Dict with full simulation results for all built-in scenarios.
        """
        return self.simulate(entries, self.BUILTIN_SCENARIOS)

    def custom_scenario(
        self,
        entries: list[dict],
        confirm_threshold: Optional[int] = None,
        threshold_min: Optional[float] = None,
        vel_enabled: bool = True,
    ) -> dict:
        """Run a custom counterfactual scenario with arbitrary parameters.

        Args:
            entries: List of trace entry dicts.
            confirm_threshold: Minimum confirm_map count needed for confirm_pass.
            threshold_min: Minimum confidence for threshold pass.
            vel_enabled: Whether VEL gate is active.

        Returns:
            Dict with simulation results.
        """
        baseline_trades, baseline_cycles = self._baseline(entries)
        total_cycles = len(entries)

        would_execute_cycles: list[int] = []

        for entry in entries:
            cycle = entry.get("cycle", 0)
            if self._evaluate_custom(entry, confirm_threshold, threshold_min, vel_enabled):
                would_execute_cycles.append(cycle)

        would_execute_set = set(would_execute_cycles)
        new_trades = sorted(would_execute_set - baseline_cycles)
        trades = len(would_execute_set)
        delta = trades - baseline_trades

        result: dict[str, Any] = {
            "trades": trades,
            "delta": delta,
            "trade_cycles": sorted(would_execute_cycles),
            "new_trades": new_trades,
            "params": {
                "confirm_threshold": confirm_threshold,
                "threshold_min": threshold_min,
                "vel_enabled": vel_enabled,
            },
        }

        if baseline_trades == 0:
            result["increase_pct"] = "N/A" if trades == 0 else "INF%"
        else:
            result["increase_pct"] = f"{int(round((delta / baseline_trades) * 100))}%"

        return result

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    @staticmethod
    def _baseline(entries: list[dict]) -> tuple[int, set[int]]:
        """Calculate baseline trades and their cycle numbers."""
        baseline_cycles: set[int] = set()
        for entry in entries:
            if entry.get("pipeline_funnel", {}).get("executed", 0) == 1:
                baseline_cycles.add(entry.get("cycle", 0))
        return len(baseline_cycles), baseline_cycles

    # ------------------------------------------------------------------
    # Scenario evaluators
    # ------------------------------------------------------------------

    def _simulate_confirm_1(
        self, entries: list[dict], baseline_cycles: set[int]
    ) -> dict:
        """Simulate confirm_threshold=1 scenario."""
        return self._simulate_generic(
            entries, baseline_cycles, "confirm_1",
            confirm_evaluator=self._confirm_1_passes,
        )

    def _simulate_vel_off(
        self, entries: list[dict], baseline_cycles: set[int]
    ) -> dict:
        """Simulate VEL disabled scenario."""
        return self._simulate_generic(
            entries, baseline_cycles, "vel_off",
            vel_enabled=False,
        )

    def _simulate_threshold_035(
        self, entries: list[dict], baseline_cycles: set[int]
    ) -> dict:
        """Simulate threshold_min=0.35 scenario."""
        return self._simulate_generic(
            entries, baseline_cycles, "threshold_035",
            threshold_min=0.35,
        )

    def _simulate_confirm_1_vel_off(
        self, entries: list[dict], baseline_cycles: set[int]
    ) -> dict:
        """Simulate confirm_threshold=1 AND VEL disabled."""
        return self._simulate_generic(
            entries, baseline_cycles, "confirm_1_vel_off",
            confirm_evaluator=self._confirm_1_passes,
            vel_enabled=False,
        )

    def _simulate_generic(
        self,
        entries: list[dict],
        baseline_cycles: set[int],
        scenario_name: str,
        confirm_evaluator: Optional[callable] = None,
        vel_enabled: bool = True,
        threshold_min: Optional[float] = None,
    ) -> dict:
        """Generic simulation with overridable confirm evaluator."""
        would_execute_cycles: list[int] = []

        for entry in entries:
            cycle = entry.get("cycle", 0)
            if self._evaluate_modified(
                entry, confirm_evaluator=confirm_evaluator,
                vel_enabled=vel_enabled, threshold_min=threshold_min,
            ):
                would_execute_cycles.append(cycle)

        would_execute_set = set(would_execute_cycles)
        new_trades = sorted(would_execute_set - baseline_cycles)
        trades = len(would_execute_set)
        baseline_trades = len(baseline_cycles)
        delta = trades - baseline_trades

        result: dict[str, Any] = {
            "trades": trades,
            "delta": delta,
            "trade_cycles": sorted(would_execute_cycles),
            "new_trades": new_trades,
        }

        if trades == 0 and baseline_trades == 0:
            result["increase_pct"] = "0%"
        elif baseline_trades == 0:
            result["increase_pct"] = "INF%"
        else:
            result["increase_pct"] = f"{int(round((delta / baseline_trades) * 100))}%"

        vel_was_blocking = any(
            not e.get("vel", {}).get("allowed", True)
            for e in entries
            if e.get("vel", {}).get("checked", False)
        )
        if scenario_name == "vel_off" and not vel_was_blocking:
            result["note"] = "VEL was not blocking any trades"
        elif scenario_name == "confirm_1_vel_off":
            confirm_1_trades = self._simulate_confirm_1(entries, baseline_cycles)["trades"]
            if trades == confirm_1_trades:
                result["note"] = "Same as confirm_1 — VEL not active"

        return result

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confirm_1_passes(entry: dict) -> bool:
        """Confirm passes if any confirm_map entry has count >= 1."""
        confirm_map = entry.get("confirm_gate", {}).get("confirm_map", {})
        return any(v >= 1 for v in confirm_map.values())

    @staticmethod
    def _confirm_baseline_passes(entry: dict) -> bool:
        """Confirm passes using original confirm_pass field."""
        return entry.get("confirm_gate", {}).get("confirm_pass", False)

    @staticmethod
    def _recount_threshold_passes(entry: dict, threshold_min: float) -> int:
        """Re-count threshold passes from signals_detail."""
        signals = entry.get("signals_detail", [])
        return sum(1 for s in signals if s.get("confidence", 0) >= threshold_min)

    def _evaluate_modified(
        self,
        entry: dict,
        confirm_evaluator: Optional[callable] = None,
        vel_enabled: bool = True,
        threshold_min: Optional[float] = None,
    ) -> bool:
        """Evaluate whether a cycle would execute under modified gates.

        Args:
            entry: A single trace entry.
            confirm_evaluator: Custom confirm check callable, or None for baseline.
            vel_enabled: Whether VEL gate is active.
            threshold_min: Custom threshold minimum, or None for baseline.

        Returns:
            True if the cycle would have resulted in execution.
        """
        # --- Threshold gate ---
        if threshold_min is not None:
            new_count = self._recount_threshold_passes(entry, threshold_min)
            if new_count == 0:
                return False

        # --- Confirm gate ---
        if confirm_evaluator is not None:
            if not confirm_evaluator(entry):
                return False
        else:
            if not self._confirm_baseline_passes(entry):
                return False

        # --- Governor gate ---
        governor = entry.get("governor", {})
        if not governor.get("authorized", False):
            return False

        # --- VEL gate ---
        if vel_enabled:
            vel = entry.get("vel", {})
            if not vel.get("allowed", True):
                return False

        # --- Circuit breaker gate ---
        cb = entry.get("circuit_breaker", {})
        if not cb.get("allowed", True):
            return False

        return True

    def _evaluate_custom(
        self,
        entry: dict,
        confirm_threshold: Optional[int] = None,
        threshold_min: Optional[float] = None,
        vel_enabled: bool = True,
    ) -> bool:
        """Evaluate a custom parameter combination.

        Args:
            entry: A single trace entry.
            confirm_threshold: Minimum value in confirm_map for confirm_pass.
            threshold_min: Minimum confidence for threshold pass.
            vel_enabled: Whether VEL gate is active.

        Returns:
            True if the cycle would have resulted in execution.
        """
        # --- Threshold gate ---
        if threshold_min is not None:
            new_count = self._recount_threshold_passes(entry, threshold_min)
            if new_count == 0:
                return False

        # --- Confirm gate ---
        if confirm_threshold is not None:
            confirm_map = entry.get("confirm_gate", {}).get("confirm_map", {})
            if not any(v >= confirm_threshold for v in confirm_map.values()):
                return False
        else:
            if not self._confirm_baseline_passes(entry):
                return False

        # --- Governor gate ---
        governor = entry.get("governor", {})
        if not governor.get("authorized", False):
            return False

        # --- VEL gate ---
        if vel_enabled:
            vel = entry.get("vel", {})
            if not vel.get("allowed", True):
                return False

        # --- Circuit breaker gate ---
        cb = entry.get("circuit_breaker", {})
        if not cb.get("allowed", True):
            return False

        return True


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    """Print a formatted summary table."""
    print("=" * 60)
    print("  COUNTERFACTUAL ACTIVATION SIMULATOR — SUMMARY")
    print("=" * 60)
    print(f"  Total cycles analyzed:  {result['total_cycles']}")
    print(f"  Baseline trades:        {result['baseline_trades']}")
    print(f"  Dominant scenario:      {result.get('dominant_scenario', 'N/A')}")
    print()
    print(f"  {'Scenario':<22} {'Trades':<8} {'Delta':<8} {'% Inc':<8}  Notes")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    for sc_name, sc_data in result.get("scenarios", {}).items():
        trades = sc_data["trades"]
        delta = sc_data["delta"]
        pct = sc_data.get("increase_pct", "0%")
        note = sc_data.get("note", "")
        new_str = ", ".join(str(c) for c in sc_data.get("new_trades", [])[:5])
        if len(sc_data.get("new_trades", [])) > 5:
            new_str += f" ... (+{len(sc_data['new_trades']) - 5} more)"
        print(f"  {sc_name:<22} {trades:<8} {delta:<8} {pct:<8}  {note}")
        if new_str:
            print(f"  {'':22} {'':8} {'':8} {'':8}  new trades: [{new_str}]")

    print("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sim = CounterfactualSimulator()
    entries = sim.load_trace()
    if not entries:
        print("No trace entries found.")
        return

    result = sim.scenario_summary(entries)
    _print_summary(result)


if __name__ == "__main__":
    main()
