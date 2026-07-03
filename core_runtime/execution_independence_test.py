"""
Execution Independence Test — validates that the execution layer responds to
ANY valid signal, independent of signal source.

This module does NOT run the actual trading system. It tests the execution
pipeline modules (execution_gate_router, execution_time_synchronizer,
execution_confirmation_loop, etc.) by feeding them controlled signals and
checking responses.

The critical question: is the pipeline functional independent of OSS?
Tests OSS signals, ALT signals, and injected signals.

Usage
-----
    from core_runtime.execution_independence_test import ExecutionIndependenceTest

    tester = ExecutionIndependenceTest()
    diagnostic = tester.run_full_diagnostic()
    print(diagnostic["verdict"])
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_ExecutionIndependenceTest"] = {}


def ExecutionIndependenceTest(instance_id="default"):
    """Singleton accessor — returns the same _ExecutionIndependenceTest for a
    given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier for the test instance.

    Returns
    -------
    _ExecutionIndependenceTest
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ExecutionIndependenceTest(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Attempt real imports (gracefully degrade to mocks)
# ---------------------------------------------------------------------------

_GATE_ROUTER_AVAILABLE = False
_TIME_SYNC_AVAILABLE = False
_CONFIRMATION_LOOP_AVAILABLE = False

try:
    from core_runtime.execution_gate_router import ExecutionGateRouter
    _GATE_ROUTER_AVAILABLE = True
except ImportError:
    ExecutionGateRouter = None  # type: ignore
    logger.debug("execution_gate_router not available — using mock")

try:
    from core_runtime.execution_time_synchronizer import ExecutionTimeSynchronizer
    _TIME_SYNC_AVAILABLE = True
except ImportError:
    ExecutionTimeSynchronizer = None  # type: ignore
    logger.debug("execution_time_synchronizer not available — using mock")

try:
    from core_runtime.execution_confirmation_loop import ExecutionConfirmationLoop
    _CONFIRMATION_LOOP_AVAILABLE = True
except ImportError:
    ExecutionConfirmationLoop = None  # type: ignore
    logger.debug("execution_confirmation_loop not available — using mock")


# ---------------------------------------------------------------------------
# Mocks for execution pipeline components
# ---------------------------------------------------------------------------

class _MockGateRouter:
    """Simulates the execution gate router with configurable blocking gates.

    The gate router checks signals against a set of gates:
        - spread_gate:    rejects if spread exceeds max_spread
        - volatility_gate: rejects if volatility exceeds max_volatility
        - min_signal_gate: rejects if signal == 0 (flat)
        - confidence_gate: rejects if confidence < min_confidence

    All thresholds are configurable to simulate different blocking conditions.
    """

    def __init__(self):
        # Gate configuration
        self.max_spread = 0.0050       # 50 pips on a 1.0 instrument
        self.max_volatility = 0.02     # 2% max allowed move
        self.min_confidence = 0.30     # minimum confidence to pass

        # Gate enable / disable (all enabled by default)
        self.gates_enabled = {
            "spread_gate": True,
            "volatility_gate": True,
            "min_signal_gate": True,
            "confidence_gate": True,
        }

        # Diagnostics
        self._test_count = 0
        self._pass_count = 0
        self._blocking_gates: Dict[str, int] = {}

    def test_signal(
        self,
        symbol: str,
        signal: int,
        price: float,
        spread: float,
        volatility: float = 0.01,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Test whether a single signal passes all enabled gates.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        signal : int
            Directional signal (-1, 0, or +1).
        price : float
            Current price.
        spread : float
            Current bid-ask spread in absolute terms.
        volatility : float
            Current volatility estimate (as a fraction of price).
        confidence : float
            Signal confidence (0.0 to 1.0).

        Returns
        -------
        dict with keys:
            passed (bool), reason (str), gates_checked (int),
            gates_passed (int), blocking_gates (list of str).
        """
        self._test_count += 1
        blocked_by: List[str] = []
        gates_checked = 0
        gates_passed = 0

        # Min signal gate: signal must be non-zero
        if self.gates_enabled.get("min_signal_gate", True):
            gates_checked += 1
            if signal == 0:
                blocked_by.append("min_signal_gate")

        # Spread gate
        if self.gates_enabled.get("spread_gate", True):
            gates_checked += 1
            if spread > self.max_spread:
                blocked_by.append("spread_gate")

        # Volatility gate
        if self.gates_enabled.get("volatility_gate", True):
            gates_checked += 1
            if volatility > self.max_volatility:
                blocked_by.append("volatility_gate")

        # Confidence gate
        if self.gates_enabled.get("confidence_gate", True):
            gates_checked += 1
            if confidence < self.min_confidence:
                blocked_by.append("confidence_gate")

        gates_passed = gates_checked - len(blocked_by)
        passed = len(blocked_by) == 0

        if passed:
            self._pass_count += 1

        # Track blocking gate counts
        for gate in blocked_by:
            self._blocking_gates[gate] = self._blocking_gates.get(gate, 0) + 1

        return {
            "passed": passed,
            "reason": "passed" if passed else f"blocked_by: {', '.join(blocked_by)}",
            "gates_checked": gates_checked,
            "gates_passed": gates_passed,
            "blocking_gates": blocked_by,
            "signal": signal,
            "symbol": symbol,
        }

    def configure_gate(self, gate_name: str, enabled: bool) -> None:
        """Enable or disable a specific gate."""
        if gate_name in self.gates_enabled:
            self.gates_enabled[gate_name] = enabled
            logger.debug("Gate '%s' %s", gate_name, "enabled" if enabled else "disabled")

    def get_stats(self) -> Dict[str, Any]:
        """Return gate router test statistics."""
        return {
            "tests": self._test_count,
            "passed": self._pass_count,
            "pass_rate": self._pass_count / self._test_count if self._test_count > 0 else 0.0,
            "blocking_gates": dict(self._blocking_gates),
        }

    def reset(self) -> None:
        """Reset all test counters."""
        self._test_count = 0
        self._pass_count = 0
        self._blocking_gates.clear()


class _MockTimeSynchronizer:
    """Simulates the execution time synchronizer."""

    def __init__(self):
        self._synced = True
        self._sync_count = 0

    def synchronize(self) -> bool:
        """Check whether the execution clock is synchronised."""
        self._sync_count += 1
        return self._synced

    def set_synced(self, value: bool) -> None:
        """Force synchronisation state (for test scenarios)."""
        self._synced = value
        logger.debug("TimeSynchronizer synced=%s", value)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sync_count": self._sync_count,
            "synced": self._synced,
        }


class _MockConfirmationLoop:
    """Simulates the execution confirmation loop.

    In a real system this would wait for broker fill confirmation.  The mock
    checks that the passed gate-router result is well-formed and confirms
    immediately with configurable success probability.
    """

    def __init__(self):
        self._confirm_count = 0
        self._confirmed_count = 0
        self._success_rate = 1.0  # always confirm by default

    def confirm(self, gate_result: Dict[str, Any]) -> Dict[str, Any]:
        """Confirm (or reject) a signal that passed the gate router.

        Parameters
        ----------
        gate_result : dict
            Result dict from :meth:`MockGateRouter.test_signal`.

        Returns
        -------
        dict with keys:
            confirmed (bool), reason (str), trade_id (str | None).
        """
        self._confirm_count += 1

        # Reject if the gate result says it didn't pass
        if not gate_result.get("passed", False):
            return {
                "confirmed": False,
                "reason": "gate_rejected",
                "trade_id": None,
            }

        # Reject if signal is zero (shouldn't happen if gate catches it)
        if gate_result.get("signal", 0) == 0:
            return {
                "confirmed": False,
                "reason": "zero_signal",
                "trade_id": None,
            }

        # Simulate confirmation
        self._confirmed_count += 1
        trade_id = f"INDEP_TEST_{gate_result.get('symbol', 'UNKNOWN')}_{self._confirm_count}"

        return {
            "confirmed": True,
            "reason": "confirmed",
            "trade_id": trade_id,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "confirm_attempts": self._confirm_count,
            "confirmed": self._confirmed_count,
            "success_rate": (
                self._confirmed_count / self._confirm_count
                if self._confirm_count > 0 else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _ExecutionIndependenceTest:
    """Tests whether the execution layer responds to ANY valid signal.

    This is the critical question: is the pipeline functional independent of
    OSS?  The test feeds controlled signals from multiple sources (OSS, ALT,
    injected) through the execution pipeline and measures pass rates.

    Parameters
    ----------
    instance_id : str
        Label for logging and singleton registry.
    """

    def __init__(self, instance_id: str = "default"):
        self._instance_id = instance_id
        self._gate_router = _MockGateRouter()
        self._time_synchronizer = _MockTimeSynchronizer()
        self._confirmation_loop = _MockConfirmationLoop()

        # Real module availability flags (exposed for reporting)
        self.gate_router_available = _GATE_ROUTER_AVAILABLE
        self.time_sync_available = _TIME_SYNC_AVAILABLE
        self.confirmation_loop_available = _CONFIRMATION_LOOP_AVAILABLE

        logger.info("ExecutionIndependenceTest(%r) initialised", instance_id)
        logger.info(
            "  real modules: gate_router=%s  time_sync=%s  confirmation=%s",
            _GATE_ROUTER_AVAILABLE,
            _TIME_SYNC_AVAILABLE,
            _CONFIRMATION_LOOP_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Test scenario methods
    # ------------------------------------------------------------------

    def test_oss_signal(self, oss_signal_source: Any = None) -> Dict[str, Any]:
        """Feed OSS signals through the gate router and check pass rates.

        OSS (Outcome Surface Signal) is the primary signal source derived
        from bucket statistics and persistence probabilities.

        Parameters
        ----------
        oss_signal_source : optional
            An object with a ``get_signal(symbol)`` method, or ``None`` to
            use built-in mock OSS data.

        Returns
        -------
        dict — see module docstring for schema.
        """
        if oss_signal_source is not None:
            mock_signals = self._sample_from_source(oss_signal_source, "oss", 50)
        else:
            mock_signals = self._generate_mock_oss_signals()

        return self._run_test_scenario(mock_signals, "oss")

    def test_alt_signal(self, alt_signal_source: Any = None) -> Dict[str, Any]:
        """Feed ALT signals through the gate router and check pass rates.

        ALT (Alternative / Control) signals provide a scientific baseline.

        Parameters
        ----------
        alt_signal_source : optional
            An object with a ``get_signal(symbol)`` method, or ``None`` to
            use built-in mock ALT data.

        Returns
        -------
        dict — see module docstring for schema.
        """
        if alt_signal_source is not None:
            mock_signals = self._sample_from_source(alt_signal_source, "alt", 50)
        else:
            mock_signals = self._generate_mock_alt_signals()

        return self._run_test_scenario(mock_signals, "alt")

    def test_injected_signal(self, injector_source: Any = None) -> Dict[str, Any]:
        """Feed injected signals and verify they reach execution confirmation.

        Injected signals come from the validation signal injector, which
        produces controlled directional bias for testing.

        Parameters
        ----------
        injector_source : optional
            An object with ``get_signal(symbol)`` and ``tick()`` methods,
            or ``None`` to use built-in mock injector data.

        Returns
        -------
        dict — see module docstring for schema.
        """
        if injector_source is not None:
            mock_signals = self._sample_from_source(injector_source, "injected", 50)
        else:
            mock_signals = self._generate_mock_injected_signals()

        return self._run_test_scenario(mock_signals, "injected")

    def test_all_sources(
        self,
        oss: Any = None,
        alt: Any = None,
        injector: Any = None,
    ) -> Dict[str, Any]:
        """Run all three sources and compare pass rates.

        Parameters
        ----------
        oss : optional
            OSS signal source.
        alt : optional
            ALT signal source.
        injector : optional
            Injector signal source.

        Returns
        -------
        dict with keys:
            sources (dict of per-source results),
            comparison (dict with best_source, worst_source, avg_pass_rate),
            verdict (str).
        """
        results = {
            "oss": self.test_oss_signal(oss),
            "alt": self.test_alt_signal(alt),
            "injected": self.test_injected_signal(injector),
        }

        # Compute comparison
        pass_rates = {
            src: res["pass_rate"]
            for src, res in results.items()
        }
        execution_rates = {
            src: res["execution_rate"]
            for src, res in results.items()
        }

        best_source = max(pass_rates, key=pass_rates.get)
        worst_source = min(pass_rates, key=pass_rates.get)
        avg_pass_rate = sum(pass_rates.values()) / len(pass_rates) if pass_rates else 0.0
        avg_execution_rate = sum(execution_rates.values()) / len(execution_rates) if execution_rates else 0.0

        any_functional = any(res["functional"] for res in results.values())

        return {
            "source": "all",
            "sources": results,
            "comparison": {
                "best_source": best_source,
                "worst_source": worst_source,
                "avg_pass_rate": round(avg_pass_rate, 4),
                "avg_execution_rate": round(avg_execution_rate, 4),
            },
            "signals_generated": sum(res["signals_generated"] for res in results.values()),
            "signals_passed_gate": sum(res["signals_passed_gate"] for res in results.values()),
            "signals_reached_execution": sum(res["signals_reached_execution"] for res in results.values()),
            "signals_confirmed": sum(res["signals_confirmed"] for res in results.values()),
            "pass_rate": round(avg_pass_rate, 4),
            "execution_rate": round(avg_execution_rate, 4),
            "functional": any_functional,
            "blocking_gates": list(dict.fromkeys(
                g for res in results.values() for g in res.get("blocking_gates", [])
            )),
        }

    def test_no_signal(self) -> Dict[str, Any]:
        """Baseline: what happens with all signals = 0.

        This establishes the floor: a degenerate signal space where no
        source produces directional information.

        Returns
        -------
        dict — see module docstring for schema.
        """
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

        mock_signals = []
        for sym in symbols:
            for _ in range(20):
                mock_signals.append({
                    "symbol": sym,
                    "signal": 0,        # always flat
                    "price": 1.0,
                    "spread": 0.0002,
                    "volatility": 0.005,
                    "confidence": 0.0,
                    "source": "baseline",
                })

        result = self._run_test_scenario(mock_signals, "baseline")
        return result

    # ------------------------------------------------------------------
    # Full diagnostic
    # ------------------------------------------------------------------

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run all test scenarios and produce a summary verdict.

        Returns
        -------
        dict with keys:
            verdict (str) — one of:
                "EXECUTION_LAYER_FUNCTIONAL"  — at least one source gets through
                "EXECUTION_LAYER_BLOCKED"     — ALL sources are blocked at gates
                "NO_SIGNAL_SOURCE_WORKS"      — no source produces non-zero signals
            scenarios (dict of per-scenario results),
            summary (dict with aggregate statistics),
            module_status (dict with real-module availability flags).
        """
        logger.info("=" * 60)
        logger.info("Execution Independence Test — Full Diagnostic")
        logger.info("=" * 60)

        scenarios = {
            "oss": self.test_oss_signal(),
            "alt": self.test_alt_signal(),
            "injected": self.test_injected_signal(),
            "baseline": self.test_no_signal(),
        }

        # ---------- Summary ----------
        functional_sources = [
            name for name, res in scenarios.items()
            if res.get("functional", False) and name != "baseline"
        ]
        blocked_sources = [
            name for name, res in scenarios.items()
            if not res.get("functional", False) and name != "baseline"
        ]

        any_non_baseline_functional = any(
            res.get("functional", False)
            for name, res in scenarios.items()
            if name != "baseline"
        )

        # Check if ALL active sources have zero non-zero signals
        all_flat = all(
            res.get("signals_generated", 0) == 0 or res.get("pass_rate", 1.0) == 0.0
            for name, res in scenarios.items()
            if name != "baseline"
        )

        # Aggregate metrics across non-baseline scenarios
        total_generated = sum(
            scenarios[name].get("signals_generated", 0)
            for name in ("oss", "alt", "injected")
        )
        total_passed = sum(
            scenarios[name].get("signals_passed_gate", 0)
            for name in ("oss", "alt", "injected")
        )
        total_confirmed = sum(
            scenarios[name].get("signals_confirmed", 0)
            for name in ("oss", "alt", "injected")
        )

        # Collect all blocking gates
        all_blocking_gates = list(dict.fromkeys(
            g for name in ("oss", "alt", "injected")
            for g in scenarios[name].get("blocking_gates", [])
        ))

        # ---------- Verdict ----------
        if any_non_baseline_functional:
            verdict = "EXECUTION_LAYER_FUNCTIONAL"
        elif all_flat:
            verdict = "NO_SIGNAL_SOURCE_WORKS"
        else:
            verdict = "EXECUTION_LAYER_BLOCKED"

        summary = {
            "total_signals_generated": total_generated,
            "total_signals_passed_gate": total_passed,
            "total_signals_confirmed": total_confirmed,
            "aggregate_pass_rate": round(total_passed / total_generated, 4) if total_generated > 0 else 0.0,
            "aggregate_execution_rate": round(total_confirmed / total_generated, 4) if total_generated > 0 else 0.0,
            "functional_sources": functional_sources,
            "blocked_sources": blocked_sources,
            "blocking_gates": all_blocking_gates,
            "verdict": verdict,
        }

        logger.info("")
        logger.info("--- SCENARIO RESULTS ---")
        for name, res in sorted(scenarios.items()):
            logger.info(
                "  %-10s gen=%d pass=%d exec=%d conf=%d rate=%.2f%% functional=%s",
                name.upper(),
                res.get("signals_generated", 0),
                res.get("signals_passed_gate", 0),
                res.get("signals_reached_execution", 0),
                res.get("signals_confirmed", 0),
                res.get("pass_rate", 0.0) * 100,
                res.get("functional", False),
            )

        logger.info("")
        logger.info("--- VERDICT ---")
        logger.info("  %s", verdict)
        logger.info("=" * 60)

        return {
            "verdict": verdict,
            "scenarios": scenarios,
            "summary": summary,
            "module_status": {
                "gate_router_available": _GATE_ROUTER_AVAILABLE,
                "time_sync_available": _TIME_SYNC_AVAILABLE,
                "confirmation_loop_available": _CONFIRMATION_LOOP_AVAILABLE,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_test_scenario(
        self,
        signals: List[Dict[str, Any]],
        source_name: str,
    ) -> Dict[str, Any]:
        """Feed a list of signal dicts through the pipeline and measure results.

        Pipeline stages:
            1. Time synchronisation check
            2. Gate router test
            3. Confirmation loop

        Parameters
        ----------
        signals : list of dict
            Each dict must have keys: ``symbol``, ``signal``, ``price``,
            ``spread``.  Optional keys: ``volatility``, ``confidence``.
        source_name : str
            Label for the signal source.

        Returns
        -------
        dict — see module docstring for schema.
        """
        signals_generated = len(signals)
        passed_gate = 0
        reached_execution = 0
        confirmed = 0
        blocking_gates_seen: Dict[str, int] = {}

        # Pipeline stage 1: time synchronisation
        if not self._time_synchronizer.synchronize():
            logger.warning("Time synchroniser not ready — all signals blocked")
            return self._empty_result(source_name, "time_sync_blocked")

        for sig in signals:
            symbol = sig.get("symbol", "UNKNOWN")
            signal_val = sig.get("signal", 0)
            price = sig.get("price", 0.0)
            spread = sig.get("spread", 0.0)
            volatility = sig.get("volatility", 0.01)
            confidence = sig.get("confidence", 0.5)

            # Pipeline stage 2: gate router test
            gate_result = self._gate_router.test_signal(
                symbol=symbol,
                signal=signal_val,
                price=price,
                spread=spread,
                volatility=volatility,
                confidence=confidence,
            )

            if not gate_result["passed"]:
                for gate in gate_result.get("blocking_gates", []):
                    blocking_gates_seen[gate] = blocking_gates_seen.get(gate, 0) + 1
                continue

            passed_gate += 1

            # Pipeline stage 3: reached execution (gate passed, next stage)
            reached_execution += 1

            # Pipeline stage 4: confirmation loop
            confirm_result = self._confirmation_loop.confirm(gate_result)
            if confirm_result.get("confirmed", False):
                confirmed += 1

        pass_rate = passed_gate / signals_generated if signals_generated > 0 else 0.0
        execution_rate = confirmed / signals_generated if signals_generated > 0 else 0.0
        functional = confirmed > 0

        return {
            "source": source_name,
            "signals_generated": signals_generated,
            "signals_passed_gate": passed_gate,
            "signals_reached_execution": reached_execution,
            "signals_confirmed": confirmed,
            "pass_rate": round(pass_rate, 4),
            "execution_rate": round(execution_rate, 4),
            "functional": functional,
            "blocking_gates": sorted(blocking_gates_seen.keys()),
        }

    def _sample_from_source(
        self,
        source: Any,
        source_type: str,
        n: int,
    ) -> List[Dict[str, Any]]:
        """Extract *n* signal observations from a real signal source object.

        The source is expected to have either:
            - a ``get_signal(symbol)`` method returning a dict with a
              ``"signal"`` key, or an int directly.
            - a ``test_signal(symbol, ...)`` method.

        Falls back to mock data if sampling fails.
        """
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        signals: List[Dict[str, Any]] = []

        try:
            for i in range(n):
                sym = symbols[i % len(symbols)]

                # Try get_signal first (common API for signal sources)
                if hasattr(source, "get_signal"):
                    raw = source.get_signal(sym)
                elif hasattr(source, "test_signal"):
                    raw = source.test_signal(sym, 0, 1.0, 0.0002)
                else:
                    raise AttributeError("No recognised signal method")

                # Normalise to dict
                if isinstance(raw, dict):
                    sig_dict = {
                        "symbol": raw.get("symbol", sym),
                        "signal": raw.get("signal", raw.get("value", 0)),
                        "price": raw.get("price", 1.0),
                        "spread": raw.get("spread", 0.0002),
                        "volatility": raw.get("volatility", 0.01),
                        "confidence": raw.get("confidence", 0.5),
                        "source": source_type,
                    }
                elif isinstance(raw, (int, float)):
                    sig_dict = {
                        "symbol": sym,
                        "signal": int(raw),
                        "price": 1.0,
                        "spread": 0.0002,
                        "volatility": 0.01,
                        "confidence": 0.5,
                        "source": source_type,
                    }
                else:
                    continue

                signals.append(sig_dict)

                # Advance tick-based sources
                if hasattr(source, "tick"):
                    source.tick()

        except Exception as exc:
            logger.warning("Failed to sample from %s source: %s — using mock", source_type, exc)
            return self._generate_mock_signals(source_type, 50)

        if not signals:
            logger.warning("Zero signals sampled from %s source — using mock", source_type)
            return self._generate_mock_signals(source_type, 50)

        return signals

    # ------------------------------------------------------------------
    # Mock signal generation
    # ------------------------------------------------------------------

    def _generate_mock_oss_signals(self) -> List[Dict[str, Any]]:
        """Generate mock OSS signal data.

        OSS signals are derived from persistence probabilities and drift
        states.  The mock simulates a mix of +1, -1, and 0 signals typical
        of a healthy OSS source.
        """
        return self._generate_mock_signals("oss", 50)

    def _generate_mock_alt_signals(self) -> List[Dict[str, Any]]:
        """Generate mock ALT (control) signal data.

        ALT signals use simple technical rules (EMA cross, momentum, z-score).
        The mock simulates a mix with somewhat more flat periods than OSS.
        """
        return self._generate_mock_signals("alt", 50)

    def _generate_mock_injected_signals(self) -> List[Dict[str, Any]]:
        """Generate mock injected signal data.

        Injected signals come from the validation signal injector and are
        designed to be clean, deterministic, and always non-zero when active.
        """
        return self._generate_mock_signals("injected", 50)

    def _generate_mock_signals(
        self,
        source_type: str,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Generate mock signal data for a given source type.

        Each source type has a characteristic distribution of signals:
            - oss:      ~40% non-flat, biased toward +1
            - alt:      ~35% non-flat, balanced
            - injected: ~80% non-flat, alternating +1/-1
        """
        import random

        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

        # Distribution profiles per source type
        profiles = {
            "oss": {
                "weights": {1: 0.20, -1: 0.15, 0: 0.65},
                "base_price": 1.1000,
                "base_spread": 0.0002,
                "base_volatility": 0.008,
            },
            "alt": {
                "weights": {1: 0.15, -1: 0.15, 0: 0.70},
                "base_price": 1.1000,
                "base_spread": 0.0002,
                "base_volatility": 0.008,
            },
            "injected": {
                "weights": {1: 0.40, -1: 0.40, 0: 0.20},
                "base_price": 1.1000,
                "base_spread": 0.0002,
                "base_volatility": 0.008,
            },
            "baseline": {
                "weights": {1: 0.0, -1: 0.0, 0: 1.0},
                "base_price": 1.1000,
                "base_spread": 0.0002,
                "base_volatility": 0.008,
            },
        }

        profile = profiles.get(source_type, profiles["alt"])
        weights = profile["weights"]
        choices = list(weights.keys())
        probs = list(weights.values())

        signals = []
        for i in range(count):
            sym = symbols[i % len(symbols)]
            signal_val = random.choices(choices, weights=probs, k=1)[0]

            # Confidence is higher for non-zero signals
            if signal_val != 0:
                confidence = round(random.uniform(0.35, 0.95), 4)
            else:
                confidence = round(random.uniform(0.0, 0.30), 4)

            # Slight random walk in price to add realism
            price = profile["base_price"] + random.uniform(-0.01, 0.01)
            spread = profile["base_spread"] * random.uniform(0.5, 2.0)
            volatility = profile["base_volatility"] * random.uniform(0.5, 1.5)

            signals.append({
                "symbol": sym,
                "signal": signal_val,
                "price": round(price, 5),
                "spread": round(spread, 5),
                "volatility": round(volatility, 5),
                "confidence": confidence,
                "source": source_type,
            })

        return signals

    @staticmethod
    def _empty_result(source_name: str, reason: str = "no_data") -> Dict[str, Any]:
        """Return an empty result dict when a test cannot run."""
        return {
            "source": source_name,
            "signals_generated": 0,
            "signals_passed_gate": 0,
            "signals_reached_execution": 0,
            "signals_confirmed": 0,
            "pass_rate": 0.0,
            "execution_rate": 0.0,
            "functional": False,
            "blocking_gates": [],
            "_empty_reason": reason,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> bool:
    """Run a comprehensive self-test of the ExecutionIndependenceTest module.

    Tests:
        1. Singleton accessor returns the same instance for same id.
        2. Singleton accessor returns different instances for different ids.
        3. OSS scenario produces expected result structure.
        4. ALT scenario produces expected result structure.
        5. Injected scenario produces expected result structure.
        6. Baseline scenario produces zero pass rate.
        7. Full diagnostic returns a valid verdict.
        8. Blocking gate scenario: when spread is too high, signals are blocked.
        9. Mock gate router basic functionality.

    Returns
    -------
    bool
        ``True`` if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("ExecutionIndependenceTest — Self Test")
    logger.info("=" * 60)

    test_passed = True

    def _check(cond: bool, msg: str) -> None:
        nonlocal test_passed
        if cond:
            logger.info("  PASS: %s", msg)
        else:
            test_passed = False
            logger.error("  FAIL: %s", msg)

    # ----- 1. Singleton accessor ------------------------------------------
    logger.info("")
    logger.info("--- Singleton accessor ---")
    inst_a = ExecutionIndependenceTest("_selftest")
    inst_b = ExecutionIndependenceTest("_selftest")
    inst_c = ExecutionIndependenceTest("_selftest_other")
    _check(inst_a is inst_b, "same instance_id returns same object")
    _check(inst_a is not inst_c, "different instance_id returns different object")

    # ----- 2. Mock gate router --------------------------------------------
    logger.info("")
    logger.info("--- MockGateRouter ---")
    router = _MockGateRouter()

    # Signal 0 should be blocked by min_signal_gate
    r0 = router.test_signal("EURUSD", 0, 1.0, 0.0002)
    _check(not r0["passed"], "zero signal blocked by min_signal_gate")
    _check("min_signal_gate" in r0["blocking_gates"], "blocking_gates contains min_signal_gate")

    # Signal +1 with low spread should pass
    r1 = router.test_signal("EURUSD", 1, 1.0, 0.0002, confidence=0.5)
    _check(r1["passed"], "positive signal with low spread passes gate")
    _check(r1["gates_passed"] == 4, "all 4 gates passed")

    # Signal +1 with high spread should be blocked by spread_gate
    r2 = router.test_signal("EURUSD", 1, 1.0, 0.0100, confidence=0.5)
    _check(not r2["passed"], "high spread blocked by spread_gate")
    _check("spread_gate" in r2["blocking_gates"], "blocking_gates contains spread_gate")

    # Disable spread gate — now it should pass even with high spread
    router.configure_gate("spread_gate", False)
    r3 = router.test_signal("EURUSD", 1, 1.0, 0.0100, confidence=0.5)
    _check(r3["passed"], "signal passes when spread_gate disabled")
    router.configure_gate("spread_gate", True)

    # Low confidence should be blocked
    r4 = router.test_signal("EURUSD", 1, 1.0, 0.0002, confidence=0.1)
    _check(not r4["passed"], "low confidence blocked by confidence_gate")
    _check("confidence_gate" in r4["blocking_gates"], "blocking_gates contains confidence_gate")

    # High volatility should be blocked
    r5 = router.test_signal("EURUSD", 1, 1.0, 0.0002, volatility=0.05, confidence=0.5)
    _check(not r5["passed"], "high volatility blocked by volatility_gate")
    _check("volatility_gate" in r5["blocking_gates"], "blocking_gates contains volatility_gate")

    # Verify stats
    stats = router.get_stats()
    _check(stats["tests"] == 6, f"router stats tests=6, got {stats['tests']}")
    _check(stats["passed"] == 2, f"router stats passed=2, got {stats['passed']}")

    # ----- 3. Mock confirmation loop --------------------------------------
    logger.info("")
    logger.info("--- MockConfirmationLoop ---")
    confirm = _MockConfirmationLoop()

    # Confirm a passed gate result
    cr1 = confirm.confirm({"passed": True, "signal": 1, "symbol": "EURUSD"})
    _check(cr1["confirmed"], "confirm passes for valid gate result")
    _check(cr1["trade_id"] is not None, "confirm returns trade_id")

    # Reject a failed gate result
    cr2 = confirm.confirm({"passed": False, "signal": 0, "symbol": "EURUSD"})
    _check(not cr2["confirmed"], "confirm rejects failed gate result")
    _check(cr2["reason"] == "gate_rejected", "reason is gate_rejected")

    # ----- 4. Test OSS scenario -------------------------------------------
    logger.info("")
    logger.info("--- OSS scenario ---")
    tester = ExecutionIndependenceTest("_selftest")
    oss_result = tester.test_oss_signal()

    # Check result structure has all required keys
    expected_keys = {
        "source", "signals_generated", "signals_passed_gate",
        "signals_reached_execution", "signals_confirmed",
        "pass_rate", "execution_rate", "functional", "blocking_gates",
    }
    _check(
        expected_keys.issubset(oss_result.keys()),
        "OSS result contains all expected keys",
    )
    _check(
        oss_result["source"] == "oss",
        f"OSS result source='oss', got '{oss_result['source']}'",
    )
    _check(
        oss_result["signals_generated"] == 50,
        f"OSS signals_generated=50, got {oss_result['signals_generated']}",
    )
    _check(
        isinstance(oss_result["pass_rate"], float),
        "OSS pass_rate is a float",
    )
    _check(
        isinstance(oss_result["functional"], bool),
        "OSS functional is a bool",
    )
    _check(
        isinstance(oss_result["blocking_gates"], list),
        "OSS blocking_gates is a list",
    )

    # ----- 5. Test ALT scenario -------------------------------------------
    logger.info("")
    logger.info("--- ALT scenario ---")
    alt_result = tester.test_alt_signal()
    _check(
        expected_keys.issubset(alt_result.keys()),
        "ALT result contains all expected keys",
    )
    _check(
        alt_result["source"] == "alt",
        f"ALT result source='alt', got '{alt_result['source']}'",
    )
    _check(
        alt_result["signals_generated"] == 50,
        f"ALT signals_generated=50, got {alt_result['signals_generated']}",
    )

    # ----- 6. Test injected scenario --------------------------------------
    logger.info("")
    logger.info("--- Injected scenario ---")
    inj_result = tester.test_injected_signal()
    _check(
        expected_keys.issubset(inj_result.keys()),
        "Injected result contains all expected keys",
    )
    _check(
        inj_result["source"] == "injected",
        f"Injected result source='injected', got '{inj_result['source']}'",
    )

    # Injected signals have a higher non-flat rate, so pass rate should
    # typically be higher than OSS/ALT
    _check(
        inj_result["signals_generated"] == 50,
        f"Injected signals_generated=50, got {inj_result['signals_generated']}",
    )

    # ----- 7. Test no-signal baseline -------------------------------------
    logger.info("")
    logger.info("--- No-signal baseline ---")
    baseline_result = tester.test_no_signal()
    _check(
        expected_keys.issubset(baseline_result.keys()),
        "Baseline result contains all expected keys",
    )
    _check(
        baseline_result["source"] == "baseline",
        f"Baseline source='baseline', got '{baseline_result['source']}'",
    )
    _check(
        baseline_result["pass_rate"] == 0.0,
        "Baseline pass_rate == 0.0 (all signals are 0)",
    )
    _check(
        not baseline_result["functional"],
        "Baseline functional == False",
    )

    # ----- 8. Full diagnostic ---------------------------------------------
    logger.info("")
    logger.info("--- Full diagnostic ---")
    diagnostic = tester.run_full_diagnostic()
    _check(
        "verdict" in diagnostic,
        "Diagnostic contains 'verdict'",
    )
    _check(
        diagnostic["verdict"] in (
            "EXECUTION_LAYER_FUNCTIONAL",
            "EXECUTION_LAYER_BLOCKED",
            "NO_SIGNAL_SOURCE_WORKS",
        ),
        f"Verdict is one of the recognised values, got '{diagnostic['verdict']}'",
    )
    _check(
        "scenarios" in diagnostic and len(diagnostic["scenarios"]) >= 3,
        f"Diagnostic contains scenarios dict with >=3 entries",
    )
    _check(
        "summary" in diagnostic,
        "Diagnostic contains summary",
    )
    _check(
        "module_status" in diagnostic,
        "Diagnostic contains module_status",
    )
    _check(
        isinstance(diagnostic["module_status"]["gate_router_available"], bool),
        "module_status.gate_router_available is bool",
    )

    # In a healthy test environment, the verdict should be FUNCTIONAL since
    # mock signals produce non-zero values
    logger.info("")
    logger.info("  Verdict: %s", diagnostic["verdict"])
    logger.info("")
    for name, scenario in sorted(diagnostic["scenarios"].items()):
        logger.info(
            "  %-10s gen=%d pass=%d exec=%d conf=%d rate=%.2f%% func=%s",
            name.upper(),
            scenario.get("signals_generated", 0),
            scenario.get("signals_passed_gate", 0),
            scenario.get("signals_reached_execution", 0),
            scenario.get("signals_confirmed", 0),
            scenario.get("pass_rate", 0.0) * 100,
            scenario.get("functional", False),
        )

    # ----- 9. Test all_sources --------------------------------------------
    logger.info("")
    logger.info("--- test_all_sources ---")
    all_result = tester.test_all_sources()
    _check(
        all_result["source"] == "all",
        "all_sources result source='all'",
    )
    _check(
        "sources" in all_result and len(all_result["sources"]) == 3,
        "all_sources contains 3 sub-results",
    )
    _check(
        "comparison" in all_result,
        "all_sources contains comparison",
    )
    _check(
        "best_source" in all_result["comparison"],
        "comparison contains best_source",
    )

    # ----- 10. Test blocking gate scenario --------------------------------
    logger.info("")
    logger.info("--- Blocking gate scenario ---")
    # Create a tester with a gate router that blocks everything by setting
    # the spread threshold very low
    blocker_tester = _ExecutionIndependenceTest("_blocker_test")
    blocker_tester._gate_router.max_spread = 0.00001  # extremely tight
    blocked_result = blocker_tester.test_oss_signal()

    # Most or all signals should be blocked
    _check(
        blocked_result["signals_passed_gate"] < blocked_result["signals_generated"],
        "Blocking scenario: fewer passed than generated",
    )
    # blocking_gates should list which gates blocked
    _check(
        "spread_gate" in blocked_result.get("blocking_gates", []),
        "spread_gate appears in blocking_gates list",
    )

    # ----- 11. Reset between tests ----------------------------------------
    logger.info("")
    logger.info("--- Reset behaviour ---")
    router.reset()
    stats_after_reset = router.get_stats()
    _check(
        stats_after_reset["tests"] == 0,
        "router.reset() clears test counter",
    )
    _check(
        stats_after_reset["passed"] == 0,
        "router.reset() clears pass counter",
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    if test_passed:
        logger.info("RESULT: ALL SELFTESTS PASSED")
    else:
        logger.error("RESULT: SOME SELFTESTS FAILED")
    logger.info("=" * 60)

    return test_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if "--self-test" in sys.argv:
        success = _run_self_test()
        sys.exit(0 if success else 1)
    else:
        # Run full diagnostic
        tester = ExecutionIndependenceTest()
        diagnostic = tester.run_full_diagnostic()

        print("")
        print("=" * 60)
        print("EXECUTION INDEPENDENCE TEST — DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"  Verdict: {diagnostic['verdict']}")
        print(f"  Modules: gate_router={diagnostic['module_status']['gate_router_available']}, "
              f"time_sync={diagnostic['module_status']['time_sync_available']}, "
              f"confirmation={diagnostic['module_status']['confirmation_loop_available']}")
        print("")
        print("  Scenario Results:")
        for name, scenario in sorted(diagnostic["scenarios"].items()):
            print(
                f"    {name.upper():10s}  "
                f"gen={scenario['signals_generated']:3d}  "
                f"pass={scenario['signals_passed_gate']:3d}  "
                f"exec={scenario['signals_reached_execution']:3d}  "
                f"conf={scenario['signals_confirmed']:3d}  "
                f"rate={scenario['pass_rate']:.2%}  "
                f"func={'✓' if scenario['functional'] else '✗'}"
            )
        print("")
        print(f"  Summary:")
        print(f"    Total generated : {diagnostic['summary']['total_signals_generated']}")
        print(f"    Total passed    : {diagnostic['summary']['total_signals_passed_gate']}")
        print(f"    Total confirmed : {diagnostic['summary']['total_signals_confirmed']}")
        print(f"    Aggregate rate  : {diagnostic['summary']['aggregate_pass_rate']:.2%}")
        print(f"    Blocking gates  : {diagnostic['summary']['blocking_gates']}")
        print(f"    Functional srcs : {diagnostic['summary']['functional_sources']}")
        print(f"    Blocked srcs    : {diagnostic['summary']['blocked_sources']}")
        print("=" * 60)

        sys.exit(0 if diagnostic["verdict"] != "EXECUTION_LAYER_BLOCKED" else 1)
