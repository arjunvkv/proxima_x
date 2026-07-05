"""
system_bootstrap.py — System lifecycle orchestrator.

Deterministic startup/shutdown controller for the complete cognitive trading system.
This is the ONLY entry point to the running system.

Import convention
-----------------
All intra-package imports are un-prefixed (e.g. ``from observability.core …``)
because the ``proxima_x/`` directory is added to ``sys.path`` by the launcher.
"""

import atexit
import asyncio
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("SystemBootstrap")


@dataclass
class BootstrapConfig:
    """Configuration for system bootstrap."""

    shm_name: str = "proxima_telemetry"
    decision_shm_name: str = "proxima_decision"
    ws_host: str = "localhost"
    ws_port: int = 8765
    demo_mode: bool = True
    log_level: str = "INFO"


class SystemBootstrap:
    """Deterministic system lifecycle orchestrator.

    Starts every subsystem in strict dependency order and shuts them down
    in the reverse order.  Safe to instantiate and call ``start()`` only
    once per process.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None):
        self.config = config or BootstrapConfig()
        self._running = False
        self._components: dict[str, Any] = {}
        self._setup_logging()
        self._register_cleanup()

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        """Start the full system in strict order.

        Sequence
        --------
        1. Initialize shared memory core
        2. Start telemetry bridge
        3. Start WebSocket server
        4. Initialize intelligence bus + all engines
        5. Initialize decision pipeline
        6. Initialize execution governor
        7. Enter main-loop wait until shutdown signal
        """
        print("=" * 60)
        print("  PROXIMA SYSTEM BOOT — STARTING")
        print("=" * 60)

        try:
            self._step("Initializing shared memory...")
            self._init_shared_memory()

            self._step("Starting telemetry bridge...")
            self._init_telemetry_bridge()

            self._step("Starting WebSocket server...")
            self._init_websocket()

            self._step("Initializing intelligence engines...")
            self._init_intelligence()

            self._step("Initializing decision pipeline...")
            self._init_decision()

            self._step("Initializing execution governor...")
            self._init_governor()

            self._running = True
            print("=" * 60)
            print(
                f"  SYSTEM RUNNING — ws://{self.config.ws_host}:{self.config.ws_port}"
            )
            print("  Press Ctrl+C to stop")
            print("=" * 60)

            # Keep main thread alive
            self._wait_for_shutdown()

        except Exception as e:
            print(f"[BOOT] FATAL: {e}")
            self.shutdown()
            raise

    def shutdown(self):
        """Shut down the full system in strict reverse order."""
        if not self._running:
            return
        print("\n" + "=" * 60)
        print("  SYSTEM SHUTDOWN — STOPPING")
        print("=" * 60)

        self._shutdown_step("Flushing final snapshots...")
        self._flush_final_state()

        self._shutdown_step("Stopping WebSocket server...")
        self._stop_websocket()

        self._shutdown_step("Stopping telemetry bridge...")
        self._stop_bridge()

        self._shutdown_step("Cleaning shared memory...")
        self._clean_shm()

        self._running = False
        print("=" * 60)
        print("  SYSTEM SHUTDOWN COMPLETE")
        print("=" * 60)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_component(self, name: str) -> Optional[Any]:
        return self._components.get(name)

    def list_components(self) -> list[str]:
        return list(self._components.keys())

    # ── Internal helpers ────────────────────────────────────────────────────

    def _step(self, msg: str) -> None:
        print(f"  ▶  {msg}")

    def _shutdown_step(self, msg: str) -> None:
        print(f"  ◀  {msg}")

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    def _register_cleanup(self) -> None:
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        print(f"\n[BOOT] Signal {signum} received")
        self.shutdown()

    # ── Initialization steps (startup order) ────────────────────────────────

    def _init_shared_memory(self) -> None:
        """Initialize TelemetryCore (creates the SHM segment)."""
        from observability.core.shared_memory_telemetry import TelemetryCore

        core = TelemetryCore(self.config.shm_name, create=True)
        self._components["telemetry_core"] = core
        logger.info("Shared memory initialized (create=True)")

    def _init_telemetry_bridge(self) -> None:
        """Initialize and start telemetry bridge daemon thread."""
        from observability.bridge.telemetry_bridge import TelemetryBridge

        bridge = TelemetryBridge(
            shm_name=self.config.shm_name, max_queue=64
        )
        bridge.start()
        self._components["bridge"] = bridge
        logger.info("Telemetry bridge started")

    def _init_websocket(self) -> None:
        """Start WebSocket server as a daemon thread (wraps async loop)."""
        from observability.ws.websocket_server import TelemetryWebSocketServer

        ws_config = {
            "host": self.config.ws_host,
            "port": self.config.ws_port,
            "shm_name": self.config.shm_name,
        }

        ws_stop_event = threading.Event()

        def _run_ws() -> None:
            """Async wrapper that runs the WebSocket server in a thread."""
            async def _start_ws():
                server = TelemetryWebSocketServer(**ws_config)
                await server.start()
                # Store server reference so stop event can trigger it
                _run_ws.server = server
                broadcast_task = asyncio.create_task(server.broadcast_loop())
                try:
                    # Wait until told to stop
                    while not ws_stop_event.is_set():
                        await asyncio.sleep(0.3)
                finally:
                    broadcast_task.cancel()
                    try:
                        await broadcast_task
                    except asyncio.CancelledError:
                        pass
                    await server.stop()

            asyncio.run(_start_ws())

        ws_thread = threading.Thread(
            target=_run_ws,
            daemon=True,
            name="ws-server",
        )
        ws_thread.start()
        # Give the thread a moment to start the server
        time.sleep(0.2)

        self._components["ws_thread"] = ws_thread
        self._components["ws_stop_event"] = ws_stop_event
        logger.info(
            "WebSocket server starting on %s:%d",
            self.config.ws_host,
            self.config.ws_port,
        )

    def _init_intelligence(self) -> None:
        """Initialize all intelligence engines and register on the bus."""
        from intelligence.intelligence_bus import IntelligenceBus
        from intelligence.regime_transition_detector import (
            RegimeTransitionDetector,
        )
        from intelligence.anomaly_detector import AnomalyDetector
        from intelligence.causal_graph_builder import CausalGraphBuilder
        from intelligence.vector_compressor import VectorCompressor
        from intelligence.system_health import SystemHealthMonitor

        bus = IntelligenceBus()
        bus.register_regime_detector(RegimeTransitionDetector())
        bus.register_anomaly_detector(AnomalyDetector())
        bus.register_causal_graph_builder(CausalGraphBuilder())
        bus.register_vector_compressor(VectorCompressor())
        bus.register_health_monitor(SystemHealthMonitor())

        self._components["intelligence_bus"] = bus
        logger.info("Intelligence layer initialized")

    def _init_decision(self) -> None:
        """Initialize decision pipeline components."""
        from decision.conflict_resolver import ConflictResolver
        from decision.meta_policy_engine import MetaPolicyEngine
        from decision.decision_synthesizer import DecisionSynthesizer
        from decision.execution_intent import ExecutionIntentTranslator
        from observability.decision_stream import DecisionStreamWriter

        self._components["conflict_resolver"] = ConflictResolver()
        self._components["meta_policy"] = MetaPolicyEngine()
        self._components["decision_synthesizer"] = DecisionSynthesizer()
        self._components["intent_translator"] = ExecutionIntentTranslator()
        self._components["decision_writer"] = DecisionStreamWriter()
        logger.info("Decision layer initialized")

    def _init_governor(self) -> None:
        """Initialize execution governor and its dependencies."""
        from execution.governor.execution_governor import ExecutionGovernor
        from execution.governor.risk_constraint_engine import (
            RiskConstraintEngine,
        )
        from execution.governor.regime_execution_matrix import (
            RegimeExecutionMatrix,
        )
        from execution.governor.intent_validator import IntentValidator
        from execution.governor.execution_finalizer import ExecutionFinalizer

        governor = ExecutionGovernor()
        governor.register_risk_constraint_engine(RiskConstraintEngine())
        governor.register_regime_matrix(RegimeExecutionMatrix())
        governor.register_intent_validator(IntentValidator())

        self._components["governor"] = governor
        self._components["finalizer"] = ExecutionFinalizer()
        logger.info("Execution governor initialized")

    def _wait_for_shutdown(self) -> None:
        """Busy-sleep until shutdown signal is received."""
        while self._running:
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                break

    # ── Shutdown steps (reverse order) ──────────────────────────────────────

    def _flush_final_state(self) -> None:
        """Write a final state dump to disk."""
        try:
            import datetime
            import json

            state = {
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "shutdown",
                "components": list(self._components.keys()),
            }
            with open("system_shutdown_state.json", "w") as f:
                json.dump(state, f, indent=2)
            logger.info("Final state dump written to system_shutdown_state.json")
        except Exception as e:
            logger.warning("Could not write final state: %s", e)

    def _stop_websocket(self) -> None:
        """Signal the WebSocket server thread to stop."""
        stop_event = self._components.get("ws_stop_event")
        if stop_event is not None:
            stop_event.set()
            logger.info("WebSocket server stop requested")
        else:
            logger.info("No WebSocket server running")

    def _stop_bridge(self) -> None:
        """Stop the telemetry bridge daemon thread."""
        bridge = self._components.get("bridge")
        if bridge is not None and hasattr(bridge, "stop"):
            try:
                bridge.stop()
                logger.info("Telemetry bridge stopped")
            except Exception as e:
                logger.warning("Error stopping bridge: %s", e)

    def _clean_shm(self) -> None:
        """Clean up shared memory segments."""
        # Close the writer-side TelemetryCore first
        core = self._components.get("telemetry_core")
        if core is not None and hasattr(core, "unlink"):
            try:
                core.unlink()
                logger.info("TelemetryCore unlinked")
            except Exception as e:
                logger.warning("TelemetryCore unlink issue: %s", e)

        # Force-unlink orphan SHM segments
        from multiprocessing import shared_memory as _shm

        for name in [self.config.shm_name, self.config.decision_shm_name]:
            try:
                seg = _shm.SharedMemory(name=name, create=False)
                seg.close()
                seg.unlink()
                logger.info("Cleaned SHM: %s", name)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("SHM cleanup issue (%s): %s", name, e)
