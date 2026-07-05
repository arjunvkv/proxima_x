import queue
import threading
import time
import struct
from typing import Optional
from ..core.shared_memory_telemetry import TelemetryCore
from ..schema.telemetry_schema import TelemetrySnapshot


class TelemetryBridge:
    """
    Bridges structured TelemetrySnapshot objects to the shared memory binary frame.
    
    Runs in a daemon thread. Consumes snapshots from a thread-safe queue
    and writes them to the SHM frame via TelemetryCore.
    
    Usage:
        bridge = TelemetryBridge()
        bridge.start()
        
        # From trading thread:
        bridge.push(snapshot)
        
        # On shutdown:
        bridge.stop()
    """
    
    def __init__(self, shm_name: str = "proxima_telemetry", max_queue: int = 64):
        self._shm_name = shm_name
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._core: Optional[TelemetryCore] = None
    
    def start(self) -> None:
        """Start the bridge daemon thread."""
        self._core = TelemetryCore(self._shm_name, create=False)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry-bridge")
        self._thread.start()
    
    def push(self, snapshot: TelemetrySnapshot) -> None:
        """
        Push a snapshot to the bridge queue.
        
        Non-blocking — if queue is full, the snapshot is dropped.
        This is intentional: never block the trading thread.
        """
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            pass  # Drop on overflow — never block
    
    def _run(self) -> None:
        """Daemon thread: consume snapshots and write to SHM."""
        if not self._core:
            return
        
        while self._running:
            try:
                snapshot: TelemetrySnapshot = self._queue.get(timeout=1.0)
                self._write_snapshot(snapshot)
            except queue.Empty:
                continue
            except Exception:
                pass
        
        self._cleanup()
    
    def _write_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """
        Convert TelemetrySnapshot to SHM frame and write.
        
        Extracts:
        - engine_vector (list of 32 floats) -> frame engine_vector
        - system_health fields -> scalars
        """
        if not self._core:
            return
        
        # Extract from the snapshot
        ev = getattr(snapshot, 'engine_vector', list(range(32)))
        if len(ev) < 32:
            ev = list(ev) + [0.0] * (32 - len(ev))
        ev = tuple(ev[:32])
        
        # Extract scalars from system_health and other fields
        sh = getattr(snapshot, 'system_health', None)
        alignment = getattr(sh, 'stability_score', 0.0) if sh else 0.0
        stability = getattr(sh, 'system_integrity', 0.0) if sh else 0.0
        
        regime = getattr(snapshot, 'regime', None)
        regime_state = getattr(regime, 'regime_state', 0.0) if regime else 0.0
        
        tpi = getattr(snapshot, 'tpi', None)
        tpi_conf = getattr(tpi, 'live_decay', {}).get('h1_hit_rate', 50.0) / 100.0 if tpi and hasattr(tpi, 'live_decay') else 0.5
        
        shadow = getattr(snapshot, 'shadow', None)
        shadow_alignment = getattr(shadow, 'shadow_alignment', 0.0) if shadow else 0.0
        sof_score = getattr(shadow, 'sof_score', 0.0) if shadow else 0.0
        
        kill_pressure = getattr(sh, 'kill_switch_pressure', 0.0) if sh else 0.0
        rollout_progress = getattr(sh, 'rollout_progress', 0.0) if sh else 0.0
        
        exec_topo = getattr(snapshot, 'execution_topology', None)
        exec_intensity = float(getattr(exec_topo, 'execution_rate', 0.0)) if exec_topo else 0.0
        risk_exposure = float(getattr(exec_topo, 'risk_exposure', 0.0)) if exec_topo else 0.0
        
        integrity = getattr(sh, 'system_integrity', 1.0) if sh else 1.0
        
        # Write to SHM
        self._core._frame_offset = None  # Reset so begin_cycle picks correct buffer
        self._core.begin_cycle(
            getattr(snapshot, 'cycle_id', 0),
            getattr(snapshot, 'timestamp', time.time())
        )
        
        self._core.write_engine_vector(ev)
        self._core.write_scalars(
            alignment=alignment,
            stability=stability,
            entropy=regime_state,
            regime_state=regime_state,
            tpi_confidence=tpi_conf,
            shadow_alignment=shadow_alignment,
            sof_score=sof_score,
            kill_switch_pressure=kill_pressure,
            rollout_progress=rollout_progress,
            execution_intensity=exec_intensity,
            risk_exposure=risk_exposure,
            system_integrity=integrity,
        )
        
        self._core.end_cycle()
    
    def stop(self) -> None:
        """Signal the bridge thread to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
    
    def _cleanup(self) -> None:
        """Clean up SHM resources."""
        if self._core:
            try:
                self._core.unlink()
            except Exception:
                pass
