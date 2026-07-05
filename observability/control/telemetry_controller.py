"""Telemetry Controller — lifecycle manager for the entire telemetry system.

Handles shared memory creation/attachment, WebSocket server threading,
signal-based cleanup, orphan SHM reclamation, and graceful restart.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import sys
import threading
from typing import Optional

from ..core.shared_memory_telemetry import TelemetryCore
from ..ws.websocket_server import TelemetryWebSocketServer

logger = logging.getLogger(__name__)


class TelemetryController:
    """
    Manages the entire telemetry system lifecycle.

    Handles:
    - Creating/attaching to shared memory
    - Starting the WebSocket server
    - Signal handling (SIGINT, SIGTERM) for clean shutdown
    - Orphan SHM cleanup
    - Graceful restart

    Usage:
        controller = TelemetryController()
        controller.start()
        # ... system runs ...
        controller.stop()  # Or SIGTERM triggers auto-stop

    Or as a context manager:
        with TelemetryController():
            # ... system runs ...
    """

    def __init__(
        self,
        shm_name: str = "proxima_telemetry",
        ws_host: str = "0.0.0.0",
        ws_port: int = 8765,
        ws_fps: int = 30,
        auto_cleanup: bool = True,
    ):
        self._shm_name = shm_name
        self._ws_host = ws_host
        self._ws_port = ws_port
        self._ws_fps = ws_fps
        self._auto_cleanup = auto_cleanup

        self._core: Optional[TelemetryCore] = None
        self._ws_server: Optional[TelemetryWebSocketServer] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize SHM and start the WebSocket server in a background thread."""
        # --- shared memory ---
        try:
            self._core = TelemetryCore(self._shm_name, create=True)
            logger.info(
                "Telemetry SHM '%s' created (%d bytes)",
                self._shm_name,
                TelemetryCore.SHM_SIZE,
            )
        except Exception as e:
            logger.warning("Telemetry SHM init failed (may already exist): %s", e)
            try:
                self._core = TelemetryCore(self._shm_name, create=False)
                logger.info("Telemetry SHM '%s' attached", self._shm_name)
            except Exception as e2:
                logger.error("Telemetry SHM attach failed: %s", e2)
                self._core = None

        # --- atexit cleanup guard ---
        if self._auto_cleanup:
            atexit.register(self._cleanup_shm)

        # --- Signal handlers for graceful shutdown (best-effort) ---
        # SIGINT / SIGTERM only work from the main thread; silently skip otherwise.
        try:
            signal.signal(signal.SIGINT, lambda _sig, _frame: self.stop())
            signal.signal(signal.SIGTERM, lambda _sig, _frame: self.stop())
            logger.debug("Signal handlers registered (SIGINT, SIGTERM)")
        except (ValueError, RuntimeError):
            logger.debug("Signal handler registration skipped (not in main thread)")

        # --- WebSocket daemon thread ---
        self._ws_thread = threading.Thread(
            target=self._run_ws_server,
            daemon=True,
            name="telemetry-ws",
        )
        self._ws_thread.start()

        self._running = True
        logger.info(
            "Telemetry controller started (ws://%s:%d)",
            self._ws_host,
            self._ws_port,
        )

    def stop(self) -> None:
        """Gracefully stop the telemetry system."""
        self._running = False

        # Stop the WebSocket server (best-effort via the background loop)
        if self._ws_server is not None and self._loop is not None and not self._loop.is_closed():
            try:
                # Schedule server stop + loop stop in the WS thread's event loop
                asyncio.run_coroutine_threadsafe(
                    self._stop_ws_server_and_loop(),
                    self._loop,
                )
            except Exception as e:
                logger.warning("WebSocket stop scheduling error: %s", e)

        # Wait for the WS thread to finish
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3.0)

        # Clean up shared memory
        self._cleanup_shm()

        logger.info("Telemetry controller stopped")

    # ------------------------------------------------------------------
    # SHM cleanup helpers
    # ------------------------------------------------------------------

    def _cleanup_shm(self) -> None:
        """Clean up shared memory resources."""
        if self._core is not None:
            try:
                self._core.unlink()
                logger.info("Telemetry SHM cleaned up")
            except Exception as e:
                logger.debug("SHM cleanup (already freed): %s", e)
            self._core = None

    def cleanup_orphan_shm(self) -> bool:
        """
        Clean up any orphaned shared memory blocks from crashed processes.

        Returns True if cleanup was performed.

        On Linux: /dev/shm/ contains orphaned POSIX SHM files
        On Windows: SharedMemory cleanup is handled by OS on process exit

        This is a best-effort cleanup.
        """
        platform = os.name  # 'posix' or 'nt'
        try:
            # Try to attach and unlink without creating
            test = TelemetryCore(self._shm_name, create=False)
            test.unlink()
            logger.info(
                "Cleaned up orphan SHM '%s' (platform=%s)",
                self._shm_name,
                platform,
            )
            return True
        except FileNotFoundError:
            # No orphan — good
            logger.debug("No orphan SHM '%s' found", self._shm_name)
            return False
        except Exception as e:
            logger.debug("Orphan SHM check: %s", e)
            return False

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------

    def get_shm_status(self) -> dict:
        """Get current SHM status for health checking."""
        if self._core is None:
            return {"status": "not_initialized"}

        try:
            snap = self._core.read_snapshot()
            if snap is not None:
                return {
                    "status": "active",
                    "frame_id": snap.get("frame_id", -1),
                    "timestamp": snap.get("timestamp", 0.0),
                    "active_buffer": int(snap.get("active_buffer_index", -1)),
                }
            return {"status": "no_frame"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TelemetryController":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal: WebSocket server runner (runs in daemon thread)
    # ------------------------------------------------------------------

    def _run_ws_server(self) -> None:
        """Run the async WebSocket server in a dedicated event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._ws_server = TelemetryWebSocketServer(
            host=self._ws_host,
            port=self._ws_port,
            fps=self._ws_fps,
            shm_name=self._shm_name,
        )

        try:
            self._loop.run_until_complete(self._ws_server.start())
            self._broadcast_task = self._loop.create_task(
                self._ws_server.broadcast_loop()
            )
            self._loop.run_forever()
        except Exception as e:
            logger.error("WebSocket server error: %s", e)
        finally:
            self._loop.close()

    async def _stop_ws_server_and_loop(self) -> None:
        """Coroutine that stops the WS server and then the event loop."""
        try:
            await self._ws_server.stop()
        except Exception as e:
            logger.warning("Error stopping WebSocket server: %s", e)
        finally:
            self._loop.stop()


# ======================================================================
# Standalone entry point
# ======================================================================


def main():
    """Standalone entry point for testing the telemetry system."""
    logging.basicConfig(level=logging.INFO)

    controller = TelemetryController()
    controller.start()

    print("Telemetry controller running. Press Ctrl+C to stop.")
    print(f"WebSocket: ws://{controller._ws_host}:{controller._ws_port}")

    try:
        # Keep main thread alive
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
