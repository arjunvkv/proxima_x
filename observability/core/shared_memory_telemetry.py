"""Shared-memory telemetry core with double-buffered lock-free IPC.

This module provides a zero-copy, lock-free telemetry transport between
Python processes using ``multiprocessing.shared_memory.SharedMemory``.

A double-buffer layout (A/B frames) with an atomic active-buffer-index swap
guarantees that a single writer can produce frames that a reader can consume
without locks or semaphores.

Layout (432 bytes total)
------------------------
    Offset   Size   Content
    0         64    Header (see TelemetryCore docstring)
    64       184    Frame buffer 0
    248      184    Frame buffer 1

Header format (``<QdQQQQ2Q``, 64 bytes)::

    +0   uint64  frame_id
    +8   float64 timestamp
    +16  uint64  active_buffer_index   (0 or 1)
    +24  uint64  write_lock_state      (0=stable, 1=writing, 2=committed)
    +32  uint64  cycle_counter
    +40  uint64  overflow_counter
    +48  uint64  [padding]
    +56  uint64  [padding]

Frame format (``<32f13f4x``, 184 bytes per buffer)::

    +0   float32[32]  engine_vector
    +128 float32[12]  alignment … system_integrity
    +176 float32[1]   [padding float]
    +180 byte[4]      [padding]
"""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple

__all__ = [
    "TelemetryCore",
    "SHM_SIZE",
    "HEADER_SIZE",
    "FRAME_SIZE",
    "BUF0_OFFSET",
    "BUF1_OFFSET",
    "FLOAT32_COUNT",
    "FLOAT32_WITH_PAD",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHM_SIZE = 432
HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248
FLOAT32_COUNT = 44  # 32 vector + 12 scalars
FLOAT32_WITH_PAD = 45  # 32 + 12 + 1 pad float


# ---------------------------------------------------------------------------
# Pre-compiled structs
# ---------------------------------------------------------------------------

_HEADER = struct.Struct("<QdQQQQ2Q")  # 64 bytes
_FRAME = struct.Struct("<32f13f4x")  # 184 bytes
_VEC = struct.Struct("<32f")  # 128 bytes (first portion of frame)
_SCALAR = struct.Struct("<13f4x")  # 56 bytes (second portion of frame)
_U64 = struct.Struct("<Q")  # single uint64


# ---------------------------------------------------------------------------
# TelemetryCore
# ---------------------------------------------------------------------------

class TelemetryCore:
    """Lock-free shared-memory telemetry writer / reader.

    **Writer workflow**::

        with TelemetryCore("my_shm", create=True) as core:
            core.begin_cycle(1, time.time())
            core.write_engine_vector(tuple(float(i) for i in range(32)))
            core.write_scalars(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                               0.9, 1.0, 1.1, 1.2)
            core.end_cycle()
            snap = core.read_snapshot()

    **Reader workflow**::

        with TelemetryCore("my_shm", create=False) as core:
            snap = core.read_snapshot()
            if snap is not None:
                # process snapshot
                ...

    Parameters
    ----------
    shm_name : str
        Name of the shared memory block.
    create : bool
        ``True`` for the writer (creates the block), ``False`` for a reader
        (attaches to an existing block).
    """

    __slots__ = (
        "_shm_name",
        "_shm",
        "_buf",
        "_frame_offset",
        "_pending_frame_id",
    )

    def __init__(self, shm_name: str = "proxima_telemetry", create: bool = True) -> None:
        self._shm_name = shm_name
        self._shm = SharedMemory(name=shm_name, create=create, size=SHM_SIZE if create else 0)
        self._buf = self._shm.buf
        self._frame_offset: Optional[int] = None
        self._pending_frame_id: Optional[int] = None

        if create:
            _HEADER.pack_into(self._buf, 0, 0, 0.0, 0, 0, 0, 0, 0, 0)

    # -- Writer interface ---------------------------------------------------

    def begin_cycle(self, cycle_id: int, timestamp: float) -> None:
        """Start writing to the inactive buffer.

        Sets *write_lock_state* to 1 (writing) and records *timestamp* and
        *cycle_counter* in the header.  The *cycle_id* is stored internally
        and will be written as *frame_id* in :meth:`end_cycle`.

        Parameters
        ----------
        cycle_id : int
            Monotonically increasing frame identifier.
        timestamp : float
            Unix timestamp for this cycle (``time.time()``).
        """
        hdr = _HEADER.unpack_from(self._buf, 0)
        active_idx = hdr[2]
        overflow_counter = hdr[5]

        inactive_idx = 1 - active_idx
        self._frame_offset = BUF0_OFFSET if inactive_idx == 0 else BUF1_OFFSET
        self._pending_frame_id = cycle_id

        _HEADER.pack_into(
            self._buf,
            0,
            hdr[0],       # frame_id (previous value, unchanged)
            timestamp,     # timestamp
            active_idx,    # keep current active buffer index
            1,             # write_lock_state = 1  (writing)
            cycle_id,      # cycle_counter
            overflow_counter,
            0,
            0,
        )

    def write_engine_vector(self, vector: Tuple[float, ...]) -> None:
        """Write the 32-dimensional engine vector to the current frame.

        If *vector* is shorter than 32 the remaining elements are set to 0.0.
        If longer, only the first 32 are used.

        Parameters
        ----------
        vector : tuple[float, …]
            At least 32 float values.
        """
        vals = list(vector[:32])
        # Silence ``extend`` static-analysis warning.
        n_pad = 32 - len(vals)
        if n_pad > 0:
            vals.extend([0.0] * n_pad)
        _VEC.pack_into(self._buf, self._frame_offset, *vals)

    def write_scalars(
        self,
        alignment: float,
        stability: float,
        entropy: float,
        regime_state: float,
        tpi_confidence: float,
        shadow_alignment: float,
        sof_score: float,
        kill_switch_pressure: float,
        rollout_progress: float,
        execution_intensity: float,
        risk_exposure: float,
        system_integrity: float,
    ) -> None:
        """Write the 12 telemetry scalars to the current frame.

        Parameters are written in the order listed to offsets 128–176 of the
        frame buffer.
        """
        _SCALAR.pack_into(
            self._buf,
            self._frame_offset + 128,
            alignment,
            stability,
            entropy,
            regime_state,
            tpi_confidence,
            shadow_alignment,
            sof_score,
            kill_switch_pressure,
            rollout_progress,
            execution_intensity,
            risk_exposure,
            system_integrity,
            0.0,  # padding float
        )

    def end_cycle(self) -> None:
        """Commit the frame and atomically swap the active buffer.

        **Write order** (critical for lock-free correctness):

        1. Write *frame_id* into the header.
        2. Set *write_lock_state* to 2 (committed).
        3. Flip *active_buffer_index* — this is the last write so that a
           reader sees a consistent state.
        """
        _U64.pack_into(self._buf, 0, self._pending_frame_id)
        _U64.pack_into(self._buf, 24, 2)  # write_lock_state = committed
        hdr = _HEADER.unpack_from(self._buf, 0)
        new_active = 1 - hdr[2]
        _U64.pack_into(self._buf, 16, new_active)

    # -- Reader interface ---------------------------------------------------

    def read_snapshot(self) -> Optional[Dict]:
        """Atomically read the latest committed frame.

        Uses a retry loop (up to 3 attempts) that compares the
        *active_buffer_index* before and after reading the frame data.
        If the index is stable the snapshot is accepted; otherwise the
        read is retried.

        Returns
        -------
        dict or None
            A dictionary with keys ``frame_id``, ``timestamp``,
            ``active_buffer_index``, ``write_lock_state``,
            ``cycle_counter``, ``overflow_counter``, ``engine_vector``,
            and the 12 scalar names, or ``None`` if no consistent snapshot
            could be obtained after 3 attempts.
        """
        for _ in range(3):
            hdr = _HEADER.unpack_from(self._buf, 0)
            active_idx = hdr[2]
            frame_offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET

            frame_data = _FRAME.unpack_from(self._buf, frame_offset)

            hdr2 = _HEADER.unpack_from(self._buf, 0)
            if hdr2[2] == active_idx:
                vector = frame_data[:32]
                s = frame_data[32:44]  # 12 scalars (skip the 13th pad float)

                return {
                    "frame_id": hdr[0],
                    "timestamp": hdr[1],
                    "active_buffer_index": active_idx,
                    "write_lock_state": hdr[3],
                    "cycle_counter": hdr[4],
                    "overflow_counter": hdr[5],
                    "engine_vector": vector,
                    "alignment": s[0],
                    "stability": s[1],
                    "entropy": s[2],
                    "regime_state": s[3],
                    "tpi_confidence": s[4],
                    "shadow_alignment": s[5],
                    "sof_score": s[6],
                    "kill_switch_pressure": s[7],
                    "rollout_progress": s[8],
                    "execution_intensity": s[9],
                    "risk_exposure": s[10],
                    "system_integrity": s[11],
                }

        return None

    # -- Lifecycle ----------------------------------------------------------

    def unlink(self) -> None:
        """Close and unlink the shared memory block.

        Safe to call multiple times.  Only has an effect on the process that
        created the block.
        """
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def __enter__(self) -> "TelemetryCore":
        return self

    def __exit__(self, *args) -> None:
        self.unlink()
