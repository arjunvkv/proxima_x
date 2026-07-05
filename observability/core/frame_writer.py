"""Low-level binary frame writer for shared memory telemetry.

Encapsulates pure binary reads/writes into shared memory buffers using
only ``struct.pack_into`` / ``struct.unpack_from``.  This is a thinner
abstraction over :class:`observability.core.shared_memory_telemetry.TelemetryCore`,
providing convenience methods for encoding / decoding the fixed 184-byte
frame layout as well as scalar uint64 / float64 fields.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple
from multiprocessing.shared_memory import SharedMemory

__all__ = [
    "FrameWriter",
]


class FrameWriter:
    """Low-level binary frame writer for shared memory telemetry.

    Provides convenience methods for encoding various telemetry data types
    into the fixed 184-byte frame layout, ensuring type safety and
    bounds checking.

    Frame layout (184 bytes)
    ------------------------
    - Offset   0:   128 bytes (32 × float32) = engine vector
    - Offset 128:    52 bytes (13 × float32) = scalars + padding
    - Offset 180:     4 bytes (padding)

    The frame is written / read with a single struct format: ``<32f13f4x``.
    """

    # Shared format strings -------------------------------------------------
    VECTOR_FMT = "<32f"      # 128 bytes
    SCALAR_FMT = "<13f4x"    # 56 bytes (13 floats + 4 byte pad = 56)
    FULL_FRAME_FMT = "<32f13f4x"  # 184 bytes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, shm: SharedMemory) -> None:
        """Bind the writer to an already-opened SharedMemory instance.

        Parameters
        ----------
        shm : SharedMemory
            An already-opened ``SharedMemory`` instance.  The underlying
            buffer **must** be at least 432 bytes (the total shared memory
            size expected by the telemetry layout), but this class itself
            only requires enough room for the offsets it is asked to write
            at.
        """
        self._buf = shm.buf

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write_engine_vector(self, offset: int, vector: Tuple[float, ...]) -> None:
        """Write 32 float32 values at *offset*, padding or truncating as needed.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.
        vector : tuple[float, ...]
            Float values to write.  If shorter than 32 the remaining slots
            are filled with 0.0; if longer only the first 32 are used.
        """
        vals = list(vector[:32])
        n_pad = 32 - len(vals)
        if n_pad > 0:
            vals.extend([0.0] * n_pad)
        struct.pack_into(self.VECTOR_FMT, self._buf, offset, *vals)

    def write_scalar_slice(self, offset: int, scalars: Tuple[float, ...]) -> None:
        """Write scalar values at *offset*, padding or truncating to 13 floats.

        The 13th float is a padding slot (not interpreted as a telemetry
        scalar).  This method is the lower-level counterpart to
        ``TelemetryCore.write_scalars(...)``.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.
        scalars : tuple[float, ...]
            Float values to write.  If fewer than 13, the remainder are
            filled with 0.0; if more than 13, only the first 13 are used.
        """
        vals = list(scalars[:13])
        n_pad = 13 - len(vals)
        if n_pad > 0:
            vals.extend([0.0] * n_pad)
        struct.pack_into(self.SCALAR_FMT, self._buf, offset, *vals)

    def write_full_frame(
        self,
        offset: int,
        vector: Tuple[float, ...],
        scalars: Tuple[float, ...],
    ) -> None:
        """Write both vector and scalars in a single struct pack (184 bytes).

        This is the fastest write path — a single ``pack_into`` call for the
        entire frame.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.
        vector : tuple[float, ...]
            Float values for the engine vector (first 32 used, remainder
            padded with 0.0).
        scalars : tuple[float, ...]
            Float values for the scalar block (first 13 used, remainder
            padded with 0.0).
        """
        vec = list(vector[:32])
        n_pad_v = 32 - len(vec)
        if n_pad_v > 0:
            vec.extend([0.0] * n_pad_v)

        scl = list(scalars[:13])
        n_pad_s = 13 - len(scl)
        if n_pad_s > 0:
            scl.extend([0.0] * n_pad_s)

        struct.pack_into(self.FULL_FRAME_FMT, self._buf, offset, *(vec + scl))

    def write_uint64(self, offset: int, value: int) -> None:
        """Write a single little-endian uint64 at *offset*.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.
        value : int
            Unsigned 64-bit integer value to write.
        """
        struct.pack_into("<Q", self._buf, offset, value)

    def write_float64(self, offset: int, value: float) -> None:
        """Write a single little-endian float64 at *offset*.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.
        value : float
            Double-precision float value to write.
        """
        struct.pack_into("<d", self._buf, offset, value)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def read_vector(self, offset: int) -> Tuple[float, ...]:
        """Read 32 float32 values starting at *offset*.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.

        Returns
        -------
        tuple[float, ...]
            A 32-element tuple of float32 values.
        """
        return struct.unpack_from(self.VECTOR_FMT, self._buf, offset)

    def read_scalars(self, offset: int) -> Tuple[float, ...]:
        """Read 13 float32 values starting at *offset* (12 real + 1 pad).

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.

        Returns
        -------
        tuple[float, ...]
            A 13-element tuple of float32 values (the last is a padding
            float and should be ignored by callers).
        """
        return struct.unpack_from(self.SCALAR_FMT, self._buf, offset)

    def read_full_frame(
        self,
        offset: int,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Read the full 184-byte frame in a single ``unpack_from`` call.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...]]
            A pair ``(vector_32, scalars_13)`` where *vector_32* is a
            32-element tuple and *scalars_13* is a 13-element tuple.
        """
        data = struct.unpack_from(self.FULL_FRAME_FMT, self._buf, offset)
        return data[:32], data[32:]

    def read_uint64(self, offset: int) -> int:
        """Read a single little-endian uint64 from *offset*.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.

        Returns
        -------
        int
            The unsigned 64-bit value.
        """
        return struct.unpack_from("<Q", self._buf, offset)[0]

    def read_float64(self, offset: int) -> float:
        """Read a single little-endian float64 from *offset*.

        Parameters
        ----------
        offset : int
            Byte offset into the shared memory buffer.

        Returns
        -------
        float
            The double-precision float value.
        """
        return struct.unpack_from("<d", self._buf, offset)[0]
