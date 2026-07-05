"""
vector_compressor.py — PCA-based compression of the 32D engine vector.

Reduces the raw 32D telemetry vector into 8-12 latent dimensions using PCA
(from scratch — no numpy).  Creates regime, behaviour, and drift embeddings
for causal interpretation of the telemetry stream.

Frame layout (432 bytes) — matches ``anomaly_detector.py``::

    Offset   Size   Content
    0         64    Header  (<QdQQQQ2Q)
    64       184    Frame buffer 0  (<32f13f4x)
    248      184    Frame buffer 1  (<32f13f4x)

The engine vector is the first 32 floats in the active frame buffer.
"""

from __future__ import annotations

import math
import random
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CompressedStateVector",
    "VectorCompressor",
]

# ---------------------------------------------------------------------------
# Constants — frame layout (mirrors anomaly_detector.py)
# ---------------------------------------------------------------------------

_HEADER_FORMAT = struct.Struct("<QdQQQQ2Q")  # 64 bytes
_FRAME_FORMAT = struct.Struct("<32f13f4x")   # 184 bytes

HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248

# Dimensionality of the raw engine vector
_ENGINE_DIMS = 32

# ---------------------------------------------------------------------------
# CompressedStateVector
# ---------------------------------------------------------------------------


@dataclass
class CompressedStateVector:
    """Latent representation of the engine telemetry vector.

    Attributes
    ----------
    dims : int
        Number of latent dimensions (typically 8).
    components : list[float]
        The latent component values (length == ``dims``).
    explained_variance : float
        Cumulative variance ratio explained by the retained components (0-1).
    embedding_type : str
        One of ``"regime"``, ``"behavior"``, ``"drift"``.
    timestamp : float
        Unix timestamp of the source frame.
    """
    dims: int
    components: List[float]
    explained_variance: float
    embedding_type: str
    timestamp: float


# ---------------------------------------------------------------------------
# VectorCompressor
# ---------------------------------------------------------------------------


class VectorCompressor:
    """Reduce the 32D engine vector to a compact latent representation.

    Uses PCA trained incrementally from raw 432-byte SHM frames.  All
    linear algebra (mean-centering, covariance, power iteration, projection)
    is implemented in pure Python — no external dependencies.

    Parameters
    ----------
    target_dims : int
        Number of latent dimensions to retain (default 8).
    max_samples : int
        Maximum number of frames retained for PCA training (default 1000).
    """

    def __init__(self, target_dims: int = 8, max_samples: int = 1000) -> None:
        if target_dims < 1 or target_dims > _ENGINE_DIMS:
            raise ValueError(
                f"target_dims must be in [1, {_ENGINE_DIMS}], got {target_dims}"
            )
        if max_samples < target_dims + 1:
            raise ValueError(
                f"max_samples ({max_samples}) must be > target_dims ({target_dims})"
            )

        self._target_dims = target_dims
        self._max_samples = max_samples

        # Ring buffer of extracted engine vectors (list of 32 floats each)
        self._samples: Deque[List[float]] = deque(maxlen=max_samples)

        # PCA state (populated by .train())
        self._mean: List[float] = [0.0] * _ENGINE_DIMS
        self._eigenvectors: List[List[float]] = []  # shape (target_dims, 32)
        self._eigenvalues: List[float] = []  # length target_dims
        self._explained_variance: float = 0.0
        self._trained: bool = False

        # Behaviour-mode recent window (last 200)
        self._behaviour_window: Deque[List[float]] = deque(maxlen=200)

        # Cached latest timestamp and engine vector
        self._latest_timestamp: float = 0.0
        self._latest_vector: List[float] = [0.0] * _ENGINE_DIMS

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, frame: bytes) -> None:
        """Parse a raw 432-byte SHM frame and accumulate the engine vector.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore``.
        """
        if len(frame) < HEADER_SIZE + FRAME_SIZE:
            return  # malformed — silently ignore

        # Parse header to determine active buffer
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = hdr[2]
        self._latest_timestamp = hdr[1]

        # Read the active frame buffer
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)

        # Extract the 32 engine floats
        engine_vector = [float(v) for v in raw[:_ENGINE_DIMS]]
        self._latest_vector = engine_vector

        # Accumulate into the training buffer
        self._samples.append(engine_vector)

        # Also accumulate into the behaviour window
        self._behaviour_window.append(engine_vector)

    def train(self) -> Dict[str, float]:
        """Run incremental PCA on accumulated samples.

        Performs mean-centering, covariance computation, and power-iteration
        eigen-decomposition to extract the top ``target_dims`` components.

        Returns
        -------
        dict
            Keys ``"explained_variance"`` (float) and ``"components_shape"``
            (str, e.g. ``"(8, 32)"``).
        """
        n = len(self._samples)
        if n < self._target_dims + 1:
            raise RuntimeError(
                f"Need at least {self._target_dims + 1} samples for PCA, "
                f"have {n}"
            )

        # 1. Build the data matrix (list of 32D vectors) and mean-center
        X = list(self._samples)  # shape (n, 32)
        self._mean = self._compute_mean(X)
        Xc = self._center(X, self._mean)  # shape (n, 32)

        # 2. Compute the 32×32 covariance matrix
        #    C = (1/(n-1)) * Xc.T @ Xc
        C = self._covariance_matrix(Xc, n)

        # 3. Power iteration — extract top-K eigenvectors / eigenvalues
        trace_C = self._trace(C)
        eigenvectors: List[List[float]] = []
        eigenvalues: List[float] = []

        for k in range(self._target_dims):
            eigval, eigvec = self._power_iterate(C)
            eigenvalues.append(eigval)
            eigenvectors.append(eigvec)

            # Deflate: C = C - λ * v @ v.T
            C = self._deflate(C, eigval, eigvec)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

        # 4. Explained variance ratio = sum(top K eigenvalues) / trace(original C)
        #    trace(original C) = trace of the pre-deflation C
        #    We saved it before deflation, but after deflation trace changes.
        #    Since sum(all eigenvalues) = trace(original C), we use the
        #    pre-computed trace.
        sum_top_k = sum(eigenvalues)
        self._explained_variance = sum_top_k / trace_C if trace_C > 0 else 0.0

        self._trained = True

        return {
            "explained_variance": self._explained_variance,
            "components_shape": f"({self._target_dims}, {_ENGINE_DIMS})",
        }

    def compress(
        self,
        frame: bytes,
        embedding_type: str = "regime",
    ) -> CompressedStateVector:
        """Project the latest 32D vector from *frame* into latent space.

        Parameters
        ----------
        frame : bytes
            Raw 432-byte SHM frame.
        embedding_type : str
            One of ``"regime"``, ``"behavior"``, ``"drift"``.

        Returns
        -------
        CompressedStateVector
        """
        if not self._trained:
            raise RuntimeError("VectorCompressor must be trained before compressing")

        if embedding_type not in ("regime", "behavior", "drift"):
            raise ValueError(
                f"embedding_type must be 'regime', 'behavior', or 'drift', "
                f"got {embedding_type!r}"
            )

        # Parse frame to get engine vector + timestamp
        vector, timestamp = self._parse_frame(frame)

        # Obtain the appropriate mean vector for this embedding type
        if embedding_type == "regime":
            mean = self._mean
        elif embedding_type == "behavior":
            mean = self._compute_mean(list(self._behaviour_window)) if self._behaviour_window else self._mean
        elif embedding_type == "drift":
            # For drift, the mean is set to the baseline mean (caller provides
            # baseline via get_drift_embedding).  We default to the global mean
            # here; the convenience method overrides.
            mean = self._mean
        else:
            mean = self._mean  # unreachable

        # Center the vector
        centered = [vector[i] - mean[i] for i in range(_ENGINE_DIMS)]

        # Project onto eigenvectors:  z_j = sum_i centered[i] * W[i][j]
        components = self._project(centered)

        return CompressedStateVector(
            dims=self._target_dims,
            components=components,
            explained_variance=self._explained_variance,
            embedding_type=embedding_type,
            timestamp=timestamp,
        )

    def get_regime_embedding(self) -> CompressedStateVector:
        """Convenience: compress the latest sample with regime embedding type.

        Returns
        -------
        CompressedStateVector
        """
        if not self._samples:
            raise RuntimeError("No samples available — call .feed() first")

        # Build a synthetic frame from the latest stored vector
        # (We don't have the original bytes, so we reconstruct a plausible
        #  frame for parsing.  The timestamp comes from our cached value.)
        return self._compress_vector(
            self._latest_vector,
            self._latest_timestamp,
            embedding_type="regime",
        )

    def get_drift_embedding(self, baseline_frames: List[bytes]) -> CompressedStateVector:
        """Compress the latest sample as a drift from *baseline_frames*.

        The drift embedding is the projection of ``(latest - baseline_mean)``,
        capturing how the current state deviates from a reference period.

        Parameters
        ----------
        baseline_frames : list[bytes]
            Raw 432-byte frames defining the baseline period.

        Returns
        -------
        CompressedStateVector
        """
        if not baseline_frames:
            raise ValueError("baseline_frames must not be empty")
        if not self._samples:
            raise RuntimeError("No samples available — call .feed() first")

        # Compute baseline mean from the parsed frames
        baseline_vectors: List[List[float]] = []
        for fr in baseline_frames:
            vec, _ = self._parse_frame(fr)
            baseline_vectors.append(vec)
        baseline_mean = self._compute_mean(baseline_vectors)

        # Latest vector minus baseline mean
        centered = [
            self._latest_vector[i] - baseline_mean[i]
            for i in range(_ENGINE_DIMS)
        ]
        components = self._project(centered)

        return CompressedStateVector(
            dims=self._target_dims,
            components=components,
            explained_variance=self._explained_variance,
            embedding_type="drift",
            timestamp=self._latest_timestamp,
        )

    def reconstruct(self, compressed: CompressedStateVector) -> List[float]:
        """Reconstruct a 32D vector from its compressed representation.

        Parameters
        ----------
        compressed : CompressedStateVector
            The latent representation to expand.

        Returns
        -------
        list[float]
            Reconstructed 32D engine vector.
        """
        if not self._trained:
            raise RuntimeError("VectorCompressor must be trained before reconstructing")

        if len(compressed.components) != self._target_dims:
            raise ValueError(
                f"Expected {self._target_dims} components, "
                f"got {len(compressed.components)}"
            )

        # Determine the appropriate mean for this embedding type
        if compressed.embedding_type == "regime":
            mean = self._mean
        elif compressed.embedding_type == "behavior":
            mean = self._compute_mean(list(self._behaviour_window)) if self._behaviour_window else self._mean
        elif compressed.embedding_type == "drift":
            # For drift, we cannot perfectly reconstruct without the baseline
            # mean.  Fall back to the global mean (this is an acknowledged
            # limitation for drift embeddings).
            mean = self._mean
        else:
            mean = self._mean

        # Reconstruct: x_recon[i] = mean[i] + sum_j components[j] * W[j][i]
        reconstructed = list(mean)
        for j in range(self._target_dims):
            ev = self._eigenvectors[j]
            cj = compressed.components[j]
            for i in range(_ENGINE_DIMS):
                reconstructed[i] += cj * ev[i]

        return reconstructed

    def reconstruction_error(
        self,
        original: bytes,
        compressed: CompressedStateVector,
    ) -> float:
        """Compute RMSE between the original and reconstructed 32D vector.

        Parameters
        ----------
        original : bytes
            Raw 432-byte SHM frame.
        compressed : CompressedStateVector
            The compressed representation (presumably from the same frame).

        Returns
        -------
        float
            Root mean squared error across all 32 dimensions.
        """
        original_vector, _ = self._parse_frame(original)
        reconstructed = self.reconstruct(compressed)

        if len(original_vector) != len(reconstructed):
            raise ValueError("Vector dimension mismatch in reconstruction error")

        n = len(original_vector)
        sq_error = sum(
            (original_vector[i] - reconstructed[i]) ** 2
            for i in range(n)
        )
        return math.sqrt(sq_error / n)

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_frame(frame: bytes) -> Tuple[List[float], float]:
        """Extract the 32D engine vector and timestamp from a raw frame.

        Returns
        -------
        (engine_vector, timestamp)
        """
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = hdr[2]
        timestamp: float = hdr[1]

        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)

        engine_vector = [float(v) for v in raw[:_ENGINE_DIMS]]
        return engine_vector, timestamp

    @staticmethod
    def _compute_mean(vectors: Sequence[List[float]]) -> List[float]:
        """Compute the element-wise mean across a list of vectors.

        Parameters
        ----------
        vectors : sequence of list[float]
            Each inner list must have length ``_ENGINE_DIMS``.

        Returns
        -------
        list[float]
            Mean vector of length ``_ENGINE_DIMS``.
        """
        n = len(vectors)
        if n == 0:
            return [0.0] * _ENGINE_DIMS
        mean = [0.0] * _ENGINE_DIMS
        for vec in vectors:
            for i in range(_ENGINE_DIMS):
                mean[i] += vec[i]
        inv_n = 1.0 / n
        for i in range(_ENGINE_DIMS):
            mean[i] *= inv_n
        return mean

    @staticmethod
    def _center(
        X: Sequence[List[float]],
        mean: List[float],
    ) -> List[List[float]]:
        """Return a new matrix with the mean subtracted from each row."""
        n = len(X)
        Xc: List[List[float]] = [[0.0] * _ENGINE_DIMS for _ in range(n)]
        for row in range(n):
            for col in range(_ENGINE_DIMS):
                Xc[row][col] = X[row][col] - mean[col]
        return Xc

    @staticmethod
    def _covariance_matrix(Xc: List[List[float]], n: int) -> List[List[float]]:
        """Compute the 32×32 covariance matrix.

        ``C[i][j] = (1/(n-1)) * sum_k Xc[k][i] * Xc[k][j]``
        """
        inv_n1 = 1.0 / (n - 1) if n > 1 else 1.0
        C: List[List[float]] = [[0.0] * _ENGINE_DIMS for _ in range(_ENGINE_DIMS)]
        for i in range(_ENGINE_DIMS):
            for j in range(i, _ENGINE_DIMS):
                s = 0.0
                for k in range(n):
                    s += Xc[k][i] * Xc[k][j]
                C[i][j] = s * inv_n1
                C[j][i] = C[i][j]  # symmetric
        return C

    @staticmethod
    def _trace(mat: List[List[float]]) -> float:
        """Trace of a square matrix (sum of diagonal elements)."""
        d = len(mat)
        t = 0.0
        for i in range(d):
            t += mat[i][i]
        return t

    @staticmethod
    def _mat_vec_mul(mat: List[List[float]], vec: List[float]) -> List[float]:
        """Multiply a square matrix by a vector: ``result[i] = sum_j mat[i][j] * vec[j]``."""
        d = len(mat)
        out = [0.0] * d
        for i in range(d):
            s = 0.0
            row = mat[i]
            for j in range(d):
                s += row[j] * vec[j]
            out[i] = s
        return out

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        """Dot product of two vectors."""
        return sum(ai * bi for ai, bi in zip(a, b))

    @staticmethod
    def _norm(v: List[float]) -> float:
        """L2 norm of a vector."""
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def _normalize(v: List[float]) -> List[float]:
        """Return a unit vector in the direction of *v*."""
        norm = VectorCompressor._norm(v)
        if norm < 1e-15:
            return [0.0] * len(v)
        inv_norm = 1.0 / norm
        return [x * inv_norm for x in v]

    @staticmethod
    def _power_iterate(
        C: List[List[float]],
        max_iter: int = 200,
        tol: float = 1e-10,
    ) -> Tuple[float, List[float]]:
        """Extract the dominant eigenvalue and eigenvector via power iteration.

        Parameters
        ----------
        C : list[list[float]]
            32×32 symmetric matrix.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance (L2 change in eigenvector).

        Returns
        -------
        (eigenvalue, eigenvector)
        """
        d = len(C)
        # Random initial vector
        v = [random.gauss(0.0, 1.0) for _ in range(d)]
        v = VectorCompressor._normalize(v)

        eigenvalue_old = 0.0
        for _ in range(max_iter):
            # C @ v
            Cv = VectorCompressor._mat_vec_mul(C, v)

            # Rayleigh quotient: λ = (v.T @ C @ v) / (v.T @ v)
            vT_Cv = VectorCompressor._dot(v, Cv)
            vT_v = VectorCompressor._dot(v, v)
            eigenvalue = vT_Cv / vT_v if vT_v > 0 else 0.0

            # Normalize
            v_new = VectorCompressor._normalize(Cv)

            # Check convergence (L2 change)
            diff = VectorCompressor._norm(
                [v_new[i] - v[i] for i in range(d)]
            )
            v = v_new

            if diff < tol and abs(eigenvalue - eigenvalue_old) < tol:
                break

            eigenvalue_old = eigenvalue

        # Ensure eigenvector has consistent sign (make the largest-magnitude
        # element positive for determinism)
        max_abs_idx = max(range(d), key=lambda i: abs(v[i]))
        if v[max_abs_idx] < 0:
            v = [-x for x in v]

        return eigenvalue, v

    @staticmethod
    def _deflate(
        C: List[List[float]],
        eigenvalue: float,
        eigenvector: List[float],
    ) -> List[List[float]]:
        """Deflate a symmetric matrix: ``C = C - λ * v @ v.T``.

        This removes the contribution of the known eigenpair so that the
        next power iteration converges to the next dominant mode.
        """
        d = len(C)
        C_new: List[List[float]] = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                C_new[i][j] = C[i][j] - eigenvalue * eigenvector[i] * eigenvector[j]
        return C_new

    def _project(self, centered: List[float]) -> List[float]:
        """Project a mean-centered 32D vector onto the eigenvector matrix.

        ``z_j = sum_i centered[i] * W[j][i]``
        """
        components = [0.0] * self._target_dims
        for j in range(self._target_dims):
            ev = self._eigenvectors[j]
            s = 0.0
            for i in range(_ENGINE_DIMS):
                s += centered[i] * ev[i]
            components[j] = s
        return components

    def _compress_vector(
        self,
        vector: List[float],
        timestamp: float,
        embedding_type: str = "regime",
    ) -> CompressedStateVector:
        """Internal: compress a raw (already-parsed) engine vector."""
        if not self._trained:
            raise RuntimeError("VectorCompressor must be trained before compressing")

        # Choose the mean
        if embedding_type == "regime":
            mean = self._mean
        elif embedding_type == "behavior":
            mean = self._compute_mean(list(self._behaviour_window)) if self._behaviour_window else self._mean
        elif embedding_type == "drift":
            mean = self._mean
        else:
            mean = self._mean

        centered = [vector[i] - mean[i] for i in range(_ENGINE_DIMS)]
        components = self._project(centered)

        return CompressedStateVector(
            dims=self._target_dims,
            components=components,
            explained_variance=self._explained_variance,
            embedding_type=embedding_type,
            timestamp=timestamp,
        )

    # ── Inspection helpers ──────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """Whether PCA has been trained on accumulated samples."""
        return self._trained

    @property
    def num_samples(self) -> int:
        """Number of frames accumulated."""
        return len(self._samples)

    @property
    def target_dims(self) -> int:
        """Target latent dimensionality."""
        return self._target_dims

    @property
    def mean_vector(self) -> List[float]:
        """The mean vector from the last PCA training."""
        return list(self._mean)

    @property
    def eigenvalues(self) -> List[float]:
        """Top-K eigenvalues from the last PCA training."""
        return list(self._eigenvalues)

    @property
    def eigenvectors(self) -> List[List[float]]:
        """Top-K eigenvectors from the last PCA training (each length 32)."""
        return [list(ev) for ev in self._eigenvectors]
