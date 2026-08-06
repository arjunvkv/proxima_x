"""mvs.utils.vector_ops — small numerical helpers for the MVS rebuilders."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def shannon_entropy(counts: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy (in bits by default) of a count/frequency histogram.

    Parameters
    ----------
    counts : sequence of float
        Non-negative bin counts (or any weights summing > 0).
    base : float
        Logarithm base — 2 for bits, e (nats), 10 (dits).

    Returns
    -------
    float
        Entropy in the chosen base. 0.0 for degenerate (all mass in one bin)
        or empty input.
    """
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    total = arr.sum()
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log(p)) / np.log(base))


def rolling_mean(arr: Sequence[float], window: int) -> np.ndarray:
    """Simple trailing-window mean with edge handling (prepends NaNs-free)."""
    a = np.asarray(arr, dtype=np.float64)
    w = max(1, int(window))
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(a.size, dtype=np.float64)
    for i in range(a.size):
        out[i] = a[max(0, i - w + 1): i + 1].mean()
    return out