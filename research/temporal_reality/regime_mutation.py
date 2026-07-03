"""
RegimeMutationAnalyzer: detect regime transitions and their relationship with adaptive_time.

Energy-based regime classification:
  - trend:   high energy_storage, directional
  - range:   low energy_storage, low dissipation
  - quiet:   all energy components low
  - volatile: high energy_dissipation, high creation

Transitions tracked:
  - trend_to_range, range_to_trend, quiet_to_volatile, volatile_to_quiet

For each transition, adaptive_time is measured before (window steps prior),
during (at the transition point), and after (window steps following).

Uses numpy + numba for performance; no Python loops where possible.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransitionEvent:
    """A single regime transition event with adaptive_time measurements."""
    timestamp: int
    before: float
    during: float
    after: float


@dataclass
class RegimeMutationReport:
    """Complete report from RegimeMutationAnalyzer.compute()."""
    asset: str
    transitions: Dict[str, List[TransitionEvent]]
    summary: Dict[str, Dict[str, float]]
    verdict: str


# ---------------------------------------------------------------------------
# Regime constants
# ---------------------------------------------------------------------------

QUIET: int = 0
RANGE: int = 1
TREND: int = 2
VOLATILE: int = 3


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _classify_regime_numba(
    creation: np.ndarray,
    storage: np.ndarray,
    dissipation: np.ndarray,
    c_low: float, c_high: float,
    s_low: float, s_high: float,
    d_low: float, d_high: float,
) -> np.ndarray:
    """Classify each timestep into a regime using pre-computed thresholds.

    0 = quiet, 1 = range, 2 = trend, 3 = volatile.

    Priority order (prevents ambiguity):
      1. storage > s_high            -> trend
      2. dissipation > d_high AND
         creation > c_high           -> volatile
      3. storage < s_low AND
         dissipation < d_low         -> range
      4. otherwise                   -> quiet
    """
    n = len(creation)
    regimes = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = storage[i]
        d = dissipation[i]
        c = creation[i]
        if s > s_high:
            regimes[i] = TREND
        elif d > d_high and c > c_high:
            regimes[i] = VOLATILE
        elif s < s_low and d < d_low:
            regimes[i] = RANGE
        else:
            regimes[i] = QUIET
    return regimes


@njit(cache=True)
def _measure_transitions_numba(
    adaptive_time: np.ndarray,
    indices: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract before-mean, during, and after-mean for each transition index."""
    n = len(adaptive_time)
    m = len(indices)
    before = np.empty(m, dtype=np.float64)
    during = np.empty(m, dtype=np.float64)
    after = np.empty(m, dtype=np.float64)

    for j in range(m):
        idx = indices[j]
        during[j] = adaptive_time[idx]

        # before: mean of adaptive_time[idx-window : idx]
        lo = max(0, idx - window)
        before[j] = np.mean(adaptive_time[lo:idx]) if idx > lo else adaptive_time[idx]

        # after: mean of adaptive_time[idx+1 : idx+1+window]
        hi = min(n, idx + 1 + window)
        after[j] = np.mean(adaptive_time[idx + 1:hi]) if hi > idx + 1 else adaptive_time[idx]

    return before, during, after


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RegimeMutationAnalyzer:
    """Analyze the relationship between adaptive_time and regime transitions.

    Parameters
    ----------
    window : int
        Number of steps before/after a transition used to compute the
        baseline adaptive_time (default 20, typical range 10-20).
    """

    # (from_regime, to_regime) for each named transition
    _TRANSITION_MAP: Dict[str, Tuple[int, int]] = {
        "trend_to_range": (TREND, RANGE),
        "range_to_trend": (RANGE, TREND),
        "quiet_to_volatile": (QUIET, VOLATILE),
        "volatile_to_quiet": (VOLATILE, QUIET),
    }

    def __init__(self, window: int = 20) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        adaptive_time: np.ndarray,
        energy_creation: np.ndarray,
        energy_storage: np.ndarray,
        energy_dissipation: np.ndarray,
        asset: str = "",
    ) -> RegimeMutationReport:
        """Run the full regime-mutation analysis pipeline.

        Parameters
        ----------
        adaptive_time : np.ndarray, shape (T,)
            The adaptive_time signal to analyse across transitions.
        energy_creation : np.ndarray, shape (T,)
        energy_storage : np.ndarray, shape (T,)
        energy_dissipation : np.ndarray, shape (T,)

        Returns
        -------
        RegimeMutationReport
        """
        self._validate_inputs(
            adaptive_time, energy_creation, energy_storage, energy_dissipation,
        )

        regimes = self._classify_regime(
            energy_creation, energy_storage, energy_dissipation,
        )
        at = adaptive_time.astype(np.float64, copy=False)

        transitions: Dict[str, List[TransitionEvent]] = {
            name: [] for name in self._TRANSITION_MAP
        }

        for name, (from_reg, to_reg) in self._TRANSITION_MAP.items():
            idx = self._find_transitions(regimes, from_reg, to_reg)
            if idx.size == 0:
                continue

            before, during, after = _measure_transitions_numba(at, idx, self.window)
            for j in range(len(idx)):
                transitions[name].append(
                    TransitionEvent(
                        timestamp=int(idx[j]),
                        before=float(before[j]),
                        during=float(during[j]),
                        after=float(after[j]),
                    )
                )

        summary = self._summarize(transitions)
        verdict = self._verdict(summary)
        return RegimeMutationReport(
            asset=asset,
            transitions=transitions,
            summary=summary,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        energy_creation: np.ndarray,
        energy_storage: np.ndarray,
        energy_dissipation: np.ndarray,
    ) -> np.ndarray:
        """Classify each timestep into 0=quiet, 1=range, 2=trend, 3=volatile.

        Thresholds are derived from the 30th/70th percentiles of each
        energy component over the full series.
        """
        c_low, c_high = np.percentile(energy_creation, [30, 70])
        s_low, s_high = np.percentile(energy_storage, [30, 70])
        d_low, d_high = np.percentile(energy_dissipation, [30, 70])

        return _classify_regime_numba(
            energy_creation, energy_storage, energy_dissipation,
            float(c_low), float(c_high),
            float(s_low), float(s_high),
            float(d_low), float(d_high),
        )

    # ------------------------------------------------------------------
    # Transition detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_transitions(
        regimes: np.ndarray,
        from_regime: int,
        to_regime: int,
    ) -> np.ndarray:
        """Return indices (into *regimes*) where a transition occurs.

        The transition point is the first timestep of the *to* regime.
        """
        diffs = np.diff(regimes.astype(np.int64, copy=False))
        target = to_regime - from_regime
        # diffs[i] corresponds to transition between regimes[i] and regimes[i+1].
        # The transition index in the original array is i+1.
        raw = np.flatnonzero(diffs == target).astype(np.int64) + 1
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(*arrays: np.ndarray) -> None:
        if not arrays:
            return
        n = len(arrays[0])
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
            if arr.ndim != 1:
                raise ValueError(f"Expected 1-d array, got shape {arr.shape}")
            if len(arr) != n:
                raise ValueError(
                    f"Array length mismatch: {len(arr)} vs expected {n}"
                )

    @staticmethod
    def _summarize(
        transitions: Dict[str, List[TransitionEvent]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-transition-type averages of before / during / after."""
        summary: Dict[str, Dict[str, float]] = {}
        for name, events in transitions.items():
            if not events:
                continue
            b = np.mean([e.before for e in events])
            d = np.mean([e.during for e in events])
            a = np.mean([e.after for e in events])
            summary[name] = {
                "avg_before": float(b),
                "avg_during": float(d),
                "avg_after": float(a),
            }
        return summary

    @staticmethod
    def _verdict(
        summary: Dict[str, Dict[str, float]],
    ) -> str:
        """Determine the overall relationship verdict.

        Logic
        -----
        For each transition type with data, compute the maximum relative
        deviation between before/during/after.  Then:

        - If *before* is the standout (max deviation involves before) in
          the majority of types -> "LEADS".
        - Else if *during* is the standout in the majority -> "MUTATES".
        - Otherwise -> "does NOT relate".
        """
        if not summary:
            return "adaptive_time does NOT relate to regimes"

        leading = 0
        mutating = 0
        unrelated = 0
        eps = 1e-12

        for vals in summary.values():
            b, d, a = vals["avg_before"], vals["avg_during"], vals["avg_after"]
            scale = max(abs(b), abs(d), abs(a), eps)

            rel_bd = abs(b - d) / scale
            rel_ba = abs(b - a) / scale
            rel_db = abs(d - b) / scale
            rel_da = abs(d - a) / scale

            before_dev = max(rel_bd, rel_ba)
            during_dev = max(rel_db, rel_da)

            threshold = 0.05  # 5 % relative change required

            if before_dev > threshold and before_dev >= during_dev:
                leading += 1
            elif during_dev > threshold:
                mutating += 1
            else:
                unrelated += 1

        # Plurality vote
        if leading >= mutating and leading >= unrelated:
            return "adaptive_time LEADS regime mutations"
        if mutating >= leading and mutating >= unrelated:
            return "adaptive_time MUTATES with regimes"
        return "adaptive_time does NOT relate to regimes"
