"""
Phase V — Real-time Amplitude Projection
==========================================
Projects conditional amplitude distributions from a trained surface
onto incoming microstructure states in real time.

``AmplitudeProjectionEngine`` wraps a trained ``AmplitudeSurfaceEngine``
and provides hash-then-lookup projection across all available horizons.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from typing import Dict, List, Optional

import numpy as np

from research.amplitude.schemas import (
    AmplitudeState,
    AmplitudeSurfaceEntry,
    StateHasher,
)
from research.amplitude.amplitude_surface import AmplitudeSurfaceEngine

# The exceed-probability thresholds used by the surface engine.
_EXCEED_KS: tuple = (1.0, 1.5, 2.0, 3.0, 4.0)


class AmplitudeProjectionEngine:
    """Real-time amplitude inference from state for Phase V.

    Projects the conditional amplitude distribution for a given
    :class:`AmplitudeState` across all horizons that exist in the
    attached surface engine.  Horizons without observations receive a
    zero-filled fallback with ``confidence="low"``.
    """

    def __init__(
        self, surface_engine: Optional[AmplitudeSurfaceEngine] = None
    ) -> None:
        self._surface_engine: Optional[AmplitudeSurfaceEngine] = None
        if surface_engine is not None:
            self.attach_surface(surface_engine)

    # ------------------------------------------------------------------
    # Surface attachment
    # ------------------------------------------------------------------

    def attach_surface(self, surface_engine: AmplitudeSurfaceEngine) -> None:
        """Attach a trained surface engine for projection queries.

        Parameters
        ----------
        surface_engine : AmplitudeSurfaceEngine
            A fitted surface engine whose entries will be queried.
        """
        self._surface_engine = surface_engine

    # ------------------------------------------------------------------
    # Projection — single state
    # ------------------------------------------------------------------

    def project(self, state: AmplitudeState) -> Dict[int, dict]:
        """Project amplitude for *state* across all horizons.

        Parameters
        ----------
        state : AmplitudeState
            The microstructure state to project from.

        Returns
        -------
        dict[int, dict]
            Mapping from horizon (seconds) to a projection dict with
            keys:

            - ``horizon``          — the forward horizon (seconds)
            - ``mean_abs``         — mean absolute move (bps)
            - ``p90_abs``          — 90th percentile absolute move (bps)
            - ``spread_multiple``  — mean_abs / spread
            - ``aer``              — conditional / unconditional std
            - ``exceed_prob``      — k → exceed probability
            - ``n``                — observation count
            - ``confidence``       — ``"low"`` | ``"medium"`` | ``"high"``
        """
        if self._surface_engine is None:
            return {}

        # 1. Deterministic state hash (same surface bucket).
        state_hash: str = StateHasher.hash(state)

        # 2. Gather every horizon that exists somewhere in the surface.
        all_horizons: set[int] = set()
        for horizons in self._surface_engine._surfaces.values():
            all_horizons.update(horizons.keys())

        # 3. Look up the entries for this particular state hash.
        state_entries: Dict[int, AmplitudeSurfaceEntry] = (
            self._surface_engine._surfaces.get(state_hash) or {}
        )

        # 4. Build output for every known horizon.
        result: Dict[int, dict] = {}
        for horizon in sorted(all_horizons):
            entry = state_entries.get(horizon)
            if entry is not None:
                result[horizon] = self._entry_to_projection(entry)
            else:
                result[horizon] = self._fallback_projection(horizon)

        return result

    # ------------------------------------------------------------------
    # Projection — batch
    # ------------------------------------------------------------------

    def project_batch(
        self, states: List[AmplitudeState]
    ) -> List[Dict[int, dict]]:
        """Project amplitude for a batch of states.

        Parameters
        ----------
        states : list[AmplitudeState]
            Sequence of states to project.

        Returns
        -------
        list[dict[int, dict]]
            One projection mapping per input state (same order).
        """
        return [self.project(state) for state in states]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_projection(entry: AmplitudeSurfaceEntry) -> dict:
        """Convert a surface entry into the projection output format."""
        return {
            "horizon": entry.horizon,
            "mean_abs": entry.mean_abs,
            "p90_abs": entry.p90_abs,
            "spread_multiple": entry.spread_multiple,
            "aer": entry.aer,
            "exceed_prob": dict(entry.exceed_prob),  # shallow copy
            "n": entry.n,
            "confidence": _confidence_label(entry.n),
        }

    @staticmethod
    def _fallback_projection(horizon: int) -> dict:
        """Return a zero-filled projection when a horizon is unseen."""
        return {
            "horizon": horizon,
            "mean_abs": 0.0,
            "p90_abs": 0.0,
            "spread_multiple": 0.0,
            "aer": 0.0,
            "exceed_prob": {k: 0.0 for k in _EXCEED_KS},
            "n": 0,
            "confidence": "low",
        }

    # ------------------------------------------------------------------
    # Static decision logic
    # ------------------------------------------------------------------

    @staticmethod
    def compute_trade_decision(
        projection: Dict[int, dict],
        min_spread_multiple: float = 2.0,
        min_aer: float = 1.5,
    ) -> dict:
        """Evaluate whether the projected amplitude warrants a trade.

        Scans all horizons in *projection* and picks the candidate with
        the highest combined score (spread_multiple + AER) that meets
        both thresholds.

        Parameters
        ----------
        projection : dict[int, dict]
            Output from :meth:`project()`.
        min_spread_multiple : float
            Minimum ``spread_multiple`` required (default 2.0).
        min_aer : float
            Minimum ``aer`` required (default 1.5).

        Returns
        -------
        dict
            With keys:

            - ``trade``         — ``True`` if a qualifying horizon exists
            - ``best_horizon``  — selected horizon (seconds), or ``-1``
            - ``reason``        — human-readable explanation
        """
        best_horizon: int = -1
        best_score: float = -np.inf

        for horizon, proj in projection.items():
            sm: float = proj.get("spread_multiple", 0.0)
            aer: float = proj.get("aer", 0.0)

            if sm >= min_spread_multiple and aer >= min_aer:
                score = sm + aer
                if score > best_score:
                    best_score = score
                    best_horizon = horizon

        if best_horizon >= 0:
            return {
                "trade": True,
                "best_horizon": best_horizon,
                "reason": (
                    f"Horizon {best_horizon}s: "
                    f"spread_multiple={projection[best_horizon]['spread_multiple']:.2f}, "
                    f"aer={projection[best_horizon]['aer']:.2f}"
                ),
            }

        return {
            "trade": False,
            "best_horizon": -1,
            "reason": (
                f"No horizon meets thresholds: "
                f"spread_multiple >= {min_spread_multiple} and "
                f"aer >= {min_aer}"
            ),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _confidence_label(n: int) -> str:
    """Map observation count to a qualitative confidence label.

    Parameters
    ----------
    n : int
        Number of observations in the surface bucket.

    Returns
    -------
    str
        ``"low"`` when fewer than 10, ``"medium"`` when fewer than 50,
        ``"high"`` otherwise.
    """
    if n < 10:
        return "low"
    if n < 50:
        return "medium"
    return "high"
