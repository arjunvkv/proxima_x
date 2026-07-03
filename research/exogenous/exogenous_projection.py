"""
Program VI — Exogenous Amplitude Projection
=============================================
Projects conditional amplitude distributions from a trained exogenous surface
onto incoming microstructure states in real time.

``ExogenousProjectionEngine`` wraps a trained ``ExogenousAmplitudeSurface``
and provides key-then-lookup projection by exogenous key or by full state.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from typing import Optional

from research.exogenous.schemas import ExogenousState, make_key
from research.exogenous.exogenous_surface import ExogenousAmplitudeSurface


class ExogenousProjectionEngine:
    """Real-time exogenous amplitude inference for Program VI.

    Projects the conditional amplitude distribution for a given exogenous
    key or :class:`ExogenousState` at a specific horizon.  When the key
    is unknown (no surface entry exists), returns a zero-filled dictionary.
    """

    def __init__(
        self, surface: Optional[ExogenousAmplitudeSurface] = None
    ) -> None:
        self._surface: Optional[ExogenousAmplitudeSurface] = None
        if surface is not None:
            self.attach(surface)

    # ------------------------------------------------------------------
    # Surface attachment
    # ------------------------------------------------------------------

    def attach(self, surface: ExogenousAmplitudeSurface) -> None:
        """Attach a trained exogenous surface for projection queries.

        Parameters
        ----------
        surface : ExogenousAmplitudeSurface
            A fitted surface whose entries will be queried.
        """
        self._surface = surface

    # ------------------------------------------------------------------
    # Projection — by key
    # ------------------------------------------------------------------

    def project(self, exogenous_key: str, horizon: int) -> dict:
        """Project amplitude for a given exogenous key at *horizon*.

        Parameters
        ----------
        exogenous_key : str
            Composite key (e.g. from ``make_key()``).
        horizon : int
            Forward horizon in seconds.

        Returns
        -------
        dict
            With keys:

            - ``mean_abs``         — mean absolute move (bps)
            - ``aer``              — amplitude excess ratio
            - ``spread_multiple``  — mean_abs / mean_spread
            - ``p_exceed_2x``      — P(|move| > 2 × spread)
            - ``p_exceed_3x``      — P(|move| > 3 × spread)
            - ``n``                — observation count

            Returns a zero-filled dict when the key is not found.
        """
        if self._surface is None:
            return self._zero_projection()

        entry = self._surface.lookup(exogenous_key, horizon)
        if not entry:
            return self._zero_projection()

        exceed_prob = entry.get("exceed_prob", {})

        return {
            "mean_abs": entry.get("mean_abs", 0.0),
            "aer": entry.get("aer", 0.0),
            "spread_multiple": entry.get("spread_multiple", 0.0),
            "p_exceed_2x": exceed_prob.get(2.0, 0.0),
            "p_exceed_3x": exceed_prob.get(3.0, 0.0),
            "n": entry.get("n", 0),
        }

    # ------------------------------------------------------------------
    # Projection — by state
    # ------------------------------------------------------------------

    def project_state(self, state: ExogenousState, horizon: int) -> dict:
        """Build the exogenous key from *state* and project at *horizon*.

        Parameters
        ----------
        state : ExogenousState
            The microstructure state to project from.  The composite key is
            derived deterministically via ``make_key()``.
        horizon : int
            Forward horizon in seconds.

        Returns
        -------
        dict
            Same shape as :meth:`project`.
        """
        key = make_key(state)
        return self.project(key, horizon)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_projection() -> dict:
        """Return a zero-filled projection for an unknown key."""
        return {
            "mean_abs": 0.0,
            "aer": 0.0,
            "spread_multiple": 0.0,
            "p_exceed_2x": 0.0,
            "p_exceed_3x": 0.0,
            "n": 0,
        }
