"""Program VI — Exogenous Amplitude Discovery.

LiquidityVoidDetector — detects liquidity voids via concurrent extreme
values in inter-arrival time, spread, and quote density.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from collections import deque
from typing import Optional

import numpy as np


class LiquidityVoidDetector:
    """Detects liquidity voids using percentile-based thresholds.

    A "liquidity void" is detected when inter-arrival time and spread both
    exceed high percentile thresholds while quote density falls below a low
    percentile threshold simultaneously.

    Parameters
    ----------
    window : int
        Rolling window for percentile statistics (default 100).
    iar_percentile : float
        Percentile threshold for inter-arrival time (default 0.95).
    spread_percentile : float
        Percentile threshold for spread (default 0.90).
    density_percentile : float
        Percentile threshold for quote density (default 0.10).
    """

    def __init__(
        self,
        window: int = 100,
        iar_percentile: float = 0.95,
        spread_percentile: float = 0.90,
        density_percentile: float = 0.10,
    ) -> None:
        self.window = window
        self.iar_percentile = iar_percentile
        self.spread_percentile = spread_percentile
        self.density_percentile = density_percentile

        # Per-symbol state
        self._state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_spread(tick: dict) -> float:
        """Extract spread from tick, computing from bid/ask if needed."""
        spread = tick.get("spread")
        if spread is not None:
            return float(spread)
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            return float(ask) - float(bid)
        return 0.0

    @staticmethod
    def _compute_density(timestamps: deque, now: float) -> float:
        """Compute quote density as ticks per second over the last 10 s.

        Parameters
        ----------
        timestamps : deque of float
            Recent tick timestamps in chronological order.
        now : float
            Current timestamp.

        Returns
        -------
        float
            Ticks per second (0 if no timestamps within window).
        """
        cutoff = now - 10.0
        count = sum(1 for t in timestamps if t >= cutoff)
        return count / 10.0

    @staticmethod
    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _std(vals: list[float], ddof: int = 1) -> float:
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=ddof))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, tick: dict) -> dict:
        """Process a single tick and return void-detection results.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        tick : dict
            Tick data expected to contain at minimum:

            - ``ts`` : float — epoch timestamp (seconds)
            - ``bid``, ``ask`` : float — used to compute spread if not
              provided directly via the ``spread`` key.

        Returns
        -------
        dict
            ``void_detected`` (bool), ``iar_z`` (float),
            ``spread_z`` (float), ``density_z`` (float), ``n_samples`` (int).
        """
        # Lazy initialise per-symbol state
        if symbol not in self._state:
            self._state[symbol] = {
                "last_ts": None,
                "iars": deque(maxlen=self.window),
                "spreads": deque(maxlen=self.window),
                "densities": deque(maxlen=self.window),
                "timestamps": deque(),
            }

        s = self._state[symbol]
        ts = float(tick["ts"])
        spread = self._get_spread(tick)

        # ── inter-arrival time ─────────────────────────────────────────────
        iar = 0.0
        if s["last_ts"] is not None and ts > s["last_ts"]:
            iar = ts - s["last_ts"]

        # Snapshot pre-update deque state BEFORE any appends, so the
        # percentile thresholds and z-scores reflect the historical baseline
        # without contamination from the current tick.
        n_before = len(s["iars"])
        iar_baseline = list(s["iars"])
        sp_baseline = list(s["spreads"])
        den_baseline = list(s["densities"])

        # ── update iars ─────────────────────────────────────────────────────
        if s["last_ts"] is not None and ts > s["last_ts"]:
            s["iars"].append(iar)
        s["spreads"].append(spread)

        # ── track timestamps for density ────────────────────────────────────
        s["timestamps"].append(ts)
        cutoff = ts - 60.0
        while s["timestamps"] and s["timestamps"][0] < cutoff:
            s["timestamps"].popleft()

        # ── quote density (ticks / s over last 10 s) ───────────────────────
        density = self._compute_density(s["timestamps"], ts)
        s["densities"].append(density)

        s["last_ts"] = ts

        n_after = len(s["iars"])

        # Short-circuit until we have a full window of iar samples
        if n_before < self.window:
            return {
                "void_detected": False,
                "iar_z": 0.0,
                "spread_z": 0.0,
                "density_z": 0.0,
                "n_samples": n_after,
            }

        # ── percentile thresholds (against pre-update baseline) ────────────
        iar_threshold = float(np.percentile(iar_baseline, self.iar_percentile * 100))
        spread_threshold = float(np.percentile(sp_baseline, self.spread_percentile * 100))
        density_threshold = float(np.percentile(den_baseline, self.density_percentile * 100))

        # ── z-scores (against pre-update baseline) ─────────────────────────
        iar_mean = self._mean(iar_baseline)
        iar_stdv = self._std(iar_baseline)
        if iar_stdv > 0:
            iar_z = (iar - iar_mean) / iar_stdv
        else:
            iar_z = 0.0 if iar == iar_mean else 999.0

        sp_mean = self._mean(sp_baseline)
        sp_stdv = self._std(sp_baseline)
        if sp_stdv > 0:
            spread_z = (spread - sp_mean) / sp_stdv
        else:
            spread_z = 0.0 if spread == sp_mean else 999.0

        den_mean = self._mean(den_baseline)
        den_stdv = self._std(den_baseline)
        if den_stdv > 0:
            density_z = (density - den_mean) / den_stdv
        else:
            density_z = 0.0 if density == den_mean else 999.0

        # ── triple-condition check ─────────────────────────────────────────
        void = (
            iar > iar_threshold
            and spread > spread_threshold
            and density < density_threshold
        )

        return {
            "void_detected": void,
            "iar_z": iar_z,
            "spread_z": spread_z,
            "density_z": density_z,
            "n_samples": n_after,
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset state for *symbol*, or for all symbols if *symbol* is None."""
        if symbol is None:
            self._state.clear()
        else:
            self._state.pop(symbol, None)
