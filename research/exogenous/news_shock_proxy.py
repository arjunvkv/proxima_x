"""Program VI — Exogenous Amplitude Discovery.

NewsShockProxy — detects potential news-driven shocks via concurrent
spikes in tick velocity, spread, and absolute price move.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from collections import deque
from typing import Optional


class NewsShockProxy:
    """Detects potential news-driven shocks using a triple-threshold check.

    A "news shock proxy" is fired when tick velocity, spread, and absolute
    price move all exceed their respective multi-sigma thresholds simultaneously.

    Parameters
    ----------
    window : int
        Rolling window for baseline statistics (default 50).
    velocity_sigma : float
        Sigma threshold for tick-velocity z-score (default 5.0).
    spread_sigma : float
        Sigma threshold for spread z-score (default 3.0).
    move_sigma : float
        Sigma threshold for absolute-move z-score (default 4.0).
    """

    def __init__(
        self,
        window: int = 50,
        velocity_sigma: float = 5.0,
        spread_sigma: float = 3.0,
        move_sigma: float = 4.0,
    ) -> None:
        self.window = window
        self.velocity_sigma = velocity_sigma
        self.spread_sigma = spread_sigma
        self.move_sigma = move_sigma

        # Per-symbol state
        self._state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(vals: list[float]) -> float:
        """Population mean (len(vals) as denominator)."""
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _std(vals: list[float], ddof: int = 1) -> float:
        """Sample standard deviation (ddof=1)."""
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / (len(vals) - ddof)
        return var ** 0.5

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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, tick: dict) -> dict:
        """Process a single tick and return shock-detection results.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        tick : dict
            Tick data expected to contain at minimum:

            - ``ts`` : float — epoch timestamp (seconds)
            - ``price`` : float — mid / last price
            - ``bid``, ``ask`` : float — used to compute spread if not
              provided directly via the ``spread`` key.

        Returns
        -------
        dict
            ``shock_detected`` (bool), ``velocity_z`` (float),
            ``spread_z`` (float), ``move_z`` (float), ``n_samples`` (int).
        """
        # Lazy initialise per-symbol state
        if symbol not in self._state:
            self._state[symbol] = {
                "last_ts": None,
                "last_price": None,
                "iars": deque(maxlen=self.window),
                "velocities": deque(maxlen=self.window),
                "spreads": deque(maxlen=self.window),
                "moves": deque(maxlen=self.window),
            }

        s = self._state[symbol]
        ts = float(tick["ts"])
        price = float(tick.get("price", 0.0))
        spread = self._get_spread(tick)

        # ── compute current metrics ────────────────────────────────────────
        iar = 0.0
        abs_move = 0.0
        velocity = 0.0

        if s["last_ts"] is not None and ts > s["last_ts"]:
            iar = ts - s["last_ts"]
        if s["last_price"] is not None:
            abs_move = abs(price - s["last_price"])
        if iar > 0:
            velocity = abs_move / iar

        # Snapshot pre-update deque state BEFORE appending current tick.
        # This ensures the z-score baseline never includes the tick being
        # evaluated (avoiding self-bias / dilution).
        n_before = len(s["velocities"])
        v_baseline = list(s["velocities"])
        sp_baseline = list(s["spreads"])
        m_baseline = list(s["moves"])

        # ── update historical deques ───────────────────────────────────────
        if s["last_ts"] is not None:
            s["iars"].append(iar)
            s["velocities"].append(velocity)
        if s["last_price"] is not None:
            s["moves"].append(abs_move)
        s["spreads"].append(spread)

        s["last_ts"] = ts
        s["last_price"] = price

        n_after = len(s["velocities"])

        # Short-circuit until we have a full window of velocity samples
        if n_before < self.window:
            return {
                "shock_detected": False,
                "velocity_z": 0.0,
                "spread_z": 0.0,
                "move_z": 0.0,
                "n_samples": n_after,
            }

        # ── z-scores (against pre-update baseline) ─────────────────────────
        v_mean = self._mean(v_baseline)
        v_std = self._std(v_baseline)
        if v_std > 0:
            velocity_z = (velocity - v_mean) / v_std
        else:
            velocity_z = 0.0 if velocity == v_mean else 999.0

        sp_mean = self._mean(sp_baseline)
        sp_std = self._std(sp_baseline)
        if sp_std > 0:
            spread_z = (spread - sp_mean) / sp_std
        else:
            spread_z = 0.0 if spread == sp_mean else 999.0

        m_mean = self._mean(m_baseline)
        m_std = self._std(m_baseline)
        if m_std > 0:
            move_z = (abs_move - m_mean) / m_std
        else:
            move_z = 0.0 if abs_move == m_mean else 999.0

        # ── triple-threshold check ─────────────────────────────────────────
        shock = (
            velocity_z > self.velocity_sigma
            and spread_z > self.spread_sigma
            and move_z > self.move_sigma
        )

        return {
            "shock_detected": shock,
            "velocity_z": velocity_z,
            "spread_z": spread_z,
            "move_z": move_z,
            "n_samples": n_after,
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset state for *symbol*, or for all symbols if *symbol* is None."""
        if symbol is None:
            self._state.clear()
        else:
            self._state.pop(symbol, None)
