"""FrictionValidator — applies CEP execution physics to cross-asset propagation results.

Adapts the Counterfactual Execution Physics (CEP) framework to evaluate
how execution friction degrades the edge identified by
:class:`~research.dpl_x.propagation_mapper.PropagationMapper`.

Usage
-----
    from research.dpl_x.friction_validator import FrictionValidator

    validator = FrictionValidator()
    result = validator.evaluate(propagation_result, "ecn_fx")
    # or evaluate against all 5 profiles at once:
    all_results = FrictionValidator.evaluate_all(propagation_result)
"""
import sys; sys.path.insert(0, ".")

from typing import Any

import numpy as np
import polars as pl  # noqa: F401 — kept for ecosystem consistency

from research.cep.counterfactual_execution import CounterfactualExecutionEngine
from research.cep.execution_profile import (
    ExecutionProfile,
    load_profiles as _load_profiles,
)
from research.cep.metrics import CEPMetrics


# ---------------------------------------------------------------------------
# FrictionValidator
# ---------------------------------------------------------------------------

class FrictionValidator:
    """Validates cross-asset propagation edges under realistic execution friction.

    Takes a :class:`~research.dpl_x.propagation_mapper.PropagationResult`,
    applies the :class:`~research.cep.counterfactual_execution.CounterfactualExecutionEngine`
    with a chosen execution profile, and reports how costs degrade profit
    factor, expectancy, and net edge.
    """

    # Directional bucket ranges for the OSS state space (0–9).
    #   * 0–3  → bearish  (short signal)
    #   * 4–5  → neutral  (skipped)
    #   * 6–9  → bullish  (long signal)
    _BULLISH_BUCKETS: tuple[int, int] = (6, 9)
    _BEARISH_BUCKETS: tuple[int, int] = (0, 3)

    def __init__(self) -> None:
        self._profile_cache: dict[str, ExecutionProfile] | None = None
        self._metrics_calc = CEPMetrics()

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    def load_profiles(self) -> dict[str, ExecutionProfile]:
        """Load the 5 execution profiles from ``execution_profiles.yaml``.

        The profiles are cached after the first call.

        Returns
        -------
        dict[str, ExecutionProfile]
            Mapping of profile name → dataclass.  The five built-in
            profiles are ``retail_fx``, ``prime_fx``, ``ecn_fx``,
            ``cme_fx``, and ``crypto_perps``.
        """
        if self._profile_cache is None:
            self._profile_cache = _load_profiles()
        return dict(self._profile_cache)

    # ------------------------------------------------------------------
    # Single-profile evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        propagation_result: Any,
        profile_name: str,
    ) -> dict[str, Any]:
        """Apply execution physics to *propagation_result* under *profile_name*.

        Parameters
        ----------
        propagation_result : PropagationResult
            Output from :meth:`PropagationMapper.map
            <research.dpl_x.propagation_mapper.PropagationMapper.map>`.
            Must expose attributes ``effects``, ``source``, ``target``,
            ``horizon`` (as a dataclass or compatible object).
        profile_name : str
            One of ``"retail_fx"``, ``"prime_fx"``, ``"ecn_fx"``,
            ``"cme_fx"``, ``"crypto_perps"``.

        Returns
        -------
        dict
            Keys:
            ``pf_after_cost``
                Profit factor after execution costs.
            ``expectancy_after_cost``
                Mean net PnL per trade (in price units).
            ``slippage_ratio``
                Average absolute slippage divided by average absolute
                gross PnL — a measure of how much friction consumes the
                raw edge.
            ``net_edge``
                Expectancy after cost expressed in basis points of the
                entry price (bps).
            ``profile``
                The profile name used.
            ``source``
                Source asset symbol from the propagation result.
            ``target``
                Target asset symbol from the propagation result.
            ``horizon``
                Forward-return horizon in seconds.

        Raises
        ------
        ValueError
            If *profile_name* is not one of the available profiles.
        """
        profiles = self.load_profiles()
        if profile_name not in profiles:
            raise ValueError(
                f"Unknown execution profile {profile_name!r}. "
                f"Available profiles: {sorted(profiles)}"
            )

        profile = profiles[profile_name]
        engine = CounterfactualExecutionEngine(profile)

        effects: dict = propagation_result.effects
        horizon: int = propagation_result.horizon

        trades: list[dict[str, Any]] = []
        rng = np.random.default_rng(42)
        NORMALIZED_PRICE = 100.0

        for bucket_id_str, effect in effects.items():
            bucket_id = int(bucket_id_str)

            # --- Determine trade direction --------------------------------
            if self._BULLISH_BUCKETS[0] <= bucket_id <= self._BULLISH_BUCKETS[1]:
                signal = 1   # long
            elif self._BEARISH_BUCKETS[0] <= bucket_id <= self._BEARISH_BUCKETS[1]:
                signal = -1  # short
            else:
                continue  # skip neutral buckets (4, 5)

            exp_ret = float(effect.get("expected_return", 0.0))
            std_ret = float(effect.get("std_return", 0.0))
            n_samples = int(effect.get("n", 0))

            if n_samples == 0:
                continue

            # Generate synthetic gross returns that reflect the bucket's
            # observed distribution.  The forward return from the
            # propagation result is (future_price - price) / price, so a
            # gross return of *r* maps directly to exit_price =
            # entry_price × (1 + r).
            std_ret = max(std_ret, 1e-10)
            gross_returns: np.ndarray = rng.normal(exp_ret, std_ret, size=n_samples)

            for gross_ret in gross_returns:
                exit_price = NORMALIZED_PRICE * (1.0 + gross_ret)

                trade = engine.simulate_trade(
                    signal=signal,
                    entry_ts=0,
                    entry_price=NORMALIZED_PRICE,
                    exit_prices=[exit_price],
                    exit_ts=horizon,
                    signal_name="OSS",
                )
                if trade is not None:
                    trades.append(trade)

        # --- Aggregate metrics --------------------------------------------
        metrics = self._metrics_calc.compute(trades)

        pf_after_cost: float = metrics["profit_factor"]
        expectancy_after_cost: float = metrics["expectancy"]

        # Slippage ratio — how much of the gross edge is eroded by
        # slippage alone.
        if trades:
            avg_slippage = float(
                np.mean([abs(t["slippage"]) for t in trades])
            )
            avg_gross = float(
                np.mean([abs(t["gross_pnl"]) for t in trades])
            )
            slippage_ratio = (
                avg_slippage / avg_gross if avg_gross > 1e-15 else 0.0
            )
        else:
            avg_slippage = 0.0
            slippage_ratio = 0.0

        # Net edge expressed in basis points of the normalized entry price.
        net_edge = (
            (expectancy_after_cost / NORMALIZED_PRICE) * 10000
            if trades
            else 0.0
        )

        return {
            "pf_after_cost": pf_after_cost,
            "expectancy_after_cost": expectancy_after_cost,
            "slippage_ratio": slippage_ratio,
            "net_edge": net_edge,
            "profile": profile_name,
            "source": propagation_result.source,
            "target": propagation_result.target,
            "horizon": horizon,
        }

    # ------------------------------------------------------------------
    # Multi-profile evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_all(
        propagation_result: Any,
        profiles: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate *propagation_result* against one or more profiles.

        Parameters
        ----------
        propagation_result : PropagationResult
        profiles : list[str] or None
            Subset of profile names to evaluate.  If ``None``, all five
            built-in profiles are tested.

        Returns
        -------
        list[dict]
            One result dict per profile (see :meth:`evaluate`).

        Raises
        ------
        ValueError
            If any requested profile name is not found.
        """
        validator = FrictionValidator()
        all_profiles = validator.load_profiles()

        if profiles is None:
            target_names = sorted(all_profiles)
        else:
            missing = [p for p in profiles if p not in all_profiles]
            if missing:
                raise ValueError(
                    f"Unknown profile(s): {missing}. "
                    f"Available profiles: {sorted(all_profiles)}"
                )
            target_names = profiles

        results: list[dict[str, Any]] = []
        for pname in target_names:
            results.append(validator.evaluate(propagation_result, pname))
        return results
