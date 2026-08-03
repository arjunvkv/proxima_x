from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class MCResult:
    n_simulations: int
    mean_final_equity: float
    std_final_equity: float
    median_final_equity: float
    percentile_5: float
    percentile_95: float
    max_drawdown_stats: dict = field(default_factory=dict)
    sharpe_stats: dict = field(default_factory=dict)
    all_equity_curves: list[list[float]] = field(default_factory=list)
    probability_of_profit: float = 0.0
    details: dict = field(default_factory=dict)


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000, seed: int = 42) -> None:
        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        trade_returns: list[float],
        initial_equity: float = 10000.0,
        resample: bool = True,
    ) -> MCResult:
        returns_arr = np.array(trade_returns, dtype=np.float64)
        n_trades = len(returns_arr)

        if n_trades == 0:
            return MCResult(
                n_simulations=self.n_simulations,
                mean_final_equity=initial_equity,
                std_final_equity=0.0,
                median_final_equity=initial_equity,
                percentile_5=initial_equity,
                percentile_95=initial_equity,
                max_drawdown_stats={"mean": 0.0, "std": 0.0, "max": 0.0},
                sharpe_stats={"mean": 0.0, "std": 0.0},
                all_equity_curves=[],
                probability_of_profit=0.0,
                details={"n_trades_available": 0},
            )

        final_equities: list[float] = []
        all_mdds: list[float] = []
        all_sharpes: list[float] = []
        all_curves: list[list[float]] = []

        for _ in range(self.n_simulations):
            if resample:
                sampled = self.bootstrap_returns(trade_returns)
            else:
                sampled = self.shuffle_returns(trade_returns)

            equity_curve = self._compute_equity_curve(sampled, initial_equity)
            all_curves.append(equity_curve)

            final_eq = equity_curve[-1]
            final_equities.append(final_eq)

            mdd = self.calc_max_drawdown(equity_curve)
            all_mdds.append(mdd)

            sharpe = self.calc_sharpe(sampled)
            all_sharpes.append(sharpe)

        eq_arr = np.array(final_equities)
        mdd_arr = np.array(all_mdds)
        sharpe_arr = np.array(all_sharpes)

        n_profitable = int(np.sum(eq_arr > initial_equity))

        return MCResult(
            n_simulations=self.n_simulations,
            mean_final_equity=float(np.mean(eq_arr)),
            std_final_equity=float(np.std(eq_arr)),
            median_final_equity=float(np.median(eq_arr)),
            percentile_5=float(np.percentile(eq_arr, 5)),
            percentile_95=float(np.percentile(eq_arr, 95)),
            max_drawdown_stats={
                "mean": float(np.mean(mdd_arr)),
                "std": float(np.std(mdd_arr)),
                "max": float(np.max(mdd_arr)),
                "min": float(np.min(mdd_arr)),
                "median": float(np.median(mdd_arr)),
                "percentile_95": float(np.percentile(mdd_arr, 95)),
            },
            sharpe_stats={
                "mean": float(np.mean(sharpe_arr)),
                "std": float(np.std(sharpe_arr)),
                "max": float(np.max(sharpe_arr)),
                "min": float(np.min(sharpe_arr)),
                "median": float(np.median(sharpe_arr)),
                "percentile_95": float(np.percentile(sharpe_arr, 95)),
                "positive_ratio": float(np.sum(sharpe_arr > 0) / self.n_simulations),
            },
            all_equity_curves=all_curves,
            probability_of_profit=float(n_profitable / self.n_simulations),
            details={
                "n_trades_available": n_trades,
                "initial_equity": initial_equity,
                "resample": resample,
            },
        )

    def run_with_strategy_params(
        self,
        trade_returns_generator: Callable,
        param_variations: list[dict],
    ) -> list[MCResult]:
        results: list[MCResult] = []
        for params in param_variations:
            returns = trade_returns_generator(params)
            result = self.run(returns)
            result.details["params"] = params
            results.append(result)
        return results

    def shuffle_returns(self, returns: list[float]) -> list[float]:
        arr = np.array(returns, dtype=np.float64)
        self._rng.shuffle(arr)
        return arr.tolist()

    def bootstrap_returns(self, returns: list[float]) -> list[float]:
        arr = np.array(returns, dtype=np.float64)
        n = len(arr)
        indices = self._rng.integers(0, n, size=n)
        return arr[indices].tolist()

    def calc_max_drawdown(self, equity_curve: list[float]) -> float:
        arr = np.array(equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(arr)
        drawdowns = (peaks - arr) / peaks
        max_dd = float(np.max(drawdowns))
        return max_dd if np.isfinite(max_dd) else 0.0

    def calc_sharpe(self, returns: list[float], risk_free_rate: float = 0.0) -> float:
        arr = np.array(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return 0.0
        mean_ret = float(np.mean(arr))
        daily_sharpe = (mean_ret - risk_free_rate) / std
        return daily_sharpe * np.sqrt(252)

    def _compute_equity_curve(
        self,
        returns: list[float],
        initial_equity: float,
    ) -> list[float]:
        eq = initial_equity
        curve: list[float] = [eq]
        for r in returns:
            eq = eq * (1.0 + r)
            curve.append(eq)
        return curve
