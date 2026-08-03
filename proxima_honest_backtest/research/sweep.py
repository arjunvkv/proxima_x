from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class SweepResult:
    symbol: str
    best_params: dict
    best_metric: float
    all_trials: list[dict] = field(default_factory=list)
    n_trials: int = 0
    elapsed_seconds: float = 0.0
    method: str = ""


class ParameterSweep:
    def __init__(
        self,
        param_space: dict,
        metric_fn: Callable,
        n_trials: int = 100,
        method: str = "auto",
        seed: int = 42,
    ) -> None:
        self.param_space = param_space
        self.metric_fn = metric_fn
        self.n_trials = n_trials
        self.seed = seed

        if method == "auto":
            try:
                import optuna  # noqa: F401
                self.method = "optuna"
            except ImportError:
                self.method = "random"
        else:
            self.method = method

        self._rng = np.random.default_rng(seed)

    def run(
        self,
        symbols: list[str],
        progress_callback: Optional[Callable] = None,
    ) -> list[SweepResult]:
        results: list[SweepResult] = []
        total = len(symbols)
        for i, symbol in enumerate(symbols):
            result = self.run_single(symbol)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i + 1, total, symbol, result)
        return results

    def run_single(self, symbol: str) -> SweepResult:
        start = time.perf_counter()

        if self.method == "optuna":
            result = self._optuna_optimize(symbol)
        elif self.method == "grid":
            result = self._grid_search(symbol)
        elif self.method == "random":
            result = self._random_search(symbol)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        result.elapsed_seconds = time.perf_counter() - start
        return result

    def _optuna_optimize(self, symbol: str) -> SweepResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for name, domain in self.param_space.items():
                if isinstance(domain, (list, tuple)) and all(
                    isinstance(v, (int, float, str)) for v in domain
                ):
                    if all(isinstance(v, str) for v in domain):
                        params[name] = trial.suggest_categorical(name, list(domain))
                    elif all(isinstance(v, int) for v in domain):
                        params[name] = trial.suggest_int(name, int(min(domain)), int(max(domain)))
                    else:
                        params[name] = trial.suggest_float(name, float(min(domain)), float(max(domain)))
                elif isinstance(domain, (list, tuple)) and len(domain) == 2:
                    low, high = domain
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = trial.suggest_int(name, low, high)
                    else:
                        params[name] = trial.suggest_float(name, float(low), float(high))
                else:
                    raise ValueError(f"Unsupported param domain for {name}: {domain}")
            return self.metric_fn(params, symbol)

        study.optimize(objective, n_trials=self.n_trials)

        all_trials: list[dict] = []
        for t in study.trials:
            if t.values is not None and len(t.values) > 0:
                all_trials.append({
                    "params": t.params,
                    "metric": t.values[0],
                    "number": t.number,
                })

        return SweepResult(
            symbol=symbol,
            best_params=study.best_params,
            best_metric=float(study.best_value),
            all_trials=all_trials,
            n_trials=len(study.trials),
            elapsed_seconds=0.0,
            method="optuna",
        )

    def _grid_search(self, symbol: str) -> SweepResult:
        grid_values: list[list[Any]] = []
        param_names: list[str] = []

        for name, domain in self.param_space.items():
            param_names.append(name)
            if isinstance(domain, (list, tuple)) and all(isinstance(v, str) for v in domain):
                grid_values.append(list(domain))
            elif isinstance(domain, (list, tuple)) and len(domain) == 2:
                low, high = domain
                if isinstance(low, int) and isinstance(high, int):
                    grid_values.append(list(range(low, high + 1)))
                else:
                    grid_values.append(list(np.linspace(float(low), float(high), num=self.n_trials)))
            else:
                grid_values.append(list(domain))

        best_metric = -float("inf")
        best_params: dict = {}
        all_trials: list[dict] = []
        n_trials = 0

        for combination in itertools.product(*grid_values):
            params = dict(zip(param_names, combination))
            metric = self.metric_fn(params, symbol)
            all_trials.append({"params": params, "metric": metric})
            n_trials += 1
            if metric > best_metric:
                best_metric = metric
                best_params = params

        return SweepResult(
            symbol=symbol,
            best_params=best_params,
            best_metric=best_metric,
            all_trials=all_trials,
            n_trials=n_trials,
            elapsed_seconds=0.0,
            method="grid",
        )

    def _random_search(self, symbol: str) -> SweepResult:
        best_metric = -float("inf")
        best_params: dict = {}
        all_trials: list[dict] = []

        for _ in range(self.n_trials):
            params: dict[str, Any] = {}
            for name, domain in self.param_space.items():
                if isinstance(domain, (list, tuple)) and all(isinstance(v, str) for v in domain):
                    idx = self._rng.integers(0, len(domain))
                    params[name] = domain[idx]
                elif isinstance(domain, (list, tuple)) and len(domain) == 2:
                    low, high = domain
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = int(self._rng.integers(low, high + 1))
                    else:
                        params[name] = float(self._rng.uniform(float(low), float(high)))
                else:
                    raise ValueError(f"Unsupported param domain for {name}: {domain}")

            metric = self.metric_fn(params, symbol)
            all_trials.append({"params": params, "metric": metric})
            if metric > best_metric:
                best_metric = metric
                best_params = params

        return SweepResult(
            symbol=symbol,
            best_params=best_params,
            best_metric=best_metric,
            all_trials=all_trials,
            n_trials=self.n_trials,
            elapsed_seconds=0.0,
            method="random",
        )

    def summarize(self, results: list[SweepResult]) -> dict:
        n_symbols = len(results)
        metrics = [r.best_metric for r in results]

        best_params_by_symbol: dict[str, dict] = {}
        for r in results:
            best_params_by_symbol[r.symbol] = r.best_params

        return {
            "best_params_by_symbol": best_params_by_symbol,
            "avg_metric": float(np.mean(metrics)) if metrics else 0.0,
            "best_metric": float(np.max(metrics)) if metrics else 0.0,
            "worst_metric": float(np.min(metrics)) if metrics else 0.0,
            "std_metric": float(np.std(metrics)) if len(metrics) > 1 else 0.0,
            "n_symbols": n_symbols,
            "method": self.method,
            "n_trials_per_symbol": self.n_trials,
        }
