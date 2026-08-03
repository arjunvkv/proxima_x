from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class WFResult:
    passed: bool
    windows: List[Dict[str, Any]] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    oos_consistency: float = 0.0
    sharpe_decay: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def _sharpe_ratio(returns: List[float]) -> float:
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * math.sqrt(len(arr)))


class WalkForwardValidator:
    def __init__(
        self,
        n_windows: int = 5,
        test_size: float = 0.2,
        embargo_days: int = 5,
    ) -> None:
        if n_windows < 1:
            raise ValueError("n_windows must be at least 1")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")
        self.n_windows = n_windows
        self.test_size = test_size
        self.embargo_days = embargo_days

    def run(
        self,
        returns: List[float],
        timestamps: List[datetime],
        strategy_func: Optional[Callable[..., Any]] = None,
        param_grid: Optional[Dict[str, Any]] = None,
    ) -> WFResult:
        if len(returns) != len(timestamps):
            raise ValueError("returns and timestamps must have same length")
        if len(returns) < self.n_windows * 2:
            raise ValueError("insufficient data for requested number of windows")

        windows = self._create_windows(timestamps)

        window_results: List[Dict[str, Any]] = []

        for window in windows:
            train_returns = returns[window["train_start"] : window["train_end"]]
            test_returns = returns[window["test_start"] : window["test_end"]]

            is_sharpe = _sharpe_ratio(train_returns)
            oos_sharpe = _sharpe_ratio(test_returns)

            window_info = {
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "train_start_date": timestamps[window["train_start"]],
                "train_end_date": timestamps[window["train_end"] - 1],
                "test_start_date": timestamps[window["test_start"]],
                "test_end_date": timestamps[window["test_end"] - 1],
                "in_sample_sharpe": is_sharpe,
                "out_of_sample_sharpe": oos_sharpe,
                "n_train": len(train_returns),
                "n_test": len(test_returns),
            }

            if param_grid is not None and strategy_func is not None:
                best_params, best_is_sharpe = self._optimize_params(
                    train_returns, strategy_func, param_grid
                )
                window_info["best_params"] = best_params
                window_info["best_is_sharpe"] = best_is_sharpe

            window_results.append(window_info)

        avg_oos = float(np.mean([w["out_of_sample_sharpe"] for w in window_results]))
        positive_oos = sum(1 for w in window_results if w["out_of_sample_sharpe"] > 0)
        oos_consistency = positive_oos / len(window_results) if window_results else 0.0

        sharpe_decay = 0.0
        if window_results:
            ratios = []
            for w in window_results:
                is_s = w["in_sample_sharpe"]
                oos_s = w["out_of_sample_sharpe"]
                if is_s != 0:
                    ratios.append(oos_s / is_s)
                else:
                    ratios.append(0.0)
            sharpe_decay = float(np.mean(ratios))

        passed = oos_consistency >= 0.5 and avg_oos > 0

        return WFResult(
            passed=passed,
            windows=window_results,
            avg_oos_sharpe=avg_oos,
            oos_consistency=oos_consistency,
            sharpe_decay=sharpe_decay,
            details={
                "n_windows": self.n_windows,
                "test_size": self.test_size,
                "embargo_days": self.embargo_days,
                "n_observations": len(returns),
            },
        )

    def run_simple(
        self,
        returns: List[float],
        timestamps: List[datetime],
        train_ratio: float = 0.7,
    ) -> WFResult:
        if len(returns) != len(timestamps):
            raise ValueError("returns and timestamps must have same length")
        if len(returns) < 2:
            raise ValueError("insufficient data")
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        split_idx = int(len(returns) * train_ratio)

        embargo_offset = 0
        if split_idx < len(timestamps):
            train_last = timestamps[split_idx - 1]
            embargo_boundary = train_last + timedelta(days=self.embargo_days)
            for i in range(split_idx, len(timestamps)):
                if timestamps[i] >= embargo_boundary:
                    embargo_offset = i - split_idx
                    break

        test_start = split_idx + embargo_offset
        if test_start >= len(timestamps):
            test_start = split_idx

        train_returns = returns[:split_idx]
        test_returns = returns[test_start:]

        is_sharpe = _sharpe_ratio(train_returns)
        oos_sharpe = _sharpe_ratio(test_returns)

        window = {
            "train_start": 0,
            "train_end": split_idx,
            "test_start": test_start,
            "test_end": len(returns),
            "train_start_date": timestamps[0],
            "train_end_date": timestamps[split_idx - 1],
            "test_start_date": timestamps[test_start],
            "test_end_date": timestamps[-1],
            "in_sample_sharpe": is_sharpe,
            "out_of_sample_sharpe": oos_sharpe,
            "n_train": len(train_returns),
            "n_test": len(test_returns),
        }

        passed = oos_sharpe > 0

        return WFResult(
            passed=passed,
            windows=[window],
            avg_oos_sharpe=oos_sharpe,
            oos_consistency=1.0 if oos_sharpe > 0 else 0.0,
            sharpe_decay=oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0,
            details={
                "n_windows": 1,
                "test_size": 1.0 - train_ratio,
                "embargo_days": self.embargo_days,
                "n_observations": len(returns),
            },
        )

    def _create_windows(
        self, timestamps: List[datetime]
    ) -> List[Dict[str, int]]:
        n = len(timestamps)
        windows: List[Dict[str, int]] = []
        window_size = n // self.n_windows
        test_len = max(1, int(window_size * self.test_size))

        for i in range(self.n_windows):
            train_end = i * window_size + window_size - test_len
            if i == self.n_windows - 1:
                train_end = n - test_len

            train_start = i * window_size

            test_start = train_end + self._apply_embargo(
                train_end, train_end + 1, timestamps
            )

            test_end = min(test_start + test_len, n)

            if test_start >= n or train_start >= train_end:
                continue

            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

        return windows

    def _apply_embargo(
        self,
        train_end_idx: int,
        test_start_idx: int,
        timestamps: List[datetime],
    ) -> int:
        if train_end_idx >= len(timestamps) or test_start_idx >= len(timestamps):
            return 0

        train_last = timestamps[train_end_idx]
        embargo_boundary = train_last + timedelta(days=self.embargo_days)

        offset = 0
        for i in range(test_start_idx, len(timestamps)):
            if timestamps[i] >= embargo_boundary:
                offset = i - test_start_idx
                break

        return offset

    def _optimize_params(
        self,
        train_returns: List[float],
        strategy_func: Callable[..., Any],
        param_grid: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], float]:
        best_sharpe = -float("inf")
        best_params: Optional[Dict[str, Any]] = None

        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combination in product(*values):
            params = dict(zip(keys, combination))
            try:
                result = strategy_func(train_returns, **params)
                if isinstance(result, (int, float)):
                    sharpe_val = float(result)
                else:
                    sharpe_val = 0.0
            except Exception:
                sharpe_val = -float("inf")

            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_params = params

        if best_params is None:
            best_params = {}
            best_sharpe = _sharpe_ratio(train_returns)

        return best_params, best_sharpe
