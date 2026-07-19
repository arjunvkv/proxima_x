"""Predictive skill metrics for WLS decomposition validation."""

import numpy as np


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def predictive_skill(actual: np.ndarray, predicted: np.ndarray, naive: np.ndarray = None) -> dict:
    if naive is None:
        naive = np.zeros_like(actual)
    model_mse = mse(actual, predicted)
    naive_mse = mse(actual, naive)
    skill = 1.0 - (model_mse / naive_mse) if naive_mse > 0 else 0.0
    model_mae = mae(actual, predicted)
    naive_mae = mae(actual, naive)
    mae_skill = 1.0 - (model_mae / naive_mae) if naive_mae > 0 else 0.0
    return {
        "model_mse": model_mse,
        "naive_mse": naive_mse,
        "mse_skill": skill,
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "mae_skill": mae_skill,
        "direction_accuracy": _direction_accuracy(actual, predicted),
    }


def _direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    correct = np.sum((actual > 0) == (predicted > 0))
    return float(correct) / len(actual)


def hit_rate(actual: np.ndarray, predicted: np.ndarray, top_pct: float = 0.25) -> dict:
    pred_rank = np.argsort(predicted)
    n = len(predicted)
    top_n = max(1, int(n * top_pct))
    top_idx = pred_rank[-top_n:]
    bottom_idx = pred_rank[:top_n]
    top_actual = actual[top_idx]
    bottom_actual = actual[bottom_idx]
    return {
        "top_mean_return": float(np.mean(top_actual)),
        "bottom_mean_return": float(np.mean(bottom_actual)),
        "spread": float(np.mean(top_actual) - np.mean(bottom_actual)),
        "top_hit_rate": float(np.mean(top_actual > 0)),
        "bottom_hit_rate": float(np.mean(bottom_actual < 0)),
    }


def information_coefficient(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return 0.0
    return float(np.corrcoef(predicted, actual)[0, 1])
