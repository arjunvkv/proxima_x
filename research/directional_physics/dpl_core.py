"""DPL shared infrastructure: load data, compute ES, memory, residuals, future returns."""
import sys, json, warnings, importlib
from pathlib import Path
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent  # proxima_x/
sys.path.insert(0, str(SRC))

from research.adaptive_alpha_engine.aae_validator import AAEValidator, HORIZONS, _future_returns, _zscore
from research.energy_reality.energy_validator import EnergyValidator
from research.residual_energy.residual_validator import ResidualEnergyValidator

SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
HORIZON_LABELS = {1: "H1", 5: "H5", 20: "H20", 50: "H50", 100: "H100", 500: "H500"}


class DPLData:
    """Loads all data for a symbol: ES, memory_density, AT, residuals, future returns."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.aae = AAEValidator()
        self.energy = EnergyValidator(symbol)
        self.rep = ResidualEnergyValidator(symbol)

        self.signals = self.energy.load(symbol)          # triggers AAE load + compute_signals
        self.data = self.energy.data
        self.price = self.energy.price
        self.es = self.energy.es_signal()
        self.fut_ret = self.energy.fut_ret               # (n, 6) for H=[1,5,20,50,100,500]

        self.memory_density = self.signals.get("memory_density", np.zeros_like(self.es))
        self.adaptive_time = self.signals.get("adaptive_time", np.zeros_like(self.es))
        self.states = self.signals.get("states", np.zeros_like(self.es, dtype=np.int64))
        self.state_mutation = self.signals.get("state_mutation_rate", np.zeros_like(self.es))
        self.regime_change = self.signals.get("regime_change_probability", np.zeros_like(self.es))
        self.returns = self.signals.get("returns", np.zeros_like(self.es))

        self.rep.load(symbol)
        self.rep.build_residuals(force=False)
        self.residuals = {k: v for k, v in self.rep.residuals.items()}

        self.vol_metrics = self.energy.vol_metrics
        self.high = self.data["high"]
        self.low = self.data["low"]

    def valid_mask(self, min_es_pct: float = 80) -> np.ndarray:
        """High-ES states (above percentile)."""
        thr = np.nanpercentile(self.es, min_es_pct) if not np.all(np.isnan(self.es)) else 0
        return ~np.isnan(self.es) & ~np.isnan(self.fut_ret[:, 2]) & (self.es > thr)

    def future_return(self, horizon_idx: int = 2) -> np.ndarray:
        return self.fut_ret[:, horizon_idx]

    def abs_future_return(self, horizon_idx: int = 2) -> np.ndarray:
        return np.abs(self.fut_ret[:, horizon_idx])

    def es_percentile(self) -> np.ndarray:
        """Rolling percentile rank of ES (0-1)."""
        n = len(self.es)
        result = np.full(n, np.nan)
        window = 252
        for i in range(window, n):
            chunk = self.es[i - window:i]
            result[i] = np.sum(chunk <= self.es[i]) / window
        return result


def compute_gradient(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple gradient (change over window)."""
    g = np.full_like(x, np.nan)
    g[window:] = x[window:] - x[:-window]
    return g


def compute_acceleration(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Second derivative."""
    g1 = compute_gradient(x, window)
    g2 = compute_gradient(g1, window)
    return g2


def compute_curvature(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Curvature = second diff / (1 + first_diff^2)^1.5."""
    g1 = np.gradient(x)
    g2 = np.gradient(g1)
    denom = (1 + g1 ** 2) ** 1.5
    denom[denom == 0] = np.nan
    return g2 / denom


if __name__ == "__main__":
    d = DPLData("EURJPY")
    print(f"EURJPY: ES len={len(d.es)}, fut_ret shape={d.fut_ret.shape}, residuals={list(d.residuals.keys())}")
