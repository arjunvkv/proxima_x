"""TDD (Temporal Distortion Dynamics) core: tick loading, event rate, acceleration, distortion."""
import sys, warnings
from pathlib import Path
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

SYMBOLS = ["EURJPY", "USDJPY", "GBPUSD"]
SYMBOL_TICK_FILES = {
    "EURJPY": "data/ticks/EURJPY_ticks.parquet",
    "USDJPY": "data/ticks/USDJPY_ticks.parquet",
    "GBPUSD": "data/ticks/GBPUSD_ticks.parquet",
    "EURUSD": "data/ticks/EURUSD_ticks.parquet",
}

HORIZON_LABELS = {5: "H5", 20: "H20", 50: "H50"}


class TDDCore:
    """Loads tick data, computes event rate λ(t), acceleration α(t), distortion δ(t)."""

    def __init__(self, symbol: str, event_threshold: float = 0.0):
        self.symbol = symbol
        self.event_threshold = event_threshold
        self.ticks = None
        self.events = None
        self.timestamps = None
        self.lmbda = None
        self.alpha = None
        self.delta = None
        self.future_returns = {}
        self.bar_times = None

    def load_ticks(self, tick_path: str = None) -> int:
        """Load tick parquet, return count."""
        if tick_path is None:
            tick_path = SYMBOL_TICK_FILES.get(self.symbol)
            if tick_path is None:
                raise ValueError(f"No tick file mapping for {self.symbol}")
        root = Path(__file__).resolve().parent.parent.parent.parent  # goes up to Agentic_Trading/
        full_path = root / tick_path
        if not full_path.exists():
            raise FileNotFoundError(f"Tick file not found: {full_path}")
        self.ticks = pl.read_parquet(str(full_path))
        self.ticks = self.ticks.sort("timestamp")
        return len(self.ticks)

    def detect_events(self, use_bid: bool = True, use_ask: bool = True):
        """Detect price change events from ticks."""
        if self.ticks is None:
            raise ValueError("Load ticks first")
        prices = []
        if use_bid and "bid" in self.ticks.columns:
            prices.append(self.ticks["bid"].to_numpy())
        if use_ask and "ask" in self.ticks.columns:
            prices.append(self.ticks["ask"].to_numpy())
        if not prices:
            raise ValueError("No price columns available")
        mid = np.column_stack(prices).mean(axis=1)
        changed = np.zeros(len(mid), dtype=bool)
        changed[1:] = np.abs(np.diff(mid)) > self.event_threshold
        self.events = self.ticks.filter(changed)
        self.timestamps = self.events["timestamp"].to_numpy()
        return len(self.events)

    def compute_event_rate(self, window_seconds: int = 60):
        """Compute rolling event rate λ(t) = events per second using searchsorted (O(n log n))."""
        if self.timestamps is None or len(self.timestamps) == 0:
            raise ValueError("Detect events first")
        ts = self.timestamps.astype(np.float64) / 1_000_000
        n = len(ts)
        window_start = ts - window_seconds
        left_idx = np.searchsorted(ts, window_start, side="left")
        right_idx = np.arange(n)
        self.lmbda = (right_idx - left_idx + 1).astype(np.float64) / window_seconds
        self._window_seconds = window_seconds
        return self.lmbda

    def compute_acceleration(self, smooth: int = 5):
        """Compute α(t) = smoothed derivative of λ(t)."""
        if self.lmbda is None:
            raise ValueError("Compute event rate first")
        self.alpha = np.full_like(self.lmbda, np.nan)
        if smooth < len(self.lmbda):
            self.alpha[smooth:] = self.lmbda[smooth:] - self.lmbda[:-smooth]
        return self.alpha

    def compute_distortion(self, baseline_percentile: float = 50):
        """Compute δ(t) = current λ / baseline λ (ratio of current to typical)."""
        if self.lmbda is None:
            raise ValueError("Compute event rate first")
        baseline = np.nanpercentile(self.lmbda, baseline_percentile)
        if baseline == 0:
            baseline = np.nanmean(self.lmbda)
        self.delta = self.lmbda / baseline if baseline > 0 else np.ones_like(self.lmbda)
        return self.delta

    def build_bar_grid(self, bar_seconds: int = 300):
        """Build regular calendar-time bar grid using searchsorted (vectorized)."""
        if self.timestamps is None:
            raise ValueError("Detect events first")
        ts_sec = self.timestamps.astype(np.float64) / 1_000_000
        t_min = ts_sec[0]
        t_max = ts_sec[-1]
        bar_starts = np.arange(t_min, t_max, bar_seconds)
        n_bars = len(bar_starts) - 1
        bar_lmbda = np.full(n_bars, np.nan)
        bar_alpha = np.full(n_bars, np.nan)
        bar_delta = np.full(n_bars, np.nan)
        bar_time = np.full(n_bars, np.nan)
        bar_time[:] = (bar_starts[:-1] + bar_starts[1:]) / 2
        left_idx = np.searchsorted(ts_sec, bar_starts[:-1], side="left")
        right_idx = np.searchsorted(ts_sec, bar_starts[1:], side="left")
        for i in range(n_bars):
            l, r = left_idx[i], right_idx[i]
            if r > l:
                bar_lmbda[i] = np.nanmean(self.lmbda[l:r]) if self.lmbda is not None else np.nan
                if self.alpha is not None:
                    bar_alpha[i] = np.nanmean(self.alpha[l:r])
                if self.delta is not None:
                    bar_delta[i] = np.nanmean(self.delta[l:r])
        self.bar_times = bar_time
        self.bar_lmbda = bar_lmbda
        self.bar_alpha = bar_alpha
        self.bar_delta = bar_delta
        self._bar_seconds = bar_seconds
        return bar_time, bar_lmbda, bar_alpha, bar_delta

    def compute_future_returns(self, horizons: list = None, price_col: str = "bid"):
        """Compute future returns at each bar from tick mid-prices."""
        if self.ticks is None or self.bar_times is None:
            raise ValueError("Load ticks and build bar grid first")
        if horizons is None:
            horizons = [5, 20, 50]
        prices = self.ticks[price_col].to_numpy() if price_col in self.ticks.columns else self.ticks["bid"].to_numpy()
        ts_sec = self.ticks["timestamp"].to_numpy().astype(np.float64) / 1_000_000
        n_bars = len(self.bar_times)
        self.future_returns = {}
        for h in horizons:
            fut = np.full(n_bars, np.nan)
            h_seconds = h * self._bar_seconds
            for i in range(n_bars):
                if np.isnan(self.bar_times[i]):
                    continue
                target_time = self.bar_times[i] + h_seconds
                idx = np.searchsorted(ts_sec, target_time)
                if idx < len(prices):
                    # Price at current bar
                    cur_idx = np.searchsorted(ts_sec, self.bar_times[i])
                    if cur_idx < len(prices):
                        fut[i] = (prices[idx] - prices[cur_idx]) / prices[cur_idx]
            self.future_returns[h] = fut
        return self.future_returns

    def run_full_pipeline(self, tick_path: str = None, window_seconds: int = 60,
                          bar_seconds: int = 300, smooth: int = 5,
                          horizons: list = None):
        """Run the full TDD pipeline end-to-end."""
        n = self.load_ticks(tick_path)
        self.detect_events()
        self.compute_event_rate(window_seconds)
        self.compute_acceleration(smooth)
        self.compute_distortion()
        self.build_bar_grid(bar_seconds)
        self.compute_future_returns(horizons)
        return n


def compute_directional_metrics(alpha: np.ndarray, delta: np.ndarray, future_ret: np.ndarray,
                                 label: str, alpha_thresholds: list = None, delta_thresholds: list = None):
    """Compute P(up | α > a_thresh AND δ > d_thresh) for various thresholds."""
    from collections import OrderedDict
    if alpha_thresholds is None:
        alpha_thresholds = [0] + [np.nanpercentile(np.abs(alpha), p) for p in [50, 75, 90]]
    if delta_thresholds is None:
        delta_thresholds = [1.0] + [np.nanpercentile(delta, p) for p in [50, 75, 90]]
    results = OrderedDict()
    for a_thresh in alpha_thresholds:
        for d_thresh in delta_thresholds:
            key = f"α>{a_thresh:.4f}_δ>{d_thresh:.4f}"
            mask = (alpha > a_thresh) & (delta > d_thresh) & ~np.isnan(future_ret)
            n_total = np.sum(mask)
            if n_total < 10:
                results[key] = {"label": label, "n": n_total, "p_up": np.nan, "alpha_thresh": a_thresh, "delta_thresh": d_thresh}
                continue
            p_up = np.mean(future_ret[mask] > 0)
            results[key] = {
                "label": label,
                "n": n_total,
                "p_up": round(p_up, 4),
                "alpha_thresh": round(a_thresh, 4),
                "delta_thresh": round(d_thresh, 4),
                "mean_ret": round(np.mean(future_ret[mask]), 6),
            }
    return results


def compute_inflection_metrics(alpha: np.ndarray, future_ret: np.ndarray, label: str):
    """Test P(up | α crosses zero, i.e., inflection point)."""
    n = len(alpha)
    cross_up = np.full(n, False)
    cross_down = np.full(n, False)
    for i in range(1, n):
        if ~np.isnan(alpha[i]) and ~np.isnan(alpha[i - 1]):
            if alpha[i - 1] < 0 and alpha[i] > 0:
                cross_up[i] = True
            elif alpha[i - 1] > 0 and alpha[i] < 0:
                cross_down[i] = True
    results = {}
    for name, mask in [("zero_cross_up", cross_up), ("zero_cross_down", cross_down), ("any_cross", cross_up | cross_down)]:
        valid = mask & ~np.isnan(future_ret)
        n_total = np.sum(valid)
        if n_total >= 10:
            p_up = np.mean(future_ret[valid] > 0)
            results[name] = {"label": label, "n": int(n_total), "p_up": round(p_up, 4), "mean_ret": round(np.mean(future_ret[valid]), 6)}
        else:
            results[name] = {"label": label, "n": int(n_total), "p_up": np.nan, "mean_ret": np.nan}
    return results


def compute_sync_metrics(alpha: np.ndarray, delta: np.ndarray, future_ret: np.ndarray, label: str):
    """Test P(up | both acceleration and elevated distortion — 'temporal sync')."""
    sync_up = (alpha > 0) & (delta > 1.0) & ~np.isnan(future_ret)
    sync_down = (alpha < 0) & (delta < 1.0) & ~np.isnan(future_ret)
    results = {}
    for name, mask in [("sync_up_accel_high_delta", sync_up), ("sync_down_decel_low_delta", sync_down)]:
        n_total = np.sum(mask)
        if n_total >= 10:
            p_up = np.mean(future_ret[mask] > 0)
            results[name] = {"label": label, "n": int(n_total), "p_up": round(p_up, 4), "mean_ret": round(np.mean(future_ret[mask]), 6)}
        else:
            results[name] = {"label": label, "n": int(n_total), "p_up": np.nan, "mean_ret": np.nan}
    return results
