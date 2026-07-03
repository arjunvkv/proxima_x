from __future__ import annotations
import numpy as np
from research.energy_reality.energy_validator import EnergyValidator, ERLResult, TARGET_ASSETS


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = np.nanmean(arr[i - window + 1:i + 1])
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = np.nanstd(arr[i - window + 1:i + 1])
    return result


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = np.nanmax(arr[i - window + 1:i + 1])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = np.nanmin(arr[i - window + 1:i + 1])
    return result


def _rolling_percentile(arr: np.ndarray, window: int, pct: float) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        chunk = arr[i - window + 1:i + 1]
        result[i] = float(np.nanpercentile(chunk, pct))
    return result


def _true_range(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(c)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    return tr


BENCHMARK_NAMES = [
    "ATR Breakout",
    "Donchian Breakout",
    "Volatility Expansion",
    "Volatility Compression Release",
    "Momentum",
    "Simple Trend Following",
]


class BenchmarkComparison:
    def __init__(self, validator: EnergyValidator):
        self.validator = validator

    def _atr_breakout(self, c: np.ndarray, h: np.ndarray, lo: np.ndarray) -> np.ndarray:
        window = 20
        tr = _true_range(h, lo, c)
        atr = _rolling_mean(tr, window)
        ma = _rolling_mean(c, window)
        signal = (c - ma) / (atr + 1e-12)
        return signal

    def _donchian_breakout(self, c: np.ndarray, h: np.ndarray, lo: np.ndarray) -> np.ndarray:
        window = 20
        top = _rolling_max(h, window)
        bot = _rolling_min(lo, window)
        signal = (c - bot) / (top - bot + 1e-12)
        return signal

    def _volatility_expansion(self, r: np.ndarray) -> np.ndarray:
        window = 20
        signal = _rolling_std(r, window)
        return signal

    def _volatility_compression_release(self, r: np.ndarray) -> np.ndarray:
        window = 20
        rv = _rolling_std(r, window)
        compression = -rv
        low_comp = _rolling_percentile(compression, window, 10)
        signal = np.where(compression < low_comp, 1.0, 0.0)
        return signal

    def _momentum(self, c: np.ndarray) -> np.ndarray:
        window = 20
        n = len(c)
        signal = np.full(n, np.nan, dtype=np.float64)
        for i in range(window, n):
            signal[i] = c[i] / c[i - window] - 1.0
        return signal

    def _simple_trend_following(self, c: np.ndarray) -> np.ndarray:
        fast = 10
        slow = 40
        fast_ma = _rolling_mean(c, fast)
        slow_ma = _rolling_mean(c, slow)
        return fast_ma - slow_ma

    def run(self) -> ERLResult:
        all_results = {}
        es_ranks = []

        header = f"{'Asset':<10} {'ES Sharpe':>10}"
        for bn in BENCHMARK_NAMES:
            header += f" {bn:>28}"
        header += f" {'ES Rank':>8}"
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for asset in TARGET_ASSETS:
            self.validator.load(asset)
            p = self.validator.price
            c = p
            h = self.validator.data["high"]
            lo = self.validator.data["low"]
            r = self.validator.data["returns"]

            es_sig = self.validator.es_signal()
            es_alpha = self.validator.eval_alpha(es_sig, 2)
            es_sharpe = es_alpha["sharpe"]

            benchmarks = {
                "ES": es_alpha,
            }
            bench_sigs = {
                "ATR Breakout": self._atr_breakout(c, h, lo),
                "Donchian Breakout": self._donchian_breakout(c, h, lo),
                "Volatility Expansion": self._volatility_expansion(r),
                "Volatility Compression Release": self._volatility_compression_release(r),
                "Momentum": self._momentum(c),
                "Simple Trend Following": self._simple_trend_following(c),
            }

            for bn, sig in bench_sigs.items():
                benchmarks[bn] = self.validator.eval_alpha(sig, 2)

            all_results[asset] = benchmarks

            by_sharpe = sorted(benchmarks.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)
            rank = 1
            for name, _ in by_sharpe:
                if name == "ES":
                    break
                rank += 1
            es_ranks.append(rank)

            line = f"{asset:<10} {es_sharpe:>10.4f}"
            for bn in BENCHMARK_NAMES:
                sharpe = benchmarks[bn]["sharpe"]
                line += f" {sharpe:>28.4f}"
            line += f" {rank:>8d}"
            lines.append(line)

        lines.append(separator)
        avg_rank = float(np.mean(es_ranks))
        passes = avg_rank <= 2.0
        verdict = "PASS" if passes else "FAIL"

        lines.append(f"\nAverage ES Rank: {avg_rank:.2f}")
        lines.append(f"Threshold: <= 2.00")
        lines.append(f"Verdict: {verdict}")

        print("\n".join(lines))

        return ERLResult(
            rq_name="ERL-3: Institutional Benchmark Test",
            status=verdict,
            metrics={
                "per_asset": all_results,
                "average_es_rank": avg_rank,
                "passes": passes,
                "verdict": verdict,
            },
        )
