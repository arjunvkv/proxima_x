import numpy as np
import pywt
import logging
from typing import Callable

logger = logging.getLogger("proxima_x.direction_lab")


class DirectionLab:
    WAVELETS = ["haar", "db4", "db8"]

    def __init__(self, price_loader: Callable = None):
        self._loader = price_loader
        self._matrix_cache = {}

    def compute_matrix(self, symbols: list[str],
                       wavelets: list[str] = None,
                       max_level: int = 6,
                       horizon: int = 20,
                       min_bars: int = 504) -> dict:
        if wavelets is None:
            wavelets = self.WAVELETS

        matrix = {
            "config": {
                "wavelets": wavelets,
                "max_level": max_level,
                "horizon": horizon,
                "min_bars": min_bars,
            },
            "symbols": {},
            "summary": {
                "best_symbol_wavelet_level": None,
                "best_directional_accuracy": 0.0,
                "overall_null_rejected": False,
            },
        }

        all_results = []

        for sym in symbols:
            sym_results = {}
            prices = self._load_prices(sym)
            if prices is None or len(prices) < min_bars:
                logger.warning(f"Not enough data for {sym}: {len(prices) if prices is not None else 0}")
                continue

            for wv in wavelets:
                wv_results = self._analyze_wavelet(prices, wv, max_level, horizon)
                sym_results[f"{wv}"] = wv_results
                for lvl, metrics in wv_results.items():
                    all_results.append((sym, wv, lvl, metrics))

            matrix["symbols"][sym] = sym_results

        if all_results:
            best = max(all_results, key=lambda x: x[3].get("directional_accuracy", 0))
            matrix["summary"]["best_symbol_wavelet_level"] = {
                "symbol": best[0],
                "wavelet": best[1],
                "level": best[2],
                "directional_accuracy": round(best[3].get("directional_accuracy", 0), 4),
                "expected_return": round(best[3].get("expected_return", 0), 6),
                "signal_count": best[3].get("signal_count", 0),
                "information_gain": round(best[3].get("information_gain", 0), 6),
            }
            matrix["summary"]["best_directional_accuracy"] = matrix["summary"]["best_symbol_wavelet_level"]["directional_accuracy"]

            # Null hypothesis test: count how many have accuracy > 0.52
            above_null = sum(1 for x in all_results if x[3].get("directional_accuracy", 0) > 0.52 and x[3].get("signal_count", 0) >= 10)
            matrix["summary"]["above_null_count"] = above_null
            matrix["summary"]["total_cells"] = len(all_results)
            matrix["summary"]["overall_null_rejected"] = above_null >= 2

        self._matrix_cache = matrix
        return matrix

    def _analyze_wavelet(self, prices: np.ndarray, wavelet: str,
                          max_level: int, horizon: int) -> dict:
        results = {}
        n = len(prices)

        try:
            coeffs = pywt.wavedec(prices, wavelet, mode="symmetric", level=max_level)
        except Exception as e:
            logger.warning(f"wavedec failed for {wavelet}: {e}")
            return results

        for level in range(1, max_level + 1):
            if level > len(coeffs) - 1:
                continue
            detail = coeffs[level]
            reconstructed_detail = self._reconstruct_level(prices, coeffs, level, wavelet)
            if len(reconstructed_detail) < n:
                reconstructed_detail = np.pad(reconstructed_detail,
                                              (n - len(reconstructed_detail), 0),
                                              mode="edge")

            metrics = self._compute_directional_metrics(prices, reconstructed_detail, horizon, n)
            metrics["detail_length"] = len(detail)
            metrics["wavelet"] = wavelet
            metrics["level"] = level
            results[f"L{level}"] = metrics

        return results

    def _reconstruct_level(self, prices, coeffs, level, wavelet):
        n = len(prices)
        detail_idx = len(coeffs) - level
        new_coeffs = []
        for i, c in enumerate(coeffs):
            if i == 0 or i != detail_idx:
                new_coeffs.append(np.zeros_like(c))
            else:
                new_coeffs.append(c.copy())
        recon = pywt.waverec(new_coeffs, wavelet)
        if len(recon) > n:
            return recon[:n]
        if len(recon) < n:
            return np.pad(recon, (0, n - len(recon)), mode="edge")
        return recon

    def _compute_directional_metrics(self, prices, detail_signal, horizon, n):
        lookback = 252
        state_values = detail_signal[-lookback:]
        future_returns = np.full(lookback, np.nan)
        for i in range(lookback - horizon):
            future_returns[i] = (prices[n - lookback + i + horizon] - prices[n - lookback + i]) / prices[n - lookback + i]

        valid = ~np.isnan(future_returns)
        if valid.sum() < 20:
            return {"directional_accuracy": 0.0, "expected_return": 0.0,
                    "signal_count": 0, "information_gain": 0.0}

        state_values = state_values[valid]
        future_returns = future_returns[valid]

        upper_thresh = np.percentile(state_values, 75)
        lower_thresh = np.percentile(state_values, 25)
        mid_thresh_low = np.percentile(state_values, 40)
        mid_thresh_high = np.percentile(state_values, 60)

        high_state = state_values > upper_thresh
        mid_high_state = (state_values > mid_thresh_high) & (state_values <= upper_thresh)
        mid_low_state = (state_values >= lower_thresh) & (state_values <= mid_thresh_low)
        low_state = state_values < lower_thresh
        neutral_state = (state_values >= mid_thresh_low) & (state_values <= mid_thresh_high)

        def _region_metrics(mask):
            if mask.sum() < 5:
                return None
            region_returns = future_returns[mask]
            correct = (region_returns > 0).mean()
            ev = region_returns.mean()
            return {"accuracy": float(correct), "expected_return": float(ev), "count": int(mask.sum())}

        upper = _region_metrics(high_state)
        upper_mid = _region_metrics(mid_high_state)
        lower_mid = _region_metrics(mid_low_state)
        lower = _region_metrics(low_state)
        neutral = _region_metrics(neutral_state)

        baseline_accuracy = (future_returns > 0).mean()

        directional_accuracy = 0.5
        expected_return = 0.0
        signal_count = 0

        if upper and lower:
            combined = np.concatenate([future_returns[high_state], -future_returns[low_state]])
            directional_accuracy = (combined > 0).mean()
            expected_return = float(np.mean(
                future_returns[high_state])) * 0.5 + float(np.mean(-future_returns[low_state])) * 0.5
            signal_count = int(high_state.sum() + low_state.sum())

        # Information gain: H(baseline) - H(conditional)
        baseline_entropy = self._entropy(baseline_accuracy)
        cond_acc = directional_accuracy if signal_count > 0 else 0.5
        cond_entropy = self._entropy(cond_acc) if signal_count > 0 else baseline_entropy
        information_gain = baseline_entropy - cond_entropy

        return {
            "directional_accuracy": round(float(directional_accuracy), 4),
            "expected_return": round(float(expected_return), 6),
            "signal_count": signal_count,
            "information_gain": round(float(information_gain), 6),
            "baseline_accuracy": round(float(baseline_accuracy), 4),
            "upper_region": upper,
            "upper_mid_region": upper_mid,
            "lower_mid_region": lower_mid,
            "lower_region": lower,
            "neutral_region": neutral,
        }

    @staticmethod
    def _entropy(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def _load_prices(self, symbol: str) -> np.ndarray:
        if self._loader:
            return self._loader(symbol)
        try:
            from proxima_ops.data import data_manager
            df = data_manager.get_rates(symbol, bars=2000)
            if df is not None and len(df) > 100:
                return df["close"].values.astype(np.float64)
        except Exception as e:
            logger.warning(f"Price load failed for {sym}: {e}")
        return None

    def format_report(self, matrix: dict) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("  FREQUENCY × SYMBOL × WAVELET DIRECTIONAL MATRIX")
        lines.append("=" * 64)
        lines.append(f"  Wavelets:   {matrix['config']['wavelets']}")
        lines.append(f"  Max Level:  {matrix['config']['max_level']}")
        lines.append(f"  Horizon:    {matrix['config']['horizon']} bars")
        lines.append("")

        best = matrix["summary"].get("best_symbol_wavelet_level")
        if best:
            lines.append("  BEST DIRECTIONAL POCKET")
            lines.append(f"  {best['symbol']} | {best['wavelet']} L{best['level']}")
            lines.append(f"  Accuracy:         {best['directional_accuracy']:.4f}")
            lines.append(f"  Expected Return:  {best['expected_return']:.6f}")
            lines.append(f"  Signal Count:     {best['signal_count']}")
            lines.append(f"  Information Gain: {best['information_gain']:.4f}")
            lines.append("")

        lines.append("  NULL HYPOTHESIS TEST")
        lines.append(f"  Total Cells:      {matrix['summary'].get('total_cells', 0)}")
        lines.append(f"  Above Null (52%): {matrix['summary'].get('above_null_count', 0)}")
        lines.append(f"  Null Rejected:    {matrix['summary'].get('overall_null_rejected', False)}")
        lines.append("")

        for sym, sym_data in matrix.get("symbols", {}).items():
            lines.append(f"  [{sym}]")
            for wv, wv_data in sym_data.items():
                best_lvl = max(wv_data.items(), key=lambda x: x[1].get("directional_accuracy", 0))
                lvl_key, m = best_lvl
                lines.append(f"    {wv} best={lvl_key}: acc={m.get('directional_accuracy', 0):.4f} "
                             f"ev={m.get('expected_return', 0):.6f} "
                             f"n={m.get('signal_count', 0)} "
                             f"ig={m.get('information_gain', 0):.4f}")
            lines.append("")

        lines.append("=" * 64)
        return "\n".join(lines)
