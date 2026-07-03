from __future__ import annotations
import numpy as np
from research.energy_reality.energy_validator import EnergyValidator, ERLResult


class Fragility:
    def __init__(self, validator: EnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ERLResult:
        self.validator.load(self.asset)
        price = self.validator.price
        returns = self.validator.signals["returns"]
        fut_ret = self.validator.fut_ret

        windows = [5, 10, 20, 30, 50, 75, 100]
        thresholds = [80, 85, 90, 95, 97, 99]

        n = len(returns)
        sharpe_grid = np.full((len(windows), len(thresholds)), np.nan)
        pp_grid = np.full((len(windows), len(thresholds)), np.nan)
        mean_grid = np.full((len(windows), len(thresholds)), np.nan)

        fwd = fut_ret[:, 2]

        for i, window in enumerate(windows):
            signal = np.full(n, np.nan, dtype=np.float64)
            for t in range(window, n):
                signal[t] = float(np.nanstd(returns[t - window:t])) * np.sqrt(252)

            for j, threshold in enumerate(thresholds):
                threshold_val = float(np.nanpercentile(signal, threshold))
                mask = signal > threshold_val
                vals = fwd[mask]
                if np.sum(mask) < 5:
                    pp_grid[i, j] = 0.5
                    mean_grid[i, j] = 0.0
                    sharpe_grid[i, j] = 0.0
                else:
                    pp = float(np.mean(vals > 0))
                    mean_val = float(np.nanmean(vals))
                    std_val = float(np.nanstd(vals))
                    sr = mean_val / max(std_val, 1e-12)
                    pp_grid[i, j] = pp
                    mean_grid[i, j] = mean_val
                    sharpe_grid[i, j] = sr

        max_sharpe = float(np.nanmax(sharpe_grid))
        max_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)
        max_sharpe_window = windows[max_idx[0]]
        max_sharpe_threshold = thresholds[max_idx[1]]

        total_pairs = len(windows) * len(thresholds)
        count = int(np.sum(sharpe_grid >= 0.8 * max_sharpe))
        plateau_size = count / total_pairs
        passes = plateau_size > 0.3
        verdict = "PASS: Parameter space shows broad plateau (ES is robust)" if passes else "FAIL: No broad plateau detected (ES is fragile)"

        self._print_tables(windows, thresholds, pp_grid, sharpe_grid,
                           max_sharpe, max_sharpe_window, max_sharpe_threshold,
                           plateau_size, passes, verdict)

        metrics = {
            "windows": windows,
            "thresholds": thresholds,
            "sharpe_grid": sharpe_grid.tolist(),
            "pp_grid": pp_grid.tolist(),
            "max_sharpe": max_sharpe,
            "max_sharpe_window": max_sharpe_window,
            "max_sharpe_threshold": max_sharpe_threshold,
            "plateau_size": plateau_size,
            "passes": bool(passes),
            "verdict": verdict,
        }

        return ERLResult(
            rq_name="ERL-4: Parameter Fragility Test",
            status="PASS" if passes else "FAIL",
            metrics=metrics,
        )

    def _print_tables(self, windows, thresholds, pp_grid, sharpe_grid,
                      max_sharpe, max_sw, max_st, plateau_size, passes, verdict):
        label = "Win\\Thr"
        header = f"{label:>8}" + "".join(f"{t:>8}" for t in thresholds)
        sep = "-" * (8 + 8 * len(thresholds))

        print("\n=== Sharpe Ratio Grid ===")
        print(header)
        print(sep)
        for i, w in enumerate(windows):
            row = f"{w:>8}" + "".join(f"{sharpe_grid[i, j]:>8.4f}" for j in range(len(thresholds)))
            print(row)

        print("\n=== Hit Rate (PP) Grid ===")
        print(header)
        print(sep)
        for i, w in enumerate(windows):
            row = f"{w:>8}" + "".join(f"{pp_grid[i, j]:>8.4f}" for j in range(len(thresholds)))
            print(row)

        print(f"\nMax Sharpe: {max_sharpe:.4f} at window={max_sw}, threshold={max_st}")
        print(f"Plateau size: {plateau_size:.2%} ({int(plateau_size * len(windows) * len(thresholds))}/{len(windows) * len(thresholds)} pairs >= 80% of max)")
        print(f"{'PASS' if passes else 'FAIL'}: {verdict}")
