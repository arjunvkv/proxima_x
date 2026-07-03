from __future__ import annotations

import numpy as np
from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES, HORIZONS, _future_returns
from sklearn.linear_model import LinearRegression


class ResidualDeployment:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()
        ES = self.validator.es
        Res = self.validator.get_residual("xgboost")
        close = self.validator.price
        h = self.validator.data["high"]
        lo = self.validator.data["low"]
        returns = self.validator.data["returns"]
        n = len(close)

        print(f"\nREP-8: Benchmark Challenge for {self.asset}")
        es_alpha = self.validator.es_alpha(2)
        res_alpha = self.validator.residual_alpha("xgboost", 2)

        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            tr[i] = max(h[i] - lo[i], abs(h[i] - close[i-1]), abs(lo[i] - close[i-1]))
        atr = np.full(n, np.nan, dtype=np.float64)
        sma20 = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            atr[i] = float(np.mean(tr[i-20:i]))
            sma20[i] = float(np.mean(close[i-20:i]))
        atr_breakout = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            atr_breakout[i] = (close[i] - sma20[i]) / max(atr[i], 1e-12)

        donchian = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            rmax = np.max(h[i-20:i])
            rmin = np.min(lo[i-20:i])
            donchian[i] = (close[i] - rmin) / max(rmax - rmin + 1e-12, 1e-12)

        momentum = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            momentum[i] = close[i] / close[i-20] - 1.0

        vol_exp = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            vol_exp[i] = float(np.std(returns[i-20:i]))

        benchmarks = {
            "ES": es_alpha,
            "Residual XGB": res_alpha,
            "ATR Breakout": self.validator.eval_alpha(atr_breakout, 2),
            "Donchian": self.validator.eval_alpha(donchian, 2),
            "Momentum": self.validator.eval_alpha(momentum, 2),
            "Vol Expansion": self.validator.eval_alpha(vol_exp, 2),
        }

        ranked = sorted(benchmarks.items(), key=lambda x: x[1].get("sharpe", 0.0), reverse=True)
        print(f"  {'Benchmark':<20} {'Sharpe':>8} {'PP':>8} {'Mean':>10} {'N':>6}")
        print(f"  {'-'*52}")
        for name, alpha in ranked:
            print(f"  {name:<20} {alpha.get('sharpe', 0):>8.3f} {alpha.get('pp', 0):>8.3f} {alpha.get('mean', 0):>10.6f} {alpha.get('n', 0):>6}")

        residual_rank = next(i for i, (n_, _) in enumerate(ranked) if n_ == "Residual XGB") + 1
        es_rank = next(i for i, (n_, _) in enumerate(ranked) if n_ == "ES") + 1
        print(f"\n  Residual Rank: #{residual_rank}")
        print(f"  ES Rank: #{es_rank}")

        print(f"\nREP-9: Fragility Analysis for {self.asset}")
        thresholds = [80, 85, 90, 95, 97, 99]
        fragility = {}
        fwd = self.validator.fut_ret[:, 2]
        for th in thresholds:
            th_val = float(np.nanpercentile(Res, th))
            mask = Res > th_val
            n_sig = int(np.sum(mask))
            if n_sig < 5:
                pp = 0.5
                mean_val = 0.0
                sharpe = 0.0
            else:
                vals = fwd[mask]
                pp = float(np.mean(vals > 0))
                mean_val = float(np.nanmean(vals))
                std_val = float(np.nanstd(vals))
                sharpe = mean_val / max(std_val, 1e-12)
            fragility[str(th)] = {"pp": pp, "sharpe": sharpe, "mean": mean_val, "n": n_sig}

        max_sharpe = max(v["sharpe"] for v in fragility.values()) if fragility else 0.0
        plateau_count = sum(1 for v in fragility.values() if v["sharpe"] > 0.8 * max_sharpe) if max_sharpe > 0 else 0
        plateau_size = plateau_count / len(thresholds)

        print(f"  {'Threshold':>10} {'PP':>8} {'Sharpe':>8} {'Mean':>10} {'N':>6}")
        print(f"  {'-'*44}")
        for th in thresholds:
            v = fragility[str(th)]
            print(f"  {th:>10} {v['pp']:>8.3f} {v['sharpe']:>8.3f} {v['mean']:>10.6f} {v['n']:>6}")

        print(f"\n  Plateau Size: {plateau_size:.1%} ({plateau_count}/{len(thresholds)} thresholds > 0.8 * max_sharpe)")
        print(f"  Max Sharpe: {max_sharpe:.4f}")

        return REPResult(
            rq_name="REP-8",
            status="completed",
            metrics={
                "asset": self.asset,
                "benchmark_results": benchmarks,
                "residual_rank": residual_rank,
                "es_rank": es_rank,
                "fragility": fragility,
                "plateau_size": plateau_size,
            },
        )
