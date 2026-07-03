from __future__ import annotations

import numpy as np
from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES, HORIZONS, _future_returns
from sklearn.linear_model import LinearRegression


class ResidualTransfer:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()
        ES = self.validator.es
        Res = self.validator.get_residual("xgboost")
        n = len(ES)
        train_bars = int(n * 0.6)
        test_bars = int(n * 0.2)
        step = test_bars
        window_results = []
        es_survival = 0
        res_survival = 0
        total_windows = 0
        train_start = 0
        while train_start + train_bars + test_bars <= n:
            train_end = train_start + train_bars
            test_start = train_end
            test_end = min(test_start + test_bars, n)
            es_threshold = float(np.nanpercentile(ES[train_start:train_end], 90))
            res_threshold = float(np.nanpercentile(Res[train_start:train_end], 90))
            es_mask = ES[test_start:test_end] > es_threshold
            res_mask = Res[test_start:test_end] > res_threshold
            es_n = int(np.sum(es_mask))
            res_n = int(np.sum(res_mask))
            es_metrics = {}
            res_metrics = {}
            for h_idx, horizon in enumerate(HORIZONS):
                fwd = self.validator.fut_ret[test_start:test_end, h_idx]
                if es_n >= 5:
                    es_vals = fwd[es_mask]
                    es_pp = float(np.mean(es_vals > 0))
                    es_mean = float(np.nanmean(es_vals))
                    es_std = float(np.nanstd(es_vals))
                    es_sharpe = es_mean / max(es_std, 1e-12)
                else:
                    es_pp = 0.5
                    es_mean = 0.0
                    es_sharpe = 0.0
                if res_n >= 5:
                    res_vals = fwd[res_mask]
                    res_pp = float(np.mean(res_vals > 0))
                    res_mean = float(np.nanmean(res_vals))
                    res_std = float(np.nanstd(res_vals))
                    res_sharpe = res_mean / max(res_std, 1e-12)
                else:
                    res_pp = 0.5
                    res_mean = 0.0
                    res_sharpe = 0.0
                es_metrics[f"H{horizon}"] = {"pp": es_pp, "mean": es_mean, "sharpe": es_sharpe, "n": es_n}
                res_metrics[f"H{horizon}"] = {"pp": res_pp, "mean": res_mean, "sharpe": res_sharpe, "n": res_n}
            total_windows += 1
            es_h20 = es_metrics.get("H20", {"pp": 0.5, "mean": 0.0})
            res_h20 = res_metrics.get("H20", {"pp": 0.5, "mean": 0.0})
            if es_h20["pp"] > 0.55 and es_h20["mean"] > 0:
                es_survival += 1
            if res_h20["pp"] > 0.55 and res_h20["mean"] > 0:
                res_survival += 1
            print(
                f"  Window {total_windows}: train=[{train_start}:{train_end}] "
                f"test=[{test_start}:{test_end}] "
                f"ES_H20_pp={es_h20['pp']:.3f} ES_H20_mean={es_h20['mean']:.6f} "
                f"Res_H20_pp={res_h20['pp']:.3f} Res_H20_mean={res_h20['mean']:.6f}"
            )
            window_results.append({
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "es_h20_pp": es_h20["pp"],
                "res_h20_pp": res_h20["pp"],
            })
            train_start += step
        es_survival_rate = es_survival / max(total_windows, 1)
        res_survival_rate = res_survival / max(total_windows, 1)
        residual_survives_better = res_survival_rate > es_survival_rate
        print(f"\nResidual Transfer Walk-Forward Results for {self.asset}:")
        print(f"  Total Windows: {total_windows}")
        print(f"  ES Survival Rate: {es_survival_rate:.2%}")
        print(f"  Residual Survival Rate: {res_survival_rate:.2%}")
        print(f"  Residual Survives Better: {residual_survives_better}")
        return REPResult(
            rq_name="REP-7",
            status="completed",
            metrics={
                "asset": self.asset,
                "es_survival_rate": es_survival_rate,
                "residual_survival_rate": res_survival_rate,
                "n_windows": total_windows,
                "window_results": window_results,
                "residual_survives_better": residual_survives_better,
            },
        )
