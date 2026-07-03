from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES, TIME_WINDOWS
from research.adaptive_alpha_engine.aae_validator import TARGET_ASSETS, HORIZONS, _future_returns


class ResidualReality:
    def __init__(self, validator: ResidualEnergyValidator):
        self.validator = validator

    def _compute_window_vol_metrics(self, data, price, window=20):
        h = data["high"]
        lo = data["low"]
        c = price
        r = data["returns"]
        n = len(price)
        vm = {}

        rv_arr = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            rv_arr[i] = float(np.nanstd(r[i - window:i])) * np.sqrt(252)
        vm["realized_vol"] = rv_arr

        pk = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            logs = np.log(h[i - window:i] / lo[i - window:i])
            pk[i] = float(np.sqrt(np.mean(logs ** 2) / (4.0 * np.log(2.0)))) * np.sqrt(252)
        vm["parkinson_vol"] = pk

        gk = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            hl = np.log(h[i - window:i] / lo[i - window:i])
            if i - window > 0:
                co = np.log(c[i - window:i] / price[i - window - 1:i - 1])
            else:
                co = np.zeros(window)
            term1 = 0.5 * np.mean(hl ** 2)
            term2 = (2.0 * np.log(2.0) - 1.0) * np.mean(co ** 2)
            gk[i] = float(np.sqrt(max(term1 - term2, 1e-12))) * np.sqrt(252)
        vm["garman_klass_vol"] = gk

        rs = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            hl = np.log(h[i - window:i] / c[i - window:i])
            lc = np.log(lo[i - window:i] / c[i - window:i])
            if i - window > 0:
                hc = np.log(h[i - window:i] / price[i - window - 1:i - 1])
                lc2 = np.log(lo[i - window:i] / price[i - window - 1:i - 1])
            else:
                hc = np.zeros(window)
                lc2 = np.zeros(window)
            term = np.mean(hl * (hl - lc) + hc * (hc - lc2))
            rs[i] = float(np.sqrt(max(term, 1e-12))) * np.sqrt(252)
        vm["rogers_satchell_vol"] = rs

        atr = np.zeros(n, dtype=np.float64)
        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            tr[i] = max(h[i] - lo[i], abs(h[i] - price[i - 1]), abs(lo[i] - price[i - 1]))
        for i in range(window, n):
            atr[i] = float(np.mean(tr[i - window:i]))
        vm["atr"] = atr

        vv = np.zeros(n, dtype=np.float64)
        for i in range(window * 2, n):
            vv[i] = float(np.std(rv_arr[i - window:i]))
        vm["vol_of_vol"] = vv

        ent = np.zeros(n, dtype=np.float64)
        bins = 20
        for i in range(window, n):
            hist, _ = np.histogram(r[i - window:i], bins=bins, density=True)
            hist = hist[hist > 0]
            ent[i] = float(-np.sum(hist * np.log(hist + 1e-12)))
        vm["entropy"] = ent

        ec = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ec[i] = ent[i] - ent[i - 1]
        vm["entropy_change"] = ec

        return vm

    def _evaluate_window(self, asset, start, end):
        aae = self.validator.energy.aae
        data = aae.load_data_window(asset, start, end)
        signals = aae.compute_signals(data)
        price = np.asarray(signals["price"], dtype=np.float64)
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        n = len(price)

        vol_metrics = self._compute_window_vol_metrics(data, price)
        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fut_ret = _future_returns(price, horizons_arr)

        vol_names = list(vol_metrics.keys())
        X_list = []
        valid = np.ones(n, dtype=bool)
        for name in vol_names:
            arr = vol_metrics[name]
            X_list.append(arr)
            valid = valid & ~np.isnan(arr)
        valid = valid & ~np.isnan(es)

        X = np.column_stack([v[valid] for v in X_list])
        y = es[valid]

        configs = [
            ("linear", LinearRegression()),
            ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
            ("xgboost", XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbosity=0)),
        ]

        residuals = {}
        for name, model in configs:
            model.fit(X, y)
            y_pred = model.predict(X)
            residual_full = np.full(n, np.nan, dtype=np.float64)
            residual_full[valid] = y - y_pred
            residuals[name] = residual_full

        es_alpha = aae.eval_alpha(es, fut_ret, 2)

        res_alphas = {}
        for rt in ["xgboost", "linear", "random_forest"]:
            res_alphas[rt] = aae.eval_alpha(residuals[rt], fut_ret, 2)

        return es_alpha, res_alphas

    def run(self) -> REPResult:
        # Part A: Cross-Asset Survival (REP-5)
        print(f"\n{'='*72}")
        print(f"  REP-5: Cross-Asset Survival")
        print(f"{'='*72}")
        print(f"\n  {'Asset':<10} {'ES PP':>8} {'XGB PP':>8} {'Lin PP':>8} {'RF PP':>8}  {'XGB/ES':>8}  {'Lin/ES':>8}  {'RF/ES':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        cross_asset = {}
        n_assets_residual_beats_es = 0

        for asset in TARGET_ASSETS:
            self.validator.load(asset)
            self.validator.build_residuals()

            es_a = self.validator.es_alpha(2)
            xgb_a = self.validator.residual_alpha("xgboost", 2)
            lin_a = self.validator.residual_alpha("linear", 2)
            rf_a = self.validator.residual_alpha("random_forest", 2)

            es_pp = es_a["pp"]
            xgb_tr = xgb_a["pp"] / es_pp if es_pp > 0 else 0.0
            lin_tr = lin_a["pp"] / es_pp if es_pp > 0 else 0.0
            rf_tr = rf_a["pp"] / es_pp if es_pp > 0 else 0.0

            cross_asset[asset] = {
                "es_alpha": es_a,
                "residual_alpha": {"xgboost": xgb_a, "linear": lin_a, "random_forest": rf_a},
                "transfer_ratio": {"xgboost": xgb_tr, "linear": lin_tr, "random_forest": rf_tr},
            }

            if xgb_a["pp"] > es_pp:
                n_assets_residual_beats_es += 1

            print(f"  {asset:<10} {es_pp:>8.4f} {xgb_a['pp']:>8.4f} {lin_a['pp']:>8.4f} {rf_a['pp']:>8.4f}  {xgb_tr:>8.4f}  {lin_tr:>8.4f}  {rf_tr:>8.4f}")

        # Part B: Cross-Time Survival (REP-6)
        print(f"\n{'='*72}")
        print(f"  REP-6: Cross-Time Survival")
        print(f"{'='*72}")
        print(f"\n  {'Window':<12} {'ES PP':>8} {'XGB PP':>8} {'ES Surv?':>9} {'XGB Surv?':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

        cross_time = {}
        n_windows_residual_survives = 0
        n_windows_es_survives = 0

        for start, end, label in TIME_WINDOWS:
            es_a, res_a = self._evaluate_window("EURJPY", start, end)

            es_pp = es_a["pp"]
            xgb_pp = res_a["xgboost"]["pp"]
            es_survives = es_pp > 0.55
            res_survives = xgb_pp > 0.55

            if es_survives:
                n_windows_es_survives += 1
            if res_survives:
                n_windows_residual_survives += 1

            cross_time[label] = {
                "es_pp": es_pp,
                "residual_pp": xgb_pp,
                "es_survives": es_survives,
                "residual_survives": res_survives,
            }

            print(f"  {label:<12} {es_pp:>8.4f} {xgb_pp:>8.4f} {str(es_survives):>9} {str(res_survives):>10}")

        n_windows = len(TIME_WINDOWS)
        es_survival_rate = n_windows_es_survives / n_windows
        residual_survival_rate = n_windows_residual_survives / n_windows
        residual_transfers_better = residual_survival_rate > es_survival_rate

        print(f"\n  ES Survival Rate:      {es_survival_rate:.2%}")
        print(f"  Residual Survival Rate: {residual_survival_rate:.2%}")
        print(f"  Residual transfers better: {residual_transfers_better}")
        print()

        metrics = {
            "cross_asset": cross_asset,
            "cross_time": cross_time,
            "n_assets_residual_beats_es": n_assets_residual_beats_es,
            "residual_survival_rate": residual_survival_rate,
            "es_survival_rate": es_survival_rate,
            "residual_transfers_better": residual_transfers_better,
        }

        return REPResult(rq_name="REP-5+6", status="COMPLETE", metrics=metrics)
