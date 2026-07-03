from __future__ import annotations
import numpy as np
from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult, HORIZONS, _future_returns


class ExecutionStress:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]
        signals = self.validator.compute_signals(data)
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))
        n = min(len(es), fr_all.shape[0])
        es, fr_all = es[:n], fr_all[:n]

        base_cost = 0.00013
        stress_levels = [1, 2, 5, 10]
        es_threshold = np.nanpercentile(es, 90)
        es_mask = es >= es_threshold

        all_results = {}

        print(f"  Execution Stress Test ({self.asset}):")
        header = f"    {'Stress':>6s} | {'H20_pp_gross':>12s} {'H20_pp_net':>11s} {'H20_sharpe_net':>14s} {'Degradation':>12s} {'Survives':>9s}"
        print(header)
        print(f"    {'-' * len(header.strip())}")

        death_stress = None

        for multiplier in stress_levels:
            adjusted_cost = base_cost * multiplier
            stress_results = {}

            for hi, h in enumerate(HORIZONS):
                fwd = fr_all[:, hi]
                gross_returns = fwd[es_mask]
                gross_returns = gross_returns[~np.isnan(gross_returns)]

                if len(gross_returns) < 5:
                    continue

                gross_mean = float(np.nanmean(gross_returns))
                gross_std = float(np.nanstd(gross_returns))
                gross_sharpe = gross_mean / max(gross_std, 1e-12)
                gross_pp = float(np.mean(gross_returns > 0))

                net_returns = gross_returns - adjusted_cost / h
                net_mean = float(np.nanmean(net_returns))
                net_std = float(np.nanstd(net_returns))
                net_sharpe = net_mean / max(net_std, 1e-12)
                net_pp = float(np.mean(net_returns > 0))

                degradation = (gross_pp - net_pp) / max(gross_pp, 1e-12) if gross_pp > 0 else 0

                stress_results[f"H{h}"] = {
                    "multiplier": multiplier,
                    "cost": adjusted_cost,
                    "gross_mean": gross_mean,
                    "gross_pp": gross_pp,
                    "gross_sharpe": gross_sharpe,
                    "net_mean": net_mean,
                    "net_pp": net_pp,
                    "net_sharpe": net_sharpe,
                    "degradation": degradation,
                    "n": len(gross_returns),
                }

            h20 = stress_results.get("H20", {})
            h20_pp_gross = h20.get("gross_pp", 0)
            h20_pp_net = h20.get("net_pp", 0)
            h20_sharpe_net = h20.get("net_sharpe", 0)
            deg = h20.get("degradation", 0)
            survives = h20_pp_net > 0.52

            if not survives and death_stress is None:
                death_stress = multiplier

            print(f"    {multiplier:>6d}x | {h20_pp_gross:>12.4f} {h20_pp_net:>11.4f} {h20_sharpe_net:>14.4f} {deg:>12.4f} {'YES' if survives else 'NO':>9s}")

            all_results[f"{multiplier}x"] = stress_results

        if death_stress is not None:
            print(f"    Alpha death at: {death_stress}x cost multiplier")
        else:
            print("    Alpha survives all stress levels")

        return AAEResult("execution_stress", "COMPLETE", metrics={
            "base_cost": base_cost,
            "results": all_results,
            "death_stress": death_stress,
        })
