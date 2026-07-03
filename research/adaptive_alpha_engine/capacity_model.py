from __future__ import annotations
import numpy as np
from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult, HORIZONS, _future_returns


class CapacityModel:
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

        es_threshold = np.nanpercentile(es, 90)
        es_mask = es >= es_threshold
        total_signals = int(np.sum(es_mask))

        n_bars = len(price)
        bars_per_day = 24
        bars_per_year = bars_per_day * 365
        total_years = n_bars / max(bars_per_year, 1)
        signals_per_year = total_signals / max(total_years, 1)

        h20_idx = HORIZONS.index(20)
        fwd = fr_all[:, h20_idx]
        es_returns = fwd[es_mask]
        es_returns = es_returns[~np.isnan(es_returns)]
        gross_mean_return = float(np.nanmean(es_returns)) if len(es_returns) > 0 else 0.0

        holding_period_bars = 20
        holding_period_days = holding_period_bars / bars_per_day
        capital_turnover = signals_per_year * (1 / max(holding_period_days, 1)) * 2

        base_cost = 0.00013
        capital_levels = [100000, 1000000, 10000000, 100000000]

        print(f"  Capacity Model ({self.asset}):")
        print(f"    Total ES signals:     {total_signals}")
        print(f"    Signals/year:         {signals_per_year:.1f}")
        print(f"    Holding period:       {holding_period_days:.2f} days")
        print(f"    Capital turnover:     {capital_turnover:.1f}x")
        print(f"    Gross mean return H20: {gross_mean_return:.6f}")
        print()
        header = f"    {'Capital':>12s} {'Eff Cost':>9s} {'Slippage':>8s} {'Net Return':>10s} {'Profitable':>10s}"
        print(header)
        print(f"    {'-' * len(header.strip())}")

        capacity_ceiling = None
        results = {}

        for cap in capital_levels:
            slippage_mult = cap / 1000000 * 0.1
            effective_cost = base_cost * (1 + slippage_mult)
            net_return = gross_mean_return - effective_cost / 20
            profitable = net_return > 0

            if not profitable and capacity_ceiling is None:
                capacity_ceiling = cap

            print(f"    {cap:>12,d} {effective_cost:>9.6f} {slippage_mult:>8.4f} {net_return:>10.6f} {'YES' if profitable else 'NO':>10s}")

            results[f"cap_{cap}"] = {
                "capital": cap,
                "slippage_multiplier": slippage_mult,
                "effective_cost": effective_cost,
                "net_return": net_return,
                "profitable": profitable,
            }

        if capacity_ceiling:
            print(f"\n    Capacity ceiling: ~${capacity_ceiling:,.0f}")
        else:
            print("\n    No capacity ceiling found within tested range")

        return AAEResult("capacity_model", "COMPLETE", metrics={
            "total_signals": total_signals,
            "signals_per_year": signals_per_year,
            "holding_period_days": holding_period_days,
            "capital_turnover": capital_turnover,
            "base_cost": base_cost,
            "gross_mean_return_H20": gross_mean_return,
            "results": results,
            "capacity_ceiling": capacity_ceiling,
        })
