from __future__ import annotations
import numpy as np
from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES, HORIZONS


LAYERS = ["momentum", "trend", "hurst", "entropy", "memory_density", "adaptive_time",
          "state_mutation", "regime_change", "compression", "information_pressure"]


class ResidualDecomposition:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()

        residual = self.validator.get_residual("xgboost")
        es_signal = self.validator.es
        sig = self.validator.signals
        vm = self.validator.energy.vol_metrics
        returns = sig["returns"]
        n = len(returns)

        layers = {}

        momentum = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            momentum[i] = float(np.sum(returns[i - 19:i + 1]))
        layers["momentum"] = momentum

        trend = np.full(n, np.nan, dtype=np.float64)
        for i in range(50, n):
            trend[i] = float(np.mean(returns[i - 49:i + 1]))
        layers["trend"] = trend

        layers["hurst"] = vm["realized_vol"].copy()

        layers["entropy"] = vm["entropy"].copy()

        layers["memory_density"] = np.asarray(sig["memory_density"], dtype=np.float64)

        layers["adaptive_time"] = np.asarray(sig["adaptive_time"], dtype=np.float64)

        layers["state_mutation"] = np.asarray(sig["state_mutation_rate"], dtype=np.float64)

        layers["regime_change"] = np.asarray(sig["regime_change_probability"], dtype=np.float64)

        compression = np.full(n, np.nan, dtype=np.float64)
        es_arr = np.asarray(sig["energy_storage"], dtype=np.float64)
        for i in range(20, n):
            compression[i] = float(np.std(es_arr[i - 19:i + 1]))
        layers["compression"] = compression

        info_pressure = np.full(n, np.nan, dtype=np.float64)
        at_arr = np.asarray(sig["adaptive_time"], dtype=np.float64)
        for i in range(20, n):
            info_pressure[i] = float(np.std(at_arr[i - 19:i + 1]))
        layers["information_pressure"] = info_pressure

        decomposition = {}
        mi_values = []
        for layer_name in LAYERS:
            layer_sig = layers[layer_name]
            corr = self.validator.correlation(residual, layer_sig)
            mi = self.validator.mutual_info(residual, layer_sig)
            decomposition[layer_name] = {"correlation": corr, "mutual_info": mi, "r2_approx": 0.0}
            mi_values.append(mi)

        max_mi = max(mi_values) if mi_values else 1.0
        if max_mi < 1e-12:
            max_mi = 1.0
        for layer_name in LAYERS:
            decomposition[layer_name]["r2_approx"] = decomposition[layer_name]["mutual_info"] / max_mi

        most_correlated = max(LAYERS, key=lambda l: abs(decomposition[l]["correlation"]))
        max_corr = decomposition[most_correlated]["correlation"]
        es_residual_corr = self.validator.correlation(es_signal, residual)

        print("=" * 80)
        print("REP-3: RESIDUAL DECOMPOSITION")
        print(f"Asset: {self.asset}")
        print("=" * 80)
        print(f"  Residual type: xgboost")
        print(f"  ES-Residual correlation: {es_residual_corr:.4f}")
        print()
        header = f"  {'Layer':<22s} {'Correlation':>12s} {'MI':>10s} {'R2_approx':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for layer_name in LAYERS:
            d = decomposition[layer_name]
            marker = " <<<" if layer_name == most_correlated else ""
            print(f"  {layer_name:<22s} {d['correlation']:>12.4f} {d['mutual_info']:>10.6f} {d['r2_approx']:>10.4f}{marker}")
        print()
        print(f"  Most correlated layer: {most_correlated} (r = {max_corr:.4f})")
        print(f"  Is residual just memory_density? {abs(decomposition['memory_density']['correlation']):.4f} >= {0.7:.4f}")
        print(f"  Is residual just adaptive_time?  {abs(decomposition['adaptive_time']['correlation']):.4f} >= {0.7:.4f}")
        print("=" * 80)
        print()

        return REPResult("residual_decomposition", "COMPLETE", metrics={
            "decomposition": decomposition,
            "most_correlated_layer": most_correlated,
            "max_correlation": max_corr,
            "es_residual_correlation": es_residual_corr,
        })
