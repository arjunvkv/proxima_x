"""RQ2: How does compression emerge? Birth, growth, decay, release."""

from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


@numba.jit(nopython=True, cache=True)
def _find_birth_events(signal: NDArray[np.float64], threshold: float) -> NDArray[np.int32]:
    n = len(signal)
    result = np.zeros(n, dtype=np.int32)
    above = signal > threshold
    for i in range(1, n):
        if above[i] and not above[i - 1]:
            result[i] = 1
    return result


class CompressionLifecycle:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        compression = np.asarray(signals["compression"], dtype=np.float64)
        returns = np.asarray(signals["returns"], dtype=np.float64)
        energy_storage = np.asarray(signals["energy_storage"], dtype=np.float64)
        memory_density = np.asarray(signals["memory_density"], dtype=np.float64)
        n = len(compression)

        threshold = float(np.nanmean(compression[20:]) + 0.5 * np.nanstd(compression[20:]))
        drop_threshold = float(np.nanstd(compression[20:]))

        birth_binary = _find_birth_events(compression[int(n * 0.1):], threshold)
        n_births = int(np.sum(birth_binary))

        diff = np.diff(compression, prepend=compression[0])
        growth_events = diff > 0
        decay_events = diff < 0
        growth_rate = float(np.mean(diff[growth_events])) if np.any(growth_events) else 0.0
        decay_rate = float(np.mean(np.abs(diff[decay_events]))) if np.any(decay_events) else 0.0

        release_count = 0
        for i in range(1, n):
            drop = compression[i - 1] - compression[i]
            if drop > drop_threshold * 2:
                release_count += 1
        release_rate = release_count / max(n, 1)

        from research.compression_physics.compression_validator import _find_peak_lag
        # What happens AROUND birth events?
        birth_signal = birth_binary.astype(np.float64)

        lag_birth_to_es, r_birth_es = _find_peak_lag(birth_signal[:len(energy_storage)], energy_storage[:len(energy_storage)], 50)
        lag_birth_to_md, r_birth_md = _find_peak_lag(birth_signal[:len(memory_density)], memory_density[:len(memory_density)], 50)

        # What leads up to compression build-up?
        rise_signal = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if compression[i] > compression[i - 5] * 1.5 and compression[i] > np.median(compression[20:]):
                rise_signal[i] = 1.0

        # Pre-birth return patterns
        pre_birth_vol = []
        birth_indices = np.where(birth_binary > 0)[0]
        for idx in birth_indices[:100]:
            if idx > 20 and idx < n:
                pre_birth_vol.append(float(np.std(returns[idx - 20:idx])))

        avg_pre_birth_vol = float(np.mean(pre_birth_vol)) if pre_birth_vol else 0.0
        overall_vol = float(np.std(returns[20:]))

        metrics = {
            "birth_rate": n_births / max(n, 1),
            "growth_rate": growth_rate,
            "decay_rate": decay_rate,
            "release_rate": release_rate,
            "n_births": n_births,
            "n_releases": release_count,
            "avg_pre_birth_vol": avg_pre_birth_vol,
            "overall_vol": overall_vol,
            "vol_ratio_pre_birth_vs_overall": avg_pre_birth_vol / max(overall_vol, 1e-12),
            "birth_to_energy_storage_lag": lag_birth_to_es,
            "birth_to_energy_storage_corr": r_birth_es,
            "birth_to_memory_density_lag": lag_birth_to_md,
            "birth_to_memory_density_corr": r_birth_md,
        }

        print(f"  Compression lifecycle:")
        print(f"    Births: {n_births} ({metrics['birth_rate']:.4f}/day)")
        print(f"    Growth rate: {growth_rate:.6f}, Decay rate: {decay_rate:.6f}")
        print(f"    Releases: {release_count} ({release_rate:.4f}/day)")
        print(f"    Pre-birth vol ratio: {metrics['vol_ratio_pre_birth_vs_overall']:.2f}x overall")

        return CPIResult("compression_lifecycle", "PASSED", metrics=metrics)
