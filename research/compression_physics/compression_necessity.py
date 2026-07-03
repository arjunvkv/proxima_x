"""RQ3: Can energy exist without compression? Remove compression, measure downstream loss."""

from __future__ import annotations

from typing import Any

import numpy as np

from research.compression_physics.compression_validator import CompressionValidator, CPIResult, _find_peak_lag


class CompressionNecessity:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def _direct_corr_loss(self, target: str, excluded: str, signals: dict) -> dict:
        target_sig = np.asarray(signals.get(target, np.zeros(100)), dtype=np.float64)
        excluded_sig = np.asarray(signals.get(excluded, np.zeros(100)), dtype=np.float64)
        n = min(len(target_sig), len(excluded_sig))

        _, with_excluded = _find_peak_lag(excluded_sig[:n], target_sig[:n], self._max_lag)

        all_sources = ["compression", "energy_storage", "memory_density", "adaptive_time",
                       "state_mutation_rate", "regime_change_probability",
                       "entropy_change", "memory_alignment", "tension_score",
                       "memory_gradient", "information_pressure", "cohort_alignment"]

        best_without = 0.0
        best_without_gen = None
        for src in all_sources:
            if src == target or src == excluded:
                continue
            if src not in signals:
                continue
            sig = np.asarray(signals[src], dtype=np.float64)
            _, r = _find_peak_lag(sig[:n], target_sig[:n], self._max_lag)
            if abs(r) > abs(best_without):
                best_without = r
                best_without_gen = src

        loss = (abs(with_excluded) - abs(best_without)) / max(abs(with_excluded), 1e-12)
        return {
            "with_excluded_r": with_excluded,
            "best_without_r": best_without,
            "best_without_generator": best_without_gen,
            "loss": loss,
        }

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        targets = ["energy_storage", "memory_density", "adaptive_time", "state_mutation_rate", "regime_change_probability"]
        losses = {}
        for t in targets:
            losses[t] = self._direct_corr_loss(t, "compression", signals)

        es_loss = losses["energy_storage"]["loss"]
        at_loss = losses["adaptive_time"]["loss"]
        avg_loss = sum(v["loss"] for v in losses.values()) / max(len(losses), 1)

        metrics = {
            "losses": losses,
            "avg_loss": avg_loss,
            "energy_storage_loss": es_loss,
            "adaptive_time_loss": at_loss,
            "energy_storage_best_alternative": losses["energy_storage"]["best_without_generator"],
            "energy_storage_best_alt_r": losses["energy_storage"]["best_without_r"],
            "energy_storage_compression_r": losses["energy_storage"]["with_excluded_r"],
        }

        print(f"  Compression necessity - loss when compression removed:")
        for t, v in losses.items():
            print(f"    {t:25s}: comp_r={v['with_excluded_r']:.4f}, best_alt={v['best_without_generator']:20s} alt_r={v['best_without_r']:.4f}, loss={v['loss']:.4f}")

        if es_loss > 0.3:
            status = "PASSED"
            print(f"  Energy DOES depend on compression (loss={es_loss:.4f})")
        elif es_loss > 0.1:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"
            print(f"  Energy can exist WITHOUT compression (loss={es_loss:.4f})")

        return CPIResult("compression_necessity", status, metrics=metrics)
