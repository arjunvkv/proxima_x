"""RQ5: Do variables shift distributions or merely risk?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_extraction.alpha_validator import (
    AlphaValidator, AELResult, PRIMARY_VARIABLES, HORIZONS,
    _future_returns, _future_drawdown, _future_runup, _numba_skew, _numba_kurtosis,
)


class DistributionShift:
    def __init__(self, validator: AlphaValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AELResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        price = data["price"]

        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        fr_all = _future_returns(price, horizons_arr)

        shifts = {}
        for var in PRIMARY_VARIABLES:
            sig = np.asarray(signals[var], dtype=np.float64)
            n = min(len(sig), fr_all.shape[0])
            sig, fr = sig[:n], fr_all[:n]

            _, masks = self.validator.decile_bins(sig)

            var_shifts = {}
            for hi, h in enumerate(HORIZONS):
                fwd = fr[:, hi]
                dd = _future_drawdown(price[:len(fwd)], h)
                ru = _future_runup(price[:len(fwd)], h)

                unconditional_fwd = fwd[~np.isnan(fwd)]
                u_mean = float(np.nanmean(unconditional_fwd))
                u_std = float(np.nanstd(unconditional_fwd))
                u_skew = _numba_skew(unconditional_fwd)
                u_dd = float(np.nanmean(dd[~np.isnan(fwd)]))
                u_ru = float(np.nanmean(ru[~np.isnan(fwd)]))

                d1 = masks[0]
                d10 = masks[-1]

                shifts_h = {}
                for label, mask in [("low", d1), ("high", d10)]:
                    if np.sum(mask) < 5:
                        continue
                    m_fwd = fwd[mask]
                    m_dd = dd[mask]
                    m_ru = ru[mask]

                    shifts_h[label] = {
                        "return_location_shift": float(np.nanmean(m_fwd)) - u_mean,
                        "return_scale_shift": float(np.nanstd(m_fwd)) - u_std,
                        "tail_shift": _numba_skew(m_fwd) - u_skew,
                        "drawdown_shift": float(np.nanmean(m_dd)) - u_dd,
                        "runup_shift": float(np.nanmean(m_ru)) - u_ru,
                        "mean": float(np.nanmean(m_fwd)),
                        "std": float(np.nanstd(m_fwd)),
                        "skew": _numba_skew(m_fwd),
                        "drawdown": float(np.nanmean(m_dd)),
                        "runup": float(np.nanmean(m_ru)),
                    }
                var_shifts[f"H{h}"] = shifts_h
            shifts[var] = var_shifts

            h20 = var_shifts.get("H20", {})
            high = h20.get("high", {})
            low = h20.get("low", {})
            print(f"    {var:25s}: high loc_shift={high.get('return_location_shift', 0):.6f}, "
                  f"scale_shift={high.get('return_scale_shift', 0):.6f}, "
                  f"tail_shift={high.get('tail_shift', 0):.4f}")

        return AELResult("distribution_shift", "COMPLETE", metrics={"shifts": shifts})
