from __future__ import annotations
import numpy as np
from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES, HORIZONS


class ResidualValidation:
    def __init__(self, validator: ResidualEnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> REPResult:
        self.validator.load(self.asset)
        self.validator.build_residuals()

        es_signal = self.validator.es
        es_multi = self.validator.multi_horizon_alpha(es_signal)

        residual_multi = {}
        residual_dd = {}
        for rt in RESIDUAL_TYPES:
            res = self.validator.get_residual(rt)
            residual_multi[rt] = self.validator.multi_horizon_alpha(res)

        es_dd = self._compute_drawdowns(es_signal)
        for rt in RESIDUAL_TYPES:
            residual_dd[rt] = self._compute_drawdowns(self.validator.get_residual(rt))

        improvement_ratios = {}
        for rt in RESIDUAL_TYPES:
            es_pp = es_multi["20"]["pp"]
            res_pp = residual_multi[rt]["20"]["pp"]
            improvement_ratios[rt] = float((res_pp - es_pp) / max(abs(es_pp), 1e-12))

        beats_es_h20 = any(
            residual_multi[rt]["20"]["sharpe"] > es_multi["20"]["sharpe"]
            for rt in RESIDUAL_TYPES
        )

        best_horizon = "20"
        best_gap = -1e9
        for rt in RESIDUAL_TYPES:
            for h in HORIZONS:
                gap = residual_multi[rt][str(h)]["sharpe"] - es_multi[str(h)]["sharpe"]
                if gap > best_gap:
                    best_gap = gap
                    best_horizon = str(h)

        print("=" * 80)
        print("REP-2: RESIDUAL ALPHA VALIDATION")
        print(f"Asset: {self.asset}")
        print("=" * 80)
        header = f"  {'Horizon':>8s} {'ES Mean':>8s} {'ES PP':>7s} {'ES Sharpe':>9s} {'ES MaxDD':>9s}"
        for rt in RESIDUAL_TYPES:
            header += f" {rt.capitalize():>10s} Mean{'':>2s}{'PP':>7s}{'Sharpe':>9s}{'MaxDD':>9s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for h in HORIZONS:
            es_r = es_multi[str(h)]
            es_d = es_dd.get(str(h), {})
            row = f"  {h:>8d} {es_r['mean']:>8.4f} {es_r['pp']:>7.3f} {es_r['sharpe']:>9.4f} {es_d.get('max_dd', 0):>9.4f}"
            for rt in RESIDUAL_TYPES:
                r_r = residual_multi[rt][str(h)]
                r_d = residual_dd[rt].get(str(h), {})
                row += f" {rt.capitalize():>10s} {r_r['mean']:>8.4f} {r_r['pp']:>7.3f} {r_r['sharpe']:>9.4f} {r_d.get('max_dd', 0):>9.4f}"
            print(row)
        print()

        print("  Improvement Ratios (Residual PP - ES PP) / ES PP @ H20:")
        for rt, ratio in improvement_ratios.items():
            marker = " <<<" if ratio > 0 else ""
            print(f"    {rt:>15s}: {ratio:+.4f}{marker}")
        print(f"\n  Best horizon (max sharpe gain): {best_horizon}")
        print(f"  Beats ES at H20: {beats_es_h20}")
        print("=" * 80)
        print()

        return REPResult("residual_validation", "COMPLETE", metrics={
            "es_multi_horizon": es_multi,
            "residual_multi_horizon": residual_multi,
            "es_drawdowns": es_dd,
            "residual_drawdowns": residual_dd,
            "improvement_ratios": improvement_ratios,
            "best_horizon": best_horizon,
            "beats_es": beats_es_h20,
        })

    def _compute_drawdowns(self, signal: np.ndarray) -> dict[str, dict]:
        result = {}
        for hi, h in enumerate(HORIZONS):
            fwd = self.validator.fut_ret[:, hi]
            n = min(len(signal), len(fwd))
            sig, fw = signal[:n], fwd[:n]
            threshold = np.nanpercentile(sig, 90)
            top_mask = sig >= threshold
            rets = fw[top_mask]
            rets = rets[~np.isnan(rets)]
            if len(rets) < 5:
                result[str(h)] = {"max_dd": 0.0, "avg_dd": 0.0}
            else:
                cum = np.cumsum(rets)
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                result[str(h)] = {"max_dd": float(np.max(dd)), "avg_dd": float(np.mean(dd))}
        return result
