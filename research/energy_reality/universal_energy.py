from __future__ import annotations
import numpy as np
from research.energy_reality.energy_validator import EnergyValidator, ERLResult, TARGET_ASSETS


class UniversalEnergy:
    def __init__(self, validator: EnergyValidator):
        self.validator = validator

    def run(self) -> ERLResult:
        per_asset = {}
        flip_asset = None
        flip_pp = 0.0
        all_positive = True

        for asset in TARGET_ASSETS:
            self.validator.load(asset)
            es = self.validator.es_signal()

            alpha_h5 = self.validator.eval_alpha(es, 1)
            alpha_h20 = self.validator.eval_alpha(es, 2)
            alpha_h50 = self.validator.eval_alpha(es, 3)

            per_asset[asset] = {
                "H5": alpha_h5,
                "H20": alpha_h20,
                "H50": alpha_h50,
            }

            pp = alpha_h20["pp"]
            if pp < 0.50:
                flip_asset = asset
                flip_pp = pp
                all_positive = False
            elif pp <= 0.55:
                all_positive = False

        if flip_asset is not None:
            passes = False
            verdict = f"FAIL: Sign flips on {flip_asset} (pp={flip_pp:.3f})"
        elif all_positive:
            passes = True
            verdict = "PASS: Sign survives across all assets — ES is a universal market phenomenon"
        else:
            passes = False
            verdict = "FAIL: Not all assets maintain sign survival (pp > 0.55)"

        pps = np.array([per_asset[a]["H20"]["pp"] for a in TARGET_ASSETS])
        sharpes = np.array([per_asset[a]["H20"]["sharpe"] for a in TARGET_ASSETS])

        mean_pp = float(np.mean(pps))
        mean_sharpe = float(np.mean(sharpes))
        min_pp = float(np.min(pps))
        min_sharpe = float(np.min(sharpes))
        pp_std = float(np.std(pps))
        sharpe_std = float(np.std(sharpes))

        self._print_table(per_asset, mean_pp, mean_sharpe, min_pp, min_sharpe, pp_std, sharpe_std, verdict)

        metrics = {
            "per_asset": per_asset,
            "all_positive": bool(all_positive),
            "mean_pp": mean_pp,
            "mean_sharpe": mean_sharpe,
            "min_pp": min_pp,
            "min_sharpe": min_sharpe,
            "pp_std": pp_std,
            "sharpe_std": sharpe_std,
            "passes": passes,
            "verdict": verdict,
        }

        return ERLResult(rq_name="ERL-6: Universal Energy Test", status="PASS" if passes else "FAIL", metrics=metrics)

    def _print_table(self, per_asset: dict, mean_pp: float, mean_sharpe: float, min_pp: float, min_sharpe: float, pp_std: float, sharpe_std: float, verdict: str) -> None:
        header = f"{'Asset':<10} {'Mean H20':>10} {'PP H20':>8} {'Sharpe H20':>12} {'n':>8} {'PP H5':>8} {'PP H50':>8}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for asset in TARGET_ASSETS:
            a = per_asset[asset]
            h20 = a["H20"]
            h5 = a["H5"]
            h50 = a["H50"]
            print(f"{asset:<10} {h20['mean']:>10.6f} {h20['pp']:>8.4f} {h20['sharpe']:>12.4f} {h20['n']:>8} {h5['pp']:>8.4f} {h50['pp']:>8.4f}")
        print("")
        print(f"Cross-asset mean PP: {mean_pp:.4f} | mean Sharpe: {mean_sharpe:.4f}")
        print(f"Cross-asset min PP: {min_pp:.4f} | min Sharpe: {min_sharpe:.4f}")
        print(f"Cross-asset PP std: {pp_std:.4f} | Sharpe std: {sharpe_std:.4f}")
        print(f"Verdict: {verdict}")
