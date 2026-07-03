from __future__ import annotations
import numpy as np
from research.energy_reality.energy_validator import EnergyValidator, ERLResult


class LiveDegradation:
    def __init__(self, validator: EnergyValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ERLResult:
        self.validator.load(self.asset)
        ES = self.validator.es_signal()
        base = self.validator.eval_alpha(ES, 2)

        stress_levels = [1, 2, 5, 10, 20]
        stress_results = {}

        print(f"\n=== ERL-5: Live Degradation Simulation ===")
        print(f"Asset: {self.asset}")
        print(f"\nBaseline alpha (horizon_idx=2):")
        print(f"  mean={base['mean']:.6f}  pp={base['pp']:.4f}  sharpe={base['sharpe']:.4f}  std={base['std']:.6f}  n={base['n']}")

        header = f"{'Stress':>6} | {'SlippagePP':>10} | {'DelayPP':>8} | {'SpreadPP':>9} | {'MissedPP':>9} | {'CombPP':>7} | {'CombSharpe':>10} | {'Survives':>8}"
        sep = "-" * 88
        print(f"\n{header}\n{sep}")

        for sm in stress_levels:
            delay = min(sm, 5)
            slippage_cost = 0.0001 * sm
            spread_cost = 0.0002 * sm
            drop_frac = min(sm * 0.02, 0.5)

            fut_slip = self.validator.fut_ret.copy()
            fut_slip[:, 2] -= slippage_cost
            slip_alpha = self.validator.aae.eval_alpha(ES, fut_slip, 2)

            delayed = np.roll(ES, delay)
            delayed[:delay] = 0
            delay_alpha = self.validator.eval_alpha(delayed, 2)

            fut_spread = self.validator.fut_ret.copy()
            fut_spread[:, 2] -= spread_cost
            spread_alpha = self.validator.aae.eval_alpha(ES, fut_spread, 2)

            np.random.seed(42)
            mask = np.random.random(len(ES)) > drop_frac
            missed_signal = ES * mask
            missed_alpha = self.validator.eval_alpha(missed_signal, 2)

            fut_comb = self.validator.fut_ret.copy()
            fut_comb[:, 2] -= (slippage_cost + spread_cost)
            delayed_comb = np.roll(ES, delay)
            delayed_comb[:delay] = 0
            np.random.seed(42)
            mask_comb = np.random.random(len(delayed_comb)) > drop_frac
            comb_signal = delayed_comb * mask_comb
            comb_alpha = self.validator.aae.eval_alpha(comb_signal, fut_comb, 2)

            survives = comb_alpha["pp"] > 0.55 or comb_alpha["mean"] > 0

            stress_results[str(sm)] = {
                "slippage": slip_alpha,
                "delay": delay_alpha,
                "spread": spread_alpha,
                "missed": missed_alpha,
                "combined": comb_alpha,
                "survives": survives,
            }

            print(f"{sm:>6} | {slip_alpha['pp']:>10.4f} | {delay_alpha['pp']:>8.4f} | {spread_alpha['pp']:>9.4f} | {missed_alpha['pp']:>9.4f} | {comb_alpha['pp']:>7.4f} | {comb_alpha['sharpe']:>10.4f} | {str(survives):>8}")

        survives_20x = stress_results["20"]["combined"]["pp"] > 0.55
        passes = survives_20x

        if passes:
            verdict = "PASS: Alpha survives realistic market degradation at 20x stress"
        else:
            verdict = f"FAIL: Combined pp at 20x = {stress_results['20']['combined']['pp']:.4f}, threshold 0.55"

        print(f"\n{verdict}")

        metrics = {
            "baseline_alpha": base,
            "stress_results": stress_results,
            "survives_20x": survives_20x,
            "passes": passes,
            "verdict": verdict,
        }

        return ERLResult(
            rq_name="ERL-5: Live Degradation Simulation",
            status="PASS" if passes else "FAIL",
            metrics=metrics,
        )
