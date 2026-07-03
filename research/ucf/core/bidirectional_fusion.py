from typing import Dict, List, Tuple
import math


class BidirectionalFusionLayer:
    def fuse_states(
        self,
        technical: dict,
        fundamental: dict,
        exposure: dict,
        regime: str,
        weights: dict,
    ) -> dict:
        t_conv = max(0.0, min(1.0, technical.get("conviction", 0.0)))
        t_dir = self._clamp_direction(technical.get("direction", 0))
        t_stab = max(0.0, min(1.0, technical.get("stability", 0.0)))

        f_conv = max(0.0, min(1.0, fundamental.get("conviction", 0.0)))
        f_dir = self._clamp_direction(fundamental.get("direction", 0))
        f_stab = max(0.0, min(1.0, fundamental.get("stability", 0.0)))

        e_conv = max(0.0, min(1.0, exposure.get("conviction", 0.0)))
        e_dir = self._clamp_direction(exposure.get("direction", 0))
        e_stab = max(0.0, min(1.0, exposure.get("stability", 0.0)))

        w_t = weights.get("technical", 0.34)
        w_f = weights.get("fundamental", 0.33)
        w_e = weights.get("exposure", 0.33)
        w_total = w_t + w_f + w_e
        if w_total > 0.0:
            w_t = w_t / w_total
            w_f = w_f / w_total
            w_e = w_e / w_total
        else:
            w_t = w_f = w_e = 1.0 / 3.0

        directions = [t_dir, f_dir, e_dir]
        agreement = self.compute_agreement(directions)

        agreement_bonus = 0.0
        if all(d == directions[0] for d in directions) and directions[0] != 0:
            agreement_bonus = 0.15
        elif sum(1 for d in directions if d != 0) >= 2 and (
            (t_dir == f_dir and t_dir != 0)
            or (t_dir == e_dir and t_dir != 0)
            or (f_dir == e_dir and f_dir != 0)
        ):
            agreement_bonus = 0.05
        elif all(d != 0 for d in directions) and len(set(directions)) == 3:
            agreement_bonus = -0.10

        weighted_sum = (w_t * t_conv) + (w_f * f_conv) + (w_e * e_conv)

        regime_override_active = False
        if t_dir != 0 and f_dir != 0 and t_dir != f_dir and regime == "stable":
            t_conv = t_conv * 0.9
            f_conv = f_conv * 0.9
            weighted_sum = (w_t * t_conv) + (w_f * f_conv) + (w_e * e_conv)
            regime_override_active = True

        if regime == "transition":
            f_extra = w_t * 0.2
            w_f_adj = w_f + f_extra
            w_t_adj = w_t * 0.8
            w_total_adj = w_t_adj + w_f_adj + w_e
            if w_total_adj > 0.0:
                w_t = w_t_adj / w_total_adj
                w_f = w_f_adj / w_total_adj
                w_e = w_e / w_total_adj
                weighted_sum = (w_t * t_conv) + (w_f * f_conv) + (w_e * e_conv)
            regime_override_active = True

        fused_conviction = max(0.0, min(1.0, weighted_sum + agreement_bonus))

        weighted_dir_sum = (w_t * t_dir) + (w_f * f_dir) + (w_e * e_dir)
        if weighted_dir_sum > 0.5:
            fused_direction = 1
        elif weighted_dir_sum < -0.5:
            fused_direction = -1
        else:
            zero_weight = 0.0
            for d, w in zip(directions, [w_t, w_f, w_e]):
                if d == 0:
                    zero_weight += w
            non_zero_dirs = [d for d in directions if d != 0]
            if len(non_zero_dirs) == 0:
                fused_direction = 0
            else:
                maj_dir = 1 if sum(non_zero_dirs) > 0 else -1
                if zero_weight >= 0.5:
                    fused_direction = 0
                else:
                    fused_direction = maj_dir

        stability_values = [t_stab, f_stab, e_stab]
        stability_weights = [w_t, w_f, w_e]
        agreement_factor = abs(agreement)
        fused_stability = 0.0
        for sv, sw in zip(stability_values, stability_weights):
            fused_stability += sv * sw
        if agreement_factor < 0.5:
            penalty = 1.0 - (0.5 - agreement_factor) * 2.0
            fused_stability = fused_stability * penalty
        fused_stability = max(0.0, min(1.0, fused_stability))

        contributions = self.compute_contribution_fractions(
            {"technical": w_t, "fundamental": w_f, "exposure": w_e},
            {"fused_conviction": fused_conviction},
        )

        final_t_contrib = t_conv * w_t / (fused_conviction + 1e-10) if fused_conviction > 0 else w_t
        final_f_contrib = f_conv * w_f / (fused_conviction + 1e-10) if fused_conviction > 0 else w_f
        final_e_contrib = e_conv * w_e / (fused_conviction + 1e-10) if fused_conviction > 0 else w_e
        contrib_total = final_t_contrib + final_f_contrib + final_e_contrib
        if contrib_total > 0.0:
            final_t_contrib = final_t_contrib / contrib_total
            final_f_contrib = final_f_contrib / contrib_total
            final_e_contrib = final_e_contrib / contrib_total
        else:
            final_t_contrib = final_f_contrib = final_e_contrib = 1.0 / 3.0

        return {
            "fused_conviction": fused_conviction,
            "fused_direction": fused_direction,
            "fused_stability": fused_stability,
            "component_contributions": {
                "technical": round(final_t_contrib, 6),
                "fundamental": round(final_f_contrib, 6),
                "exposure": round(final_e_contrib, 6),
            },
            "agreement": agreement,
            "regime_override_active": regime_override_active,
        }

    def fuse_batch(
        self,
        states: Dict[str, dict],
        regime: str,
        weights: dict,
    ) -> Dict[str, dict]:
        results = {}
        for symbol, signal in states.items():
            tech = signal.get("technical", {})
            fund = signal.get("fundamental", {})
            exp = signal.get("exposure", {})
            results[symbol] = self.fuse_states(tech, fund, exp, regime, weights)
        return results

    def compute_agreement(self, directions: List[int]) -> float:
        clamped = [self._clamp_direction(d) for d in directions]
        non_zero = [d for d in clamped if d != 0]
        if len(non_zero) == 0:
            return 0.0
        if all(d == non_zero[0] for d in non_zero):
            return 1.0
        if len(non_zero) >= 2 and (
            (clamped[0] == clamped[1] and clamped[0] != 0)
            or (clamped[0] == clamped[2] and clamped[0] != 0)
            or (clamped[1] == clamped[2] and clamped[1] != 0)
        ):
            return 0.33
        return -1.0

    def compute_contribution_fractions(self, weights: dict, fused: dict) -> dict:
        w_total = sum(weights.values())
        if w_total <= 0.0:
            return {"technical": 1.0 / 3.0, "fundamental": 1.0 / 3.0, "exposure": 1.0 / 3.0}
        return {
            "technical": weights.get("technical", 0.0) / w_total,
            "fundamental": weights.get("fundamental", 0.0) / w_total,
            "exposure": weights.get("exposure", 0.0) / w_total,
        }

    def _clamp_direction(self, value: int) -> int:
        if value > 0:
            return 1
        elif value < 0:
            return -1
        return 0
