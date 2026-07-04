from typing import Dict, List, Tuple
import math


class BidirectionalFusionLayer:
    def __init__(self) -> None:
        self.history_states = {}
        self.regime_history = {}

    def reset_history(self) -> None:
        self.history_states.clear()
        self.regime_history.clear()

    def fuse_states(
        self,
        technical: dict,
        fundamental: dict,
        exposure: dict,
        regime: str,
        weights: dict,
        symbol: str = "default",
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

        # Count-based agreement/disagreement logic
        num_pos = directions.count(1)
        num_neg = directions.count(-1)
        num_zero = directions.count(0)

        agreement_bonus = 0.0
        if num_pos > 0 and num_neg > 0:
            # Opposing signals: 1 and -1 conflict, apply disagreement penalty
            if num_zero == 1:
                # Complete conflict, e.g. [1, -1, 0]
                agreement_bonus = -0.10
            else:
                # Partial conflict, e.g. [1, 1, -1] or [-1, -1, 1]
                agreement_bonus = -0.05
        elif num_pos == 3 or num_neg == 3:
            # Full agreement, e.g. [1, 1, 1] or [-1, -1, -1]
            agreement_bonus = 0.15
        elif num_pos == 2 or num_neg == 2:
            # Partial agreement, e.g. [1, 1, 0] or [-1, -1, 0]
            agreement_bonus = 0.05

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

        # Enforce no dominance constraint: no single component contribution can exceed 0.65
        contribs = [final_t_contrib, final_f_contrib, final_e_contrib]
        max_contrib = 0.65
        if any(c > max_contrib for c in contribs):
            excess = 0.0
            non_dominant_indices = []
            for i, c in enumerate(contribs):
                if c > max_contrib:
                    excess += (c - max_contrib)
                    contribs[i] = max_contrib
                else:
                    non_dominant_indices.append(i)
            if non_dominant_indices and excess > 0.0:
                non_dom_sum = sum(contribs[i] for i in non_dominant_indices)
                if non_dom_sum > 0.0:
                    for i in non_dominant_indices:
                        contribs[i] += excess * (contribs[i] / non_dom_sum)
                else:
                    share = excess / len(non_dominant_indices)
                    for i in non_dominant_indices:
                        contribs[i] += share
            c_sum = sum(contribs)
            if c_sum > 0.0:
                contribs = [c / c_sum for c in contribs]
            else:
                contribs = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            final_t_contrib, final_f_contrib, final_e_contrib = contribs

        # Observe, measure, and calculate parallel analytical fields
        divergence_signature = {
            "conviction_diff_t_f": abs(t_conv - f_conv),
            "conviction_diff_t_e": abs(t_conv - e_conv),
            "conviction_diff_f_e": abs(f_conv - e_conv),
            "direction_diff_t_f": abs(t_dir - f_dir),
            "direction_diff_t_e": abs(t_dir - e_dir),
            "direction_diff_f_e": abs(f_dir - e_dir),
            "stability_diff_t_f": abs(t_stab - f_stab),
            "stability_diff_t_e": abs(t_stab - e_stab),
            "stability_diff_f_e": abs(f_stab - e_stab),
        }

        conflict_residue_vector = []
        if (t_dir == 1 and f_dir == -1) or (t_dir == -1 and f_dir == 1):
            conflict_residue_vector.append("tech_vs_fund")
        if (t_dir == 1 and e_dir == -1) or (t_dir == -1 and e_dir == 1):
            conflict_residue_vector.append("tech_vs_exposure")
        if (f_dir == 1 and e_dir == -1) or (f_dir == -1 and e_dir == 1):
            conflict_residue_vector.append("fund_vs_exposure")

        # Agreement Polarization Index calculation (API = ΔPredictive_Richness / ΔEntropy_Loss)
        def entropy_bin(p: float) -> float:
            p_c = max(1e-9, min(1.0 - 1e-9, p))
            return - (p_c * math.log2(p_c) + (1.0 - p_c) * math.log2(1.0 - p_c))

        input_entropy = (entropy_bin(t_conv) + entropy_bin(f_conv) + entropy_bin(e_conv)) / 3.0
        output_entropy = entropy_bin(fused_conviction)
        entropy_loss = max(1e-6, input_entropy - output_entropy)
        predictive_richness = abs(fused_conviction - 0.5)
        agreement_polarization_index = predictive_richness / entropy_loss

        # Multi-Resolution conviction updates
        if symbol not in self.history_states:
            self.history_states[symbol] = []
        self.history_states[symbol].append(fused_conviction)
        if len(self.history_states[symbol]) > 200:
            self.history_states[symbol] = self.history_states[symbol][-200:]

        if symbol not in self.regime_history:
            self.regime_history[symbol] = {}
        if regime not in self.regime_history[symbol]:
            self.regime_history[symbol][regime] = []
        self.regime_history[symbol][regime].append(fused_conviction)
        if len(self.regime_history[symbol][regime]) > 200:
            self.regime_history[symbol][regime] = self.regime_history[symbol][regime][-200:]

        # Micro: instantaneous
        fusion_micro = fused_conviction

        # Meso: rolling 20 ticks
        meso_window = self.history_states[symbol][-20:]
        fusion_meso = sum(meso_window) / len(meso_window) if meso_window else fused_conviction

        # Macro: rolling 50 ticks in current regime
        macro_window = self.regime_history[symbol][regime][-50:]
        fusion_macro = sum(macro_window) / len(macro_window) if macro_window else fused_conviction

        return {
            "fused_conviction": fused_conviction,
            "fused_direction": fused_direction,
            "fused_stability": fused_stability,
            "component_contributions": {
                "technical": round(final_t_contrib, 6),
                "fundamental": round(final_f_contrib, 6),
                "exposure": round(final_e_contrib, 6),
                "technical_contribution": round(final_t_contrib, 6),
                "fundamental_contribution": round(final_f_contrib, 6),
                "exposure_contribution": round(final_e_contrib, 6),
            },
            "agreement": agreement,
            "regime_override_active": regime_override_active,
            "divergence_signature": divergence_signature,
            "conflict_residue_vector": conflict_residue_vector,
            "agreement_polarization_index": agreement_polarization_index,
            "fusion_micro": fusion_micro,
            "fusion_meso": fusion_meso,
            "fusion_macro": fusion_macro,
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
            results[symbol] = self.fuse_states(tech, fund, exp, regime, weights, symbol)
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
