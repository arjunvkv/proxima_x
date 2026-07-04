from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

from .adaptive_weight_engine import AdaptiveWeightEngine
from .bidirectional_fusion import BidirectionalFusionLayer


class UnifiedConvictionField:
    def __init__(self) -> None:
        self.weight_engine = AdaptiveWeightEngine()
        self.fusion_layer = BidirectionalFusionLayer()
        self.history: list[dict[str, Any]] = []
        self.context_shadow_state: dict[str, Any] = {
            "history": [],
            "rolling_stats": {}
        }

    def compute(
        self,
        symbols: list[str],
        technical_convictions: dict[str, dict[str, Any]],
        fundamental_convictions: dict[str, dict[str, Any]],
        exposure_convictions: dict[str, dict[str, Any]],
        regime_context: dict[str, Any],
    ) -> dict[str, Any]:
        weights = self.weight_engine.compute_weights(regime_context)

        states: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            tech = technical_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            fund = fundamental_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            expo = exposure_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            states[symbol] = {
                "technical": tech,
                "fundamental": fund,
                "exposure": expo,
            }

        fusion_weights = {
            "technical": weights["technical_weight"],
            "fundamental": weights["fundamental_weight"],
            "exposure": weights["exposure_weight"],
        }
        regime = regime_context.get("regime", "neutral")
        fused_results = self.fusion_layer.fuse_batch(states, regime, fusion_weights)
        timestamp = time.time()

        field: dict[str, Any] = {}
        direction_counts: Counter[int] = Counter()
        total_agreement = 0.0

        for symbol in symbols:
            fused = fused_results.get(symbol, {})
            tech = technical_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            fund = fundamental_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            expo = exposure_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})

            conviction_score = fused.get("fused_conviction", 0.0)
            direction = fused.get("fused_direction", 0)
            stability = fused.get("fused_stability", 0.5)
            comp_contrib = fused.get(
                "component_contributions",
                {
                    "technical": 0.0,
                    "fundamental": 0.0,
                    "exposure": 0.0,
                },
            )
            component_breakdown = {
                "technical": comp_contrib.get("technical", 0.0),
                "fundamental": comp_contrib.get("fundamental", 0.0),
                "exposure": comp_contrib.get("exposure", 0.0),
                "technical_contribution": comp_contrib.get("technical", 0.0),
                "fundamental_contribution": comp_contrib.get("fundamental", 0.0),
                "exposure_contribution": comp_contrib.get("exposure", 0.0),
            }
            agreement = fused.get("agreement", 0.0)

            directions = [tech.get("direction", 0), fund.get("direction", 0), expo.get("direction", 0)]
            entropy = self._compute_entropy(directions)

            regime_adapted = regime != "neutral"

            field[symbol] = {
                "conviction_score": max(0.0, min(1.0, conviction_score)),
                "direction": direction,
                "stability": max(0.0, min(1.0, stability)),
                "entropy": max(0.0, min(1.0, entropy)),
                "component_breakdown": component_breakdown,
                "agreement": max(-1.0, min(1.0, agreement)),
                "regime_adapted": regime_adapted,
            }

            direction_counts[direction] += 1
            total_agreement += agreement

        field_coherence = total_agreement / len(symbols) if symbols else 0.0
        dominant_direction = direction_counts.most_common(1)[0][0] if direction_counts else 0

        # 1. Regime-Conditioned Memory Retention (A3)
        regime_lower = str(regime).lower()
        rf_mapping = {
            "stable": 1.0,
            "transition": 0.5,
            "risk_on": 0.8,
            "risk_off": 0.6,
            "neutral": 1.0,
        }
        ret_factor = rf_mapping.get(regime_lower, 1.0)
        max_history_len = max(10, int(100 * ret_factor))

        result: dict[str, Any] = {
            "field": field,
            "weights": {
                "technical_weight": weights["technical_weight"],
                "fundamental_weight": weights["fundamental_weight"],
                "macro_weight": weights["macro_weight"],
                "exposure_weight": weights["exposure_weight"],
                "confidence": weights["confidence"],
            },
            "regime": regime,
            "field_coherence": field_coherence,
            "dominant_direction": dominant_direction,
            "timestamp": timestamp,
            "retention_factor": ret_factor,
        }

        # 2. UCF ↔ FSV Memory Symmetry Layer & State Drift Metric (B1, B2)
        current_drifts = {}
        current_ucf = {}
        current_fsv = {}
        for symbol in symbols:
            ucf_conv = self._clean_nan(field.get(symbol, {}).get("conviction_score", 0.5), 0.5)
            fsv_conv = self._clean_nan(fundamental_convictions.get(symbol, {}).get("conviction", 0.5), 0.5)
            current_drifts[symbol] = ucf_conv - fsv_conv
            current_ucf[symbol] = ucf_conv
            current_fsv[symbol] = fsv_conv

        self.context_shadow_state["history"].append({
            "timestamp": timestamp,
            "drifts": current_drifts,
            "ucf": current_ucf,
            "fsv": current_fsv
        })
        if len(self.context_shadow_state["history"]) > max_history_len:
            self.context_shadow_state["history"] = self.context_shadow_state["history"][-max_history_len:]

        drift_velocities = {}
        drift_accelerations = {}
        shadow_history = self.context_shadow_state["history"]
        n_shadow = len(shadow_history)

        for symbol in symbols:
            v = 0.0
            a = 0.0
            if n_shadow >= 2:
                d_curr = shadow_history[-1]["drifts"].get(symbol, 0.0)
                d_prev = shadow_history[-2]["drifts"].get(symbol, 0.0)
                v = d_curr - d_prev
            if n_shadow >= 3:
                d_prev2 = shadow_history[-3]["drifts"].get(symbol, 0.0)
                v_prev = d_prev - d_prev2
                a = v - v_prev
            drift_velocities[symbol] = v
            drift_accelerations[symbol] = a

        rolling_stats = {}
        for symbol in symbols:
            drifts_list = [h["drifts"].get(symbol, 0.0) for h in shadow_history]
            ucf_list = [h["ucf"].get(symbol, 0.0) for h in shadow_history]
            fsv_list = [h["fsv"].get(symbol, 0.0) for h in shadow_history]
            
            m_drift = sum(drifts_list) / len(drifts_list) if drifts_list else 0.0
            m_ucf = sum(ucf_list) / len(ucf_list) if ucf_list else 0.0
            m_fsv = sum(fsv_list) / len(fsv_list) if fsv_list else 0.0
            
            var_drift = sum((x - m_drift) ** 2 for x in drifts_list) / len(drifts_list) if drifts_list else 0.0
            var_ucf = sum((x - m_ucf) ** 2 for x in ucf_list) / len(ucf_list) if ucf_list else 0.0
            var_fsv = sum((x - m_fsv) ** 2 for x in fsv_list) / len(fsv_list) if fsv_list else 0.0
            
            rolling_stats[symbol] = {
                "mean_drift": m_drift,
                "std_drift": math.sqrt(var_drift),
                "mean_ucf": m_ucf,
                "std_ucf": math.sqrt(var_ucf),
                "mean_fsv": m_fsv,
                "std_fsv": math.sqrt(var_fsv),
                "count": len(drifts_list)
            }
        self.context_shadow_state["rolling_stats"] = rolling_stats

        avg_drift_velocity = sum(drift_velocities.values()) / len(drift_velocities) if drift_velocities else 0.0
        avg_drift_acceleration = sum(drift_accelerations.values()) / len(drift_accelerations) if drift_accelerations else 0.0

        result["drift_metrics"] = {
            "symbol_drift_velocity": drift_velocities,
            "symbol_drift_acceleration": drift_accelerations,
            "avg_drift_velocity": avg_drift_velocity,
            "avg_drift_acceleration": avg_drift_acceleration,
            "context_shadow_state": dict(rolling_stats)
        }

        # 3. Over-Governance Detection & Predictive Entropy Gain (D1, D2)
        input_convictions = []
        for s in symbols:
            input_convictions.append(self._clean_nan(technical_convictions.get(s, {}).get("conviction", 0.0), 0.0))
            input_convictions.append(self._clean_nan(fundamental_convictions.get(s, {}).get("conviction", 0.0), 0.0))
            input_convictions.append(self._clean_nan(exposure_convictions.get(s, {}).get("conviction", 0.0), 0.0))

        n_inputs = len(input_convictions)
        mean_input = sum(input_convictions) / n_inputs if n_inputs else 0.0
        var_input = sum((x - mean_input) ** 2 for x in input_convictions) / n_inputs if n_inputs else 0.0
        std_input = math.sqrt(var_input)

        output_convictions = [self._clean_nan(field[s]["conviction_score"], 0.0) for s in symbols]
        n_outputs = len(output_convictions)
        mean_output = sum(output_convictions) / n_outputs if n_outputs else 0.0
        var_output = sum((x - mean_output) ** 2 for x in output_convictions) / n_outputs if n_outputs else 0.0
        std_output = math.sqrt(var_output)

        signal_diversity_suppression = max(0.0, std_input - std_output)
        governance_compression_ratio = std_output / (std_input + 1e-6)
        over_governance_detected = bool(governance_compression_ratio < 0.3 and std_input > 0.05)

        avg_input_entropy = sum(self._clean_nan(field[s]["entropy"], 0.0) for s in symbols) / len(symbols) if symbols else 0.0
        total_score = sum(output_convictions)
        if total_score > 0 and len(symbols) > 1:
            probs = [score / total_score for score in output_convictions]
            output_entropy = -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(symbols))
        else:
            output_entropy = 0.0

        predictive_entropy_gain = avg_input_entropy - output_entropy
        forecasting_sharpness = max(output_convictions) if output_convictions else 0.0
        forecasting_vs_coherence_ratio = forecasting_sharpness / (field_coherence + 1e-6)

        result["governance_metrics"] = {
            "input_conviction_std": std_input,
            "output_conviction_std": std_output,
            "signal_diversity_suppression": signal_diversity_suppression,
            "governance_compression_ratio": governance_compression_ratio,
            "over_governance_detected": over_governance_detected,
            "avg_input_entropy": avg_input_entropy,
            "output_entropy": output_entropy,
            "predictive_entropy_gain": predictive_entropy_gain,
            "forecasting_sharpness": forecasting_sharpness,
            "forecasting_vs_coherence_ratio": forecasting_vs_coherence_ratio,
        }

        result["divergence_signature"] = {symbol: fused_results.get(symbol, {}).get("divergence_signature") for symbol in symbols}
        result["conflict_residue_vector"] = {symbol: fused_results.get(symbol, {}).get("conflict_residue_vector") for symbol in symbols}
        result["agreement_polarization_index"] = {symbol: fused_results.get(symbol, {}).get("agreement_polarization_index") for symbol in symbols}
        result["fusion_micro"] = {symbol: fused_results.get(symbol, {}).get("fusion_micro") for symbol in symbols}
        result["fusion_meso"] = {symbol: fused_results.get(symbol, {}).get("fusion_meso") for symbol in symbols}
        result["fusion_macro"] = {symbol: fused_results.get(symbol, {}).get("fusion_macro") for symbol in symbols}

        self.history.append(result)
        if len(self.history) > max_history_len:
            self.history = self.history[-max_history_len:]

        return result

    def _clean_nan(self, val: Any, default: float = 0.0) -> float:
        try:
            if val is None or math.isnan(float(val)):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    def get_field_snapshot(self) -> dict[str, Any]:
        if not self.history:
            return {}
        return dict(self.history[-1])

    def get_coherence_timeline(self, limit: int = 50) -> list[float]:
        return [entry["field_coherence"] for entry in self.history[-limit:]]

    def get_weight_evolution(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(entry["weights"]) for entry in self.history[-limit:]]

    def reset(self) -> None:
        self.history.clear()
        self.context_shadow_state["history"].clear()
        self.context_shadow_state["rolling_stats"].clear()

    def _compute_entropy(self, directions: list[int]) -> float:
        total = len(directions)
        if total == 0:
            return 0.0
        counts: Counter[int] = Counter(directions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(3)
        return entropy / max_entropy if max_entropy > 0 else 0.0
