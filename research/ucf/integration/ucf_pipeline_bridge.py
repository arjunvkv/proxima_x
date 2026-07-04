from __future__ import annotations

import math
import time
from typing import Any

from ..core.unified_conviction_field import UnifiedConvictionField
from ..core.adaptive_weight_engine import AdaptiveWeightEngine
from ..integration.regime_adaptive_modulator import RegimeAdaptiveModulator
from ...fsv.core.fsv_schema import FundamentalStateVector


class UCFPipelineBridge:
    def __init__(self) -> None:
        self.weight_engine = AdaptiveWeightEngine()
        self.ucf = UnifiedConvictionField()
        self.modulator = RegimeAdaptiveModulator()
        self._last_result: dict[str, Any] | None = None
        self.context_shadow_state: dict[str, Any] = {
            "history": [],
            "rolling_stats": {}
        }

    def retention_factor(self, regime: str) -> float:
        return self.modulator.retention_factor(regime)


    def process(
        self,
        symbols: list[str],
        technical_states: dict[str, dict[str, Any]],
        fsv_states: dict[str, dict[str, Any]] | None = None,
        cev_state: dict[str, dict[str, Any]] | None = None,
        regime_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        engine = AdaptiveWeightEngine()
        ucf = UnifiedConvictionField()

        if fsv_states is None:
            fsv_states = {s: {"conviction": 0.5, "direction": 0, "stability": 0.5} for s in symbols}
        if cev_state is None:
            cev_state = {s: {"conviction": 0.5, "direction": 0, "stability": 0.5} for s in symbols}
        if regime_state is None:
            regime_state = {}

        regime_context = self._build_regime_context_from_state(regime_state)

        raw_result = ucf.compute(
            symbols,
            technical_states,
            fundamental_convictions=fsv_states,
            exposure_convictions=cev_state,
            regime_context=regime_context,
        )

        instability = 1.0 - regime_context.get("regime_stability", 0.5)
        modulator = RegimeAdaptiveModulator()

        modulation_entries = []
        for symbol in symbols:
            entry = dict(raw_result["field"].get(symbol, {}))
            fsv_dir = fsv_states.get(symbol, {}).get("direction", 0)
            fused_dir = raw_result["field"].get(symbol, {}).get("direction", 0)
            fsv_conv = fsv_states.get(symbol, {}).get("conviction", 0.0)
            entry["fsv_alignment"] = fsv_dir * fused_dir * fsv_conv
            modulation_entries.append(entry)

        modulation_result = {"field": modulation_entries}
        modulated = modulator.modulate_field(modulation_result, regime_context.get("regime", "neutral"), instability)

        for i, symbol in enumerate(symbols):
            if symbol in raw_result["field"]:
                raw_result["field"][symbol]["conviction_score"] = modulated["field"][i]["conviction_score"]

        sorted_symbols = sorted(
            symbols,
            key=lambda s: raw_result["field"].get(s, {}).get("conviction_score", 0.0),
            reverse=True,
        )

        ranked_symbols = [
            {
                "symbol": s,
                "ucf_score": raw_result["field"][s]["conviction_score"],
                "direction": raw_result["field"][s]["direction"],
                "stability": raw_result["field"][s]["stability"],
                "agreement": raw_result["field"][s].get("agreement", 0.0),
            }
            for s in sorted_symbols
        ]

        selected = sorted_symbols[0] if sorted_symbols else (symbols[0] if symbols else "")

        output: dict[str, Any] = {
            "ranked_symbols": ranked_symbols,
            "field": dict(raw_result["field"]),
            "selected_symbol": selected,
            "regime": regime_context.get("regime", "neutral"),
            "weights_used": dict(raw_result.get("weights", {})),
            "modulation_applied": True,
            "is_blocking": False,
            "is_selection": True,
            "fallback_used": False,
            "timestamp": time.time(),
        }

        # 1. Regime-Conditioned Memory Retention (A3)
        r_factor = self.retention_factor(output["regime"])
        max_history_len = max(10, int(100 * r_factor))
        output["retention_factor"] = r_factor

        # 2. UCF ↔ FSV Memory Symmetry Layer & State Drift Metric (B1, B2)
        current_drifts = {}
        current_ucf = {}
        current_fsv = {}
        for symbol in symbols:
            ucf_conv = self._clean_nan(output["field"].get(symbol, {}).get("conviction_score", 0.5), 0.5)
            fsv_conv = self._clean_nan(fsv_states.get(symbol, {}).get("conviction", 0.5), 0.5)
            current_drifts[symbol] = ucf_conv - fsv_conv
            current_ucf[symbol] = ucf_conv
            current_fsv[symbol] = fsv_conv

        self.context_shadow_state["history"].append({
            "timestamp": output["timestamp"],
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

        output["drift_metrics"] = {
            "symbol_drift_velocity": drift_velocities,
            "symbol_drift_acceleration": drift_accelerations,
            "avg_drift_velocity": avg_drift_velocity,
            "avg_drift_acceleration": avg_drift_acceleration,
            "context_shadow_state": dict(rolling_stats)
        }

        # 3. Over-Governance Detection & Predictive Entropy Gain (D1, D2)
        input_convictions = []
        for s in symbols:
            input_convictions.append(self._clean_nan(technical_states.get(s, {}).get("conviction", 0.0), 0.0))
            input_convictions.append(self._clean_nan(fsv_states.get(s, {}).get("conviction", 0.0), 0.0))
            if cev_state:
                input_convictions.append(self._clean_nan(cev_state.get(s, {}).get("conviction", 0.0), 0.0))
            else:
                input_convictions.append(0.5)

        n_inputs = len(input_convictions)
        mean_input = sum(input_convictions) / n_inputs if n_inputs else 0.0
        var_input = sum((x - mean_input) ** 2 for x in input_convictions) / n_inputs if n_inputs else 0.0
        std_input = math.sqrt(var_input)

        output_convictions = [self._clean_nan(output["field"][s]["conviction_score"], 0.0) for s in symbols]
        n_outputs = len(output_convictions)
        mean_output = sum(output_convictions) / n_outputs if n_outputs else 0.0
        var_output = sum((x - mean_output) ** 2 for x in output_convictions) / n_outputs if n_outputs else 0.0
        std_output = math.sqrt(var_output)

        signal_diversity_suppression = max(0.0, std_input - std_output)
        governance_compression_ratio = std_output / (std_input + 1e-6)
        over_governance_detected = bool(governance_compression_ratio < 0.3 and std_input > 0.05)

        avg_input_entropy = sum(self._clean_nan(output["field"][s].get("entropy", 0.0), 0.0) for s in symbols) / len(symbols) if symbols else 0.0
        total_score = sum(output_convictions)
        if total_score > 0 and len(symbols) > 1:
            probs = [score / total_score for score in output_convictions]
            output_entropy = -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(symbols))
        else:
            output_entropy = 0.0

        predictive_entropy_gain = avg_input_entropy - output_entropy
        forecasting_sharpness = max(output_convictions) if output_convictions else 0.0
        
        # Calculate agreement coherence
        agreement_list = [output["field"][s].get("agreement", 0.0) for s in symbols]
        coherence = sum(agreement_list) / len(symbols) if symbols else 0.0
        forecasting_vs_coherence_ratio = forecasting_sharpness / (coherence + 1e-6)

        output["governance_metrics"] = {
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

        self._last_result = output
        return output

    def _clean_nan(self, val: Any, default: float = 0.0) -> float:
        try:
            if val is None or math.isnan(float(val)):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default



    def _build_regime_context_from_state(self, regime_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "regime": regime_state.get("regime", "neutral"),
            "regime_stability": regime_state.get("regime_stability", 0.0),
            "fsv_entropy": regime_state.get("fsv_entropy", 0.0),
            "technical_volatility": regime_state.get("technical_volatility", 0.0),
            "recent_prediction_error": regime_state.get("recent_prediction_error", 0.0),
            "exposure_concentration": regime_state.get("exposure_concentration", 0.0),
        }

    def build_regime_context(
        self,
        fsv_engine: Any = None,
        symbols: list[str] | None = None,
        fsv_states: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "regime": "neutral",
            "regime_stability": 0.5,
            "fsv_entropy": 0.0,
            "technical_volatility": 0.0,
            "recent_prediction_error": 0.0,
            "exposure_concentration": 0.0,
        }

    def validate_output(self, result: dict[str, Any], symbols: list[str]) -> bool:
        ranked = result.get("ranked_symbols", [])
        if not ranked:
            return False
        selected = result.get("selected_symbol", "")
        if selected not in symbols:
            return False
        if result.get("is_blocking", True) is not False:
            return False
        for entry in ranked:
            score = entry.get("ucf_score", -1.0)
            if not (0.0 <= score <= 1.0):
                return False
        return True

    def process_with_fallback(
        self,
        symbols: list[str],
        technical_states: dict[str, dict[str, Any]],
        fsv_states: dict[str, dict[str, Any]] | None = None,
        cev_state: dict[str, dict[str, Any]] | None = None,
        regime_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self.process(symbols, technical_states, fsv_states, cev_state, regime_state)
        except Exception:
            safe_symbol = symbols[0] if symbols else ""
            return {
                "ranked_symbols": [{"symbol": safe_symbol, "ucf_score": 0.5, "direction": 0, "stability": 0.5, "agreement": 0.0}],
                "field": {},
                "selected_symbol": safe_symbol,
                "regime": "neutral",
                "weights_used": {},
                "modulation_applied": False,
                "is_blocking": False,
                "is_selection": True,
                "fallback_used": True,
                "timestamp": time.time(),
            }
