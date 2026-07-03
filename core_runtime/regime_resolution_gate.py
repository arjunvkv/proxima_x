"""
Regime Resolution Gate — produces a gate verdict combining resolution
classification, structure scale, and entropy map to determine optimal
operating timeframe and signal viability.
"""

import logging

logger = logging.getLogger(__name__)

_instances = {}


def RegimeResolutionGate(instance_id="default"):
    if instance_id not in _instances:
        _instances[instance_id] = _RegimeResolutionGate(instance_id)
    return _instances[instance_id]


class _RegimeResolutionGate:
    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        logger.debug("RegimeResolutionGate(%r) initialised", instance_id)

    def evaluate(self, symbol, resolution_classification, structure_scale, entropy_map):
        resolution = resolution_classification.get("resolution", "NOISE")
        res_confidence = resolution_classification.get("confidence", 0.0)

        scale = structure_scale.get("scale", "MICRO_NOISE")
        scale_confidence = structure_scale.get("confidence", 0.0)

        entropies = entropy_map.get("entropies", {})
        optimal_tf = entropy_map.get("optimal_timeframe", "1M")
        entropy_viability = entropy_map.get("signal_viability", "LOW")
        pred_horizon = entropy_map.get("predictability_horizon", 0)

        if resolution == "NOISE" and scale == "MICRO_NOISE" and entropy_viability == "LOW":
            signal_viability = "LOW"
        elif resolution in ("MICRO_STRUCTURE", "MESO_STRUCTURE"):
            signal_viability = entropy_viability
        else:
            signal_viability = entropy_viability

        has_entropy_data = len(entropies) > 0
        if has_entropy_data and entropy_viability != "LOW":
            gate_optimal_tf = optimal_tf
        else:
            gate_optimal_tf = "1M"

        confidences = [res_confidence, scale_confidence]
        if has_entropy_data:
            if entropy_viability == "HIGH":
                confidences.append(1.0)
            elif entropy_viability == "MODERATE":
                confidences.append(0.6)
            else:
                confidences.append(0.3)
        else:
            confidences.append(0.0)

        confidence = round(sum(confidences) / len(confidences), 4)

        reasoning_parts = []
        reasoning_parts.append(f"resolution={resolution}({res_confidence})")
        reasoning_parts.append(f"scale={scale}({scale_confidence})")
        reasoning_parts.append(f"entropy_viability={entropy_viability}")
        reasoning_parts.append(f"optimal_tf={gate_optimal_tf}")

        if resolution == "NOISE" and scale == "MICRO_NOISE" and entropy_viability == "LOW":
            reasoning_parts.append("all_indicators_agree_noise")
            recommended_action = "STOP"
        elif resolution in ("MICRO_STRUCTURE", "MESO_STRUCTURE") and entropy_viability == "HIGH":
            reasoning_parts.append("structure_confirmed_high_predictability")
            recommended_action = "CONTINUE_TICK"
        elif resolution == "MACRO_TREND" and entropy_viability == "HIGH":
            reasoning_parts.append("macro_trend_high_predictability")
            recommended_action = "CONTINUE_TICK"
        elif entropy_viability == "LOW":
            reasoning_parts.append("low_predictability_escalate")
            recommended_action = "ESCALATE_TIMEFRAME"
        else:
            reasoning_parts.append("moderate_conditions_monitor")
            recommended_action = "CONTINUE_TICK"

        reasoning = "; ".join(reasoning_parts)

        return {
            "optimal_timeframe": gate_optimal_tf,
            "current_tick_regime": resolution,
            "signal_viability": signal_viability,
            "confidence": confidence,
            "reasoning": reasoning,
            "recommended_action": recommended_action,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Regime Resolution Gate — Self Test")
    print("=" * 60)

    gate = RegimeResolutionGate("selftest")

    # Scenario 1: Noise regime — all indicators agree noise
    print("\n--- Scenario 1: Noise (all indicators agree noise) ---")
    res_noise = {
        "resolution": "NOISE",
        "confidence": 0.8,
        "volatility": 0.000005,
        "tick_frequency": 0.0,
        "price_range": 0.0001,
        "signal_viability": "LOW",
    }
    scale_noise = {
        "scale": "MICRO_NOISE",
        "confidence": 0.7,
        "hurst_approx": 0.5,
        "variance_ratio_10_50": 1.0,
        "variance_ratio_50_200": 1.0,
        "effective_resolution": "MICRO_NOISE",
    }
    entropy_noise = {
        "entropies": {"TICK": 1.5, "1M": 1.8, "5M": 1.9},
        "optimal_timeframe": "TICK",
        "predictability_horizon": 0,
        "signal_viability": "LOW",
        "entropy_gradient": [["TICK", 1.5], ["1M", 1.8], ["5M", 1.9]],
    }
    r1 = gate.evaluate("NOISE_SYM", res_noise, scale_noise, entropy_noise)
    print(f"  optimal_timeframe: {r1['optimal_timeframe']}")
    print(f"  current_tick_regime: {r1['current_tick_regime']}")
    print(f"  signal_viability: {r1['signal_viability']}")
    print(f"  confidence: {r1['confidence']}")
    print(f"  reasoning: {r1['reasoning']}")
    print(f"  recommended_action: {r1['recommended_action']}")
    assert r1["signal_viability"] == "LOW", f"Expected LOW viability, got {r1['signal_viability']}"
    assert r1["recommended_action"] == "STOP", f"Expected STOP, got {r1['recommended_action']}"
    print("  >>> PASS")

    # Scenario 2: Moderate regime — MICRO_STRUCTURE with moderate entropy
    print("\n--- Scenario 2: Moderate (MICRO_STRUCTURE + moderate entropy) ---")
    res_mod = {
        "resolution": "MICRO_STRUCTURE",
        "confidence": 0.6,
        "volatility": 0.0003,
        "tick_frequency": 0.0,
        "price_range": 0.005,
        "signal_viability": "MODERATE",
    }
    scale_mod = {
        "scale": "MESO_STRUCTURE",
        "confidence": 0.5,
        "hurst_approx": 0.6,
        "variance_ratio_10_50": 1.2,
        "variance_ratio_50_200": 0.8,
        "effective_resolution": "MESO_STRUCTURE",
    }
    entropy_mod = {
        "entropies": {"TICK": 0.8, "1M": 0.6, "5M": 0.9},
        "optimal_timeframe": "1M",
        "predictability_horizon": 50,
        "signal_viability": "MODERATE",
        "entropy_gradient": [["TICK", 0.8], ["1M", 0.6], ["5M", 0.9]],
    }
    r2 = gate.evaluate("MOD_SYM", res_mod, scale_mod, entropy_mod)
    print(f"  optimal_timeframe: {r2['optimal_timeframe']}")
    print(f"  current_tick_regime: {r2['current_tick_regime']}")
    print(f"  signal_viability: {r2['signal_viability']}")
    print(f"  confidence: {r2['confidence']}")
    print(f"  reasoning: {r2['reasoning']}")
    print(f"  recommended_action: {r2['recommended_action']}")
    assert r2["signal_viability"] == "MODERATE", f"Expected MODERATE viability, got {r2['signal_viability']}"
    assert r2["recommended_action"] == "CONTINUE_TICK", f"Expected CONTINUE_TICK, got {r2['recommended_action']}"
    assert r2["optimal_timeframe"] == "1M", f"Expected 1M, got {r2['optimal_timeframe']}"
    print("  >>> PASS")

    # Scenario 3: Strong trend — MACRO_TREND with high predictability
    print("\n--- Scenario 3: Strong Trend (MACRO_TREND + HIGH entropy viability) ---")
    res_trend = {
        "resolution": "MACRO_TREND",
        "confidence": 0.9,
        "volatility": 0.002,
        "tick_frequency": 0.0,
        "price_range": 0.15,
        "signal_viability": "HIGH",
    }
    scale_trend = {
        "scale": "MACRO_TREND",
        "confidence": 0.85,
        "hurst_approx": 0.85,
        "variance_ratio_10_50": 1.8,
        "variance_ratio_50_200": 2.5,
        "effective_resolution": "MACRO_TREND",
    }
    entropy_trend = {
        "entropies": {"TICK": 0.3, "1M": 0.1, "5M": 0.2, "15M": 0.15, "1H": 0.4},
        "optimal_timeframe": "1M",
        "predictability_horizon": 250,
        "signal_viability": "HIGH",
        "entropy_gradient": [["TICK", 0.3], ["1M", 0.1], ["5M", 0.2], ["15M", 0.15], ["1H", 0.4]],
    }
    r3 = gate.evaluate("TREND_SYM", res_trend, scale_trend, entropy_trend)
    print(f"  optimal_timeframe: {r3['optimal_timeframe']}")
    print(f"  current_tick_regime: {r3['current_tick_regime']}")
    print(f"  signal_viability: {r3['signal_viability']}")
    print(f"  confidence: {r3['confidence']}")
    print(f"  reasoning: {r3['reasoning']}")
    print(f"  recommended_action: {r3['recommended_action']}")
    assert r3["signal_viability"] == "HIGH", f"Expected HIGH viability, got {r3['signal_viability']}"
    assert r3["recommended_action"] == "CONTINUE_TICK", f"Expected CONTINUE_TICK, got {r3['recommended_action']}"
    assert r3["optimal_timeframe"] == "1M", f"Expected 1M, got {r3['optimal_timeframe']}"
    assert r3["confidence"] > 0.5, f"Expected confidence > 0.5, got {r3['confidence']}"
    print("  >>> PASS")

    # Singleton test
    print("\n--- Singleton test ---")
    same = RegimeResolutionGate("selftest")
    assert same is gate, "Singleton should return the same instance"
    print("  >>> PASS")

    print("\n" + "=" * 60)
    print("All self-tests PASSED.")
    print("=" * 60)
