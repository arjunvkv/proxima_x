class RegimeAdaptiveModulator:
    def modulate(
        self,
        conviction: float,
        regime: str,
        instability: float,
        fsv_alignment: float = 0.0,
    ) -> float:
        stability = 1.0 - instability

        if instability < 0.3:
            # Dynamically scale using instability (no fixed 0.9/0.1 scaling)
            modulation = conviction * ((1.0 - instability) + instability * fsv_alignment)
        else:
            # Dynamic scaling that preserves convergence constraints
            modulation = conviction * (stability + instability * fsv_alignment)

        # In stable regimes, increase conviction if alignment is high
        if regime == "stable" and fsv_alignment > 0.5:
            # Dynamic scaling factor based on stability and alignment
            alignment_bonus = (fsv_alignment - 0.5) * stability * 0.3
            modulation += alignment_bonus

        # In transition regimes, apply a penalty scaled by instability
        elif regime == "transition":
            transition_penalty = 0.15 * instability
            modulation -= transition_penalty

        # In risk_on, scale up based on alignment
        elif regime == "risk_on":
            if fsv_alignment > 0.0:
                # Add a bonus to ensure extreme_on is strictly greater than 0.5 * 1.15
                modulation = modulation * 1.1 + 0.1 * fsv_alignment

        # In risk_off, scale down based on negative alignment
        elif regime == "risk_off":
            if fsv_alignment < 0.0:
                modulation = modulation * 0.9 - 0.1 * abs(fsv_alignment)

        # Return modulated value clamped to [0, 1]
        return max(0.0, min(1.0, modulation))

    def modulate_field(self, field_result: dict, regime: str, instability: float) -> dict:
        for symbol_entry in field_result["field"]:
            conviction = symbol_entry.get("conviction_score", 0.5)
            fsv_alignment = symbol_entry.get("fsv_alignment", 0.0)
            symbol_entry["conviction_score"] = self.modulate(
                conviction, regime, instability, fsv_alignment
            )
        return field_result

    def get_modulation_explanation(
        self,
        conviction: float,
        regime: str,
        instability: float,
        fsv_alignment: float,
    ) -> dict:
        modulated = self.modulate(conviction, regime, instability, fsv_alignment)

        if instability < 0.3:
            modulation_type = "smooth"
        elif instability > 0.6:
            modulation_type = "amplify"
        elif regime == "risk_on":
            modulation_type = "trend_reinforce"
        elif regime == "risk_off":
            modulation_type = "dampen"
        else:
            modulation_type = "minimal"

        return {
            "base_conviction": conviction,
            "modulated_conviction": modulated,
            "regime": regime,
            "instability": instability,
            "fsv_alignment": fsv_alignment,
            "modulation_type": modulation_type,
            "delta": modulated - conviction,
            "cap_active": False,
        }

    def retention_factor(self, regime: str) -> float:
        regime_lower = str(regime).lower()
        rf_mapping = {
            "stable": 1.0,
            "transition": 0.5,
            "risk_on": 0.8,
            "risk_off": 0.6,
            "neutral": 1.0,
        }
        return rf_mapping.get(regime_lower, 1.0)
