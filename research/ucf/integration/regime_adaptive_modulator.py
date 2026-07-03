class RegimeAdaptiveModulator:
    def modulate(
        self,
        conviction: float,
        regime: str,
        instability: float,
        fsv_alignment: float = 0.0,
    ) -> float:
        if instability < 0.3:
            modulation = conviction * (0.9 + 0.1 * fsv_alignment)
        elif instability > 0.6:
            modulation = conviction * (0.7 + 0.3 * abs(fsv_alignment))
        elif regime == "risk_on":
            modulation = conviction * (1.0 + 0.15 * max(0, fsv_alignment))
        elif regime == "risk_off":
            modulation = conviction * (1.0 - 0.15 * max(0, -fsv_alignment))
        else:
            modulation = conviction

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
