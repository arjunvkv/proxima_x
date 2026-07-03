"""LOEF — Latent Opportunity Extraction Field.

Continuous field of executable opportunity density.
Ranks tradability surface across symbols.
"""


class LatentOpportunityExtractionField:
    """Computes opportunity density across symbols from multi-factor signals."""

    def __init__(self):
        pass

    def compute(
        self,
        signals: list,
        sil_scores: dict,
        rsi_dict: dict,
        activation: dict,
        readiness: dict,
        erf: float,
        aeem_escape: float,
        gmci: float,
        eprg_reachability: float,
    ) -> dict:
        """Compute opportunity density surface and return ranked results.

        Parameters
        ----------
        signals : list
            List of signal dicts, each expected to have a "symbol" key and
            optionally a "confidence" key.
        sil_scores : dict
            Per-symbol SIL scores (float).
        rsi_dict : dict
            Per-symbol RSI values (float).
        activation : dict
            Field activation state, checked for "pre_activation" key.
        readiness : dict
            Readiness state, checked for "ready" key.
        erf : float
            Emergent Resonance Field value (0-1).
        aeem_escape : float
            AEEM escape energy (0-1). Low escape => high opportunity.
        gmci : float
            GMC integration penalty (0-1).
        eprg_reachability : float
            EPRG reachability score (0-1).

        Returns
        -------
        dict
            Opportunity density report with keys:
            opportunity_density, density_by_symbol, top_k_symbols,
            surface_peaks, tradability_rank, is_opportunity_rich.
        """
        try:
            # --- Build lookup maps from signals list ---------------
            signal_map = {}
            for s in signals:
                sym = s.get("symbol")
                if sym:
                    signal_map[sym] = s.get("confidence", 0.0)

            # --- Collect all unique symbols -------------------------
            all_symbols = set(signal_map.keys())
            all_symbols.update(sil_scores.keys())

            density_by_symbol = {}

            for sym in all_symbols:
                # 1. Base signal confidence (or 0 if no signal)
                density = signal_map.get(sym, 0.0)

                # 2. SIL score contribution
                sil = sil_scores.get(sym, 0.0)
                density += sil * 0.15

                # 3. RSI extremity bonus
                rsi = rsi_dict.get(sym)
                if rsi is not None:
                    rsi_bonus = abs(50.0 - rsi) / 50.0 * 0.20
                    density += rsi_bonus

                # 4. Activation bonus
                if activation.get("pre_activation"):
                    density += 0.10

                # 5. Readiness bonus
                if readiness.get("ready"):
                    density += 0.05

                # 6. ERF bonus
                density += erf * 0.10

                # 7. AEEM escape bonus (low escape energy = high opportunity)
                density += (1.0 - aeem_escape) * 0.05

                # 8. GMCI penalty
                density -= gmci * 0.05

                # 9. EPRG reachability bonus
                density += eprg_reachability * 0.10

                # Clamp to [0.0, 1.0]
                density = max(0.0, min(1.0, density))

                density_by_symbol[sym] = density

            # --- Aggregate ------------------------------------------
            if density_by_symbol:
                overall_density = sum(density_by_symbol.values()) / len(density_by_symbol)
            else:
                overall_density = 0.0

            # --- Top K (3) symbols ----------------------------------
            sorted_symbols = sorted(
                density_by_symbol.items(), key=lambda x: x[1], reverse=True
            )
            top_k = [sym for sym, _ in sorted_symbols[:3]]

            # --- Surface peaks --------------------------------------
            surface_peaks = []
            for sym, dens in sorted_symbols[:3]:
                rsi = rsi_dict.get(sym)
                direction = "BUY" if rsi is not None and rsi < 50 else "SELL"
                surface_peaks.append(
                    {"symbol": sym, "density": dens, "direction": direction}
                )

            # --- Tradability rank (all symbols, desc) --------------
            tradability_rank = [sym for sym, _ in sorted_symbols]

            # --- Opportunity rich flag ------------------------------
            is_opportunity_rich = overall_density > 0.5

            return {
                "opportunity_density": overall_density,
                "density_by_symbol": density_by_symbol,
                "top_k_symbols": top_k,
                "surface_peaks": surface_peaks,
                "tradability_rank": tradability_rank,
                "is_opportunity_rich": is_opportunity_rich,
            }

        except Exception:
            return {
                "opportunity_density": 0.0,
                "density_by_symbol": {},
                "top_k_symbols": [],
                "surface_peaks": [],
                "tradability_rank": [],
                "is_opportunity_rich": False,
            }
