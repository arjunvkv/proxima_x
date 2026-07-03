"""
Currency Exposure Vector (CEV) State Engine.

Maintains real-time exposure vector across all open positions.
Tracks net directional exposure per currency (GBP, USD, EUR, JPY, AUD, CHF, NZD, CAD, XAU).
"""

import copy


class CurrencyExposureVectorEngine:
    """Maintains real-time exposure vector across all open positions.

    Tracks net directional exposure per currency. Exposure accumulates
    additively across positions, so opposing trades naturally neutralize.

    Rules:
      - BUY  <BASE><QUOTE>  → +volume to base, -volume to quote
      - SELL <BASE><QUOTE>  → -volume to base, +volume to quote
    """

    SUPPORTED_CURRENCIES = frozenset({
        "GBP", "USD", "EUR", "JPY", "AUD", "CHF", "NZD", "CAD", "XAU",
    })

    def __init__(self) -> None:
        self.exposure: dict[str, float] = {
            "GBP": 0.0,
            "USD": 0.0,
            "EUR": 0.0,
            "JPY": 0.0,
            "AUD": 0.0,
            "CHF": 0.0,
            "NZD": 0.0,
            "CAD": 0.0,
            "XAU": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_position(self, symbol: str, direction: str, volume: float) -> None:
        """Accumulate exposure from a single position.

        Parameters
        ----------
        symbol : str
            Forex symbol, e.g. "GBPAUD", "EURUSD".
        direction : str
            Trade direction: "BUY" or "SELL".
        volume : float
            Trade volume in lots (or any unit, used as-is).

        Raises
        ------
        ValueError
            If the symbol cannot be parsed into two known currencies,
            or if direction is not BUY/SELL.
        """
        direction_upper = direction.upper()
        if direction_upper not in ("BUY", "SELL"):
            raise ValueError(f"Invalid direction '{direction}'; must be BUY or SELL")

        base, quote = self._parse_symbol(symbol)

        volume = float(volume)

        if direction_upper == "BUY":
            self.exposure[base] += volume
            self.exposure[quote] -= volume
        else:  # SELL
            self.exposure[base] -= volume
            self.exposure[quote] += volume

    def reset(self) -> None:
        """Zero out all exposures.

        Call at the start of each cycle before re-computing from positions.
        """
        for currency in self.exposure:
            self.exposure[currency] = 0.0

    def get_exposure_vector(self) -> dict[str, float]:
        """Return a deep copy of the current exposure dictionary.

        Returns
        -------
        dict[str, float]
            Snapshot of current exposures keyed by currency.
        """
        return copy.deepcopy(self.exposure)

    def get_concentration_score(self, currency: str) -> float:
        """Return the absolute exposure for *currency* normalised by
        total absolute exposure across all currencies.

        Formula: ``abs(self.exposure[currency]) / total_abs``

        Parameters
        ----------
        currency : str
            Currency code, e.g. "GBP".

        Returns
        -------
        float
            Concentration score in [0.0, 1.0].  Returns 0.0 when total
            absolute exposure is zero.
        """
        total_abs = sum(abs(v) for v in self.exposure.values())
        if total_abs == 0.0:
            return 0.0
        return abs(self.exposure[currency]) / total_abs

    def get_dominant_currency(self) -> tuple[str, float]:
        """Return the currency with the highest absolute exposure and its
        concentration score.

        Returns
        -------
        tuple[str, float]
            (currency_code, concentration_score).  Returns ("", 0.0) when
            total absolute exposure is zero.
        """
        total_abs = sum(abs(v) for v in self.exposure.values())
        if total_abs == 0.0:
            return ("", 0.0)

        dominant = max(self.exposure, key=lambda c: abs(self.exposure[c]))
        score = abs(self.exposure[dominant]) / total_abs
        return (dominant, score)

    def get_summary(self) -> dict:
        """Return a full summary of the current exposure state.

        Returns
        -------
        dict
            Contains keys:
              - exposure        : dict of net exposures per currency
              - concentration   : dict of concentration scores per currency
              - dominant        : (currency, score) tuple
              - total_abs       : sum of absolute exposures
        """
        total_abs = sum(abs(v) for v in self.exposure.values())
        concentration = {
            c: self.get_concentration_score(c) for c in self.exposure
        }
        dominant_currency, dominant_score = self.get_dominant_currency()

        return {
            "exposure": copy.deepcopy(self.exposure),
            "concentration": concentration,
            "dominant": (dominant_currency, dominant_score),
            "total_abs": total_abs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        """Parse a forex symbol into (base, quote) currency pair.

        Iterates over supported currencies from longest to shortest to
        correctly handle prefixes like ``XAU`` (3 chars) vs ``USD`` (3 chars)
        and avoid greedy prefix mismatches.

        Parameters
        ----------
        symbol : str
            Uppercase symbol string, e.g. "GBPAUD".

        Returns
        -------
        tuple[str, str]
            (base_currency, quote_currency).

        Raises
        ------
        ValueError
            If the symbol does not match any known currency pair.
        """
        symbol_upper = symbol.upper()

        # Sort by length descending so that e.g. "XAU" is tried before "X"
        # as a candidate.  This prevents ``XA`` → (XAU wrong) issues.
        known = sorted(CurrencyExposureVectorEngine.SUPPORTED_CURRENCIES,
                       key=len, reverse=True)

        for base in known:
            if symbol_upper.startswith(base):
                remainder = symbol_upper[len(base):]
                for quote in known:
                    if remainder == quote:
                        return (base, quote)

        raise ValueError(
            f"Could not parse symbol '{symbol}' into a known currency pair. "
            f"Supported currencies: {sorted(CurrencyExposureVectorEngine.SUPPORTED_CURRENCIES)}"
        )
