"""
Signal Independence Value (SIV) Engine for CEV v2.

Computes Signal Independence Value between FX symbols and
Effective Information Gain (EIG) per symbol given portfolio context.
This is the mathematical foundation of CEV v2 — no penalties, no hard
constraints, only utility scoring.
"""

import math


class SIVEngine:
    """Computes SIV and EIG for FX symbol pairs and portfolios.

    This is the mathematical foundation of CEV v2 — no penalties,
    no hard constraints, only utility scoring.

    SIV measures how much independent information a symbol carries
    relative to another symbol (or portfolio).  EIG weights that
    independence by a conviction estimate.
    """

    #: Currencies recognised by the engine.
    SUPPORTED_CURRENCIES = frozenset({
        "GBP", "USD", "EUR", "JPY", "AUD", "CHF", "NZD", "CAD", "XAU",
    })

    #: Static regime profile per currency — used by MRD.
    CURRENCY_REGIMES: dict[str, dict[str, float]] = {
        "USD": {"rate_sensitivity": 0.9, "risk_sensitivity": 0.3, "commodity_link": 0.1},
        "EUR": {"rate_sensitivity": 0.7, "risk_sensitivity": 0.4, "commodity_link": 0.2},
        "GBP": {"rate_sensitivity": 0.8, "risk_sensitivity": 0.5, "commodity_link": 0.2},
        "JPY": {"rate_sensitivity": 0.6, "risk_sensitivity": 0.8, "commodity_link": 0.1},
        "AUD": {"rate_sensitivity": 0.5, "risk_sensitivity": 0.7, "commodity_link": 0.9},
        "CHF": {"rate_sensitivity": 0.7, "risk_sensitivity": 0.9, "commodity_link": 0.1},
        "NZD": {"rate_sensitivity": 0.5, "risk_sensitivity": 0.6, "commodity_link": 0.8},
        "CAD": {"rate_sensitivity": 0.6, "risk_sensitivity": 0.5, "commodity_link": 0.7},
        "XAU": {"rate_sensitivity": 0.2, "risk_sensitivity": 0.9, "commodity_link": 0.9},
    }

    #: Static volatility profile per symbol — used by VSD.
    VOLATILITY_PROFILES: dict[str, float] = {
        "GBPAUD": 0.7, "GBPCHF": 0.3, "GBPJPY": 0.8, "GBPUSD": 0.5,
        "EURUSD": 0.4, "EURGBP": 0.3, "EURJPY": 0.7, "EURCHF": 0.3,
        "USDJPY": 0.6, "USDCAD": 0.5, "USDCHF": 0.4, "AUDUSD": 0.6,
        "AUDJPY": 0.7, "AUDNZD": 0.5, "NZDUSD": 0.5, "NZDCAD": 0.5,
        "CADJPY": 0.6, "CHFJPY": 0.5, "XAUUSD": 0.9,
    }

    # ------------------------------------------------------------------
    # Symbol Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        """Parse a forex symbol into (base, quote) currency pair.

        Iterates over supported currencies from longest to shortest to
        correctly handle prefixes (e.g. ``XAU`` is 3 chars).

        Parameters
        ----------
        symbol : str
            Upper- or lower-case symbol, e.g. ``"GBPAUD"``.

        Returns
        -------
        tuple[str, str]
            (base_currency, quote_currency).

        Raises
        ------
        ValueError
            If the symbol cannot be parsed into two known currencies.
        """
        symbol_upper = symbol.upper()
        known = sorted(SIVEngine.SUPPORTED_CURRENCIES, key=len, reverse=True)

        for base in known:
            if symbol_upper.startswith(base):
                remainder = symbol_upper[len(base):]
                for quote in known:
                    if remainder == quote:
                        return (base, quote)

        raise ValueError(
            f"Could not parse symbol '{symbol}' into a known currency pair. "
            f"Supported currencies: {sorted(SIVEngine.SUPPORTED_CURRENCIES)}"
        )

    # ------------------------------------------------------------------
    # Currency Overlap Component (COC)
    # ------------------------------------------------------------------

    def compute_currency_vector(self, symbol: str) -> dict[str, float]:
        """Compute the currency exposure vector for a symbol.

        Parameters
        ----------
        symbol : str
            Forex symbol, e.g. ``"GBPAUD"``.

        Returns
        -------
        dict[str, float]
            Mapping of each supported currency to its exposure:
            base = ``+1.0``, quote = ``-1.0``, others = ``0.0``.

        Raises
        ------
        ValueError
            If the symbol cannot be parsed.
        """
        base, quote = self._parse_symbol(symbol)
        vec: dict[str, float] = {}
        for c in self.SUPPORTED_CURRENCIES:
            if c == base:
                vec[c] = 1.0
            elif c == quote:
                vec[c] = -1.0
            else:
                vec[c] = 0.0
        return vec

    @staticmethod
    def compute_coc(vec_a: dict, vec_b: dict) -> float:
        """Compute Currency Overlap Component via cosine similarity.

        Parameters
        ----------
        vec_a : dict
            Currency vector from ``compute_currency_vector``.
        vec_b : dict
            Currency vector from ``compute_currency_vector``.

        Returns
        -------
        float
            Cosine similarity clamped to ``[0, 1]``.
            Higher = more currency overlap.
        """
        # Union of all keys across both vectors
        all_keys = set(vec_a) | set(vec_b)

        dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in all_keys)

        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cos_sim = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, cos_sim))

    # ------------------------------------------------------------------
    # Macro Regime Divergence (MRD)
    # ------------------------------------------------------------------

    def compute_mrd(self, symbol_a: str, symbol_b: str) -> float:
        """Compute Macro Regime Divergence between two symbols.

        Each symbol's regime profile is a weighted combination of its
        base currency (60 %) and quote currency (40 %).  MRD is the
        cosine similarity of the two profile vectors.

        Parameters
        ----------
        symbol_a : str
            First forex symbol.
        symbol_b : str
            Second forex symbol.

        Returns
        -------
        float
            MRD in ``[0, 1]``.  Higher = more similar regime profile.
        """
        profile_a = self._regime_profile(symbol_a)
        profile_b = self._regime_profile(symbol_b)

        all_keys = set(profile_a) | set(profile_b)
        dot = sum(profile_a.get(k, 0.0) * profile_b.get(k, 0.0) for k in all_keys)
        norm_a = math.sqrt(sum(v * v for v in profile_a.values()))
        norm_b = math.sqrt(sum(v * v for v in profile_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cos_sim = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, cos_sim))

    def _regime_profile(self, symbol: str) -> dict[str, float]:
        """Build the weighted regime profile vector for *symbol*.

        Weighting: ``profile = regime[base] * 0.6 + regime[quote] * 0.4``.

        Unknown currencies receive a neutral profile
        ``{rate_sensitivity: 0.5, risk_sensitivity: 0.5, commodity_link: 0.5}``.
        """
        base, quote = self._parse_symbol(symbol)

        r_base = self.CURRENCY_REGIMES.get(
            base,
            {"rate_sensitivity": 0.5, "risk_sensitivity": 0.5, "commodity_link": 0.5},
        )
        r_quote = self.CURRENCY_REGIMES.get(
            quote,
            {"rate_sensitivity": 0.5, "risk_sensitivity": 0.5, "commodity_link": 0.5},
        )

        return {
            "rate_sensitivity": r_base["rate_sensitivity"] * 0.6
            + r_quote["rate_sensitivity"] * 0.4,
            "risk_sensitivity": r_base["risk_sensitivity"] * 0.6
            + r_quote["risk_sensitivity"] * 0.4,
            "commodity_link": r_base["commodity_link"] * 0.6
            + r_quote["commodity_link"] * 0.4,
        }

    # ------------------------------------------------------------------
    # Volatility Structure Divergence (VSD)
    # ------------------------------------------------------------------

    def compute_vsd(self, symbol_a: str, symbol_b: str) -> float:
        """Compute Volatility Structure Divergence between two symbols.

        Uses static volatility profiles; unknown symbols default to 0.5.
        VSD = 1 - abs(vol_a - vol_b), clamped to ``[0, 1]``.

        Parameters
        ----------
        symbol_a : str
            First forex symbol.
        symbol_b : str
            Second forex symbol.

        Returns
        -------
        float
            VSD in ``[0, 1]``.  Higher = more similar volatility structure
            (lower divergence).
        """
        vol_a = self.VOLATILITY_PROFILES.get(symbol_a.upper(), 0.5)
        vol_b = self.VOLATILITY_PROFILES.get(symbol_b.upper(), 0.5)

        vsd = 1.0 - abs(vol_a - vol_b)
        return max(0.0, min(1.0, vsd))

    # ------------------------------------------------------------------
    # Core SIV Computation
    # ------------------------------------------------------------------

    def compute_siv(self, symbol_a: str, symbol_b: str) -> float:
        """Compute Signal Independence Value between two symbols.

        Steps:
        1. Build currency vectors for both symbols.
        2. Compute COC, MRD, and VSD.
        3. Combine: ``independence = 1 - (0.5 * COC + 0.3 * MRD + 0.2 * VSD)``
        4. Clamp result to ``[0, 1]``.

        SIV(sym, sym) = 1.0 (a symbol is fully independent of itself).

        Parameters
        ----------
        symbol_a : str
            First forex symbol.
        symbol_b : str
            Second forex symbol.

        Returns
        -------
        float
            SIV in ``[0, 1]``.  Higher value = more independent
            (less similar).
        """
        if symbol_a.upper() == symbol_b.upper():
            return 1.0

        vec_a = self.compute_currency_vector(symbol_a)
        vec_b = self.compute_currency_vector(symbol_b)

        coc = self.compute_coc(vec_a, vec_b)
        mrd = self.compute_mrd(symbol_a, symbol_b)
        vsd = self.compute_vsd(symbol_a, symbol_b)

        independence = 1.0 - (0.5 * coc + 0.3 * mrd + 0.2 * vsd)
        return max(0.0, min(1.0, independence))

    # ------------------------------------------------------------------
    # Pairwise SIV Matrix
    # ------------------------------------------------------------------

    def compute_pairwise_siv_matrix(
        self,
        symbols: list[str],
    ) -> dict[tuple[str, str], float]:
        """Compute SIV for all unique pairs in *symbols*.

        Both orderings are stored in the result dict so that
        ``result[(a, b)]`` and ``result[(b, a)]`` are always valid.

        Parameters
        ----------
        symbols : list[str]
            List of forex symbols.

        Returns
        -------
        dict[tuple[str, str], float]
            Mapping ``(sym_a, sym_b) -> SIV`` for every pair.
            SIV(sym, sym) = 1.0.
        """
        result: dict[tuple[str, str], float] = {}
        upper = [s.upper() for s in symbols]
        n = len(upper)
        for i in range(n):
            for j in range(i, n):
                siv = self.compute_siv(upper[i], upper[j])
                result[(symbols[i], symbols[j])] = siv
                if i != j:
                    result[(symbols[j], symbols[i])] = siv
        return result

    # ------------------------------------------------------------------
    # Effective Information Gain (EIG)
    # ------------------------------------------------------------------

    def compute_independence(self, sym_a: str, sym_b: str) -> float:
        """Alias for ``compute_siv`` — Signal Independence Value.

        Returns a score in ``[0, 1]`` where 1.0 means fully independent
        (no shared information) and 0.0 means perfectly correlated.
        """
        return self.compute_siv(sym_a, sym_b)

    def compute_pairwise_siv(self, sym_a: str, sym_b: str) -> float:
        """Alias for ``compute_siv``."""
        return self.compute_siv(sym_a, sym_b)

    def compute_eig(
        self,
        conviction: float,
        symbol: str,
        portfolio_symbols: list[str],
    ) -> float:
        """Compute Effective Information Gain for *symbol* vs a portfolio.

        ``EIG = conviction * average_independence``

        where *average_independence* is the mean SIV between *symbol*
        and every member of *portfolio_symbols*.  If the portfolio is
        empty, independence defaults to 1.0.

        Parameters
        ----------
        conviction : float
            Conviction score for the symbol, expected in ``[0, 1]``.
        symbol : str
            Candidate symbol to evaluate.
        portfolio_symbols : list[str]
            Current portfolio symbols (may be empty).

        Returns
        -------
        float
            EIG in ``[0, 1]``.
        """
        if not portfolio_symbols:
            avg_independence = 1.0
        else:
            siv_vals = [self.compute_siv(symbol, p) for p in portfolio_symbols]
            avg_independence = sum(siv_vals) / len(siv_vals)

        eig = conviction * avg_independence
        return max(0.0, min(1.0, eig))

    # ------------------------------------------------------------------
    # Portfolio Redundancy Cost
    # ------------------------------------------------------------------

    def redundancy_cost(self, candidate: str, portfolio: list[str]) -> float:
        """Compute the redundancy cost of *candidate* vs *portfolio*.

        Average SIV between the candidate and each portfolio member.
        Higher return = more redundant (less independent).

        Parameters
        ----------
        candidate : str
            Candidate symbol to evaluate.
        portfolio : list[str]
            Current portfolio symbols (may be empty).

        Returns
        -------
        float
            Redundancy cost in ``[0, 1]``.  Returns 0.0 for an empty
            portfolio.
        """
        if not portfolio:
            return 0.0

        siv_vals = [self.compute_siv(candidate, p) for p in portfolio]
        return sum(siv_vals) / len(siv_vals)

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------

    def get_full_analysis(
        self,
        symbol: str,
        portfolio_symbols: list[str],
        conviction: float = None,
    ) -> dict:
        """Produce a comprehensive analysis for *symbol*.

        Parameters
        ----------
        symbol : str
            Symbol to analyse.
        portfolio_symbols : list[str]
            Current portfolio symbols.
        conviction : float, optional
            Conviction score.  If ``None``, defaults to the average
            independence of *symbol* vs the portfolio.

        Returns
        -------
        dict
            Keys:
              - **symbol**            -- the input symbol
              - **currency_vector**    -- output of ``compute_currency_vector``
              - **portfolio_size**    -- ``len(portfolio_symbols)``
              - **avg_independence**  -- mean SIV vs each portfolio member
              - **eig**               -- effective information gain
              - **redundancy_cost**   -- output of ``redundancy_cost``
        """
        currency_vector = self.compute_currency_vector(symbol)

        if portfolio_symbols:
            siv_vals = [self.compute_siv(symbol, p) for p in portfolio_symbols]
            avg_independence = sum(siv_vals) / len(siv_vals)
            red_cost = self.redundancy_cost(symbol, portfolio_symbols)
        else:
            avg_independence = 1.0
            red_cost = 0.0

        if conviction is None:
            conviction = avg_independence

        eig = self.compute_eig(conviction, symbol, portfolio_symbols)

        return {
            "symbol": symbol,
            "currency_vector": currency_vector,
            "portfolio_size": len(portfolio_symbols),
            "avg_independence": avg_independence,
            "eig": eig,
            "redundancy_cost": red_cost,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_SIV_ENGINE_INSTANCE = SIVEngine()


def parse_currency_vector(symbol: str) -> dict[str, float]:
    """Parse a forex symbol into a ``{currency: exposure}`` vector.

    Parameters
    ----------
    symbol : str
        Forex symbol, e.g. ``"GBPAUD"``.

    Returns
    -------
    dict[str, float]
        Vector with ``+1.0`` for base, ``-1.0`` for quote, ``0.0`` otherwise.
    """
    return _SIV_ENGINE_INSTANCE.compute_currency_vector(symbol)


def compute_pairwise_siv(sym_a: str, sym_b: str) -> float:
    """Compute pairwise SIV between two symbols.

    Parameters
    ----------
    sym_a : str
        First forex symbol.
    sym_b : str
        Second forex symbol.

    Returns
    -------
    float
        SIV score in ``[0, 1]``.
    """
    return _SIV_ENGINE_INSTANCE.compute_siv(sym_a, sym_b)
